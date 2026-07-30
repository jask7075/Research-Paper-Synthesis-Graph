"""ArXiv search + PDF retrieval.

Two roles:
    search_arxiv()  Corpus discovery with no API key. ArXiv's API is open, and for
                    quant-ph / cs.LG / cs.CL its coverage of the full text is total —
                    every hit has a PDF, unlike S2 where only some do.
    fetch_pdf()     Download one paper. Parsing is `pdf_parser`'s job.

Choosing between this and Semantic Scholar:
    S2     needs a key (the unauthenticated pool returns a sustained 429) but supplies
           `references` -> the Tier-A `cites` edges the citation-graph baseline needs,
           plus disambiguated author ids.
    ArXiv  needs no key and guarantees full text, but returns no reference lists and no
           author ids, so `to_graph` emits no `cites` and no Author nodes.

They compose: build the corpus from ArXiv now, then once a key arrives backfill S2
metadata by ArXiv id (S2 accepts `/paper/ARXIV:2301.12345`) to recover the citation
edges without re-downloading anything.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rpsg.ingestion.semantic_scholar import S2Author, S2Paper
from rpsg.logging import get_logger

log = get_logger(__name__)

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"

#: Trailing version marker on an ArXiv id ("2010.05863v1" -> "2010.05863"). Stripped so
#: re-ingesting a paper that has since been revised does not create a second record.
_VERSION_SUFFIX = re.compile(r"v\d+$")


def _fs_safe(arxiv_id: str) -> str:
    """Make an ArXiv id usable as a filename.

    Pre-2007 ids carry a slash ("quant-ph/9512022"), which would otherwise be read as a
    directory separator when the PDF is written to `<dir>/<paper_id>.pdf`. quant-ph has a
    lot of foundational pre-2007 work, so this is not a hypothetical.
    """
    return arxiv_id.replace("/", "_")


def search_arxiv(
    query: str,
    limit: int = 100,
    category: str | None = "quant-ph",
    min_year: int | None = None,
    sort: str = "relevance",
    page_size: int = 100,
) -> list[S2Paper]:
    """Search ArXiv and return records in the pipeline's paper-metadata shape.

    `S2Paper` is reused deliberately: it is the pipeline's paper record regardless of
    where the metadata came from, so nothing downstream needs to know the source. The
    fields ArXiv cannot supply are left empty rather than guessed — `references` stays
    `[]` (no citation edges) and authors carry no `authorId` (so `to_graph` skips them
    rather than inventing an identity).

    `min_year` is pushed into the ArXiv query as a `submittedDate` range rather than
    filtered client-side, so it does not silently eat into `limit`.
    """
    import arxiv

    terms = [query]
    if category:
        terms.insert(0, f"cat:{category}")
    if min_year:
        terms.append(f"submittedDate:[{min_year}0101 TO 20991231]")
    full_query = " AND ".join(terms)

    criterion = (
        arxiv.SortCriterion.SubmittedDate if sort == "date" else arxiv.SortCriterion.Relevance
    )
    client = arxiv.Client(page_size=min(page_size, limit), delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(query=full_query, max_results=limit, sort_by=criterion)

    papers: list[S2Paper] = []
    for result in client.results(search):
        arxiv_id = _VERSION_SUFFIX.sub("", result.get_short_id())
        papers.append(
            S2Paper(
                paperId=_fs_safe(arxiv_id),
                title=result.title,
                abstract=result.summary,
                year=result.published.year if result.published else None,
                venue="arXiv",
                authors=[S2Author(authorId=None, name=a.name) for a in result.authors],
                references=[],
                openAccessPdf={"url": result.pdf_url} if result.pdf_url else None,
                externalIds={"ArXiv": arxiv_id, "DOI": result.doi},
            )
        )
    log.info("ArXiv search %r -> %d papers", full_query, len(papers))
    return papers


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for block in resp.iter_bytes():
                fh.write(block)


#: Smallest plausible PDF. Anything under this is a truncated or error response.
_MIN_PDF_BYTES = 1024


def _is_usable_pdf(path: Path) -> bool:
    """True when the file exists, is big enough, and starts with the PDF magic bytes."""
    try:
        if path.stat().st_size < _MIN_PDF_BYTES:
            return False
        with path.open("rb") as fh:
            return fh.read(5).startswith(b"%PDF")
    except OSError:
        return False


def fetch_pdf(
    arxiv_id: str | None,
    pdf_url: str | None,
    dest_dir: Path,
    paper_id: str,
    *,
    polite_delay: float = 3.0,
) -> Path | None:
    """Download a paper PDF. Prefers ArXiv, falls back to the S2 open-access link.

    Returns the local path, or None if no source was available. Idempotent: an existing
    non-empty file is reused, so the pipeline is safely re-runnable.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper_id}.pdf"
    if dest.exists():
        if _is_usable_pdf(dest):
            return dest  # idempotent: a good file is reused
        dest.unlink(missing_ok=True)  # a bad file from an earlier run is retried

    url = ARXIV_PDF.format(arxiv_id=arxiv_id) if arxiv_id else pdf_url
    if not url:
        log.debug("No PDF source for %s", paper_id)
        return None

    try:
        _download(url, dest)
    except Exception as exc:  # noqa: BLE001 - a missing PDF must not kill the batch
        log.warning("PDF download failed for %s (%s): %s", paper_id, url, exc)
        dest.unlink(missing_ok=True)
        return None

    # `raise_for_status` is not enough. A 200 with an empty body still opens (and so
    # creates) the destination file, leaving a 0-byte PDF that raises nothing here and
    # then aborts the parse stage later. An HTML error page served as a PDF fails the
    # same way, hence the magic-byte check.
    if not _is_usable_pdf(dest):
        log.warning("PDF for %s was empty or not a PDF (%s); discarding", paper_id, url)
        dest.unlink(missing_ok=True)
        return None

    time.sleep(polite_delay)  # ArXiv asks for ~1 request per 3s
    return dest
