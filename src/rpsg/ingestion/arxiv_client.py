"""ArXiv PDF retrieval.

S2 gives metadata and (sometimes) an open-access PDF link; ArXiv is the reliable
full-text source for cs.LG / cs.CL / quant-ph. This module only downloads — parsing is
`pdf_parser`'s job.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rpsg.logging import get_logger

log = get_logger(__name__)

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"


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
    if dest.exists() and dest.stat().st_size > 0:
        return dest

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

    time.sleep(polite_delay)  # ArXiv asks for ~1 request per 3s
    return dest
