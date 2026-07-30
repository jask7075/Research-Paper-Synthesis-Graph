"""PDF -> ordered `Section` list.

Two backends:
    GROBID (preferred)  Structured TEI with a real section tree; keeps the Appendix.
                        Run it locally: `make grobid`. This is what you want for the
                        2,000-paper batch — it gives clean section boundaries that the
                        chunker and extraction-routing depend on.
    PyMuPDF (fallback)  No section model; heuristically splits on heading-like lines.
                        Fine for a quick smoke test, not for the real corpus.

Both return `list[Section]` so the rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import httpx

from rpsg.ingestion.chunking import Section, classify_section, refine_section_types
from rpsg.logging import get_logger

log = get_logger(__name__)

# --- PyMuPDF heading detection -------------------------------------------------
# Headings are found by TYPOGRAPHY (font size / weight), not by text shape. An
# earlier text-shape rule ("short line starting with a capital") misfired badly on
# two-column PDFs: wrapped body fragments and bibliography author surnames read as
# headings, so an 8-page paper split into 44 pseudo-sections with 36 of them typed
# `other` — which silently defeats the section-routed extraction in
# `rpsg.extraction.prompts` and let reference lists into the chunk corpus.

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold
_MAX_HEADING_CHARS = 90
_MAX_HEADING_WORDS = 12
_MIN_HEADING_LETTERS = 3
#: Figure/table captions are frequently bold and would otherwise split a section.
_CAPTION = re.compile(r"^\s*(fig(ure)?|tab(le)?)\b\.?\s*\d", re.I)
#: A numbered reference-list entry, e.g. "[12] J. Doe, ...".
_REF_ENTRY = re.compile(r"^\s*\[\d{1,3}\]")
_REF_RUN_WINDOW = 12
_REF_RUN_MIN = 3


def _split_bibliography(buffer: list[str]) -> tuple[list[str], list[str]]:
    """Split a trailing reference list off a section body.

    Many preprints carry no `References` heading at all — the numbered list simply
    follows the conclusion — so heading detection alone leaves the bibliography
    glued to the last real section (observed: 51 `[n]` entries inside a 10k-char
    "Conclusions"). Cut at the first `[n]` line that begins a *run* of them, so a
    lone wrapped citation in body prose is not mistaken for the list.
    """
    for i, line in enumerate(buffer):
        if not _REF_ENTRY.match(line):
            continue
        window = buffer[i : i + _REF_RUN_WINDOW]
        if sum(1 for candidate in window if _REF_ENTRY.match(candidate)) >= _REF_RUN_MIN:
            return buffer[:i], buffer[i:]
    return buffer, []


class _Line:
    """One rendered line with the typography needed to judge heading-ness."""

    __slots__ = ("text", "size", "bold")

    def __init__(self, text: str, size: float, bold: bool) -> None:
        self.text = text
        self.size = size
        self.bold = bold


def _lines(doc) -> list[_Line]:
    """Flatten a document to `_Line`s in PyMuPDF's natural block order.

    Blocks are deliberately NOT re-sorted: PyMuPDF already emits two-column pages
    in reading order, and sorting by y-coordinate would interleave the columns.
    """
    out: list[_Line] = []
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0 == text; skip images
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                # Text uses EVERY span: some PDFs emit inter-word spaces as their
                # own whitespace-only spans, so filtering those first would glue
                # words together ("Barrenplateausin...").
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                inked = [s for s in spans if s.get("text", "").strip()]
                if not inked:
                    continue
                out.append(
                    _Line(
                        text=text,
                        size=round(max(float(s.get("size", 0.0)) for s in inked), 1),
                        bold=all(int(s.get("flags", 0)) & _BOLD_FLAG for s in inked),
                    )
                )
    return out


def _body_size(lines: list[_Line]) -> float:
    """The dominant font size, weighted by characters — i.e. body text."""
    hist: Counter[float] = Counter()
    for line in lines:
        hist[line.size] += len(line.text)
    return hist.most_common(1)[0][0] if hist else 0.0


def _is_heading(line: _Line, body_size: float) -> bool:
    """True when a line looks like a section heading on typographic evidence."""
    text = line.text
    if len(text) > _MAX_HEADING_CHARS or len(text.split()) > _MAX_HEADING_WORDS:
        return False
    if _CAPTION.match(text):
        return False
    # Mostly-symbolic lines are equation fragments, page numbers, running heads.
    letters = sum(c.isalpha() for c in text)
    if letters < _MIN_HEADING_LETTERS or letters / len(text) < 0.5:
        return False
    # A fully-bold short line is a heading regardless of its size. Do NOT gate this
    # on `size >= body_size`: in two-column preprints the abstract is often set a
    # point larger than the body, so `body_size` lands above the heading size and
    # such a gate rejects every real heading (observed: 9pt bold CMBX9 headings in
    # a paper whose dominant font is 10pt).
    if line.bold:
        return True
    return line.size >= body_size + 0.5


def parse_with_grobid(pdf_path: Path, grobid_url: str, timeout: float = 120.0) -> list[Section]:
    """Send the PDF to a GROBID `processFulltextDocument` endpoint and parse the TEI."""
    from xml.etree import ElementTree as ET  # noqa: N817 - stdlib TEI parsing

    with pdf_path.open("rb") as fh:
        resp = httpx.post(
            f"{grobid_url}/api/processFulltextDocument",
            files={"input": (pdf_path.name, fh, "application/pdf")},
            data={"segmentSentences": "0"},
            timeout=timeout,
        )
    resp.raise_for_status()

    # `raise_for_status` is not sufficient. A misconfigured or wrong host can answer 2xx
    # with an HTML landing page — observed: a Hugging Face Space returned HTTP 206 and
    # 4KB of HTML, which parsed without error and yielded zero sections. Silently
    # treating that as a successful parse would produce an empty corpus for every paper.
    content_type = resp.headers.get("content-type", "")
    if "xml" not in content_type.lower():
        raise ValueError(
            f"GROBID returned content-type {content_type!r}, not XML — "
            f"{grobid_url} is probably not a GROBID service"
        )

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(resp.text)
    sections: list[Section] = []

    # Abstract
    abstract_el = root.find(".//tei:profileDesc//tei:abstract", ns)
    if abstract_el is not None:
        text = " ".join(t.strip() for t in abstract_el.itertext() if t.strip())
        if text:
            sections.append(Section(title="Abstract", text=text, section_type="abstract"))

    # Body divisions
    for div in root.findall(".//tei:body//tei:div", ns):
        head = div.find("tei:head", ns)
        title = "".join(head.itertext()).strip() if head is not None else "Section"
        paras = [
            " ".join(t.strip() for t in p.itertext() if t.strip())
            for p in div.findall("tei:p", ns)
        ]
        text = "\n".join(x for x in paras if x)
        if text:
            sections.append(
                Section(title=title, text=text, section_type=classify_section(title))
            )

    if not sections:
        raise ValueError(f"GROBID returned no sections for {pdf_path.name}")

    log.info("GROBID parsed %s -> %d sections", pdf_path.name, len(sections))
    return sections


def parse_with_pymupdf(pdf_path: Path) -> list[Section]:
    """Local fallback: split on typographically-detected headings.

    Good enough to run the pipeline without GROBID — which matters because the
    GROBID image is amd64-only and cannot spawn its native `pdfalto` helper under
    Apple-Silicon emulation. Still not a real section model: GROBID remains the
    right choice for a large batch where you can run it natively.

    Bibliographies are handled by detecting the `References` heading and then
    suppressing further generic headings, so stray bold author names cannot end
    the references block early. The chunker drops `references` sections outright
    (`DROP_SECTION_TYPES`), while a later `Appendix` heading still resumes normal
    collection — appendices follow the bibliography in most preprints and carry
    the reproducibility facts.
    """
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        lines = _lines(doc)

    body_size = _body_size(lines)
    sections: list[Section] = []
    title = "Front Matter"
    buffer: list[str] = []
    in_references = False

    def flush() -> None:
        body, refs = _split_bibliography(buffer)
        text = "\n".join(body).strip()
        if text:
            sections.append(Section(title=title, text=text, section_type=classify_section(title)))
        if refs:
            # Emitted as `references` so the chunker drops it (DROP_SECTION_TYPES).
            sections.append(
                Section(title="References", text="\n".join(refs).strip(), section_type="references")
            )

    for line in lines:
        if not _is_heading(line, body_size):
            buffer.append(line.text)
            continue
        section_type = classify_section(line.text)
        # Inside the bibliography, only a recognised section resumes collection.
        if in_references and section_type in ("other", "references"):
            buffer.append(line.text)
            continue
        flush()
        title = line.text
        buffer = []
        in_references = section_type == "references"
    flush()

    kept = [s for s in sections if s.section_type != "references"]
    log.info(
        "PyMuPDF parsed %s -> %d sections (body font %.1fpt; %d references section(s) found)",
        pdf_path.name,
        len(sections),
        body_size,
        len(sections) - len(kept),
    )
    return sections


def parse_pdf(pdf_path: Path, grobid_url: str | None = None) -> list[Section]:
    """Parse a PDF, preferring GROBID and falling back to PyMuPDF on any failure.

    "Failure" includes a 2xx response that is not XML and a parse that yields zero
    sections — both raise in `parse_with_grobid` so they reach the fallback here.
    A degraded parse is recoverable; a silently empty one is not.
    """
    if grobid_url:
        try:
            return refine_section_types(parse_with_grobid(pdf_path, grobid_url))
        except Exception as exc:  # noqa: BLE001 - degrade rather than lose the paper
            log.warning("GROBID failed for %s (%s); falling back to PyMuPDF", pdf_path.name, exc)
    # Applied to both backends: GROBID gives better titles but is equally unable to
    # invent a `Conclusions` heading a paper does not have.
    return refine_section_types(parse_with_pymupdf(pdf_path))
