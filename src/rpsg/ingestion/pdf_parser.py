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
from pathlib import Path

import httpx

from rpsg.ingestion.chunking import Section, classify_section
from rpsg.logging import get_logger

log = get_logger(__name__)

# A heading-like line: short, title-ish, optionally numbered. Heuristic; PyMuPDF path only.
_HEADING = re.compile(r"^\s*(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:&/-]{2,60}\s*$")


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

    log.info("GROBID parsed %s -> %d sections", pdf_path.name, len(sections))
    return sections


def parse_with_pymupdf(pdf_path: Path) -> list[Section]:
    """Heuristic fallback. Splits text on heading-like lines; no reliable section tree."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    sections: list[Section] = []
    title = "Body"
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(title=title, text=text, section_type=classify_section(title)))

    for line in full_text.splitlines():
        if _HEADING.match(line) and len(line.split()) <= 8:
            flush()
            title = line.strip()
            buffer = []
        else:
            buffer.append(line)
    flush()

    log.info("PyMuPDF parsed %s -> %d sections (heuristic)", pdf_path.name, len(sections))
    return sections


def parse_pdf(pdf_path: Path, grobid_url: str | None = None) -> list[Section]:
    """Parse a PDF, preferring GROBID and falling back to PyMuPDF on any failure."""
    if grobid_url:
        try:
            return parse_with_grobid(pdf_path, grobid_url)
        except Exception as exc:  # noqa: BLE001 - degrade rather than lose the paper
            log.warning("GROBID failed for %s (%s); falling back to PyMuPDF", pdf_path.name, exc)
    return parse_with_pymupdf(pdf_path)
