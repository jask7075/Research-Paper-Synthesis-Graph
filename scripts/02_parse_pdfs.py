"""Parse downloaded PDFs into section lists.

    python scripts/02_parse_pdfs.py

Reads  data/raw/pdfs/*.pdf
Writes data/interim/sections/<paper_id>.json   (list[Section])
Prefers GROBID (RPSG_GROBID_URL) and falls back to PyMuPDF.
"""

from __future__ import annotations

import json

from rpsg.config import get_settings
from rpsg.ingestion.pdf_parser import parse_pdf
from rpsg.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    pdf_dir = settings.paths.data_raw / "pdfs"
    out_dir = settings.paths.data_interim / "sections"
    out_dir.mkdir(parents=True, exist_ok=True)

    grobid = settings.grobid_url
    pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    log.info("parsing %d PDFs (grobid=%s)", len(pdfs), grobid)

    for pdf in pdfs:
        paper_id = pdf.stem
        out = out_dir / f"{paper_id}.json"
        if out.exists():
            continue  # idempotent
        sections = parse_pdf(pdf, grobid_url=grobid)
        out.write_text(json.dumps([s.model_dump() for s in sections], indent=2))
    log.info("done")


if __name__ == "__main__":
    main()