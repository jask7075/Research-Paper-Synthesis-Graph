"""Dry-run a single PDF through parse -> chunk and print what comes out.

    python scripts/inspect_pdf.py data/raw/pdfs/1803.11173.pdf
    python scripts/inspect_pdf.py paper.pdf --show-text      # include chunk previews
    python scripts/inspect_pdf.py paper.pdf --no-grobid      # force the local fallback

Writes nothing. This is the tool for answering "what does the system actually see
when I drop this PDF in?" before spending an extraction batch on 200 of them —
section_type drives extraction routing (`rpsg.extraction.prompts`), so a corpus
where everything lands in `other` will quietly produce a graph with no Limitation
nodes and no refutes/undercuts edges.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from rpsg.config import get_settings
from rpsg.extraction.prompts import build_user_prompt
from rpsg.ingestion.chunking import DROP_SECTION_TYPES, approx_tokens, chunk_paper
from rpsg.ingestion.pdf_parser import parse_pdf


def _routed_types(section_type: str) -> str:
    """What extraction would be asked to produce for this section type."""
    prompt = build_user_prompt("x", "t", section_type, "body")
    lines = [ln for ln in prompt.splitlines() if ln.startswith("Extract these")]
    return " | ".join(ln.replace("Extract these ", "") for ln in lines) or "(none)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--show-text", action="store_true", help="print a preview of each chunk")
    ap.add_argument("--no-grobid", action="store_true", help="force the PyMuPDF fallback")
    ap.add_argument("--preview", type=int, default=160, help="preview length in chars")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"no such file: {args.pdf}")

    settings = get_settings()
    paper_id = args.pdf.stem
    sections = parse_pdf(args.pdf, grobid_url=None if args.no_grobid else settings.grobid_url)
    chunks = chunk_paper(
        paper_id,
        None,  # no abstract: that comes from the S2 metadata in stage 01
        sections,
        target_tokens=settings.chunking.target_tokens,
        overlap_tokens=settings.chunking.overlap_tokens,
    )

    print(f"\n{'=' * 78}\n{args.pdf.name}  ->  {len(sections)} sections, {len(chunks)} chunks")
    print(f"{'=' * 78}")

    by_section: dict[str, list] = collections.defaultdict(list)
    for chunk in chunks:
        by_section[chunk.section_title].append(chunk)

    for section in sections:
        dropped = section.section_type in DROP_SECTION_TYPES
        mark = "  [DROPPED — not chunked, not extracted]" if dropped else ""
        print(f"\n{section.section_type.upper():<14} {section.title[:56]!r}{mark}")
        print(f"{'':14} {len(section.text):,} chars, ~{approx_tokens(section.text)} tokens")
        if not dropped:
            print(f"{'':14} extraction asks for: {_routed_types(section.section_type)}")
        for chunk in by_section.get(section.title, []):
            print(
                f"{'':14}   chunk {chunk.char_start}-{chunk.char_end} "
                f"(~{approx_tokens(chunk.text)} tok)  id={chunk.id}"
            )
            if args.show_text:
                preview = " ".join(chunk.text.split())[: args.preview]
                print(f"{'':14}     {preview}…")

    print(f"\n{'-' * 78}\nsummary")
    counts = collections.Counter(s.section_type for s in sections)
    for stype, n in counts.most_common():
        flag = "  (dropped)" if stype in DROP_SECTION_TYPES else ""
        print(f"  sections {stype:<14} {n}{flag}")
    kept = [s for s in sections if s.section_type not in DROP_SECTION_TYPES]
    other = counts.get("other", 0)
    print(f"  extraction API calls this paper would cost: {len(kept)}")
    if other:
        print(
            f"  !! {other}/{len(sections)} sections are 'other' — those get the default "
            "node types only (no Limitation, no refutes/undercuts)"
        )


if __name__ == "__main__":
    main()
