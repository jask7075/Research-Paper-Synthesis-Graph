"""Section-aware chunking of parsed papers.

    python scripts/03_chunk.py

Reads  data/interim/sections/<paper_id>.json  + abstracts from data/external/papers.jsonl
Writes data/interim/chunks.jsonl              (all chunks, both corpora)
"""

from __future__ import annotations

import json

from rpsg.config import get_settings
from rpsg.ingestion.chunking import Section, chunk_paper
from rpsg.logging import get_logger

log = get_logger(__name__)


def _abstracts() -> dict[str, str]:
    settings = get_settings()
    path = settings.paths.data_external / "papers.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line:
            rec = json.loads(line)
            out[rec["paperId"]] = rec.get("abstract") or ""
    return out


def main() -> None:
    settings = get_settings()
    sect_dir = settings.paths.data_interim / "sections"
    abstracts = _abstracts()
    out = settings.paths.data_interim / "chunks.jsonl"

    n_chunks = 0
    with out.open("w") as fh:
        for sect_file in sorted(sect_dir.glob("*.json")):
            paper_id = sect_file.stem
            sections = [Section(**s) for s in json.loads(sect_file.read_text())]
            chunks = chunk_paper(
                paper_id,
                abstracts.get(paper_id),
                sections,
                target_tokens=settings.chunking.target_tokens,
                overlap_tokens=settings.chunking.overlap_tokens,
            )
            for c in chunks:
                fh.write(c.model_dump_json() + "\n")
            n_chunks += len(chunks)
    log.info("wrote %d chunks -> %s", n_chunks, out)


if __name__ == "__main__":
    main()