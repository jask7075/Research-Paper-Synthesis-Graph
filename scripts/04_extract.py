"""Extract Tier B/C + reproducibility nodes/edges (the one-time API batch).

    python scripts/04_extract.py            # needs ANTHROPIC_API_KEY

Reads  data/interim/sections/<paper_id>.json
Writes data/processed/extractions.jsonl     (one ExtractionResult per paper)

Cost note: this is the run where paying for a frontier-small API model buys you out of the
extraction-quality bottleneck. ~$20-45 over 2,000 papers on claude-haiku-4-5 (half that on
the Batch API). Idempotent per paper so it is safe to re-run after prompt changes.
"""

from __future__ import annotations

import json

from rpsg.config import get_settings
from rpsg.extraction.extractor import Extractor
from rpsg.ingestion.chunking import Section
from rpsg.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set (see .env.example).")

    sect_dir = settings.paths.data_interim / "sections"
    out = settings.paths.data_processed / "extractions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        done = {json.loads(line)["paper_id"] for line in out.read_text().splitlines() if line}

    extractor = Extractor(model=settings.models.extraction_model)
    with out.open("a") as fh:
        for sect_file in sorted(sect_dir.glob("*.json")):
            paper_id = sect_file.stem
            if paper_id in done:
                continue
            sections = [Section(**s) for s in json.loads(sect_file.read_text())]
            result = extractor.extract_paper(paper_id, sections)
            fh.write(result.model_dump_json() + "\n")
            tiers = result.by_tier()
            log.info("extracted %s: %d nodes %d edges %s", paper_id,
                     len(result.nodes), len(result.edges), tiers)


if __name__ == "__main__":
    main()
