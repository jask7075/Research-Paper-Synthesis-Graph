"""Extract Tier B/C + reproducibility nodes/edges (the one-time API batch).

    python scripts/04_extract.py                      # resume: skips papers already done
    python scripts/04_extract.py --papers eval/gold/repro_gold.jsonl --out /tmp/x.jsonl

Reads  data/interim/sections/<paper_id>.json
Writes data/processed/extractions.jsonl     (one ExtractionResult per paper)

Cost note: this is the run where paying for a hosted model buys you out of the
extraction-quality bottleneck. Single-digit dollars over a 200-paper corpus on a
small model. Idempotent per paper so it is safe to re-run after prompt changes.

`--papers` + `--out` exist to price a routing change before buying it. A prompt edit
invalidates every stored extraction, but re-running 271 papers to find out whether the edit
worked is the expensive way round: the repro layer is scored against 21 gold papers, so
re-extracting those 21 into a scratch file measures the change for under a tenth of the
cost and leaves the corpus extraction untouched while the answer is still unknown.

Note that `--out` writes a PARTIAL corpus. `05_build_stores.py` must never be pointed at
one -- a graph built from 21 papers would silently score as if the other 250 held nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.extraction.extractor import Extractor
from rpsg.ingestion.chunking import Section
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger

log = get_logger(__name__)


def _wanted(spec: str) -> set[str]:
    """Paper ids from a jsonl with `paper_id`, or a comma-separated list of ids."""
    p = Path(spec)
    if p.exists():
        return {
            json.loads(line)["paper_id"]
            for line in p.read_text().splitlines()
            if line.strip()
        }
    return {x.strip() for x in spec.split(",") if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=None,
                    help="restrict to these ids (a jsonl with paper_id, or id,id,...)")
    ap.add_argument("--out", default=None, help="write here instead of extractions.jsonl")
    args = ap.parse_args()

    settings = get_settings()
    # The API-key check lives in the provider adapter (rpsg.llm), so it stays
    # correct whichever provider `models.extraction_model` routes to.
    sect_dir = settings.paths.data_interim / "sections"
    out = Path(args.out) if args.out else settings.paths.data_processed / "extractions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    wanted = _wanted(args.papers) if args.papers else None

    done = set()
    if out.exists():
        done = {json.loads(line)["paper_id"] for line in out.read_text().splitlines() if line}

    extractor = Extractor(model=settings.models.extraction_model)
    with out.open("a") as fh:
        for sect_file in sorted(sect_dir.glob("*.json")):
            paper_id = sect_file.stem
            if paper_id in done or (wanted is not None and paper_id not in wanted):
                continue
            sections = [Section(**s) for s in json.loads(sect_file.read_text())]
            result = extractor.extract_paper(paper_id, sections)
            fh.write(result.model_dump_json() + "\n")
            tiers = result.by_tier()
            log.info("extracted %s: %d nodes %d edges %s", paper_id,
                     len(result.nodes), len(result.edges), tiers)

    if wanted is not None:
        missing = wanted - {
            json.loads(line)["paper_id"] for line in out.read_text().splitlines() if line
        }
        if missing:
            log.warning("%d requested papers have no sections file: %s",
                        len(missing), ", ".join(sorted(m[:10] for m in missing)))
    print("\n" + USAGE.summary())


if __name__ == "__main__":
    main()
