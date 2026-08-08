"""Populate cross-paper `refutes` / `undercuts` edges. Costs money; run deliberately.

    python scripts/find_contradictions.py --dry-run      # count candidates, call nothing
    python scripts/find_contradictions.py --floor 0.90
    python scripts/find_contradictions.py --review       # print accepted edges and exit

Reads  data/processed/extractions.jsonl
Writes data/processed/contradictions.json  (stage 05 applies it on the next build)

Kept out of `05_build_stores.py` for the same reason `merge_entities.py` is: a store
rebuild should be free and repeatable, and adjudication is neither. The build reads this
file if it exists and skips the pass if it does not.

Why this exists: per-paper extraction can only see contradictions a paper states about
itself, which yielded 8 `refutes` and 25 `undercuts` across the corpus. Measured
consequence -- 1 of 9 refutation gold queries surfaces its contradiction, on every arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.extraction.contradiction import (
    adjudicate,
    candidate_pairs,
    summarize,
    to_edges,
)
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger
from rpsg.stores.embedder import SentenceTransformerEmbedder

log = get_logger(__name__)


def _nodes(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            for n in json.loads(line)["nodes"]:
                out.setdefault(n["id"], n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.90, help="candidate similarity floor")
    ap.add_argument("--neighbours", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="adjudicate only the top N candidates")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    out_path = settings.paths.data_processed / "contradictions.json"
    cache_path = settings.paths.data_processed / "contradiction_verdicts.json"

    if args.review:
        if not out_path.exists():
            raise SystemExit(f"no edges at {out_path}")
        data = json.loads(out_path.read_text())
        print(f"{len(data['edges'])} edges\n")
        for e in data["edges"][:50]:
            print(f"  [{e['type']:9} {e['confidence']:.3f}] {e['src'][:44]}")
            print(f"                      -> {e['dst'][:44]}")
            if e.get("evidence"):
                print(f"      {e['evidence'][0][:88]}")
        return

    nodes = _nodes(settings.paths.data_processed / "extractions.jsonl")
    embedder = SentenceTransformerEmbedder(
        settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
    )
    pairs = candidate_pairs(
        list(nodes.values()), embedder, floor=args.floor, neighbours=args.neighbours
    )
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} cross-paper candidate pairs above {args.floor}")
    if args.dry_run:
        print(f"~${len(pairs) * 0.00012:.2f} to adjudicate if none are cached")
        return

    found = adjudicate(
        pairs,
        nodes,
        model=settings.models.extraction_model,
        cache_path=cache_path,
        workers=args.workers,
    )
    edges = to_edges(found)
    out_path.write_text(
        json.dumps({"floor": args.floor, "approved": False, "edges": edges}, indent=2)
    )
    print("\n" + summarize(found))
    print(f"\nedges -> {out_path}, marked approved: false")
    print("audit with scripts/audit_contradictions.py before setting approved: true")
    print("\n" + USAGE.summary())


if __name__ == "__main__":
    main()