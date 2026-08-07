"""Populate the semantic-merge verdict cache. Costs money; run deliberately.

    python scripts/merge_entities.py --floor 0.95        # ~3,500 pairs, ~$0.30
    python scripts/merge_entities.py --dry-run           # candidates only, no API calls
    python scripts/merge_entities.py --review            # print accepted merges and exit

Reads  data/processed/extractions.jsonl
Writes data/processed/merge_verdicts.json  (stage 05 applies it on the next build)

Kept out of `05_build_stores.py` on purpose: a store rebuild should be free and
repeatable, and adjudication is neither. The build reads this cache if it exists and skips
the tier if it does not, so the expensive step happens when asked for rather than as a side
effect of rebuilding.

Why two stages at all: on this corpus embedding similarity cannot separate same from
different. Nearest-neighbour cosine over 6,193 Method nodes never falls below 0.70, and
"Gradient-free ... optimization" scores 0.993 against "Gradient-based ... optimization"
while a genuine QMDD spelling variant scores 0.976. Embeddings supply recall; the model
supplies the decision, rejecting ~67% of what they nominate.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.extraction.semantic_merge import adjudicate, candidate_pairs, merge_map
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
    ap.add_argument("--floor", type=float, default=0.95, help="candidate similarity floor")
    ap.add_argument("--neighbours", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count candidates, call nothing")
    ap.add_argument("--review", action="store_true", help="print cached accepted merges")
    args = ap.parse_args()

    settings = get_settings()
    cache_path = settings.paths.data_processed / "merge_verdicts.json"

    if args.review:
        if not cache_path.exists():
            raise SystemExit(f"no cache at {cache_path}")
        cache = json.loads(cache_path.read_text())
        accepted = [(k, v) for k, v in cache.items() if v.get("same")]
        print(f"{len(accepted)} accepted of {len(cache)} adjudicated\n")
        for key, verdict in accepted[:60]:
            a, _, b = key.partition("␟")
            print(f"  {a[:46]:48} == {b[:46]}")
            if verdict.get("reason"):
                print(f"      {verdict['reason'][:88]}")
        return

    nodes = _nodes(settings.paths.data_processed / "extractions.jsonl")
    embedder = SentenceTransformerEmbedder(
        settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
    )
    pairs = candidate_pairs(
        list(nodes.values()), embedder, floor=args.floor, neighbours=args.neighbours
    )
    print(f"{len(pairs)} candidate pairs above {args.floor}")
    if args.dry_run:
        # Rough only: cached pairs cost nothing, and this cannot know which are cached
        # without loading the cache, which --dry-run deliberately does not touch.
        print(f"~${len(pairs) * 0.00009:.2f} to adjudicate if none are cached")
        return

    verdicts = adjudicate(
        pairs,
        {i: n["name"] for i, n in nodes.items()},
        model=settings.models.extraction_model,
        cache_path=cache_path,
        workers=args.workers,
    )
    accepted = [v for v in verdicts if v.same]
    mapping = merge_map(verdicts)
    print(f"accepted {len(accepted)}/{len(verdicts)} ({len(accepted) / len(verdicts):.1%})")
    print(f"ids merged {len(mapping)} ({len(mapping) / len(nodes):.2%} of {len(nodes):,})")
    print("by type:", dict(collections.Counter(k.split(":")[0] for k in mapping)))
    print(f"\ncache -> {cache_path}; stage 05 applies it on the next build")
    print("\n" + USAGE.summary())


if __name__ == "__main__":
    main()