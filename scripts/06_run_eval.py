"""Score a system against the gold set — the Iteration 1 exit criterion.

    python scripts/06_run_eval.py --system vector_fulltext
    python scripts/06_run_eval.py --system vector_abstract --no-judge --hash-embed

Reads  eval/gold/queries.jsonl, the built vector store
Writes eval/runs/<timestamp>_<system>/{answers,traces,scores}.jsonl + report.md
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.gold_schema import load_gold
from rpsg.eval.runner import run_system
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger
from rpsg.retrieval.build import SYSTEMS as _SYSTEMS
from rpsg.retrieval.build import build_system
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(_SYSTEMS), default="vector_fulltext")
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--gold", default="queries.jsonl",
                    help="gold file under eval/gold. The active 10 by default; §3.5 scores "
                         "on queries.full34.jsonl, once")
    ap.add_argument("--hash-embed", action="store_true", help="offline embedder (smoke test)")
    ap.add_argument("--max-retrievals", type=int, default=None,
                    help="agentic arms: hard ceiling on retrievals per query")
    ap.add_argument("--no-graph-hints", action="store_true",
                    help="agentic arms: plan without consulting the typed graph")
    ap.add_argument("--no-anchor", action="store_true",
                    help="agentic arms: drop the deterministic retrieval on the original "
                         "query (§3.1). The ablation the anchor was adopted against")
    ap.add_argument("--stage-writes", action="store_true",
                    help="agentic arms: persist the decomposition as STAGED (§3.2). Off for "
                         "3.5's scored run — the deliverable must not write to the graph "
                         "it is measured against")
    args = ap.parse_args()

    settings = get_settings()
    gold = load_gold(str(settings.paths.eval_gold / args.gold))
    log.info("loaded %d gold queries", len(gold))

    embedder = (
        HashEmbedder(dim=settings.embeddings.dim)
        if args.hash_embed
        else SentenceTransformerEmbedder(
            settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
        )
    )
    # Shared with `scripts/ask.py`, so the arm scored here and the arm asked interactively
    # are configured by the same code. `top_k` reaches the vector arms only; see
    # `rpsg.retrieval.build`.
    system = build_system(
        args.system,
        embedder=embedder,
        top_k=args.top_k,
        max_retrievals=args.max_retrievals,
        graph_hints=not args.no_graph_hints,
        anchor=not args.no_anchor,
        stage_writes=args.stage_writes,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = "" if args.gold == "queries.jsonl" else f"_{Path(args.gold).stem}"
    run_dir = settings.paths.eval_runs / f"{stamp}_{args.system}{tag}"
    run_system(system, gold, run_dir, use_judge=not args.no_judge)
    log.info("run complete -> %s", run_dir)
    print((run_dir / "report.md").read_text())
    print(USAGE.summary())


if __name__ == "__main__":
    main()