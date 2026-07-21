"""Score a system against the gold set — the Iteration 1 exit criterion.

    python scripts/06_run_eval.py --system vector_fulltext
    python scripts/06_run_eval.py --system vector_abstract --no-judge --hash-embed

Reads  eval/gold/queries.jsonl, the built vector store
Writes eval/runs/<timestamp>_<system>/{answers,traces,scores}.jsonl + report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from rpsg.config import get_settings
from rpsg.eval.gold_schema import load_gold
from rpsg.eval.runner import run_system
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import VectorRAGSystem
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)

_CORPUS = {"vector_abstract": "abstract", "vector_fulltext": "fulltext"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(_CORPUS), default="vector_fulltext")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--hash-embed", action="store_true", help="offline embedder (smoke test)")
    args = ap.parse_args()

    settings = get_settings()
    gold = load_gold(str(settings.paths.eval_gold / "queries.jsonl"))
    log.info("loaded %d gold queries", len(gold))

    embedder = (
        HashEmbedder(dim=settings.embeddings.dim)
        if args.hash_embed
        else SentenceTransformerEmbedder(
            settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
        )
    )
    store = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
    store.load()

    system = VectorRAGSystem(
        name=args.system,
        embedder=embedder,
        store=store,
        corpus=_CORPUS[args.system],
        top_k=args.top_k,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.paths.eval_runs / f"{stamp}_{args.system}"
    run_system(system, gold, run_dir, use_judge=not args.no_judge)
    log.info("run complete -> %s", run_dir)
    print((run_dir / "report.md").read_text())


if __name__ == "__main__":
    main()