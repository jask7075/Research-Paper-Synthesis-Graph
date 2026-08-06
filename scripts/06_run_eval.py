"""Score a system against the gold set — the Iteration 1 exit criterion.

    python scripts/06_run_eval.py --system vector_fulltext
    python scripts/06_run_eval.py --system vector_abstract --no-judge --hash-embed

Reads  eval/gold/queries.jsonl, the built vector store
Writes eval/runs/<timestamp>_<system>/{answers,traces,scores}.jsonl + report.md
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from rpsg.config import get_settings
from rpsg.eval.gold_schema import load_gold
from rpsg.eval.runner import run_system
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import System, VectorRAGSystem
from rpsg.retrieval.typed_graph import TypedGraphSystem
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder
from rpsg.stores.graph_store import KuzuGraphStore
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)

_CORPUS = {"vector_abstract": "abstract", "vector_fulltext": "fulltext"}
_SYSTEMS = (*_CORPUS, "typed_graph")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(_SYSTEMS), default="vector_fulltext")
    ap.add_argument("--top-k", type=int, default=60)
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
    system: System
    if args.system == "typed_graph":
        # Reads the graph, not the vector index. `hops` and `max_nodes` take the module
        # defaults, both set from the retrieval sweep rather than chosen.
        system = TypedGraphSystem(
            name=args.system,
            embedder=embedder,
            store=KuzuGraphStore(str(settings.paths.kuzu_db)),
        )
    else:
        vector_store = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
        vector_store.load()
        system = VectorRAGSystem(
            name=args.system,
            embedder=embedder,
            store=vector_store,
            corpus=_CORPUS[args.system],
            top_k=args.top_k,
        )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.paths.eval_runs / f"{stamp}_{args.system}"
    run_system(system, gold, run_dir, use_judge=not args.no_judge)
    log.info("run complete -> %s", run_dir)
    print((run_dir / "report.md").read_text())
    print(USAGE.summary())


if __name__ == "__main__":
    main()