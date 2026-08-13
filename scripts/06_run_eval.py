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
from rpsg.retrieval.agentic import AgenticSystem
from rpsg.retrieval.baselines import System, VectorRAGSystem
from rpsg.retrieval.citation_graph import CitationGraphSystem
from rpsg.retrieval.typed_graph import TypedGraphSystem
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder
from rpsg.stores.graph_store import KuzuGraphStore
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)

_CORPUS = {"vector_abstract": "abstract", "vector_fulltext": "fulltext"}
_SYSTEMS = (*_CORPUS, "typed_graph", "typed_graph_chunks",
            "citation_graph", "citation_graph_seeded",
            # 3.1. `_no_critique` is §3.5's required ablation: without it, "the loop helps"
            # cannot be separated from "planning helps".
            "agentic", "agentic_no_critique")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(_SYSTEMS), default="vector_fulltext")
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--hash-embed", action="store_true", help="offline embedder (smoke test)")
    ap.add_argument("--max-retrievals", type=int, default=None,
                    help="agentic arms: hard ceiling on retrievals per query")
    ap.add_argument("--no-graph-hints", action="store_true",
                    help="agentic arms: plan without consulting the typed graph")
    ap.add_argument("--stage-writes", action="store_true",
                    help="agentic arms: persist the decomposition as STAGED (§3.2). Off for "
                         "3.5's scored run — the deliverable must not write to the graph "
                         "it is measured against")
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
    if args.system.startswith("agentic"):
        # Same vector index and corpus as `vector_fulltext`, so 3.5's comparison isolates
        # the loop rather than the retrieval substrate.
        vs = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
        vs.load()
        kwargs = {}
        if args.max_retrievals is not None:
            kwargs["max_retrievals"] = args.max_retrievals
        system = AgenticSystem(
            name=args.system,
            embedder=embedder,
            store=vs,
            corpus="fulltext",
            graph_store=(
                None if args.no_graph_hints else KuzuGraphStore(str(settings.paths.kuzu_db))
            ),
            critique=not args.system.endswith("_no_critique"),
            stage_writes=args.stage_writes,
            **kwargs,
        )
    elif args.system.startswith("citation_graph"):
        # The ablation: `cites` edges from S2 metadata instead of extracted typed edges,
        # everything else held constant. `_seeded` starts from vector retrieval rather
        # than title similarity, so the pair separates "citations are weak" from "title
        # seeding is weak".
        vs = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
        vs.load()
        system = CitationGraphSystem(
            name=args.system,
            embedder=embedder,
            store=KuzuGraphStore(str(settings.paths.kuzu_db)),
            vector_store=vs,
            seed_from="chunks" if args.system.endswith("_seeded") else "title",
        )
    elif args.system.startswith("typed_graph"):
        # Reads the graph, not the vector index. `hops` and `max_nodes` take the module
        # defaults, both set from the retrieval sweep rather than chosen.
        routed = args.system == "typed_graph_chunks"
        vs = None
        if routed:
            # Router variant: traversal picks the papers, chunks supply the evidence, so
            # both arms are compared on the same evidence unit.
            vs = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
            vs.load()
        system = TypedGraphSystem(
            name=args.system,
            embedder=embedder,
            store=KuzuGraphStore(str(settings.paths.kuzu_db)),
            vector_store=vs,
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