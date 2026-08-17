"""Ask the corpus a question — the interactive entrypoint, on any arm.

    python scripts/ask.py "what mitigates barren plateaus in VQE?"
    python scripts/ask.py "..." --system agentic          # plan, retrieve per part, critique
    python scripts/ask.py "..." --system typed_graph      # traverse the graph instead
    python scripts/ask.py "..." --retrieval-only          # vector arms: no LLM call, no cost
    python scripts/ask.py "..." --show-evidence           # print the excerpts sent to the model
    python scripts/ask.py --list-systems

Arms are built by `rpsg.retrieval.build.build_system`, the same function
`scripts/06_run_eval.py` uses, so the arm you ask here is configured identically to the
arm the report scores. This script was previously hardcoded to `VectorRAGSystem` while
claiming in this docstring to be "a different front end, not a different pipeline" — true
when that was the only arm, and quietly false from the moment three more landed.

Run from the repository root. `.env` is resolved relative to the working directory, so
`cd scripts && python ask.py ...` finds no secrets and fails at the synthesis call after
retrieval has already run.

Note on `--top-k`: it applies to the vector arms only. The graph arms take measured
`seeds`/`hops`/`max_nodes` defaults and the agentic arm a measured `top_k`, and overriding
those from a CLI default would change a swept value into an arbitrary one.
"""

from __future__ import annotations

import argparse

from rpsg.config import get_settings
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger
from rpsg.retrieval.build import RETRIEVES_CHUNKS_UP_FRONT, SYSTEMS, build_system
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder

log = get_logger(__name__)

#: Arms that read the vector index. Checked before building so a first-time user gets a
#: "build the index" message instead of a bare faiss RuntimeError on a missing file.
_NEEDS_VECTOR_INDEX = frozenset(SYSTEMS) - {"typed_graph"}
_NEEDS_GRAPH = frozenset(
    n for n in SYSTEMS if n.startswith(("typed_graph", "citation_graph", "agentic"))
)


def _print_trajectory(trace: dict) -> None:
    """The agentic arm's plan and what it cost. This is the arm's whole point, and an
    answer alone does not show whether the loop did anything."""
    if trace.get("planner_failed"):
        print("\n!! planner failed — degraded to a single retrieval on the original query")
    subs = trace.get("sub_questions") or []
    print(f"\n--- plan ({len(subs)} sub-question(s)) ---")
    for i, sub in enumerate(subs, 1):
        print(f"  {i}. {sub}")
    if trace.get("plan_reasoning"):
        print(f"  reasoning: {trace['plan_reasoning']}")
    if trace.get("graph_hints"):
        print(f"  graph hints used: {', '.join(map(str, trace['graph_hints'][:6]))}")
    print(
        f"\n  retrievals: {trace.get('retrievals_used', 0)} used"
        f", {trace.get('retrievals_refused', 0)} refused"
        f"   anchor: {'on' if trace.get('anchor_used') else 'off'}"
    )
    if trace.get("critique_ran"):
        added = trace.get("critique_added_papers") or []
        gaps = trace.get("gaps") or []
        print(f"  critique: ran, {len(gaps)} gap(s) found, {len(added)} required paper(s) added")
        if trace.get("critique_assessment"):
            print(f"  assessment: {trace['critique_assessment']}")
    else:
        print("  critique: disabled")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="the question to ask")
    ap.add_argument("--system", choices=list(SYSTEMS), default="vector_fulltext")
    ap.add_argument("--list-systems", action="store_true", help="print the arms and exit")
    ap.add_argument("--top-k", type=int, default=None, help="vector arms only (default 20)")
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="show what was retrieved and stop — makes no LLM call. Vector arms only",
    )
    ap.add_argument("--show-evidence", action="store_true", help="print the excerpts verbatim")
    ap.add_argument(
        "--hash-embed",
        action="store_true",
        help="deterministic offline embedder; exercises the plumbing, not real similarity",
    )
    # Agentic flags, named as in 06_run_eval.py so the two CLIs stay learnable together.
    ap.add_argument("--max-retrievals", type=int, default=None,
                    help="agentic arms: hard ceiling on retrievals per query")
    ap.add_argument("--no-graph-hints", action="store_true",
                    help="agentic arms: plan without consulting the typed graph")
    ap.add_argument("--no-anchor", action="store_true",
                    help="agentic arms: drop the deterministic retrieval on the original query")
    ap.add_argument("--stage-writes", action="store_true",
                    help="agentic arms: persist the decomposition as STAGED nodes")
    args = ap.parse_args()

    if args.list_systems:
        print("arms (--system):")
        for name in SYSTEMS:
            marks = []
            if name in RETRIEVES_CHUNKS_UP_FRONT:
                marks.append("supports --retrieval-only")
            if name in _NEEDS_GRAPH:
                marks.append("needs the Kuzu graph")
            print(f"  {name:<24} {'; '.join(marks)}")
        return
    if not args.question:
        ap.error("a question is required (or use --list-systems)")

    settings = get_settings()
    if args.retrieval_only and args.system not in RETRIEVES_CHUNKS_UP_FRONT:
        # Refused rather than ignored: silently answering in full when the user asked for
        # no LLM call would spend money they declined to spend.
        ap.error(
            f"--retrieval-only is not available for {args.system!r}: it plans or traverses "
            "before it retrieves, so there is no pre-synthesis chunk list. Vector arms only: "
            f"{', '.join(sorted(RETRIEVES_CHUNKS_UP_FRONT))}"
        )
    if args.top_k is not None and args.system not in RETRIEVES_CHUNKS_UP_FRONT:
        log.warning(
            "--top-k is ignored for %s; its retrieval breadth is a measured default", args.system
        )

    # Checked explicitly: faiss.read_index raises a bare RuntimeError on a missing file,
    # which is not a useful message to hand a first-time user.
    if args.system in _NEEDS_VECTOR_INDEX and not settings.paths.vector_index.exists():
        raise SystemExit(
            f"no vector index at {settings.paths.vector_index}\n"
            "Build one first:  python scripts/05_build_stores.py [--hash-embed]"
        )
    if args.system in _NEEDS_GRAPH and not settings.paths.kuzu_db.exists():
        raise SystemExit(
            f"no graph at {settings.paths.kuzu_db}\n"
            f"{args.system!r} traverses the typed graph. Build it first:  "
            "python scripts/05_build_stores.py"
        )

    embedder = (
        HashEmbedder(dim=settings.embeddings.dim)
        if args.hash_embed
        else SentenceTransformerEmbedder(
            settings.embeddings.model_name,
            settings.embeddings.dim,
            settings.embeddings.batch_size,
        )
    )
    system = build_system(
        args.system,
        embedder=embedder,
        top_k=args.top_k,
        max_retrievals=args.max_retrievals,
        graph_hints=not args.no_graph_hints,
        anchor=not args.no_anchor,
        stage_writes=args.stage_writes,
    )

    print(f"\nQUESTION  {args.question}")
    print(f"ARM       {system.name}")

    if args.system in RETRIEVES_CHUNKS_UP_FRONT:
        hits = system._retrieve(args.question)  # noqa: SLF001 - same object the runner uses
        print(f"RETRIEVED {len(hits)} chunks from corpus={system._corpus}")  # noqa: SLF001
        for hit in hits:
            print(f"  {hit.score:+.3f}  {hit.chunk.section_type:<12} {hit.chunk.id}")
        if not hits:
            raise SystemExit(
                "\nNothing retrieved. Either the index is empty, or it holds no chunks for "
                f"corpus={system._corpus!r} (abstract chunks only exist if stage 01 "  # noqa: SLF001
                "fetched metadata)."
            )
        if args.retrieval_only:
            if args.show_evidence:
                print("\n--- evidence ---")
                for hit in hits:
                    print(f"\n[paper:{hit.chunk.paper_id}] ({hit.chunk.section_type})")
                    print(" ".join(hit.chunk.text.split())[:400], "…")
            return

    out = system.answer(args.question)
    if out.trace:
        _print_trajectory(out.trace)
    if args.show_evidence:
        print(f"\n--- evidence sent to the model ({len(out.evidence)} chars) ---")
        print(out.evidence[:4000])
    print(f"\n--- answer ({settings.models.synthesis_model}) ---\n")
    print(out.text)
    print(f"\n--- grounded on {len(out.cited_paper_ids)} paper(s) ---")
    for pid in out.cited_paper_ids:
        print(f"  {pid}")
    print("\n" + USAGE.summary())


if __name__ == "__main__":
    main()
