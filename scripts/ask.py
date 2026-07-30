"""Ask the corpus a question — the interactive entrypoint.

    python scripts/ask.py "what mitigates barren plateaus in VQE?"
    python scripts/ask.py "..." --retrieval-only      # no LLM call, no cost
    python scripts/ask.py "..." --hash-embed          # offline embedder (plumbing only)
    python scripts/ask.py "..." --show-evidence       # print the excerpts sent to the model

Requires a built vector index (`scripts/05_build_stores.py`). Everything the eval
harness scores goes through the same `VectorRAGSystem`, so what you see here is
what stage 06 measures — this script is a different front end, not a different
pipeline.
"""

from __future__ import annotations

import argparse

from rpsg.config import get_settings
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import VectorRAGSystem
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--corpus", choices=["fulltext", "abstract"], default="fulltext")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="show what was retrieved and stop — makes no LLM call",
    )
    ap.add_argument("--show-evidence", action="store_true", help="print the excerpts verbatim")
    ap.add_argument(
        "--hash-embed",
        action="store_true",
        help="deterministic offline embedder; exercises the plumbing, not real similarity",
    )
    args = ap.parse_args()

    settings = get_settings()
    embedder = (
        HashEmbedder(dim=settings.embeddings.dim)
        if args.hash_embed
        else SentenceTransformerEmbedder(
            settings.embeddings.model_name,
            settings.embeddings.dim,
            settings.embeddings.batch_size,
        )
    )
    # Checked explicitly: faiss.read_index raises a bare RuntimeError on a missing
    # file, which is not a useful message to hand a first-time user.
    if not settings.paths.vector_index.exists():
        raise SystemExit(
            f"no vector index at {settings.paths.vector_index}\n"
            "Build one first:  python scripts/05_build_stores.py [--hash-embed]"
        )
    store = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
    store.load()

    system = VectorRAGSystem(
        name=f"vector_{args.corpus}",
        embedder=embedder,
        store=store,
        corpus=args.corpus,
        top_k=args.top_k,
    )

    hits = system._retrieve(args.question)  # noqa: SLF001 - same object the runner uses
    print(f"\nQUESTION  {args.question}")
    print(f"RETRIEVED {len(hits)} chunks from corpus={args.corpus} (top_k={args.top_k})")
    for hit in hits:
        print(f"  {hit.score:+.3f}  {hit.chunk.section_type:<12} {hit.chunk.id}")

    if not hits:
        raise SystemExit(
            "\nNothing retrieved. Either the index is empty, or it holds no chunks for "
            f"corpus={args.corpus!r} (abstract chunks only exist if stage 01 fetched metadata)."
        )
    if args.retrieval_only:
        if args.show_evidence:
            print("\n--- evidence ---")
            for hit in hits:
                print(f"\n[paper:{hit.chunk.paper_id}] ({hit.chunk.section_type})")
                print(" ".join(hit.chunk.text.split())[:400], "…")
        return

    out = system.answer(args.question)
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