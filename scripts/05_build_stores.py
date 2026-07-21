"""Build the vector index and the Tier-A graph.

    python scripts/05_build_stores.py
    python scripts/05_build_stores.py --hash-embed   # offline smoke test, no model download

Reads  data/interim/chunks.jsonl, data/external/papers.jsonl, data/processed/extractions.jsonl
Writes data/processed/vectors.faiss(+meta), data/processed/rpsg.kuzu
"""

from __future__ import annotations

import argparse
import json

from rpsg.config import get_settings
from rpsg.extraction.schema import ExtractionResult
from rpsg.ingestion.semantic_scholar import S2Paper, to_graph
from rpsg.logging import get_logger
from rpsg.stores.base import Chunk
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder
from rpsg.stores.graph_store import KuzuGraphStore
from rpsg.stores.vector_store import FaissVectorStore

log = get_logger(__name__)


def build_vectors(hash_embed: bool) -> None:
    settings = get_settings()
    chunks_path = settings.paths.data_interim / "chunks.jsonl"
    chunks = [
        Chunk(**json.loads(line)) for line in chunks_path.read_text().splitlines() if line
    ]
    if hash_embed:
        embedder = HashEmbedder(dim=settings.embeddings.dim)
    else:
        embedder = SentenceTransformerEmbedder(
            settings.embeddings.model_name, settings.embeddings.dim, settings.embeddings.batch_size
        )
    store = FaissVectorStore(str(settings.paths.vector_index), settings.embeddings.dim)
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts)
    store.add(chunks, embeddings)
    store.persist()
    log.info("indexed %d chunks", len(chunks))


def build_graph() -> None:
    settings = get_settings()
    store = KuzuGraphStore(str(settings.paths.kuzu_db))
    store.init_schema()

    # Tier A from S2 metadata
    papers_path = settings.paths.data_external / "papers.jsonl"
    if papers_path.exists():
        papers = [
            S2Paper(**json.loads(line)) for line in papers_path.read_text().splitlines() if line
        ]
        nodes, edges = to_graph(papers, max_references=200)
        store.upsert_nodes(nodes)
        store.upsert_edges(edges)

    # Tier B/C + repro from extractions (curated layer)
    ext_path = settings.paths.data_processed / "extractions.jsonl"
    if ext_path.exists():
        for line in ext_path.read_text().splitlines():
            if not line:
                continue
            result = ExtractionResult(**json.loads(line))
            store.upsert_nodes(result.nodes)
            store.upsert_edges(result.edges)
    log.info("graph built at %s", settings.paths.kuzu_db)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-embed", action="store_true", help="offline hashing embedder")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--skip-vectors", action="store_true")
    args = ap.parse_args()
    if not args.skip_vectors:
        build_vectors(args.hash_embed)
    if not args.skip_graph:
        build_graph()


if __name__ == "__main__":
    main()