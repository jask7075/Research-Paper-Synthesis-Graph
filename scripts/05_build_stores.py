"""Build the vector index and the Tier-A graph.

    python scripts/05_build_stores.py
    python scripts/05_build_stores.py --hash-embed   # offline smoke test, no model download

Reads  data/interim/chunks.jsonl, data/external/papers.jsonl, data/processed/extractions.jsonl
Writes data/processed/vectors.faiss(+meta), data/processed/rpsg.kuzu
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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


def _reset_graph(db_path: Path) -> None:
    """Delete the graph before rebuilding it.

    `upsert_nodes` uses MERGE, so building into an existing database *accumulates*: nodes
    from an earlier extraction survive even when the current one no longer produces them.
    After the Hardware routing fix that made the graph a union of two extraction
    generations rather than a picture of one, which is a correctness problem before it is
    a size one — you cannot tell which run a node came from.

    It also caused the failure that surfaced this: merging a fresh 25k-node extraction on
    top of a 64 MB database exhausted Kuzu's buffer pool ("Unable to allocate memory! The
    buffer pool is full and no memory could be freed!").

    The graph is derived data, fully reconstructible from papers.jsonl + extractions.jsonl,
    so rebuilding from empty is the honest default. The sidecar .wal / .shadow files go too;
    leaving a WAL behind replays writes belonging to the database we just removed.
    """
    for p in (db_path, Path(f"{db_path}.wal"), Path(f"{db_path}.shadow")):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    log.info("cleared existing graph at %s", db_path)


def build_graph() -> None:
    settings = get_settings()
    _reset_graph(Path(settings.paths.kuzu_db))
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