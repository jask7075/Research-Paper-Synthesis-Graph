"""Local vector store (Phase 1). Default is a FAISS flat index — at ~40k chunks an exact
flat index is fast and needs no external service. A pgvector adapter stub is provided for
the Phase-1.5 portable path; Qdrant is Phase 2.

Both abstract-only and full-text chunks live in one index, discriminated by `Chunk.corpus`,
so `vector_abstract` and `vector_fulltext` baselines share infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rpsg.logging import get_logger
from rpsg.stores.base import Chunk, SearchHit, VectorStore

log = get_logger(__name__)


class FaissVectorStore(VectorStore):
    def __init__(self, index_path: str, dim: int) -> None:
        self._index_path = Path(index_path)
        self._meta_path = self._index_path.with_suffix(".meta.jsonl")
        self.dim = dim
        self._index = None  # lazy faiss index
        self._chunks: list[Chunk] = []

    def _ensure_index(self) -> None:
        if self._index is None:
            import faiss

            # Inner-product on L2-normalized vectors == cosine similarity.
            self._index = faiss.IndexFlatIP(self.dim)

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._ensure_index()
        mat = self._normalize(np.asarray(embeddings, dtype="float32"))
        self._index.add(mat)  # type: ignore[union-attr]
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int, corpus: str) -> list[SearchHit]:
        self._ensure_index()
        q = self._normalize(np.asarray([query_embedding], dtype="float32"))
        # Over-fetch, then filter by corpus (flat index has no metadata filter).
        k = min(len(self._chunks), max(top_k * 5, top_k))
        if k == 0:
            return []
        scores, idxs = self._index.search(q, k)  # type: ignore[union-attr]
        hits: list[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if chunk.corpus != corpus:
                continue
            hits.append(SearchHit(chunk=chunk, score=float(score)))
            if len(hits) >= top_k:
                break
        return hits

    def persist(self) -> None:
        import faiss

        self._ensure_index()
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with self._meta_path.open("w") as fh:
            for c in self._chunks:
                fh.write(c.model_dump_json() + "\n")
        log.info("Persisted %d chunks to %s", len(self._chunks), self._index_path)

    def load(self) -> None:
        import faiss

        self._index = faiss.read_index(str(self._index_path))
        self._chunks = [
            Chunk(**json.loads(line)) for line in self._meta_path.read_text().splitlines() if line
        ]
        log.info("Loaded %d chunks from %s", len(self._chunks), self._index_path)


class PgVectorStore(VectorStore):  # pragma: no cover - Phase-1.5 portable path
    """pgvector adapter. Implement when you want the Phase-2-portable path; the interface
    is identical so retrieval code is unchanged. Requires the `pgvector` extra + a running
    Postgres with the `vector` extension."""

    def __init__(self, dsn: str, dim: int) -> None:
        self.dsn = dsn
        self.dim = dim

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError("PgVectorStore: implement for the Phase-1.5 portable path.")

    def search(self, query_embedding: list[float], top_k: int, corpus: str) -> list[SearchHit]:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError