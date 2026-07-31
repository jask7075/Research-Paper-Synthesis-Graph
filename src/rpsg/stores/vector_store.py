"""Local vector store (Phase 1). Default is a FAISS flat index — at ~40k chunks an exact
flat index is fast and needs no external service. A pgvector adapter stub is provided for
the Phase-1.5 portable path; Qdrant is Phase 2.

Both abstract-only and full-text chunks live in one index, discriminated by `Chunk.corpus`,
so `vector_abstract` and `vector_fulltext` baselines share infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rpsg.config import get_settings
from rpsg.logging import get_logger
from rpsg.stores.base import Chunk, SearchHit, VectorStore

log = get_logger(__name__)


#: Section types whose content is inherently terse, exempt from length damping.
#: An availability statement is *complete* at 269 chars ("The code and data supporting
#: this study are openly available at ..."), unlike a 162-char "This section briefly
#: provides..." stub which is merely truncated — so length cannot tell them apart but
#: section type can. Measured: damping without this exemption dropped availability
#: sections from 5 hits to 0 across six probe queries, including the one that asks
#: literally "where is the code for these experiments available", making the
#: reproducibility layer unreachable by vector search.
_DAMPING_EXEMPT_SECTIONS = frozenset({"availability"})


def _length_damping(n_chars: int, reference_chars: int) -> float:
    """Scale a similarity score down for chunks shorter than `reference_chars`.

    Short text embeds to a less specific vector sitting nearer the corpus centroid, so it
    scores moderately against *every* query and wins top-k whenever no longer chunk is a
    strong match. Measured on a 9,913-chunk corpus with a median of 1,881 chars, 26% of
    retrieved hits were under 300 chars against 4.0% of the corpus — a ~6x
    over-representation that starves the synthesizer of context.

    A length floor is the wrong instrument here. 73% of short chunks are the sole chunk of
    a genuinely short section, and those are two populations length cannot separate: a
    269-char "Code and data availability" statement (legitimate, and the one place the
    reproducibility layer is stated plainly) versus a 162-char "This section briefly
    provides..." stub (content-free). Damping keeps the former retrievable when it truly
    matches while stopping the latter from crowding out substance.

    Linear below the reference, 1.0 at or above it — proportional to the context a chunk
    actually carries, with no penalty once it carries enough.
    """
    if reference_chars <= 0:
        return 1.0
    return min(1.0, n_chars / reference_chars)


class FaissVectorStore(VectorStore):
    def __init__(
        self, index_path: str, dim: int, length_damping_chars: int | None = None
    ) -> None:
        self._length_damping_chars = (
            get_settings().retrieval.length_damping_chars
            if length_damping_chars is None
            else length_damping_chars
        )
        self._index_path = Path(index_path)
        self._meta_path = self._index_path.with_suffix(".meta.jsonl")
        self.dim = dim
        # `Any`, not `faiss.Index`: faiss lives in the optional `vector` extra and is
        # imported lazily, so the annotation must hold whether or not it is installed.
        # (Typed as None it passed mypy only while faiss was absent — CI installs
        # `.[dev]` without `vector`, so that discrepancy was invisible there.)
        self._index: Any = None
        self._chunks: list[Chunk] = []

    def _ensure_index(self):  # noqa: ANN202 - faiss is untyped; returns a faiss index
        """Build the index on first use and return it.

        Returning the index (rather than only assigning it) is what lets callers use
        it without a `type: ignore` — the previous shape forced one at every call
        site, and those ignores had drifted to the wrong error code.
        """
        if self._index is None:
            import faiss

            # Inner-product on L2-normalized vectors == cosine similarity.
            self._index = faiss.IndexFlatIP(self.dim)
        return self._index

    @staticmethod
    def _pin_faiss_to_one_thread() -> None:
        """Stop faiss spawning OpenMP threads, which segfaults alongside torch.

        Three separate `libomp.dylib` copies ship in this environment (faiss, torch,
        sklearn). When a process has loaded torch and then runs a threaded faiss
        search, it dies with SIGSEGV — reproducibly, and only at `search()`: loading
        the index, encoding, and building a fresh `IndexFlatIP` are all fine, which
        is why stage 05 never hit it and `ask.py` always did.

        Called immediately before searching rather than at import, because setting it
        before both runtimes exist raises `OMP Error #15` instead. A global
        `OMP_NUM_THREADS=1` also fixes the crash but throttles torch, slowing the
        10k-chunk embedding pass in stage 05 for no reason; `KMP_DUPLICATE_LIB_OK`
        does not fix it at all. Single-threaded search costs nothing measurable on a
        flat index of this size.
        """
        try:
            import faiss

            faiss.omp_set_num_threads(1)
        except Exception:  # noqa: BLE001 - a missing symbol must not break retrieval
            pass

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        index = self._ensure_index()
        mat = self._normalize(np.asarray(embeddings, dtype="float32"))
        index.add(mat)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int, corpus: str) -> list[SearchHit]:
        self._pin_faiss_to_one_thread()
        index = self._ensure_index()
        q = self._normalize(np.asarray([query_embedding], dtype="float32"))
        # Over-fetch, then filter by corpus (flat index has no metadata filter). The
        # same surplus is what makes length damping possible without a second index.
        k = min(len(self._chunks), max(top_k * 5, top_k))
        if k == 0:
            return []
        scores, idxs = index.search(q, k)
        candidates: list[SearchHit] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if chunk.corpus != corpus:
                continue
            raw = float(score)
            factor = (
                1.0
                if chunk.section_type in _DAMPING_EXEMPT_SECTIONS
                else _length_damping(len(chunk.text), self._length_damping_chars)
            )
            candidates.append(SearchHit(chunk=chunk, score=raw * factor, raw_score=raw))
        # Damping reorders the pool, so it must be re-sorted before truncating. Taking
        # the first `top_k` in index order (as an early break would) would apply the
        # penalty without ever letting a longer chunk overtake a shorter one.
        candidates.sort(key=lambda hit: hit.score, reverse=True)
        return candidates[:top_k]

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