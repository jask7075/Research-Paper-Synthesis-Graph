"""Embedding backends. `SentenceTransformerEmbedder` is the Phase-1 default;
`HashEmbedder` is a deterministic, dependency-free stand-in so the pipeline and its tests
run end-to-end without downloading a model.
"""

from __future__ import annotations

import hashlib

import numpy as np

from rpsg.stores.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """SPECTER2 (or any sentence-transformers model). Requires the `vector` extra."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = dim
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


class HashEmbedder(Embedder):
    """Deterministic hashing embedder — no model download, no network.

    Not semantically meaningful; it exists so the full pipeline (chunk -> embed -> index ->
    retrieve -> score) is runnable and testable offline. Never use it for reported numbers.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype="float32")
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            out.append((vec / norm).tolist() if norm else vec.tolist())
        return out
