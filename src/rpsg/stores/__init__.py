"""Storage layer. Interfaces in `base`; local adapters (Kuzu / FAISS) alongside.

Everything downstream depends on the *interfaces*, not the adapters — so Phase 2
(Neo4j AuraDB + Qdrant) is a config swap, not a rewrite.
"""

from rpsg.stores.base import Chunk, GraphStore, SearchHit, VectorStore

__all__ = ["Chunk", "SearchHit", "GraphStore", "VectorStore"]