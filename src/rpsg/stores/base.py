"""Storage interfaces. Depend on these, not on Kuzu/FAISS/pgvector concrete classes.

Phase-2 portability contract:
    GraphStore   Kuzu (Phase 1)      -> Neo4j AuraDB (Phase 2).  Both Cypher.
    VectorStore  FAISS/pgvector (P1) -> Qdrant (Phase 2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from rpsg.extraction.schema import Edge, Node, SourceLayer


class Chunk(BaseModel):
    """A retrievable unit of text with section-aware provenance."""

    id: str
    paper_id: str
    text: str
    section_title: str
    section_type: str  # abstract | intro | method | results | discussion | limitations | other
    char_start: int
    char_end: int
    corpus: str = "fulltext"  # "fulltext" | "abstract"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float


class VectorStore(ABC):
    """ANN over chunk embeddings. The `corpus` field lets one store hold both
    abstract-only and full-text chunks so the two baselines share infrastructure."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int, corpus: str) -> list[SearchHit]: ...

    @abstractmethod
    def persist(self) -> None: ...

    @abstractmethod
    def load(self) -> None: ...


class GraphStore(ABC):
    """Typed knowledge graph. `source_layer` filtering enforces curated vs. staged:
    metrics query CURATED only; the agent may write STAGED at query time."""

    @abstractmethod
    def init_schema(self) -> None:
        """Create node/edge tables (idempotent)."""

    @abstractmethod
    def upsert_nodes(self, nodes: list[Node]) -> None: ...

    @abstractmethod
    def upsert_edges(self, edges: list[Edge]) -> None: ...

    @abstractmethod
    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Run a read query and return rows as dicts."""

    @abstractmethod
    def promote_staged(self, node_ids: list[str] | None = None) -> int:
        """Move reviewed STAGED nodes/edges into CURATED. Returns count promoted.
        Intentionally explicit — there is no auto-merge (see README design principle 3)."""

    def curated_filter(self) -> str:
        """Cypher fragment to restrict a match to the curated layer."""
        return f"source_layer = '{SourceLayer.CURATED.value}'"


class Embedder(ABC):
    """Text -> vector. Kept in the store package so retrieval depends only on interfaces."""

    dim: int = 768

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...