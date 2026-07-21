"""Semantic Scholar Graph API client — the Tier-A metadata source.

Everything this module returns is high-precision and effectively free: Paper, Author
(already disambiguated by S2 — do NOT re-extract authors with an LLM), Venue, and the
`cites` edges that constitute the citation-graph baseline.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rpsg.extraction.schema import Edge, EdgeType, Node, NodeType, SourceLayer
from rpsg.logging import get_logger

log = get_logger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"

DEFAULT_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,authors,"
    "references.paperId,openAccessPdf"
)


class S2Author(BaseModel):
    authorId: str | None = None
    name: str | None = None


class S2Paper(BaseModel):
    paperId: str
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[S2Author] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    openAccessPdf: dict[str, Any] | None = None
    externalIds: dict[str, Any] = Field(default_factory=dict)

    @property
    def pdf_url(self) -> str | None:
        return (self.openAccessPdf or {}).get("url")

    @property
    def arxiv_id(self) -> str | None:
        return self.externalIds.get("ArXiv")


class SemanticScholarClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        headers = {"x-api-key": api_key} if api_key else {}
        self._client = httpx.Client(base_url=S2_BASE, headers=headers, timeout=timeout)
        # Unauthenticated S2 is ~1 req/s; be a good citizen.
        self._min_interval = 0.1 if api_key else 1.1
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, limit: int = 100, fields: str = DEFAULT_FIELDS) -> list[S2Paper]:
        """Relevance search. S2 caps `limit` at 100 per page."""
        papers: list[S2Paper] = []
        offset = 0
        while len(papers) < limit:
            page = min(100, limit - len(papers))
            data = self._get(
                "/paper/search",
                {"query": query, "limit": page, "offset": offset, "fields": fields},
            )
            batch = data.get("data", [])
            if not batch:
                break
            papers.extend(S2Paper(**p) for p in batch)
            offset += page
            if data.get("next") is None:
                break
        log.info("S2 search '%s' -> %d papers", query, len(papers))
        return papers[:limit]

    def get_paper(self, paper_id: str, fields: str = DEFAULT_FIELDS) -> S2Paper:
        return S2Paper(**self._get(f"/paper/{paper_id}", {"fields": fields}))

    def close(self) -> None:
        self._client.close()


def to_graph(papers: list[S2Paper], max_references: int = 200) -> tuple[list[Node], list[Edge]]:
    """Convert S2 metadata into Tier-A nodes and edges.

    `cites` edges are emitted only between papers *inside the corpus* — a dangling citation
    to a paper you never ingested adds a node you can't ground anything on.
    """
    in_corpus = {p.paperId for p in papers}
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    def add(node: Node) -> None:
        nodes.setdefault(node.id, node)

    for p in papers:
        pid = f"paper:{p.paperId}"
        add(
            Node(
                id=pid,
                type=NodeType.PAPER,
                name=p.title or p.paperId,
                attrs={
                    "s2_id": p.paperId,
                    "year": p.year,
                    "abstract": p.abstract,
                    "arxiv_id": p.arxiv_id,
                    "pdf_url": p.pdf_url,
                },
                source_layer=SourceLayer.CURATED,
                confidence=1.0,
            )
        )

        if p.venue:
            vid = f"venue:{p.venue.lower().replace(' ', '-')}"
            add(Node(id=vid, type=NodeType.VENUE, name=p.venue))
            edges.append(Edge(src=pid, dst=vid, type=EdgeType.PUBLISHED_IN))

        for a in p.authors:
            if not a.authorId:
                continue  # skip un-disambiguated authors rather than guess
            aid = f"author:{a.authorId}"
            add(Node(id=aid, type=NodeType.AUTHOR, name=a.name or a.authorId))
            edges.append(Edge(src=pid, dst=aid, type=EdgeType.AUTHORED_BY))

        for ref in p.references[:max_references]:
            ref_id = ref.get("paperId")
            if ref_id and ref_id in in_corpus:
                edges.append(Edge(src=pid, dst=f"paper:{ref_id}", type=EdgeType.CITES))

    log.info("Tier-A graph: %d nodes, %d edges", len(nodes), len(edges))
    return list(nodes.values()), edges