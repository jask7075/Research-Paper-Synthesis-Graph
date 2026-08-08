"""Typed-graph retrieval: the arm the Iteration-2 thesis is about.

`VectorRAGSystem` retrieves *chunks* — contiguous spans of one paper, ranked by similarity
to the query. This retrieves *nodes* and then walks typed edges out of them, so the unit of
evidence is an extracted claim or method with its grounding quote, and relatedness comes
from an edge a paper asserted rather than from embedding proximity.

That difference is the whole experiment. On a relational question — "which methods address
X, and what limits each" — the vector arm has to find one passage that happens to state
both halves, while a traversal can start at X and follow `addresses` to the methods and
`undercuts` to their limitations, assembling an answer no single passage contains.

Both arms implement the same `System` protocol and share the synthesis half, so the only
difference under test is retrieval. They also share the handle scheme (`[P1]` rather than
40-hex ids), because the runner harvests citations by regex and a mangled id silently
corrupts `must_cite_recall`.

Seeds come from embedding the query against *node names*, not chunk text. That is a weaker
signal than chunk retrieval — names are short — and it is deliberately not backstopped with
chunk search, which would make this a hybrid and confound the comparison it exists to make.
"""

from __future__ import annotations

import json
from typing import Any

from rpsg.config import get_settings
from rpsg.llm import get_chat_client
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import _SYNTH_SYSTEM, SystemOutput, VectorRAGSystem
from rpsg.stores.base import Embedder
from rpsg.stores.graph_store import KuzuGraphStore

log = get_logger(__name__)

#: Types worth seeding from. `Claim` earns its place empirically: dropping it dropped
#: recall 0.556 -> 0.333. Claims are sentences and so is a query, so they match far better
#: than 3-word concept names -- they are the doorway into a neighbourhood, not noise. I had
#: predicted the opposite, that claim-shaped seeds crowd out the concept nodes a query asks
#: to enumerate.
_SEED_TYPES = ("Method", "Problem", "Claim", "Limitation", "Dataset", "Hardware", "Software")


class GraphHit:
    """One node reached by retrieval, with where it came from and how."""

    def __init__(self, node: dict[str, Any], hop: int, via: str | None) -> None:
        self.node = node
        self.hop = hop
        self.via = via  # the edge type traversed to reach it, None for a seed

    @property
    def paper_id(self) -> str | None:
        attrs = self.node.get("attrs")
        if isinstance(attrs, str):
            try:
                attrs = json.loads(attrs)
            except json.JSONDecodeError:
                return None
        return (attrs or {}).get("from_paper")

    @property
    def quotes(self) -> list[str]:
        ev = self.node.get("evidence")
        if isinstance(ev, str):
            try:
                ev = json.loads(ev)
            except json.JSONDecodeError:
                return []
        return [q for q in (ev or []) if isinstance(q, str) and q.strip()]


class TypedGraphSystem:
    """Seed by name similarity, expand along typed edges, answer from node evidence."""

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        store: KuzuGraphStore,
        *,
        seeds: int = 12,
        hops: int = 2,
        max_nodes: int = 150,
        synthesis_model: str | None = None,
        vector_store: Any = None,
        chunks_per_paper: int = 4,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._store = store
        self._seeds = seeds
        # 2 hops, measured. A retrieval-only sweep over seeds x hops x max_nodes put
        # required-pair recall at 0.389 (1 hop) -> 0.556 (2) -> 0.556 (3): the second hop
        # is the whole gain and the third only adds evidence volume. Neither more seeds
        # nor a larger cap moved it.
        self._hops = hops
        # A ceiling rather than a budget match: at 2 hops the neighbourhood yields ~41
        # nodes and ~9k chars, against the vector arm's ~111k at top_k=60. The graph arm
        # is not evidence-starved by this cap -- it simply reaches less, which is the
        # result, not a handicap to correct for.
        self._max_nodes = max_nodes
        self._synthesis_model = synthesis_model or get_settings().models.synthesis_model
        self._client: Any = None
        self._names: list[dict[str, Any]] = []
        self._vecs: Any = None
        # With a vector store the graph becomes a *router*: traversal chooses the papers,
        # chunks from those papers become the evidence. That puts both arms on the same
        # evidence unit, which isolates traversal from evidence formatting.
        #
        # The first scored run passed node quotes instead, and lost: retrieval reached
        # 0.556 of required pairs against the vector arm's 0.611, but converted only 33%
        # of that into citations against 60%. One-sentence quotes stripped of context
        # appear to be too thin to build a citable answer from. Mapping a quote back to
        # its chunk by substring only works for 57% of nodes — chunking normalises
        # whitespace and some quotes straddle boundaries — so the routing is done at
        # paper granularity, where it is exact.
        self._vector_store = vector_store
        self._chunks_per_paper = chunks_per_paper

    def _load_names(self) -> None:
        """Embed every seedable node name once, on first use."""
        if self._vecs is not None:
            return
        import numpy as np

        rows = self._store.query(
            "MATCH (e:Entity) WHERE e.type IN $types RETURN e.id AS id, e.name AS name, "
            "e.type AS type, e.attrs AS attrs, e.evidence AS evidence",
            {"types": list(_SEED_TYPES)},
        )
        self._names = [r for r in rows if (r.get("name") or "").strip()]
        vecs = np.asarray(
            self._embedder.encode([r["name"] for r in self._names]), dtype="float32"
        )
        self._vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        log.info("typed-graph: embedded %d node names", len(self._names))

    def _seed_nodes(self, query: str) -> list[dict[str, Any]]:
        import numpy as np

        self._load_names()
        if not self._names:
            return []
        q = np.asarray(self._embedder.encode([query])[0], dtype="float32")
        q /= np.linalg.norm(q) + 1e-9
        sims = self._vecs @ q
        top = np.argsort(-sims)[: self._seeds]
        return [self._names[int(i)] for i in top]

    def _expand(self, seed_ids: list[str]) -> list[GraphHit]:
        """Walk `hops` steps out, breadth-first, recording the edge type used.

        Direction is ignored: `addresses` runs Method -> Problem, and a query naming the
        problem needs to reach the method, so a directed walk would answer only half the
        relational questions.
        """
        seen = set(seed_ids)
        frontier = list(seed_ids)
        hits: list[GraphHit] = []
        for hop in range(1, self._hops + 1):
            if not frontier or len(seen) >= self._max_nodes:
                break
            rows = self._store.query(
                # Ordered by the returned alias, not `r.confidence`: after RETURN DISTINCT
                # the relationship variable is out of scope and Kuzu raises a binder error.
                # Highest-confidence edges first, so `max_nodes` truncates the weakest.
                "MATCH (a:Entity)-[r:REL]-(b:Entity) WHERE a.id IN $ids "
                "RETURN DISTINCT b.id AS id, b.name AS name, b.type AS type, "
                "b.attrs AS attrs, b.evidence AS evidence, r.type AS via, "
                "r.confidence AS confidence ORDER BY confidence DESC",
                {"ids": frontier},
            )
            frontier = []
            for row in rows:
                if row["id"] in seen or len(seen) >= self._max_nodes:
                    continue
                seen.add(row["id"])
                frontier.append(row["id"])
                hits.append(GraphHit(row, hop, row.get("via")))
        return hits

    def _retrieve(self, query: str) -> list[GraphHit]:
        seeds = self._seed_nodes(query)
        hits = [GraphHit(s, 0, None) for s in seeds]
        hits += self._expand([s["id"] for s in seeds])
        return hits[: self._max_nodes]

    @staticmethod
    def _format_evidence(hits: list[GraphHit]) -> tuple[str, dict[str, str]]:
        """Node evidence, labelled by paper handle.

        Nodes whose quotes are missing are dropped rather than sent as a bare name: a name
        with no grounding gives the synthesizer nothing to cite and invites it to fill the
        gap from its own knowledge.
        """
        handles: dict[str, str] = {}
        by_paper: dict[str, str] = {}
        blocks: list[str] = []
        for h in hits:
            pid, quotes = h.paper_id, h.quotes
            if not pid or not quotes:
                continue
            full = f"paper:{pid}"
            if full not in by_paper:
                handle = f"P{len(by_paper) + 1}"
                by_paper[full] = handle
                handles[handle] = full
            relation = f" via {h.via}" if h.via else " (seed)"
            blocks.append(
                f"[{by_paper[full]}] {h.node['type']}: {h.node['name']}{relation}\n"
                f"    {' '.join(quotes)[:600]}"
            )
        return "\n\n".join(blocks), handles

    def _synthesize(self, query: str, evidence: str) -> str:
        if self._client is None:
            self._client = get_chat_client(self._synthesis_model)
        return self._client.text(
            system=_SYNTH_SYSTEM,
            user=f"QUESTION: {query}\n\nEXCERPTS:\n{evidence}",
            max_tokens=4096,
        )

    def _chunk_evidence(self, query: str, papers: set[str]) -> tuple[str, dict[str, str]]:
        """Chunks from the traversed papers, ranked by query similarity.

        The store has no per-paper filter, so over-fetch and keep what the traversal
        selected — the same over-fetch-then-filter shape `FaissVectorStore` already uses
        for its corpus filter.
        """
        qvec = self._embedder.encode([query])[0]
        pool = self._vector_store.search(qvec, top_k=600, corpus="fulltext")
        handles: dict[str, str] = {}
        by_paper: dict[str, str] = {}
        per_paper: dict[str, int] = {}
        blocks: list[str] = []
        for hit in pool:
            pid = hit.chunk.paper_id
            if pid not in papers or per_paper.get(pid, 0) >= self._chunks_per_paper:
                continue
            per_paper[pid] = per_paper.get(pid, 0) + 1
            full = f"paper:{pid}"
            if full not in by_paper:
                by_paper[full] = f"P{len(by_paper) + 1}"
                handles[by_paper[full]] = full
            blocks.append(f"[{by_paper[full]}] ({hit.chunk.section_type}) {hit.chunk.text}")
        return "\n\n".join(blocks), handles

    def answer(self, query: str) -> SystemOutput:
        hits = self._retrieve(query)
        if self._vector_store is not None:
            papers = {h.paper_id for h in hits if h.paper_id}
            evidence, handles = self._chunk_evidence(query, papers)
        else:
            evidence, handles = self._format_evidence(hits)
        if not evidence:
            return SystemOutput("No relevant evidence was retrieved.", [], "")
        text, cited = VectorRAGSystem._resolve_handles(self._synthesize(query, evidence), handles)
        return SystemOutput(text=text, cited_paper_ids=cited, evidence=evidence)