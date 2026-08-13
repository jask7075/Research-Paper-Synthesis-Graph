"""Citation-graph retrieval: the ablation that says what the *typed* layer is worth.

`TypedGraphSystem` walks edges the extractor produced -- `addresses`, `builds_on`,
`undercuts` -- which cost ~$6 of extraction per corpus refresh and are only as good as the
model that wrote them. This walks `cites` edges instead: Tier-A metadata straight from
Semantic Scholar, free, and objectively correct in the sense that a citation either exists
or does not.

Everything else is held constant. Same seeding shape, same hop budget, same node cap, same
chunk routing, same synthesis prompt. The only variable is which edges are traversed.

That makes the comparison decisive in either direction. If this loses too, the finding is
about graph retrieval on this corpus and the typed layer is not the culprit. If it *wins*,
the typed layer is worse than free metadata, and the extraction was the problem rather than
the idea. Without this arm, "typed graphs did not beat vector search" cannot be separated
from "no graph beats vector search here", and those support very different theses.

Two seeding modes, because they answer different questions:

    "title"   embed the query against paper titles. Structurally matched to the typed arm,
              which seeds on node names -- both get a short string per node and no chunk
              search. This is the honest ablation.
    "chunks"  seed from vector retrieval, then expand along citations. This is what a
              practitioner would actually build, and it inherits the vector arm's seeding
              strength, so it measures whether citations *add* anything to vector search
              rather than whether they replace it.

Reporting one without the other invites the wrong reading, so `06_run_eval.py` exposes
both as `citation_graph` and `citation_graph_seeded`.
"""

from __future__ import annotations

from typing import Any

from rpsg.config import get_settings
from rpsg.extraction.schema import SourceLayer
from rpsg.llm import get_chat_client
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import _SYNTH_SYSTEM, SystemOutput, VectorRAGSystem
from rpsg.stores.base import Embedder
from rpsg.stores.graph_store import KuzuGraphStore

log = get_logger(__name__)


def _bare(node_id: str) -> str:
    """`paper:<40hex>` -> `<40hex>`. Chunks are keyed without the prefix."""
    return node_id.split(":", 1)[1] if node_id.startswith("paper:") else node_id


class CitationGraphSystem:
    """Seed papers, expand along `cites`, answer from chunks of the papers reached."""

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        store: KuzuGraphStore,
        vector_store: Any,
        *,
        seeds: int = 12,
        hops: int = 2,
        max_nodes: int = 30,
        chunks_per_paper: int = 4,
        seed_from: str = "title",
        synthesis_model: str | None = None,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._store = store
        self._vector_store = vector_store
        # Seeds and hops are matched to TypedGraphSystem's measured defaults rather than
        # re-tuned. Re-sweeping only this arm would hand it an advantage the typed arm
        # never got, and the question is what the edges are worth, not what a tuned
        # citation walker can do.
        self._seeds = seeds
        self._hops = hops
        # `max_nodes` is NOT matched at 150, and copying that number would have quietly
        # rigged the comparison. In the typed graph a node is an entity, and 150 entities
        # route to ~10 papers (measured: mean 10.1, max 28 over the gold set). Here a node
        # *is* a paper, so 150 would mean 150 papers -- 55 of them with chunks, 201k chars
        # of evidence against the typed arm's 61k. The citation arm would then win on
        # evidence volume rather than on structure, which is not the question.
        #
        # 30 is the typed arm's observed ceiling in papers, so this arm is never
        # evidence-starved relative to it and cannot outspend it either. Both use
        # chunks_per_paper=4, so equal papers means equal volume.
        self._max_nodes = max_nodes
        self._chunks_per_paper = chunks_per_paper
        self._seed_from = seed_from
        self._synthesis_model = synthesis_model or get_settings().models.synthesis_model
        self._client: Any = None
        self._papers: list[dict[str, Any]] = []
        self._vecs: Any = None

    def _load_titles(self) -> None:
        if self._vecs is not None:
            return
        import numpy as np

        rows = self._store.query(
            # CURATED only -- the layer invariant, now enforced rather than assumed.
            "MATCH (e:Entity) WHERE e.type = 'Paper' AND e.source_layer = $curated "
            "RETURN e.id AS id, e.name AS name",
            {"curated": SourceLayer.CURATED.value},
        )
        self._papers = [r for r in rows if (r.get("name") or "").strip()]
        vecs = np.asarray(
            self._embedder.encode([r["name"] for r in self._papers]), dtype="float32"
        )
        self._vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        log.info("citation-graph: embedded %d paper titles", len(self._papers))

    def _seed_ids(self, query: str) -> list[str]:
        import numpy as np

        if self._seed_from == "chunks":
            qvec = self._embedder.encode([query])[0]
            out: list[str] = []
            for hit in self._vector_store.search(qvec, top_k=200, corpus="fulltext"):
                pid = f"paper:{hit.chunk.paper_id}"
                if pid not in out:
                    out.append(pid)
                if len(out) >= self._seeds:
                    break
            return out

        self._load_titles()
        if not self._papers:
            return []
        q = np.asarray(self._embedder.encode([query])[0], dtype="float32")
        q /= np.linalg.norm(q) + 1e-9
        sims = self._vecs @ q
        return [self._papers[int(i)]["id"] for i in np.argsort(-sims)[: self._seeds]]

    def _expand(self, seed_ids: list[str]) -> list[str]:
        """Walk `cites` outward, ignoring direction.

        A paper's references and the papers citing it are both relevant neighbours: for
        "what approaches exist and how do they compare", the survey citing three methods
        and the three methods citing a common predecessor are equally useful routes. A
        directed walk would answer only half of those.
        """
        seen = set(seed_ids)
        frontier = list(seed_ids)
        for _ in range(self._hops):
            if not frontier or len(seen) >= self._max_nodes:
                break
            rows = self._store.query(
                "MATCH (a:Entity)-[r:REL]-(b:Entity) WHERE a.id IN $ids AND r.type = 'cites' "
                "AND b.source_layer = $curated AND r.source_layer = $curated "
                "AND b.type = 'Paper' RETURN DISTINCT b.id AS id",
                {"ids": frontier, "curated": SourceLayer.CURATED.value},
            )
            frontier = []
            for row in rows:
                if row["id"] in seen or len(seen) >= self._max_nodes:
                    continue
                seen.add(row["id"])
                frontier.append(row["id"])
        return list(seen)

    def _chunk_evidence(self, query: str, papers: set[str]) -> tuple[str, dict[str, str]]:
        """Chunks from the reached papers, ranked by query similarity.

        Identical to the typed arm's routing, deliberately: if the evidence were formatted
        differently the comparison would confound edges with presentation, which is the
        mistake the first typed-graph run made when it passed bare node quotes.
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
        reached = self._expand(self._seed_ids(query))
        evidence, handles = self._chunk_evidence(query, {_bare(p) for p in reached})
        if not evidence:
            return SystemOutput("No relevant evidence was retrieved.", [], "")
        if self._client is None:
            self._client = get_chat_client(self._synthesis_model)
        text = self._client.text(
            system=_SYNTH_SYSTEM,
            user=f"QUESTION: {query}\n\nEXCERPTS:\n{evidence}",
            max_tokens=4096,
        )
        answer, cited = VectorRAGSystem._resolve_handles(text, handles)
        return SystemOutput(text=answer, cited_paper_ids=cited, evidence=evidence)