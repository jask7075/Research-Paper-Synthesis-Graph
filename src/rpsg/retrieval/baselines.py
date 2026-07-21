"""Baseline systems: vector-over-abstracts and vector-over-full-text.

A `System` maps a query -> (answer text, cited paper ids, retrieved-evidence string). The
runner is system-agnostic, so `typed_graph` (Iteration 2 onward) plugs in behind the same Protocol.

The vector systems: embed the query, retrieve top-k chunks from the shared index filtered
by `corpus`, then synthesize a cited answer with the synthesis model. Citations are the
`paper:<id>` of each chunk the synthesizer was given — this is what deterministic metrics
score against, so retrieval and citation stay coupled.
"""

from __future__ import annotations

from typing import Protocol

from rpsg.config import get_settings
from rpsg.logging import get_logger
from rpsg.stores.base import Embedder, SearchHit, VectorStore

log = get_logger(__name__)


class SystemOutput:
    def __init__(self, text: str, cited_paper_ids: list[str], evidence: str) -> None:
        self.text = text
        self.cited_paper_ids = cited_paper_ids
        self.evidence = evidence


class System(Protocol):
    name: str

    def answer(self, query: str) -> SystemOutput: ...


_SYNTH_SYSTEM = """\
You are a research-synthesis assistant. Answer the question using ONLY the provided
excerpts. Cite the paper id in square brackets after each claim, e.g. [paper:abc123].
If the excerpts are insufficient for part of the question, say so explicitly and hedge —
do not fill gaps from outside knowledge. Surface any contradictions between excerpts.
"""


class VectorRAGSystem:
    """Vector retrieval + LLM synthesis. `corpus` selects abstract-only vs full-text."""

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        store: VectorStore,
        corpus: str,
        top_k: int = 20,
        synthesis_model: str | None = None,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._store = store
        self._corpus = corpus
        self._top_k = top_k
        self._synthesis_model = synthesis_model or get_settings().models.synthesis_model
        self._client = None  # lazy Anthropic client

    def _retrieve(self, query: str) -> list[SearchHit]:
        qvec = self._embedder.encode([query])[0]
        return self._store.search(qvec, top_k=self._top_k, corpus=self._corpus)

    @staticmethod
    def _format_evidence(hits: list[SearchHit]) -> tuple[str, list[str]]:
        blocks, papers = [], []
        for h in hits:
            pid = f"paper:{h.chunk.paper_id}"
            papers.append(pid)
            blocks.append(f"[{pid}] ({h.chunk.section_type}) {h.chunk.text}")
        return "\n\n".join(blocks), sorted(set(papers))

    def _synthesize(self, query: str, evidence: str) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)
        resp = self._client.messages.create(
            model=self._synthesis_model,
            max_tokens=2048,
            system=_SYNTH_SYSTEM,
            messages=[
                {"role": "user", "content": f"QUESTION: {query}\n\nEXCERPTS:\n{evidence}"}
            ],
        )
        return next((b.text for b in resp.content if b.type == "text"), "")

    def answer(self, query: str) -> SystemOutput:
        hits = self._retrieve(query)
        evidence, cited = self._format_evidence(hits)
        if not hits:
            return SystemOutput("No relevant evidence was retrieved.", [], "")
        text = self._synthesize(query, evidence)
        return SystemOutput(text=text, cited_paper_ids=cited, evidence=evidence)
