"""Baseline systems: vector-over-abstracts and vector-over-full-text.

A `System` maps a query -> (answer text, cited paper ids, retrieved-evidence string). The
runner is system-agnostic, so `typed_graph` (Iteration 2 onward) plugs in behind the same Protocol.

The vector systems: embed the query, retrieve top-k chunks from the shared index filtered
by `corpus`, then synthesize a cited answer with the synthesis model. Citations are the
`paper:<id>` of each chunk the synthesizer was given — this is what deterministic metrics
score against, so retrieval and citation stay coupled.
"""

from __future__ import annotations

import re
from typing import Protocol

from rpsg.config import get_settings
from rpsg.llm import ChatClient, get_chat_client
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
excerpts. Cite the short paper handle in square brackets after each claim, e.g. [P1].
Use one handle per bracket: write [P1][P3], not [P1, P3]. Use only handles that appear
in the excerpts.
If the excerpts are insufficient for part of the question, say so explicitly and hedge —
do not fill gaps from outside knowledge. Surface any contradictions between excerpts.
"""

#: A citation handle in the model's answer. Tolerates `[P1, P3]` even though the prompt
#: asks for `[P1][P3]`, because an unmatched bracket would otherwise survive into the
#: answer as a dangling handle.
_HANDLE = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")


class VectorRAGSystem:
    """Vector retrieval + LLM synthesis. `corpus` selects abstract-only vs full-text."""

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        store: VectorStore,
        corpus: str,
        top_k: int = 60,
        synthesis_model: str | None = None,
    ) -> None:
        self.name = name
        self._embedder = embedder
        self._store = store
        self._corpus = corpus
        self._top_k = top_k
        self._synthesis_model = synthesis_model or get_settings().models.synthesis_model
        self._client: ChatClient | None = None  # built on first synthesis call

    def _retrieve(self, query: str) -> list[SearchHit]:
        qvec = self._embedder.encode([query])[0]
        return self._store.search(qvec, top_k=self._top_k, corpus=self._corpus)

    @staticmethod
    def _format_evidence(hits: list[SearchHit]) -> tuple[str, dict[str, str]]:
        """Label excerpts with short handles; return the text and handle -> paper id.

        Papers are identified to the model as `P1`, `P2`, ... rather than by their
        40-hex-character Semantic Scholar id. Asked to copy those ids verbatim, the
        model mangles them — observed in a real answer, one id came back with an
        internal fragment duplicated and another truncated. Because `runner.py`
        harvests citations by regex from the answer text, a mangled id counts as a
        distinct cited paper and silently corrupts `must_cite_recall` and
        `citation_precision`. Short handles are copyable, and unknown ones can be
        rejected on the way back (see `_resolve_handles`).
        """
        handles: dict[str, str] = {}
        by_paper: dict[str, str] = {}
        blocks = []
        for h in hits:
            pid = f"paper:{h.chunk.paper_id}"
            if pid not in by_paper:
                handle = f"P{len(by_paper) + 1}"  # first-appearance order: deterministic
                by_paper[pid] = handle
                handles[handle] = pid
            blocks.append(f"[{by_paper[pid]}] ({h.chunk.section_type}) {h.chunk.text}")
        return "\n\n".join(blocks), handles

    @staticmethod
    def _resolve_handles(text: str, handles: dict[str, str]) -> tuple[str, list[str]]:
        """Rewrite `[P1]` back to `[paper:<id>]` and report which papers were cited.

        Restoring real ids in the stored answer keeps the downstream contract intact:
        `runner.py`'s citation regex, the deterministic metrics, and gold-set matching
        all continue to work on `paper:<id>`. A handle that was never issued is dropped
        rather than passed through, so a hallucinated citation cannot inflate the
        cited-paper set.
        """
        cited: set[str] = set()

        def replace(match: re.Match[str]) -> str:
            resolved = []
            for raw in match.group(1).split(","):
                pid = handles.get(raw.strip())
                if pid is None:
                    continue  # handle the model invented — drop it
                cited.add(pid)
                resolved.append(f"[{pid}]")
            return "".join(resolved)

        resolved_text = _HANDLE.sub(replace, text)
        # Dropping a handle would otherwise leave the space that preceded it stranded
        # before the punctuation ("...from nowhere .").
        resolved_text = re.sub(r"[ \t]+([.,;:!?])", r"\1", resolved_text)
        return resolved_text, sorted(cited)

    def _synthesize(self, query: str, evidence: str) -> str:
        if self._client is None:
            self._client = get_chat_client(self._synthesis_model)
        return self._client.text(
            system=_SYNTH_SYSTEM,
            user=f"QUESTION: {query}\n\nEXCERPTS:\n{evidence}",
            max_tokens=4096,
        )

    def answer(self, query: str) -> SystemOutput:
        hits = self._retrieve(query)
        if not hits:
            return SystemOutput("No relevant evidence was retrieved.", [], "")
        evidence, handles = self._format_evidence(hits)
        text, cited = self._resolve_handles(self._synthesize(query, evidence), handles)
        # `cited` is reported as-is, including empty. An earlier version fell back to
        # every retrieved paper when the model cited nothing, on the reasoning that
        # `citation_precision` needed a denominator — it does not, it returns 1.0 for an
        # uncited answer by design. The fallback instead credited the answer with citing
        # everything retrieved, so an answer that cited *nothing* scored
        # `must_cite_recall = 1.0` whenever retrieval had found the required papers, and
        # `wellformed.check_answer` could not see the failure either because the ids were
        # already backfilled. Reporting the truth makes recall 0.0 and raises
        # `no_citations`, which is what a total attribution failure should look like.
        return SystemOutput(text=text, cited_paper_ids=cited, evidence=evidence)
