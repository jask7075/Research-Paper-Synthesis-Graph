"""The ask-a-question service, with no web framework in it.

`app.py` is a thin translation layer: HTTP in, `AskService` out. Everything that decides
*what happens* — which arm gets built, what a question costs, what is refused and why —
lives here, so it can be tested without starting a server and without installing FastAPI
(CI installs neither the `web` nor the `vector` extras).

Two things this deliberately does not do:

* It does not re-choose defaults. `top_k` and the agentic flags are passed through to
  `build_system` exactly as `ask.py` passes them, `None` meaning "the measured default".
  A web form that quietly substituted its own numbers would show the user an arm the
  report never scored.
* It does not expose `stage_writes`. The CLI can persist a decomposition as STAGED nodes;
  a browser tab that writes to the graph on a mistyped question is a different risk, so
  the UI is read-only and the CLI keeps that flag.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from rpsg.config import get_settings
from rpsg.llm.usage import USAGE, ModelUsage, cost_usd
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import System, VectorRAGSystem
from rpsg.retrieval.build import (
    NEEDS_GRAPH,
    RETRIEVES_CHUNKS_UP_FRONT,
    SYSTEMS,
    build_system,
    missing_store,
)
from rpsg.stores.base import Embedder, SearchHit
from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder

log = get_logger(__name__)

#: How much of a chunk to send to the browser per retrieved hit. The full text of 20
#: full-text chunks is megabytes of JSON for a panel nobody scrolls to the end of; the
#: whole evidence string is available separately when the caller asks for it.
EXCERPT_CHARS = 600


class ServiceError(Exception):
    """A refusal the user can act on: no index built, no API key, unknown arm.

    Separated from unexpected failures because the two want different HTTP codes and,
    more importantly, different text. These carry a message written for the person who
    typed the question; anything else is a traceback that belongs in the log.
    """


@dataclass(frozen=True)
class ArmInfo:
    """What the UI needs to render one arm in the picker."""

    name: str
    supports_retrieval_only: bool
    needs_graph: bool
    ready: bool
    #: Why the arm cannot run, when `ready` is False. Shown instead of a silent disable.
    blocked_on: str | None = None


@dataclass(frozen=True)
class HitView:
    paper_id: str
    chunk_id: str
    section_type: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class UsageRow:
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    #: None means unpriced, not free — see `rpsg.llm.usage.cost_usd`.
    cost_usd: float | None


@dataclass
class AskResult:
    question: str
    arm: str
    retrieval_only: bool
    elapsed_s: float
    #: None only when `retrieval_only` — the run stopped before synthesis.
    answer: str | None = None
    cited_paper_ids: list[str] = field(default_factory=list)
    #: The excerpts as the model received them. Populated on request; it is large.
    evidence: str | None = None
    #: The agentic arm's plan, retrievals and critique. Empty for the static arms.
    trace: dict[str, Any] = field(default_factory=dict)
    #: Vector arms only — the graph and agentic arms have no single pre-synthesis list.
    hits: list[HitView] = field(default_factory=list)
    usage: list[UsageRow] = field(default_factory=list)


def _usage_delta(before: dict[str, ModelUsage], after: dict[str, ModelUsage]) -> list[UsageRow]:
    """Per-model usage attributable to one question.

    `USAGE` is a process-wide running total, so the figure for a single question is a
    difference of two snapshots. That is only sound while one question runs at a time,
    which is why `AskService.ask` takes the lock around the whole call rather than only
    around the arm cache.
    """
    pricing = get_settings().models.pricing or {}
    rows = []
    for model, post in sorted(after.items()):
        pre = before.get(model, ModelUsage())
        delta = ModelUsage(
            calls=post.calls - pre.calls,
            input_tokens=post.input_tokens - pre.input_tokens,
            output_tokens=post.output_tokens - pre.output_tokens,
            cached_input_tokens=post.cached_input_tokens - pre.cached_input_tokens,
        )
        if delta.calls == 0:
            continue
        rows.append(
            UsageRow(
                model=model,
                calls=delta.calls,
                input_tokens=delta.input_tokens,
                output_tokens=delta.output_tokens,
                cached_input_tokens=delta.cached_input_tokens,
                cost_usd=cost_usd(delta, pricing.get(model)),
            )
        )
    return rows


class AskService:
    """Holds the loaded embedder and the built arms for the lifetime of the server.

    Building an arm means loading a 768-dim sentence-transformer and memory-mapping a
    34MB FAISS index. Doing that per request would put ten seconds in front of every
    question, so both are built once and reused. The cache key is the full option tuple:
    changing `top_k` has to produce a different arm, not silently reuse the last one.
    """

    def __init__(self, *, hash_embed: bool = False) -> None:
        self._hash_embed = hash_embed
        self._embedder: Embedder | None = None
        self._systems: dict[tuple[Any, ...], System] = {}
        #: One question at a time. The arms hold FAISS and Kuzu handles that were never
        #: written for concurrent use, and the usage delta above needs the serialization
        #: anyway. A research UI has one user, so the queue costs nothing real.
        self._lock = threading.Lock()

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            settings = get_settings()
            self._embedder = (
                HashEmbedder(dim=settings.embeddings.dim)
                if self._hash_embed
                else SentenceTransformerEmbedder(
                    settings.embeddings.model_name,
                    settings.embeddings.dim,
                    settings.embeddings.batch_size,
                )
            )
        return self._embedder

    def _get_system(self, key: tuple[Any, ...]) -> System:
        if key not in self._systems:
            name, top_k, max_retrievals, graph_hints, anchor = key
            log.info("building arm %s (top_k=%s)", name, top_k)
            self._systems[key] = build_system(
                name,
                embedder=self._get_embedder(),
                top_k=top_k,
                max_retrievals=max_retrievals,
                graph_hints=graph_hints,
                anchor=anchor,
                stage_writes=False,
            )
        return self._systems[key]

    def arms(self) -> list[ArmInfo]:
        """Every arm, and whether its stores exist. Ordered as `SYSTEMS` declares them."""
        out = []
        for name in SYSTEMS:
            problem = missing_store(name)
            out.append(
                ArmInfo(
                    name=name,
                    supports_retrieval_only=name in RETRIEVES_CHUNKS_UP_FRONT,
                    needs_graph=name in NEEDS_GRAPH,
                    ready=problem is None,
                    blocked_on=problem,
                )
            )
        return out

    def ask(
        self,
        question: str,
        *,
        system: str = "vector_fulltext",
        top_k: int | None = None,
        retrieval_only: bool = False,
        show_evidence: bool = False,
        max_retrievals: int | None = None,
        graph_hints: bool = True,
        anchor: bool = True,
    ) -> AskResult:
        """Answer `question` on one arm. Raises `ServiceError` for anything the user can fix."""
        question = question.strip()
        if not question:
            raise ServiceError("Ask a question first.")
        if system not in SYSTEMS:
            raise ServiceError(f"unknown arm {system!r}; choose from {', '.join(SYSTEMS)}")
        if retrieval_only and system not in RETRIEVES_CHUNKS_UP_FRONT:
            # Refused, not downgraded to a full answer: the user asked for no LLM call,
            # and spending money they declined to spend is the wrong way to be helpful.
            raise ServiceError(
                f"retrieval-only is not available for {system!r}: it plans or traverses "
                "before it retrieves, so there is no pre-synthesis chunk list. Vector arms "
                f"only: {', '.join(sorted(RETRIEVES_CHUNKS_UP_FRONT))}"
            )
        if top_k is not None and system not in RETRIEVES_CHUNKS_UP_FRONT:
            raise ServiceError(
                f"top-k applies to the vector arms only; {system!r} uses measured "
                "seeds/hops/max-nodes defaults that a form default would overwrite."
            )
        if (problem := missing_store(system)) is not None:
            raise ServiceError(problem)

        key = (system, top_k, max_retrievals, graph_hints, anchor)
        with self._lock:
            started = time.perf_counter()
            before = USAGE.snapshot()
            try:
                arm = self._get_system(key)
                result = self._run(arm, question, system, retrieval_only, show_evidence)
            except ServiceError:
                raise
            except RuntimeError as exc:
                # The adapters raise this for a missing API key, which is configuration
                # the user can fix, so it reaches them as text rather than a 500.
                raise ServiceError(str(exc)) from exc
            result.elapsed_s = time.perf_counter() - started
            result.usage = _usage_delta(before, USAGE.snapshot())
        return result

    def _run(
        self,
        arm: System,
        question: str,
        system: str,
        retrieval_only: bool,
        show_evidence: bool,
    ) -> AskResult:
        result = AskResult(
            question=question, arm=arm.name, retrieval_only=retrieval_only, elapsed_s=0.0
        )

        # `isinstance` rather than a membership test on the arm name: it is the same set
        # (RETRIEVES_CHUNKS_UP_FRONT is exactly the vector arms) and it is the half the
        # type checker can see, since `_retrieve` and `_corpus` belong to the class and
        # not to the `System` protocol every arm satisfies.
        if isinstance(arm, VectorRAGSystem):
            # Same private call `ask.py` makes, for the same reason: this is the exact
            # retrieval the arm is about to synthesize from, so showing anything else
            # would be showing a second, differently-configured search.
            hits: list[SearchHit] = arm._retrieve(question)  # noqa: SLF001
            result.hits = [
                HitView(
                    paper_id=h.chunk.paper_id,
                    chunk_id=h.chunk.id,
                    section_type=h.chunk.section_type,
                    score=h.score,
                    excerpt=" ".join(h.chunk.text.split())[:EXCERPT_CHARS],
                )
                for h in hits
            ]
            if not hits:
                raise ServiceError(
                    "Nothing retrieved. Either the index is empty, or it holds no chunks "
                    f"for corpus={arm._corpus!r} (abstract chunks exist only if stage 01 "  # noqa: SLF001
                    "fetched metadata)."
                )
            if retrieval_only:
                return result

        out = arm.answer(question)
        result.answer = out.text
        result.cited_paper_ids = list(out.cited_paper_ids)
        result.trace = out.trace
        if show_evidence:
            result.evidence = out.evidence
        return result
