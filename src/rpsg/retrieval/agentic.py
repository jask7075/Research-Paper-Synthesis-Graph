"""Agentic arm: plan sub-questions, retrieve per sub-question, draft, critique, re-retrieve.

The hypothesis this exists to test, from §4.5 of the Iteration 2 report. Relational queries
score 0.202 against vector search's 0.357 -- the worst of any type, and the type the typed
graph was built for. Those queries are near-uniformly *"X, and what limits each"*, and the
edge that would reach the second half (`undercuts`) was traversed **zero times across 34
queries**, because 33 such edges exist in a graph of 11,186.

There were two routes to that. Add the missing edges -- attempted in §8.2, re-attempted in
3.6a with worked negatives, and **refuted**: edge precision held at 32.5% while the revision
discarded ~65% of the real edges. Or stop needing them: decompose *"which methods, and what
limits each"* into a retrieval for the methods and then one retrieval per method for its
limitations, so the missing edge stops mattering because the second hop is a second *query*
rather than a graph traversal. 3.6a's failure leaves this as the only surviving route.

**What is held constant, and why it matters more here than anywhere else.** This arm reuses
`VectorRAGSystem`'s synthesis prompt, its `P1`-handle scheme and its handle resolution
verbatim -- not a copy, the same functions. 3.5 compares this against the static arms, so any
difference in how evidence is formatted or how citations are written would show up as a
difference in the loop. The variable under test is the *control flow* and nothing else.

**The budget is enforced, not merely recorded.** An agent that wins by issuing twenty
retrievals has not beaten `top_k=60`; it has spent more. `max_retrievals` is a hard ceiling,
`RetrievalBudget` refuses the call that would exceed it, and the count is reported in the
trace so 3.5 can state the cost multiple beside the score.

**It fails closed.** If the planner returns nothing usable, the arm degrades to a single
retrieval on the original query -- i.e. to `vector_fulltext` -- and marks the trace. A silent
degradation would let a broken planner score as a working one, and 3.4 reads
`planner_failed` to exclude those queries from trajectory claims.

**It always retrieves on the original query first (the anchor).** Measured before this was
added: two runs of functionally identical code scored `must_cite_recall` 0.567 and 0.333 on
the same 10 queries. The cause is structural rather than a bug. `gpt-5.4-nano` does not
return identical text at `temperature=0.0` -- three draws of one plan gave three distinct
wordings -- and in this arm the planner's output *is* the retrieval query, so a reworded
sub-question embeds differently, reaches different chunks, and changes which papers can be
cited at all. A static arm embeds a fixed gold query and absorbs that noise; this arm
amplifies it.

The anchor puts a deterministic retrieval under every query, so the plan moves the margin
rather than the whole evidence set, and `_merge` keeps it first so the low `P1`-handles are
stable too. It is a design fix and not a score-chasing tweak: it was adopted on whether it
narrows the run-to-run spread, not on whether it raises the mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rpsg.config import get_settings
from rpsg.extraction.schema import SourceLayer
from rpsg.llm import ChatClient, get_chat_client
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import SystemOutput, VectorRAGSystem
from rpsg.stores.base import Embedder, SearchHit, VectorStore

log = get_logger(__name__)

#: Retrievals per query. Six covers a plan of up to five sub-questions plus one critique-
#: driven follow-up, which is the shape the loop is designed around. It is a ceiling, not a
#: target: a lookup query that decomposes into one sub-question spends one.
DEFAULT_MAX_RETRIEVALS = 6

#: Chunks per sub-question retrieval. Deliberately below the static arms' `top_k=60`, because
#: the agent issues several. At 20 x 5 sub-questions the arm sees ~100 chunks against the
#: static arm's 60 -- more, but within the same order, so a win is not simply more evidence.
#: 3.5's required breakdown is what separates those two explanations.
DEFAULT_TOP_K = 20

_PLANNER_SYSTEM = """\
You decompose a research question into the retrievals needed to answer it.

Return JSON only: {"sub_questions": ["...", "..."], "reasoning": "<20 words"}

Rules:
- Each sub-question must be answerable by searching a corpus of paper excerpts on its own.
  It becomes a standalone search query, so it must carry its own context: write "what limits
  the QAOA mixer Hamiltonian", never "what limits it".
- Decompose only where the question genuinely has parts. A single-fact lookup gets ONE
  sub-question. Inventing parts wastes retrievals and dilutes the evidence.
- A question of the form "which X, and what limits each" needs a sub-question for the X and
  then one per likely X for its limits. That shape is the reason this planner exists.
- At most 5. Prefer 2-3.
"""

_CRITIC_SYSTEM = """\
You check a draft answer against the plan it was written from, and name what is missing.

Return JSON only:
{"gaps": ["<a search query that would fill the gap>"], "assessment": "<25 words"}

A gap is a sub-question the draft does not actually answer from the excerpts, or a claim the
draft makes without evidence. Write each gap as a SEARCH QUERY, not as a complaint: the
string is passed straight to the retriever.

Return {"gaps": []} if the draft addresses the plan. An empty list is the expected outcome
for a well-answered question -- do not invent gaps to appear diligent, and do not list a gap
the excerpts have already shown the corpus cannot answer.
At most 2 gaps.
"""

_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sub_questions": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["sub_questions", "reasoning"],
}

_CRITIQUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "gaps": {"type": "array", "items": {"type": "string"}},
        "assessment": {"type": "string"},
    },
    "required": ["gaps", "assessment"],
}


class BudgetExhausted(RuntimeError):
    """Raised when a retrieval would exceed the per-query ceiling."""


@dataclass
class RetrievalBudget:
    """A hard ceiling on retrievals per query, with the spend recorded.

    A cap that is logged but not enforced is not a cap. §3.5 has to report the cost multiple
    beside the score, and a run whose agent quietly issued twenty retrievals on the hard
    queries and two on the easy ones would report a misleading mean unless the ceiling
    actually bound.
    """

    limit: int
    used: int = 0
    refused: int = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            self.refused += 1
            raise BudgetExhausted(f"retrieval budget of {self.limit} exhausted")
        self.used += 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


@dataclass
class Trajectory:
    """What the loop did, for 3.4 to score.

    Persisted per query in `traces.jsonl`. 3.4's measures read directly off these fields:
    decomposition coverage from `sub_questions`, retrieval efficiency from `retrievals_used`
    against papers found, and critique usefulness from `gaps` plus the paper sets before and
    after -- which is why `papers_before_critique` is recorded separately rather than
    reconstructed.
    """

    query: str
    sub_questions: list[str] = field(default_factory=list)
    plan_reasoning: str = ""
    graph_hints: list[str] = field(default_factory=list)
    per_sub_question: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    critique_assessment: str = ""
    papers_before_critique: list[str] = field(default_factory=list)
    papers_after_critique: list[str] = field(default_factory=list)
    retrievals_used: int = 0
    retrievals_refused: int = 0
    planner_failed: bool = False
    critique_ran: bool = False
    anchor_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sub_questions": self.sub_questions,
            "plan_reasoning": self.plan_reasoning,
            "graph_hints": self.graph_hints,
            "per_sub_question": self.per_sub_question,
            "gaps": self.gaps,
            "critique_assessment": self.critique_assessment,
            "papers_before_critique": self.papers_before_critique,
            "papers_after_critique": self.papers_after_critique,
            "retrievals_used": self.retrievals_used,
            "retrievals_refused": self.retrievals_refused,
            "planner_failed": self.planner_failed,
            "critique_ran": self.critique_ran,
            "anchor_used": self.anchor_used,
            # Derived here rather than in 3.4 so the definition lives with the data: did the
            # second pass actually reach a paper the first did not?
            "critique_added_papers": sorted(
                set(self.papers_after_critique) - set(self.papers_before_critique)
            ),
        }


class AgenticSystem:
    """Planner -> per-sub-question retrieval -> draft -> critique -> re-retrieve -> synthesise.

    Behind the same `System` Protocol as every other arm, so `06_run_eval.py` scores it
    unchanged. The runner has been system-agnostic since Iteration 1 and this is what that
    was for.
    """

    def __init__(
        self,
        name: str,
        embedder: Embedder,
        store: VectorStore,
        *,
        corpus: str = "fulltext",
        top_k: int = DEFAULT_TOP_K,
        max_retrievals: int = DEFAULT_MAX_RETRIEVALS,
        planner_model: str | None = None,
        synthesis_model: str | None = None,
        graph_store: Any = None,
        critique: bool = True,
        stage_writes: bool = False,
        anchor: bool = True,
    ) -> None:
        settings = get_settings()
        self.name = name
        self._embedder = embedder
        self._store = store
        self._corpus = corpus
        self._top_k = top_k
        self._max_retrievals = max_retrievals
        # The planner and critic run on the cheaper model by default: they emit a handful of
        # short strings, and §3.3's point is that the loop multiplies calls. Synthesis keeps
        # the static arms' model so the writing is held constant.
        self._planner_model = planner_model or settings.models.extraction_model
        self._synthesis_model = synthesis_model or settings.models.synthesis_model
        self._graph_store = graph_store
        # The ablation §3.5 requires: without it, "the loop helps" cannot be separated from
        # "planning helps".
        self._critique = critique
        # See the module docstring: a deterministic retrieval on the original query, under
        # every plan. `anchor=False` is the ablation that recovers the pre-3.5 behaviour and
        # is how the spread reduction was measured.
        self._anchor = anchor
        # §3.2. Off by default: 3.5's acceptance is that the scored run is unaffected, and
        # the cleanest way to guarantee that is for the deliverable not to write at all.
        # The equality is demonstrated by running both ways, not assumed.
        self._stage_writes = stage_writes and graph_store is not None
        self._planner: ChatClient | None = None
        self._synth: VectorRAGSystem = VectorRAGSystem(
            name=f"{name}:synth",
            embedder=embedder,
            store=store,
            corpus=corpus,
            top_k=top_k,
            synthesis_model=self._synthesis_model,
        )

    # ---- planning ---------------------------------------------------------------

    def _graph_hints(self, query: str) -> list[str]:
        """`Method` names reachable from the question along `addresses`.

        This is the one thing Iteration 2 showed traversal does reliably: `addresses`
        carries 35.7% of all traversal and connects a problem to the methods that tackle it.
        §5.2 showed the typed graph is a poor *retriever* -- free citation edges match it
        within 0.016 -- but that says nothing about whether it is a useful *planner*. Here it
        supplies candidate names to decompose over, never evidence, so a wrong hint costs a
        sub-question rather than a wrong citation.

        Best-effort: any failure returns no hints and planning proceeds without them.
        """
        if self._graph_store is None:
            return []
        try:
            rows = self._graph_store.query(
                # `Entity`, and CURATED only: the layer invariant in `stores/base.py` is that
                # metrics never see STAGED, and a planner hint feeds a scored run.
                "MATCH (p:Entity)-[r:REL]->(m:Entity) "
                "WHERE r.type = 'addresses' AND m.type = 'Method' "
                "AND m.source_layer = $curated "
                "RETURN DISTINCT m.name AS name, r.confidence AS confidence "
                "ORDER BY confidence DESC LIMIT 200",
                {"curated": SourceLayer.CURATED.value},
            )
        except Exception as exc:  # noqa: BLE001 - a planner hint must never fail a query
            log.warning("graph hints unavailable, planning without them: %s", exc)
            return []
        names = [r["name"] for r in rows if r.get("name")]
        if not names:
            return []
        # Rank by similarity to the question rather than by edge confidence alone: confidence
        # orders how firmly the paper asserted the edge, not how relevant the method is here.
        import numpy as np

        vecs = np.asarray(self._embedder.encode(names), dtype="float32")
        qv = np.asarray(self._embedder.encode([query])[0], dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        qv /= np.linalg.norm(qv) + 1e-9
        order = np.argsort(-(vecs @ qv))[:8]
        return [names[i] for i in order]

    def _plan(self, query: str, traj: Trajectory) -> list[str]:
        if self._planner is None:
            self._planner = get_chat_client(self._planner_model)
        hints = self._graph_hints(query)
        traj.graph_hints = hints
        hint_block = (
            "\n\nMethods the knowledge graph associates with this problem area, as candidate "
            "parts to decompose over (ignore any that do not fit):\n- " + "\n- ".join(hints)
            if hints
            else ""
        )
        try:
            raw = self._planner.json(
                system=_PLANNER_SYSTEM,
                user=f"QUESTION: {query}{hint_block}",
                schema=_PLAN_SCHEMA,
                schema_name="plan",
                max_tokens=1024,
                temperature=0.0,
            )
            subs = [s.strip() for s in raw.get("sub_questions", []) if s and s.strip()]
            traj.plan_reasoning = str(raw.get("reasoning", ""))[:200]
        except Exception as exc:  # noqa: BLE001 - fail closed, do not fail the query
            log.warning("planner failed for %r, falling back to one retrieval: %s", query, exc)
            subs = []
        if not subs:
            traj.planner_failed = True
            return [query]
        # Leave room for the critique, and for the anchor when it is enabled. A plan that
        # consumed the whole budget would make the loop a fan-out rather than a loop.
        reserved = 1 + (1 if self._anchor else 0)
        return subs[: max(1, self._max_retrievals - reserved)]

    # ---- retrieval --------------------------------------------------------------

    def _retrieve(self, question: str, budget: RetrievalBudget) -> list[SearchHit]:
        budget.spend()
        qvec = self._embedder.encode([question])[0]
        return self._store.search(qvec, top_k=self._top_k, corpus=self._corpus)

    @staticmethod
    def _merge(batches: list[list[SearchHit]]) -> list[SearchHit]:
        """Dedupe by chunk id, preserving first-seen order.

        Order is load-bearing: `_format_evidence` assigns `P1`, `P2`, ... by first
        appearance, so a stable order makes the handle mapping deterministic for a given
        plan. Sub-questions are retrieved in plan order, so the earlier parts of the question
        get the lower handles.
        """
        seen: set[str] = set()
        out: list[SearchHit] = []
        for batch in batches:
            for h in batch:
                cid = getattr(h.chunk, "id", None) or f"{h.chunk.paper_id}:{h.chunk.text[:64]}"
                if cid not in seen:
                    seen.add(cid)
                    out.append(h)
        return out

    # ---- the loop ---------------------------------------------------------------

    def answer(self, query: str) -> SystemOutput:
        budget = RetrievalBudget(self._max_retrievals)
        traj = Trajectory(query=query)

        subs = self._plan(query, traj)
        traj.sub_questions = subs

        batches: list[list[SearchHit]] = []
        # The anchor: one deterministic retrieval on the question as asked, before any
        # generated text is involved. Spent from the same budget, so the arm is not handed a
        # free extra retrieval the static arms do not get.
        if self._anchor:
            try:
                batches.append(self._retrieve(query, budget))
                traj.anchor_used = True
            except BudgetExhausted:
                log.warning("budget exhausted before the anchor retrieval")

        for sub in subs:
            try:
                hits = self._retrieve(sub, budget)
            except BudgetExhausted:
                log.info("budget exhausted before sub-question %r", sub)
                break
            batches.append(hits)
            traj.per_sub_question.append({
                "sub_question": sub,
                "hits": len(hits),
                "papers": sorted({h.chunk.paper_id for h in hits}),
            })

        hits = self._merge(batches)
        if not hits:
            traj.retrievals_used, traj.retrievals_refused = budget.used, budget.refused
            return SystemOutput(
                "No relevant evidence was retrieved.", [], "", trace=traj.as_dict()
            )

        traj.papers_before_critique = sorted({h.chunk.paper_id for h in hits})

        if self._critique and budget.remaining:
            gaps, assessment = self._critique_draft(query, subs, hits)
            traj.gaps, traj.critique_assessment, traj.critique_ran = gaps, assessment, True
            for gap in gaps:
                try:
                    batches.append(self._retrieve(gap, budget))
                except BudgetExhausted:
                    break
            hits = self._merge(batches)

        traj.papers_after_critique = sorted({h.chunk.paper_id for h in hits})
        traj.retrievals_used, traj.retrievals_refused = budget.used, budget.refused

        # The static arms' formatting and synthesis, unchanged — the whole point.
        evidence, handles = self._synth._format_evidence(hits)
        text, cited = self._synth._resolve_handles(
            self._synth._synthesize(query, evidence), handles
        )
        trace = traj.as_dict()
        if self._stage_writes:
            self._stage(trace)
        return SystemOutput(
            text=text, cited_paper_ids=cited, evidence=evidence, trace=trace
        )

    def _stage(self, trace: dict[str, Any]) -> None:
        """Persist the decomposition as STAGED. Never allowed to fail a query.

        A query-time write is a side effect of answering, not part of it. An answer that
        succeeded must not be lost because the graph was locked or the write was malformed.
        """
        from rpsg.retrieval.staging import decomposition_nodes, write_staged

        try:
            nodes, edges = decomposition_nodes(
                trace, qid=trace.get("qid", ""), system=self.name, model=self._planner_model
            )
            write_staged(self._graph_store, nodes, edges)
        except Exception as exc:  # noqa: BLE001 - staging must never fail an answer
            log.warning("staging failed, answer unaffected: %s", exc)

    def _critique_draft(
        self, query: str, subs: list[str], hits: list[SearchHit]
    ) -> tuple[list[str], str]:
        """Name what the evidence does not cover, as search queries.

        Critiques the *evidence against the plan* rather than a written draft. Drafting twice
        would double synthesis cost -- the expensive call -- to check something the excerpts
        already determine: if no excerpt bears on a sub-question, no draft written from those
        excerpts can answer it. §3.5 reports cost per query, so a step that doubles the
        largest cost has to earn it, and this one cannot yet show it would.
        """
        if self._planner is None:
            self._planner = get_chat_client(self._planner_model)
        evidence, _ = self._synth._format_evidence(hits)
        try:
            raw = self._planner.json(
                system=_CRITIC_SYSTEM,
                user=(
                    f"QUESTION: {query}\n\nPLAN:\n- " + "\n- ".join(subs)
                    + f"\n\nEXCERPTS RETRIEVED:\n{evidence[:24000]}"
                ),
                schema=_CRITIQUE_SCHEMA,
                schema_name="critique",
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - a failed critique is no critique, not a failure
            log.warning("critique failed for %r: %s", query, exc)
            return [], "critique failed"
        gaps = [g.strip() for g in raw.get("gaps", []) if g and g.strip()]
        return gaps[: min(2, self._max_retrievals)], str(raw.get("assessment", ""))[:200]


def load_trajectories(path: str) -> list[dict[str, Any]]:
    """Trajectories from a run's `traces.jsonl`, for 3.4."""
    from pathlib import Path

    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("trajectory"):
                out.append({"qid": row["qid"], **row["trajectory"]})
    return out