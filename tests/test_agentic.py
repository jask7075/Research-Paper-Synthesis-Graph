"""The agentic loop's control flow (§3.1). Deterministic: no API key, no network.

Three properties carry 3.5's validity, and each fails silently if broken:

  * the budget is a ceiling that BINDS, not a counter. An arm that quietly issued twenty
    retrievals would beat `top_k=60` by spending more, and the cost multiple 3.5 reports
    beside the score would be wrong rather than absent.
  * the loop fails CLOSED. A broken planner must degrade to one retrieval and say so in the
    trace, because a silent degradation scores as a working agent.
  * synthesis is the static arms' code, not a copy of it. 3.5's variable is the control
    flow; a divergent prompt or handle scheme would show up as a difference in the loop.
"""

from __future__ import annotations

import pytest

from rpsg.retrieval.agentic import (
    AgenticSystem,
    BudgetExhausted,
    RetrievalBudget,
    Trajectory,
)


class _Chunk:
    def __init__(self, pid: str, text: str, cid: str) -> None:
        self.paper_id, self.text, self.id = pid, text, cid
        self.section_type = "method"


class _Hit:
    def __init__(self, chunk: _Chunk) -> None:
        self.chunk = chunk


class _Store:
    """Returns a distinct paper per query, so merging and dedupe are observable."""

    def __init__(self) -> None:
        self.queries: list[int] = []

    def search(self, qvec, top_k: int, corpus: str) -> list[_Hit]:  # noqa: ANN001
        n = len(self.queries)
        self.queries.append(n)
        return [_Hit(_Chunk(f"paper{n}", f"text {n}", f"c{n}"))]


class _Embedder:
    def encode(self, texts: list[str]):  # noqa: ANN201
        return [[0.0, 1.0] for _ in texts]


def _system(**kw) -> AgenticSystem:
    return AgenticSystem(
        name="agentic", embedder=_Embedder(), store=_Store(),
        planner_model="gpt-5.4-nano", synthesis_model="gpt-5.4-mini", **kw
    )


# ---- the budget -------------------------------------------------------------------

def test_budget_refuses_the_call_that_would_exceed_it() -> None:
    b = RetrievalBudget(limit=2)
    b.spend()
    b.spend()
    with pytest.raises(BudgetExhausted):
        b.spend()
    assert (b.used, b.refused, b.remaining) == (2, 1, 0)


def test_budget_of_zero_permits_nothing() -> None:
    """A degenerate cap must bind rather than wrap to unlimited."""
    b = RetrievalBudget(limit=0)
    with pytest.raises(BudgetExhausted):
        b.spend()
    assert b.used == 0


def test_a_refused_retrieval_is_counted_not_silently_dropped() -> None:
    """3.5 reports cost per query. A refusal means the plan was truncated, and a trace that
    did not say so would look like a cheaper plan rather than a curtailed one."""
    b = RetrievalBudget(limit=1)
    b.spend()
    for _ in range(3):
        with pytest.raises(BudgetExhausted):
            b.spend()
    assert b.refused == 3


# ---- failing closed ---------------------------------------------------------------

def test_a_planner_returning_nothing_degrades_to_one_retrieval(monkeypatch) -> None:
    sys_ = _system()
    monkeypatch.setattr(sys_, "_graph_hints", lambda q: [])

    class _Dead:
        def json(self, **kw):  # noqa: ANN003, ANN201
            raise RuntimeError("planner is down")

    sys_._planner = _Dead()
    traj = Trajectory(query="q")
    assert sys_._plan("q", traj) == ["q"]
    assert traj.planner_failed is True


def test_an_empty_plan_is_treated_as_a_failure_not_as_zero_work(monkeypatch) -> None:
    """A planner returning `{"sub_questions": []}` is broken in the same way as one that
    raises; answering nothing at all is never the right reading."""
    sys_ = _system()
    monkeypatch.setattr(sys_, "_graph_hints", lambda q: [])

    class _Empty:
        def json(self, **kw):  # noqa: ANN003, ANN201
            return {"sub_questions": ["  ", ""], "reasoning": "none"}

    sys_._planner = _Empty()
    traj = Trajectory(query="q")
    assert sys_._plan("q", traj) == ["q"]
    assert traj.planner_failed is True


def test_the_plan_leaves_a_retrieval_for_the_critique(monkeypatch) -> None:
    """A plan that consumed the whole budget would make this a fan-out, not a loop."""
    sys_ = _system(max_retrievals=3)
    monkeypatch.setattr(sys_, "_graph_hints", lambda q: [])

    class _Greedy:
        def json(self, **kw):  # noqa: ANN003, ANN201
            return {"sub_questions": [f"s{i}" for i in range(5)], "reasoning": "r"}

    sys_._planner = _Greedy()
    traj = Trajectory(query="q")
    assert len(sys_._plan("q", traj)) == 2  # max_retrievals - 1
    assert traj.planner_failed is False


# ---- merging ----------------------------------------------------------------------

def test_merge_dedupes_by_chunk_and_preserves_first_seen_order() -> None:
    """`_format_evidence` assigns P1, P2, ... by first appearance, so order decides the
    handle mapping. Sub-questions retrieve in plan order, so earlier parts of the question
    get the lower handles."""
    a, b, c = _Chunk("p1", "a", "c1"), _Chunk("p2", "b", "c2"), _Chunk("p3", "c", "c3")
    merged = AgenticSystem._merge([[_Hit(a), _Hit(b)], [_Hit(b), _Hit(c)]])
    assert [h.chunk.id for h in merged] == ["c1", "c2", "c3"]


def test_merge_of_nothing_is_empty_not_an_error() -> None:
    assert AgenticSystem._merge([[], []]) == []


# ---- the trajectory ---------------------------------------------------------------

def test_the_trajectory_reports_what_the_critique_added() -> None:
    """3.4's `critique usefulness` measure: a critique that never changes the answer is an
    expensive no-op, and the trace has to make that visible per query."""
    t = Trajectory(query="q")
    t.papers_before_critique = ["p1", "p2"]
    t.papers_after_critique = ["p1", "p2", "p3"]
    assert t.as_dict()["critique_added_papers"] == ["p3"]


def test_a_critique_that_adds_nothing_is_recorded_as_adding_nothing() -> None:
    t = Trajectory(query="q")
    t.papers_before_critique = t.papers_after_critique = ["p1"]
    t.critique_ran = True
    d = t.as_dict()
    assert d["critique_added_papers"] == []
    assert d["critique_ran"] is True


def test_the_trajectory_carries_every_field_3_4_scores_on() -> None:
    """Pins the trace contract. 3.4 reads these off `traces.jsonl`; a renamed field would
    surface as a missing measure rather than as an error."""
    d = Trajectory(query="q").as_dict()
    for key in (
        "sub_questions", "per_sub_question", "gaps", "retrievals_used",
        "retrievals_refused", "planner_failed", "critique_ran",
        "papers_before_critique", "papers_after_critique", "critique_added_papers",
        "graph_hints", "plan_reasoning",
    ):
        assert key in d, f"trace lost {key}, which 3.4 reads"


# ---- what must stay shared with the static arms -----------------------------------

def test_synthesis_is_the_static_arms_code_and_not_a_copy() -> None:
    """3.5's variable is the control flow. If this arm formatted evidence or resolved
    handles differently, the comparison would measure the writing instead."""
    from rpsg.retrieval.baselines import VectorRAGSystem

    sys_ = _system()
    assert isinstance(sys_._synth, VectorRAGSystem)
    assert sys_._synth._format_evidence is VectorRAGSystem._format_evidence
    assert sys_._synth._resolve_handles is VectorRAGSystem._resolve_handles


def test_the_ablation_arm_really_disables_the_critique() -> None:
    assert _system(critique=False)._critique is False
    assert _system()._critique is True


def test_system_output_carries_a_trace_and_the_static_arms_leave_it_empty() -> None:
    """`runner.py` writes `trajectory` only when non-empty, which is what keeps the static
    arms' traces byte-identical to Iteration 2's."""
    from rpsg.retrieval.baselines import SystemOutput

    assert SystemOutput("t", [], "e").trace == {}
    assert SystemOutput("t", [], "e", trace={"a": 1}).trace == {"a": 1}