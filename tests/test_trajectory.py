"""Trajectory eval (§3.4). Deterministic by construction: nothing here asks a model.

That is the property under test as much as any number. 3.6c/3.6d established what a judged
criterion costs to certify -- temperature pinning, worst-sample certification, and a human
ceiling that for two of five existing criteria sat at or below the judge. These measures are
arithmetic over the trace and the gold record so they inherit none of it.
"""

from __future__ import annotations

import pytest

from rpsg.eval.gold_schema import GoldRecord, QueryType
from rpsg.eval.trajectory import (
    critique_usefulness,
    decomposition_coverage,
    plan_outcome_coupling,
    retrieval_efficiency,
    score_trajectory,
)


class _Embedder:
    """Exact-match embedder: identical strings are parallel, different ones orthogonal.

    Keeps the coverage tests about the *rule* (does any sub-question clear the threshold)
    rather than about SPECTER's opinion of two phrases. The id registry lives on the
    instance because `decomposition_coverage` encodes facets and sub-questions in two
    separate calls, and a vocabulary rebuilt per call would give them different widths.
    """

    DIM = 32

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, texts: list[str]):  # noqa: ANN201
        out = []
        for text in texts:
            idx = self._ids.setdefault(text, len(self._ids))
            vec = [0.0] * self.DIM
            vec[idx % self.DIM] = 1.0
            out.append(vec)
        return out


def _gold(**kw) -> GoldRecord:
    base = dict(
        qid="q1", query="a question", query_type=QueryType.RELATIONAL,
        facets=["facet one", "facet two"], must_cite=["pA", "pB"],
    )
    base.update(kw)
    return GoldRecord(**base)


def _traj(**kw) -> dict:
    base = {
        "sub_questions": ["facet one", "facet two"],
        "per_sub_question": [{"sub_question": "facet one", "papers": ["pA"]}],
        "papers_before_critique": ["pA"],
        "papers_after_critique": ["pA", "pB"],
        "critique_added_papers": ["pB"],
        "retrievals_used": 2,
        "critique_ran": True,
        "planner_failed": False,
    }
    base.update(kw)
    return base


# ---- decomposition coverage --------------------------------------------------------

def test_coverage_counts_a_facet_reached_by_any_sub_question() -> None:
    cov, n = decomposition_coverage(["facet one", "facet two"], ["facet one", "facet two"],
                                    _Embedder(), threshold=0.9)
    assert (cov, n) == (1.0, 2)


def test_a_facet_no_sub_question_reaches_is_uncovered() -> None:
    cov, n = decomposition_coverage(["facet one"], ["facet one", "facet two"],
                                    _Embedder(), threshold=0.9)
    assert (cov, n) == (0.5, 1)


def test_one_sub_question_may_cover_two_facets() -> None:
    """Coverage is reach, not division of labour. Penalising a sub-question that spans two
    facets would reward padding the plan with one entry per facet."""
    cov, _ = decomposition_coverage(["facet one"], ["facet one", "facet one"],
                                    _Embedder(), threshold=0.9)
    assert cov == 1.0


def test_gold_with_no_facets_returns_none_not_zero() -> None:
    """The rule `deterministic_scores` set: a metric the gold gives nothing to measure
    returns None. Scoring an unasked question as 0.0 would deflate every aggregate."""
    assert decomposition_coverage(["anything"], [], _Embedder()) == (None, 0)


def test_an_empty_plan_covers_nothing_which_is_zero_not_none() -> None:
    """Distinct from the case above: the gold DID ask, and the plan answered nothing."""
    assert decomposition_coverage([], ["facet one"], _Embedder()) == (0.0, 0)


# ---- retrieval efficiency ----------------------------------------------------------

def test_efficiency_is_required_papers_per_retrieval_call() -> None:
    eff, found, total = retrieval_efficiency(_traj(retrievals_used=2), _gold())
    assert (found, total) == (2, 2)
    assert eff == pytest.approx(1.0)


def test_spending_more_retrievals_for_the_same_papers_scores_worse() -> None:
    """The measure's whole purpose: an agent can beat the static arm on `must_cite_recall`
    while losing here, and 3.5 must be able to see that."""
    cheap, _, _ = retrieval_efficiency(_traj(retrievals_used=2), _gold())
    dear, _, _ = retrieval_efficiency(_traj(retrievals_used=6), _gold())
    assert dear < cheap


def test_efficiency_is_none_when_the_gold_names_no_required_papers() -> None:
    assert retrieval_efficiency(_traj(), _gold(must_cite=[])) == (None, 0, 0)


def test_papers_reached_falls_back_to_the_per_sub_question_union() -> None:
    """A trace written when the critique did not run has no `papers_after_critique`."""
    t = _traj(papers_after_critique=[], critique_ran=False,
              per_sub_question=[{"papers": ["pA"]}, {"papers": ["pB"]}])
    _, found, _ = retrieval_efficiency(t, _gold())
    assert found == 2


# ---- critique usefulness -----------------------------------------------------------

def test_critique_usefulness_separates_required_from_merely_added() -> None:
    """A high `any` with a zero `required` is the expensive no-op §3.4 exists to detect."""
    req, any_ = critique_usefulness(_traj(critique_added_papers=["pB", "pZ", "pY"]), _gold())
    assert (req, any_) == (1, 3)


def test_a_critique_that_adds_only_irrelevant_papers_scores_zero_required() -> None:
    req, any_ = critique_usefulness(_traj(critique_added_papers=["pZ"]), _gold())
    assert (req, any_) == (0, 1)


# ---- plan-outcome coupling ---------------------------------------------------------

def _score(qid: str, cov: float, eff: float, failed: bool = False):  # noqa: ANN202
    return score_trajectory(
        _traj(
            sub_questions=["facet one"] if cov else [],
            retrievals_used=int(1 / eff) if eff else 1,
            planner_failed=failed,
        ),
        _gold(qid=qid), _Embedder(), threshold=0.9,
    )


def test_coupling_excludes_queries_where_the_planner_failed() -> None:
    """Those degraded to a single retrieval and have no plan to correlate; including them
    would measure the fallback path."""
    scores = [_score(f"q{i}", 1.0, 1.0) for i in range(3)]
    scores.append(_score("q9", 1.0, 1.0, failed=True))
    out = plan_outcome_coupling(scores, {s.qid: 0.5 for s in scores})
    assert out["n"] == 3
    assert out["excluded_planner_failed"] == 1


def test_coupling_reports_constant_input_rather_than_a_nan() -> None:
    """A NaN rho renders as a number and reads like a measured zero association."""
    scores = [_score(f"q{i}", 1.0, 1.0) for i in range(4)]
    out = plan_outcome_coupling(scores, {s.qid: 0.5 for s in scores})
    assert out["decomposition_specificity"]["rho"] is None


def test_coupling_needs_at_least_three_points() -> None:
    scores = [_score("q1", 1.0, 1.0), _score("q2", 1.0, 1.0)]
    out = plan_outcome_coupling(scores, {s.qid: 0.5 for s in scores})
    assert out["decomposition_specificity"]["rho"] is None


def test_absolute_coverage_is_reported_invalid_when_the_null_overlaps() -> None:
    """The check the first 3.4 run lacked. Coverage read 1.000 across all 29 facets because
    SPECTER makes every phrase in one sub-field similar to every other -- at the 0.60 default
    every unrelated pair passed. A validator that cannot fail would not have caught it."""
    from rpsg.eval.trajectory import absolute_coverage_is_valid

    class _Flat:
        """Everything similar to everything: the pathology, in the extreme."""

        def encode(self, texts):  # noqa: ANN001, ANN201
            return [[1.0, 0.0] for _ in texts]

    out = absolute_coverage_is_valid(
        [["a", "b"], ["c", "d"]], [["p"], ["q"]], _Flat()
    )
    assert out["valid"] is False
    assert out["by_threshold"][0.6]["null_admitted"] == 1.0


def test_specificity_is_none_without_rivals_rather_than_a_perfect_score() -> None:
    """A paired measure with nothing to pair against is undefined, not 1.0."""
    from rpsg.eval.trajectory import decomposition_specificity

    assert decomposition_specificity(["f"], ["s"], [], _Embedder()) == (None, 0)


# ---- the whole score ---------------------------------------------------------------

def test_score_trajectory_carries_the_planner_failure_flag_through() -> None:
    s = score_trajectory(_traj(planner_failed=True), _gold(), _Embedder(), threshold=0.9)
    assert s.planner_failed is True
    assert s.qid == "q1" and s.query_type == "relational"


def test_no_measure_calls_a_model() -> None:
    """The design property. `_Embedder` has no network and no API key, and every measure
    above ran on it — so the family needs no calibration and cannot drift between runs."""
    s = score_trajectory(_traj(), _gold(), _Embedder(), threshold=0.9)
    assert s.decomposition_coverage == 1.0
    assert s.retrieval_efficiency == pytest.approx(1.0)