"""Second-annotator sampling and scoring (§3.6d).

Two properties carry the whole item, and both are silent failures if broken:

  * the sheet must not leak the first grades, the judge's scores, or the retrieved
    evidence. A leak turns a measurement into a confirmation exercise and nothing in the
    output would look wrong.
  * the sample must mirror the 34 by query type, because these 20 stand in for the set §6
    reports on.

Deterministic: no API key, no network.
"""

from __future__ import annotations

import pytest

from rpsg.eval.annotator_agreement import (
    CRITERIA,
    AnswerSample,
    agreement,
    ceiling,
    gradeable_n,
    sample_answers,
)


def _rows(n_by_type: dict[str, int]) -> list[dict]:
    rows = []
    for t, n in n_by_type.items():
        for i in range(n):
            rows.append({
                "qid": f"{t[:3]}-{i:03d}",
                "query": f"a {t} question {i}",
                "query_type": t,
                "facets": ["f1", "f2"],
                "key_claims": [{"text": "c", "papers": []}],
                "known_refutations": [{"a": "x", "b": "y"}] if t == "refutation" else [],
                "answer": f"answer {i}",
            })
    return rows


# The real distribution across the 34 hand-graded queries.
_REAL = {"relational": 14, "refutation": 9, "lookup": 6, "open-directions": 5}


def test_sample_is_proportional_to_the_full_set() -> None:
    got = sample_answers(_rows(_REAL), total=20, seed=0)
    assert len(got) == 20
    counts = {t: sum(1 for s in got if s.query_type == t) for t in _REAL}
    # Largest-remainder apportionment of 20 seats over 14/9/6/5.
    assert counts == {"relational": 8, "refutation": 5, "lookup": 4, "open-directions": 3}


def test_sampling_is_deterministic_under_seed() -> None:
    """So the sample can be extended without re-labelling what a reader already judged."""
    a = [s.qid for s in sample_answers(_rows(_REAL), total=20, seed=0)]
    b = [s.qid for s in sample_answers(_rows(_REAL), total=20, seed=0)]
    assert a == b
    assert [s.qid for s in sample_answers(_rows(_REAL), total=20, seed=1)] != a


def test_the_sample_is_shuffled_across_strata() -> None:
    """Emitting the strata in order leaks the stratification: a reader who notices the sheet
    change character partway through is no longer grading each answer on its merits."""
    types = [s.query_type for s in sample_answers(_rows(_REAL), total=20, seed=0)]
    grouped = sorted(types, key=lambda t: sorted(_REAL).index(t))
    assert types != grouped


def test_the_sample_carries_nothing_that_reveals_a_prior_grade() -> None:
    """The blinding invariant. `AnswerSample` is the whole contract: if a field is not in
    it, it cannot reach the sheet. Evidence is excluded too — the first grader never saw it
    (§3.1), so showing it would measure that asymmetry rather than reader disagreement."""
    forbidden = {"grade", "human", "judge", "evidence", "scores", "judge_attribution"}
    assert not (set(AnswerSample._fields) & forbidden)
    for s in sample_answers(_rows(_REAL), total=20, seed=0):
        assert not (set(s._asdict()) & forbidden)


def test_total_larger_than_the_population_is_capped_not_padded() -> None:
    got = sample_answers(_rows({"lookup": 3}), total=20, seed=0)
    assert len(got) == 3


def _pair(a: dict | None, b: dict | None) -> dict:
    return {"a": a, "b": b}


def test_agreement_skips_rows_either_pass_left_null() -> None:
    """`refutation_handling` is null wherever the gold encodes no contradiction. Coercing
    those to a number would manufacture agreement on questions nobody was asked."""
    labels = [
        _pair({"coverage": 5, "refutation_handling": None},
              {"coverage": 5, "refutation_handling": 3}),
        _pair({"coverage": 1, "refutation_handling": 2}, {"coverage": 1, "refutation_handling": 2}),
        _pair({"coverage": 3, "refutation_handling": 4}, {"coverage": 3, "refutation_handling": 4}),
    ]
    counts = gradeable_n(labels)
    assert counts["coverage"] == 3
    assert counts["refutation_handling"] == 2
    assert counts["synthesis"] == 0


def test_a_criterion_nobody_graded_twice_returns_none_rather_than_a_number() -> None:
    labels = [_pair({"synthesis": 4}, {"synthesis": None}) for _ in range(5)]
    assert agreement(labels, min_kappa=0.6)["synthesis"] is None


def test_perfect_agreement_scores_one() -> None:
    labels = [_pair({"coverage": v}, {"coverage": v}) for v in (1, 2, 3, 4, 5, 1, 5)]
    assert agreement(labels, min_kappa=0.6)["coverage"].quadratic_kappa == pytest.approx(1.0)


def test_the_ceiling_reading_names_a_bar_above_the_human_ceiling() -> None:
    """The reading that would retire 3.6c: if two readers agree less well than the bar
    demands of the judge, no rubric can close the gap and the criterion needs
    re-specifying rather than re-prompting."""
    low = [_pair({"attribution": a}, {"attribution": b})
           for a, b in [(1, 3), (5, 2), (3, 5), (2, 1), (4, 2), (1, 4), (5, 3), (2, 4)]]
    hh = agreement(low, min_kappa=0.6)
    assert hh["attribution"].quadratic_kappa < 0.6
    text = ceiling(hh, hh, hh, min_kappa=0.6)
    assert "BAR ABOVE CEILING" in text


def test_the_ceiling_reading_blames_the_judge_when_the_humans_agree() -> None:
    agree = [_pair({"coverage": v}, {"coverage": v}) for v in (1, 2, 3, 4, 5, 1, 5, 2)]
    disagree = [_pair({"coverage": a}, {"coverage": b})
                for a, b in [(1, 3), (5, 2), (3, 5), (2, 1), (4, 2), (1, 4), (5, 3), (2, 4)]]
    text = ceiling(
        agreement(agree, min_kappa=0.6),
        agreement(disagree, min_kappa=0.6),
        agreement(disagree, min_kappa=0.6),
        min_kappa=0.6,
    )
    assert "the judge is the weak component" in text


def test_the_ceiling_reading_flags_a_judge_that_tracks_one_grader() -> None:
    """§10 in its literal form: the judge has learned one grader's taste."""
    agree = [_pair({"coverage": v}, {"coverage": v}) for v in (1, 2, 3, 4, 5, 1, 5, 2)]
    near = [_pair({"coverage": a}, {"coverage": b})
            for a, b in [(1, 1), (2, 2), (3, 3), (4, 5), (5, 5), (1, 2), (5, 4), (2, 2)]]
    far = [_pair({"coverage": a}, {"coverage": b})
           for a, b in [(1, 3), (2, 4), (3, 2), (4, 2), (5, 3), (1, 3), (5, 2), (2, 5)]]
    text = ceiling(
        agreement(agree, min_kappa=0.6),
        agreement(near, min_kappa=0.6),
        agreement(far, min_kappa=0.6),
        min_kappa=0.6,
    )
    assert "tracks one grader" in text


def test_every_criterion_the_judge_scores_is_covered() -> None:
    from rpsg.eval.judge import CRITERIA as JUDGE_CRITERIA

    assert set(CRITERIA) == set(JUDGE_CRITERIA)