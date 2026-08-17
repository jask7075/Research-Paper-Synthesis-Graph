"""§3.5's paired statistics. Deterministic: no API key, no network.

This computes p=0.012, the headline result of Iteration 3, and until now it had no tests —
the one piece of code the thesis leans on hardest with nothing behind it. The tests that
matter here are not "does it run" but the four ways a paired comparison can silently
overstate its evidence:

  * pooling repeats as independent observations, trebling the apparent n
  * dropping ties without saying so
  * reporting a p-value from a handful of differences
  * getting the statistic wrong in a way no other test would notice

The last is why there is a hand-calculable case below.
"""

from __future__ import annotations

import pytest

from rpsg.eval.comparison import MIN_NONTIED, paired, per_query, run_means


def _rep(**scores: float | None) -> dict[str, dict]:
    return {q: {"must_cite_recall": v} for q, v in scores.items()}


# ---- averaging repeats -------------------------------------------------------------

def test_repeats_are_averaged_per_query() -> None:
    got = per_query([_rep(q1=1.0, q2=0.0), _rep(q1=0.0, q2=0.0)], "must_cite_recall")
    assert got == {"q1": 0.5, "q2": 0.0}


def test_averaging_happens_before_pairing_not_after() -> None:
    """The inflation guard. Three repeats of 34 questions is 34 paired observations, not 102.
    Pooling them would treble the apparent n while the questions stay the same 34, which is
    the error that makes a paired design look stronger than it is."""
    reps = [_rep(q1=1.0), _rep(q1=1.0), _rep(q1=1.0)]
    assert len(per_query(reps, "must_cite_recall")) == 1


def test_none_scores_are_skipped_not_coerced() -> None:
    """`deterministic_scores` returns None where the gold gives a metric nothing to measure.
    Scoring those as 0.0 would punish an arm for a question nobody asked."""
    got = per_query([_rep(q1=1.0, q2=None), _rep(q1=1.0, q2=None)], "must_cite_recall")
    assert got == {"q1": 1.0}


def test_a_query_scored_in_only_one_repeat_still_averages_over_what_exists() -> None:
    got = per_query([_rep(q1=1.0, q2=0.5), _rep(q1=0.0)], "must_cite_recall")
    assert got["q1"] == 0.5
    assert got["q2"] == 0.5


# ---- the spread --------------------------------------------------------------------

def test_run_means_gives_one_value_per_repeat() -> None:
    """Reported beside every headline figure, because §4.1's n=10 number and §6's whole
    calibration table were each a single draw."""
    means = run_means([_rep(q1=1.0, q2=0.0), _rep(q1=1.0, q2=1.0)], "must_cite_recall")
    assert means == [0.5, 1.0]


def test_a_repeat_with_nothing_scoreable_is_dropped_not_counted_as_zero() -> None:
    assert run_means([_rep(q1=None), _rep(q1=1.0)], "must_cite_recall") == [1.0]


# ---- the paired test ---------------------------------------------------------------

def test_wins_losses_and_ties_are_all_reported() -> None:
    """A test that drops ties silently overstates the evidence. Wilcoxon needs the non-tied
    differences, so ties leave the statistic by necessity — but the count has to surface."""
    a = {"q1": 1.0, "q2": 0.0, "q3": 0.5, "q4": 0.5}
    b = {"q1": 0.0, "q2": 1.0, "q3": 0.5, "q4": 0.5}
    r = paired(a, b)
    assert (r["a_wins"], r["b_wins"], r["ties"], r["n"]) == (1, 1, 2, 4)


def test_mean_diff_includes_ties_but_the_statistic_does_not() -> None:
    """Two different questions. The effect size a reader wants is over all shared queries;
    the test statistic is defined only on the non-tied ones. Conflating them would either
    overstate the effect or misreport the test."""
    a = {"q1": 1.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
    b = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
    r = paired(a, b)
    assert r["mean_diff"] == pytest.approx(0.25)   # 1.0 spread over four queries
    assert r["ties"] == 3


def test_too_few_nontied_differences_returns_no_p_value() -> None:
    """A p-value from five differences reads with the same authority as one from
    thirty-four. It must be withheld rather than qualified in prose."""
    a = {f"q{i}": 1.0 for i in range(MIN_NONTIED - 1)}
    b = {f"q{i}": 0.0 for i in range(MIN_NONTIED - 1)}
    r = paired(a, b)
    assert r["p"] is None
    assert "too few to test" in r["note"]


def test_exactly_the_threshold_does_report_a_p_value() -> None:
    a = {f"q{i}": 1.0 for i in range(MIN_NONTIED)}
    b = {f"q{i}": 0.0 for i in range(MIN_NONTIED)}
    assert paired(a, b)["p"] is not None


def test_only_queries_both_arms_scored_are_compared() -> None:
    """An arm that failed on a question must not have that question counted against the
    other arm — the pairing is over the intersection."""
    r = paired({"q1": 1.0, "q2": 1.0}, {"q1": 0.0})
    assert r["n"] == 1


def test_identical_arms_produce_all_ties_and_no_test() -> None:
    a = {f"q{i}": 0.5 for i in range(10)}
    r = paired(a, dict(a))
    assert (r["ties"], r["a_wins"], r["b_wins"], r["p"]) == (10, 0, 0, None)
    assert r["mean_diff"] == 0.0


def test_the_direction_convention_is_a_minus_b() -> None:
    """A sign error here would invert every conclusion in §3.5 while leaving every p-value
    unchanged — the failure mode no other test in this file would catch."""
    better = paired({"q1": 1.0}, {"q1": 0.0})
    worse = paired({"q1": 0.0}, {"q1": 1.0})
    assert better["mean_diff"] > 0 and better["a_wins"] == 1
    assert worse["mean_diff"] < 0 and worse["b_wins"] == 1


# ---- the statistic itself ----------------------------------------------------------

def test_wilcoxon_matches_a_hand_calculable_case() -> None:
    """Nothing else verifies that `paired` returns the number it should.

    Six differences, all positive and all distinct: +0.1 … +0.6. Signed ranks are 1..6, so
    W- = 0 and the two-sided exact p is 2 / 2**6 = 0.03125. If the implementation ever
    switched to a normal approximation, dropped the two-sided correction, or ranked by signed
    value instead of absolute value, this is the test that would notice.
    """
    a = {f"q{i}": (i + 1) / 10 for i in range(6)}
    b = {f"q{i}": 0.0 for i in range(6)}
    r = paired(a, b)
    assert r["p"] == pytest.approx(2 / 2**6)
    assert (r["a_wins"], r["b_wins"], r["ties"]) == (6, 0, 0)


def test_a_symmetric_split_is_not_significant() -> None:
    """Three wins and three losses of equal size is the null. If this ever returned a small
    p, the test would be picking up magnitude without sign."""
    a = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 0.0, "q5": 0.0, "q6": 0.0}
    b = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 1.0, "q5": 1.0, "q6": 1.0}
    r = paired(a, b)
    assert r["p"] == pytest.approx(1.0)
    assert r["mean_diff"] == 0.0


def test_the_reported_iteration_3_result_reproduces() -> None:
    """Regression lock on the headline, from the real run data.

    The 14 relational per-query scores for `agentic` and `vector_fulltext`, each averaged
    over three repeats, copied out of the §3.5 runs. They must reproduce 8 wins / 1 loss /
    5 ties and p=0.0117. If a refactor moves this, the thesis's central claim has moved.

    The magnitudes matter and not only the win count: an earlier version of this test used
    win/loss/tie-equivalent differences that were guessed rather than copied, and returned
    p=0.0352 while looking correct. Wilcoxon ranks by the *size* of each difference, so a
    lock on W/L/T alone locks nothing.
    """
    agentic = [0.0, 0.833333, 1.0, 0.666667, 1.0, 0.0, 0.666667,
               0.5, 0.666667, 0.666667, 0.0, 0.333333, 0.5, 0.555556]
    baseline = [0.0, 1.0, 0.5, 0.0, 0.5, 0.0, 0.5,
                0.5, 0.666667, 0.0, 0.0, 0.0, 0.0, 0.222222]
    a = {f"q{i}": v for i, v in enumerate(agentic)}
    b = {f"q{i}": v for i, v in enumerate(baseline)}
    r = paired(a, b)
    assert (r["a_wins"], r["b_wins"], r["ties"]) == (8, 1, 5)
    assert r["p"] == pytest.approx(0.0117, abs=5e-4)