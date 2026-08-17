"""Paired comparison of arms across repeated runs (§3.5's statistics).

Extracted from `scripts/compare_arms.py` so it can be unit-tested. It computes p=0.012 --
the headline result of Iteration 3 -- and was the one piece of code the thesis leans on
hardest with no tests behind it.

**Paired, not a comparison of run means.** Every arm answers the same 34 questions, so the
comparison that matters is per query: on how many questions did A beat B, and by how much. A
Wilcoxon signed-rank over 34 paired differences is far more powerful than a test over three
run-level averages, which is all "3 repeats" would otherwise give.

**Repeats are averaged per query BEFORE pairing.** Each arm's score for a question is the mean
of its repeats, so the pairing is one number per question per arm. Pooling repeats as
independent observations would treble the apparent n while the questions stay the same 34,
which inflates significance -- the exact error that makes a paired design look stronger than
it is.

**Ties are counted and reported.** On this gold set many questions score identically under two
arms, and a test that silently drops them overstates how much evidence there is. Wilcoxon
requires the non-tied differences, so they are dropped from the *test* by necessity -- but the
count is surfaced so a reader can see how thin the comparison is.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

#: Below this many non-tied differences, no p-value is reported. Wilcoxon's normal
#: approximation is unreliable on a handful of pairs, and a p-value from five differences
#: reads with the same authority as one from thirty-four.
MIN_NONTIED = 6


def per_query(repeats: list[dict[str, dict]], metric: str) -> dict[str, float]:
    """Mean across repeats, per query.

    `None` scores are skipped rather than coerced, following the rule
    `deterministic_scores` established: a metric the gold gives nothing to measure must not
    be scored as a zero.
    """
    acc: dict[str, list[float]] = defaultdict(list)
    for rep in repeats:
        for qid, row in rep.items():
            if row.get(metric) is not None:
                acc[qid].append(float(row[metric]))
    return {q: sum(v) / len(v) for q, v in acc.items() if v}


def run_means(repeats: list[dict[str, dict]], metric: str) -> list[float]:
    """One mean per repeat, for the spread. Reported beside every headline figure, because
    §4.1's n=10 number and §6's whole calibration table were each a single draw."""
    out = []
    for rep in repeats:
        vals = [float(r[metric]) for r in rep.values() if r.get(metric) is not None]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """Wilcoxon signed-rank over the queries both arms scored.

    `mean_diff` is over ALL shared queries including ties, because that is the effect size a
    reader wants; the test statistic uses only the non-tied ones, because that is what
    Wilcoxon is defined on. Reporting both keeps the two from being confused.
    """
    from scipy.stats import wilcoxon

    shared = sorted(set(a) & set(b))
    diffs = [a[q] - b[q] for q in shared]
    nonzero = [d for d in diffs if d != 0]
    res: dict[str, Any] = {
        "n": len(shared),
        "ties": len(diffs) - len(nonzero),
        "a_wins": sum(1 for d in nonzero if d > 0),
        "b_wins": sum(1 for d in nonzero if d < 0),
        "mean_diff": sum(diffs) / len(diffs) if diffs else 0.0,
    }
    if len(nonzero) < MIN_NONTIED:
        res["p"] = None
        res["note"] = f"only {len(nonzero)} non-tied queries — too few to test"
        return res
    res["p"] = float(wilcoxon(nonzero).pvalue)
    return res
