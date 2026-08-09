"""Sampling and scoring for a second pass over the hand-graded answers (§3.6d).

§10's standing threat: every calibrated criterion is calibrated against one annotator, so
"`coverage` is trustworthy at kappa=+0.72" means "the judge agrees with *this grader*". A
second pass says whether that generalises.

**Human-human agreement is the ceiling for judge-human agreement.** If two readers agree on
`attribution` at +0.45, no judge can be expected to clear a 0.6 bar against either of them,
and the criterion was unreachable by construction rather than badly prompted. 3.6c is why
this matters concretely: three rubric versions moved `attribution`'s mean and its ranking
and left agreement-on-level at +0.29..+0.45, with the judge never once returning a 5 in 34
answers. Whether that is a judge defect or a bar set above the human ceiling is exactly
what this module measures, and the two readings call for opposite next steps.

**The sheet never shows the first grades, the judge's scores, or which answers disagree.**
`contradiction_audit` and `extraction_audit` follow the same rule for the same reason: shown
a prior label, a reader agrees with it, and the result is a review rather than a
measurement.

**The sheet shows what the first grader could see, and nothing more.** That is the query,
the gold skeleton, and the answer -- but NOT the retrieved evidence. Iteration 2 §3.1
records the asymmetry: the judge grades `attribution` with the context in its prompt while
the human grades from the answer alone. Handing annotator 2 the evidence would measure that
asymmetry rather than disagreement between readers, and would make the two human passes
non-comparable.

Three passes share this machinery, and they answer different questions:

    inter-annotator   a second person       does calibration generalise past one grader?
    test-retest       the same person, blind  how stable are the labels themselves?
    model bound       a different model      a bound only -- §8.2 states the caveat

Only the first discharges §10. The second is the human analogue of the judge-temperature
finding in 3.6c and bounds how much label noise sits under every kappa in §6. The third is
the weakest and must be reported as bounding rather than settling.
"""

from __future__ import annotations

import random
from typing import Any, NamedTuple

from rpsg.eval.calibration import calibrate_criterion

CRITERIA = ("coverage", "attribution", "hedging_accuracy", "refutation_handling", "synthesis")


class AnswerSample(NamedTuple):
    qid: str
    query: str
    query_type: str
    facets: list[str]
    key_claims: list[dict[str, Any]]
    known_refutations: list[dict[str, Any]]
    answer: str


def sample_answers(
    rows: list[dict[str, Any]], *, total: int = 20, seed: int = 0
) -> list[AnswerSample]:
    """Draw `total` answers, stratified by query type in proportion to the full set.

    Proportional rather than equal-n, the opposite of `contradiction_audit.sample_pairs`,
    and for the opposite reason. There the classes were 18:1 imbalanced and the rare class
    was the one under suspicion, so equal-n was the only way to see it. Here the population
    *is* the thing being characterised: these 20 stand in for the 34 that calibration
    actually runs on, so over-weighting a type would produce an agreement figure that does
    not describe the set §6 reports.

    The cost is `refutation_handling`, gradeable only where the gold encodes a contradiction
    -- 9 of 34, so about 5 of 20. `gradeable_n` reports that up front rather than letting a
    kappa over five rows be read like the others.

    Deterministic under `seed`, so the sample can be extended later without re-labelling
    what was already judged.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["query_type"], []).append(r)

    # Largest-remainder apportionment: floor every quota, then hand the leftover seats to
    # the largest fractions. Rounding each independently would over- or under-fill `total`.
    n_all = len(rows)
    quotas: dict[str, float] = {t: len(v) * total / n_all for t, v in by_type.items()}
    take = {t: int(q) for t, q in quotas.items()}
    leftover = total - sum(take.values())
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - take[t]), t))[:leftover]:
        take[t] += 1

    out: list[AnswerSample] = []
    for t in sorted(by_type):
        pool = sorted(by_type[t], key=lambda r: r["qid"])
        for r in rng.sample(pool, min(len(pool), take[t])):
            out.append(AnswerSample(**{k: r[k] for k in AnswerSample._fields}))
    # Shuffle across strata before returning, as `sample_pairs` does: a sheet that changes
    # character partway through leaks its own stratification, and a reader who notices is
    # no longer grading each answer on its merits.
    rng.shuffle(out)
    return out


def gradeable_n(
    labels: list[dict[str, Any]], criteria: tuple[str, ...] = CRITERIA
) -> dict[str, int]:
    """Rows where *both* passes recorded a grade, per criterion.

    `refutation_handling` is null wherever the gold encodes no contradiction, so its n is
    always far below the others and a kappa computed over it is not comparable to the rest.
    """
    out = {}
    for c in criteria:
        out[c] = sum(
            1
            for r in labels
            if (r.get("a") or {}).get(c) is not None and (r.get("b") or {}).get(c) is not None
        )
    return out


def agreement(
    labels: list[dict[str, Any]],
    *,
    min_kappa: float,
    criteria: tuple[str, ...] = CRITERIA,
) -> dict[str, Any]:
    """Quadratic-weighted kappa between two passes, per criterion.

    Rows where either pass left a criterion null are skipped rather than coerced -- the rule
    `deterministic_scores`, `repro_gold` and `contradiction_audit` all follow.

    `min_kappa` is passed through only to populate the `trusted` flag on each result. It has
    no meaning between two humans: the threshold exists to decide whether to believe a
    judge, and two readers disagreeing is a fact about the criterion, not a failure by
    either of them.
    """
    out: dict[str, Any] = {}
    for c in criteria:
        pairs = [
            (int(a), int(b))
            for r in labels
            if (a := (r.get("a") or {}).get(c)) is not None
            and (b := (r.get("b") or {}).get(c)) is not None
        ]
        if len(pairs) < 2:
            out[c] = None
            continue
        out[c] = calibrate_criterion([x for x, _ in pairs], [y for _, y in pairs], c, min_kappa)
    return out


def ceiling(
    human_vs_human: dict[str, Any],
    judge_vs_a: dict[str, Any],
    judge_vs_b: dict[str, Any],
    *,
    min_kappa: float,
) -> str:
    """The reading that decides what to do next, per criterion.

    Three outcomes, and they call for opposite work:

      * human-human BELOW the bar -- the bar is above the ceiling. No rubric can make the
        judge agree with one reader better than two readers agree with each other, so the
        criterion should be re-specified or dropped, not re-prompted. This is the reading
        that would retire 3.6c's remaining hypothesis.
      * human-human ABOVE the bar, judge below -- the criterion is well defined and the
        judge is the weak component. Judge work is justified.
      * judge agrees with one reader markedly better than the other -- the judge has
        learned one grader's taste, which is the §10 threat in its literal form.
    """
    lines = [
        f"{'criterion':22} {'human-human':>12} {'judge-A':>9} {'judge-B':>9}   reading",
    ]
    for c in CRITERIA:
        hh, ja, jb = human_vs_human.get(c), judge_vs_a.get(c), judge_vs_b.get(c)
        if hh is None:
            lines.append(f"  {c:20} {'—':>12} {'—':>9} {'—':>9}   too few rows graded twice")
            continue
        k_hh = hh.quadratic_kappa
        k_ja = ja.quadratic_kappa if ja else float("nan")
        k_jb = jb.quadratic_kappa if jb else float("nan")
        if k_hh < min_kappa:
            reading = f"BAR ABOVE CEILING — readers agree at {k_hh:+.2f} < {min_kappa}"
        elif max(k_ja, k_jb) < min_kappa:
            reading = "criterion is sound, the judge is the weak component"
        elif abs(k_ja - k_jb) >= 0.15:
            reading = "judge tracks one grader markedly better — §10 in its literal form"
        else:
            reading = "judge agrees with both readers alike"
        lines.append(
            f"  {c:20} {k_hh:+12.2f} {k_ja:+9.2f} {k_jb:+9.2f}   {reading}"
        )
    return "\n".join(lines)


def summarize(
    result: dict[str, Any], counts: dict[str, int], *, label: str = "pass A vs pass B"
) -> str:
    lines = [f"{label}:"]
    for c in CRITERIA:
        r = result.get(c)
        if r is None:
            lines.append(f"  -- {c:20s} not enough rows graded by both")
            continue
        lines.append(
            f"     {c:20s} kappa={r.quadratic_kappa:+.2f} rho={r.spearman_rho:+.2f} "
            f"(p={r.spearman_p:.3f}, n={counts.get(c, r.n)})"
        )
    return "\n".join(lines)