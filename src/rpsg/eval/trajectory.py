"""Trajectory eval (§3.4): score the plan, not the answer.

Every metric before this one scores the *output*. `must_cite_recall` cannot distinguish an
agent that planned well and synthesised badly from one that did the reverse -- and for a
planner-critic loop that distinction is the whole object of study.

**Deliberately, nothing here asks a model.** The plan says a new judged metric family starts
untrusted, and 3.6c/3.6d then established what clearing that costs: a judge pinned to
temperature 0, certified on its worst sample rather than one draw, and checked against the
grader's own self-agreement -- which for two of five existing criteria turned out to sit at
or below the judge's. A judged trajectory criterion would inherit all of that and could not
be reported in 3.5 without it. All four measures below are arithmetic over `traces.jsonl`
and the gold record, so they need no calibration and cannot drift between runs.

The one place judgement enters is matching a sub-question to a gold facet, which is semantic
rather than exact. That is done by embedding cosine against a stated threshold, and
`coverage_sensitivity` reports the whole curve so the threshold is visible as a choice
rather than buried as a constant.

What these measures are *for*, from the plan:

    decomposition coverage   do the sub-questions collectively cover the gold facets?
    retrieval efficiency     required papers found per retrieval call -- the number that
                             makes "better quality, worse cost" visible
    critique usefulness      did the second pass add a REQUIRED paper the first missed?
                             A critique that never changes the answer is an expensive no-op
    plan-outcome coupling    do trajectory scores predict must_cite_recall? If not, either
                             the measures are wrong or the plan does not matter, and both
                             are worth knowing

`plan_outcome_coupling` is the one that can invalidate the other three, which is why it is
reported alongside them rather than as a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rpsg.eval.gold_schema import GoldRecord
from rpsg.logging import get_logger

log = get_logger(__name__)

#: Cosine above which a sub-question is treated as addressing a facet.
#:
#: **This measure is not valid on this corpus, and the constant is not the reason.**
#: `absolute_coverage_is_valid` measures the null directly: a facet matched against a
#: *different query's* plan scores 0.769 mean against 0.808 for its own, and at 0.60 every
#: single unrelated pair passes. No threshold separates them usefully -- 0.80 keeps 62% of
#: matched pairs while admitting 30% of unrelated ones, 0.85 keeps 28% and admits 11%.
#: SPECTER is trained on scientific titles and abstracts, so short phrases from one
#: sub-field are mutually similar by construction, and "does this plan address this facet"
#: is not a question cosine can answer here.
#:
#: The first 3.4 run reported decomposition coverage of 1.000 across all 29 facets. That was
#: the artefact, not a result. `decomposition_specificity` is the substitute that survives.
DEFAULT_FACET_THRESHOLD = 0.60


@dataclass
class TrajectoryScore:
    qid: str
    query_type: str
    decomposition_coverage: float | None   # NOT VALID on this corpus; see the module doc
    decomposition_specificity: float | None
    facets_total: int
    facets_covered: int
    facets_specific: int
    retrieval_efficiency: float | None
    required_found: int
    required_total: int
    retrievals_used: int
    critique_added_required: int
    critique_added_any: int
    critique_ran: bool
    planner_failed: bool

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _bare(paper: str) -> str:
    """A paper id without the `paper:` prefix.

    Gold `must_cite` entries are `paper:<id>` because that is what the answer text carries
    and what `must_cite_recall` matches against; the trajectory records `chunk.paper_id`,
    which is bare. Comparing the two sets directly returns the empty set for every query --
    silently, and as a plausible-looking 0.000 rather than an error. It did exactly that on
    the first 3.4 run, which is the argument for 3.4 existing before 3.5 rather than beside
    it.
    """
    return paper.split(":", 1)[1] if paper.startswith("paper:") else paper


def _cosine_matrix(embedder: Any, a: list[str], b: list[str]):  # noqa: ANN201
    """Row-normalised cosine of every `a` against every `b`."""
    import numpy as np

    va = np.asarray(embedder.encode(a), dtype="float32")
    vb = np.asarray(embedder.encode(b), dtype="float32")
    va /= np.linalg.norm(va, axis=1, keepdims=True) + 1e-9
    vb /= np.linalg.norm(vb, axis=1, keepdims=True) + 1e-9
    return va @ vb.T


def decomposition_coverage(
    sub_questions: list[str],
    facets: list[str],
    embedder: Any,
    threshold: float = DEFAULT_FACET_THRESHOLD,
) -> tuple[float | None, int]:
    """Fraction of gold facets some sub-question addresses, and the count covered.

    Returns `(None, 0)` when the gold names no facets, following the rule
    `deterministic_scores` established: a metric the gold gives nothing to measure returns
    None rather than 0.0, because scoring an unasked question as a failure inflates nothing
    and deflates everything.

    A facet counts as covered if ANY sub-question clears the threshold. Coverage is about
    reach, not division of labour -- one sub-question that happens to span two facets has
    covered both, and penalising that would reward padding the plan.
    """
    if not facets:
        return None, 0
    if not sub_questions:
        return 0.0, 0
    sims = _cosine_matrix(embedder, facets, sub_questions)
    covered = int((sims.max(axis=1) >= threshold).sum())
    return covered / len(facets), covered


def coverage_sensitivity(
    sub_questions: list[str],
    facets: list[str],
    embedder: Any,
    thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
) -> dict[float, float]:
    """Coverage at each threshold, so the constant cannot quietly carry the result."""
    if not facets or not sub_questions:
        return {}
    sims = _cosine_matrix(embedder, facets, sub_questions)
    best = sims.max(axis=1)
    return {t: float((best >= t).sum()) / len(facets) for t in thresholds}


def absolute_coverage_is_valid(
    facet_sets: list[list[str]],
    plans: list[list[str]],
    embedder: Any,
    thresholds: tuple[float, ...] = (0.60, 0.70, 0.75, 0.80, 0.85),
) -> dict[str, Any]:
    """Can a cosine threshold tell a facet's own plan from someone else's?

    The validation `decomposition_coverage` needs and did not have. Matched pairs are each
    facet against its own query's plan; null pairs are the same facet against every *other*
    query's plan. If the two distributions overlap, a threshold is separating nothing and a
    coverage figure computed from one is measuring the corpus's topical narrowness.

    Run this before believing any coverage number on a new corpus. It is cheap -- one
    embedding pass -- and it is the difference between 1.000 meaning "the plan covered
    everything" and 1.000 meaning "every phrase here resembles every other".
    """
    import numpy as np

    encoded = [
        (
            _normalised(embedder, facets) if facets else None,
            _normalised(embedder, plan) if plan else None,
        )
        for facets, plan in zip(facet_sets, plans, strict=True)
    ]
    matched: list[float] = []
    null: list[float] = []
    for i, (F, _) in enumerate(encoded):
        if F is None:
            continue
        for j, (_, P) in enumerate(encoded):
            if P is None:
                continue
            best = (F @ P.T).max(axis=1).tolist()
            (matched if i == j else null).extend(best)
    if not matched or not null:
        return {"n_matched": len(matched), "n_null": len(null), "valid": None}
    m, n = np.asarray(matched), np.asarray(null)
    rows = {
        float(t): {
            "matched_kept": float((m >= t).mean()),
            "null_admitted": float((n >= t).mean()),
        }
        for t in thresholds
    }
    # Usable only if some threshold keeps most real matches while rejecting most fakes.
    valid = any(r["matched_kept"] >= 0.80 and r["null_admitted"] <= 0.20 for r in rows.values())
    return {
        "n_matched": len(matched),
        "n_null": len(null),
        "matched_mean": float(m.mean()),
        "null_mean": float(n.mean()),
        "by_threshold": rows,
        "valid": valid,
    }


def decomposition_specificity(
    facets: list[str], own_plan: list[str], rival_plans: list[list[str]], embedder: Any
) -> tuple[float | None, int]:
    """Fraction of facets whose own plan matches them better than EVERY rival plan does.

    The substitute for absolute coverage, and it works for the reason absolute coverage does
    not: it is a paired comparison, so the corpus-wide similarity floor that swamps a fixed
    threshold cancels out. Each facet is scored against its own plan and against every other
    query's plan, and counts only if its own wins outright.

    Chance is `1 / (1 + len(rival_plans))` -- with nine rivals, 0.100. On the 3.1 acceptance
    run this returns 0.276 against that 0.100 (8 of 29 facets, binomial p ~ 0.007), so it
    does discriminate.

    **It is a diagnostic, not a headline, and comparing it ACROSS query types is invalid.**
    The rival pool decides both the score and the chance rate, and neither is constant by
    type. Relational queries are near-uniformly *"X, and what limits each"* (§4.5), so a
    relational plan competes against near-duplicates in form: scored against all nine rivals
    relational reads 0.083, against the six non-relational ones 0.167 -- but chance moves
    from 0.100 to 0.143 at the same time, so it sits near chance either way. Lookup reads
    0.600 against both pools because it has one same-type rival. The ordering is stable and
    the magnitudes are not comparable.

    Use it to compare *arms on the same queries* -- agentic against agentic-no-critique,
    where the rival pool is identical -- never to compare query types within one arm. 3.5
    reports retrieval efficiency and critique usefulness as its trajectory headline; this
    and absolute coverage are reported as diagnostics with their limits attached.

    **Read it as specificity, not as coverage.** 0.276 does not mean 27.6% of facets were
    addressed; it means 27.6% were addressed in a way that distinguishes this plan from a
    plan written for a different question. A plan can address a facet in terms so generic
    that a rival plan matches it equally well, and that is counted here as a failure to
    specialise rather than a failure to cover.
    """
    if not facets or not own_plan:
        return None, 0
    if not rival_plans:
        return None, 0
    import numpy as np

    F = _normalised(embedder, facets)
    own = (F @ _normalised(embedder, own_plan).T).max(axis=1)
    rivals = np.stack(
        [(F @ _normalised(embedder, r).T).max(axis=1) for r in rival_plans if r]
    )
    wins = int((own > rivals.max(axis=0)).sum())
    return wins / len(facets), wins


def _normalised(embedder: Any, texts: list[str]):  # noqa: ANN201
    import numpy as np

    v = np.asarray(embedder.encode(texts), dtype="float32")
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def retrieval_efficiency(
    trajectory: dict[str, Any], gold: GoldRecord
) -> tuple[float | None, int, int]:
    """Required papers reached per retrieval call.

    The number that separates "the agent answers better" from "the agent spends more". A
    static arm issues one retrieval; this arm issues several, so an agent finding twice the
    required papers with five times the calls is *worse* on this measure while looking
    better on `must_cite_recall`. §3.5 reports both for that reason.

    `None` when the gold names no required papers -- there is no denominator and no numerator.
    """
    required = {_bare(p) for p in (gold.must_cite or [])}
    if not required:
        return None, 0, 0
    reached = _papers_reached(trajectory)
    found = len(required & reached)
    used = int(trajectory.get("retrievals_used") or 0)
    return (found / used if used else None), found, len(required)


def _papers_reached(trajectory: dict[str, Any]) -> set[str]:
    """Every paper the loop retrieved, across sub-questions and the critique pass.

    Read from `papers_after_critique` where present because that is the post-merge truth,
    falling back to the per-sub-question union for a trace written before the critique ran.
    """
    after = trajectory.get("papers_after_critique") or []
    if after:
        return {_bare(p) for p in after}
    reached: set[str] = set()
    for step in trajectory.get("per_sub_question") or []:
        reached.update(_bare(p) for p in (step.get("papers") or []))
    return reached


def critique_usefulness(
    trajectory: dict[str, Any], gold: GoldRecord
) -> tuple[int, int]:
    """(required papers the critique added, any papers it added).

    Both, because they answer different questions. "Any" says the second pass changed the
    evidence at all; "required" says the change was worth making. A critique with a high
    `any` and a zero `required` is retrieving more and reaching nothing the gold asked for,
    which is precisely the expensive no-op §3.4 exists to detect.
    """
    added = {_bare(p) for p in (trajectory.get("critique_added_papers") or [])}
    required = {_bare(p) for p in (gold.must_cite or [])}
    return len(added & required), len(added)


def score_trajectory(
    trajectory: dict[str, Any],
    gold: GoldRecord,
    embedder: Any,
    threshold: float = DEFAULT_FACET_THRESHOLD,
    rival_plans: list[list[str]] | None = None,
) -> TrajectoryScore:
    subs = trajectory.get("sub_questions") or []
    cov, covered = decomposition_coverage(subs, gold.facets or [], embedder, threshold)
    spec, n_spec = (
        decomposition_specificity(gold.facets or [], subs, rival_plans, embedder)
        if rival_plans
        else (None, 0)
    )
    eff, found, req_total = retrieval_efficiency(trajectory, gold)
    add_req, add_any = critique_usefulness(trajectory, gold)
    return TrajectoryScore(
        qid=gold.qid,
        query_type=gold.query_type.value,
        decomposition_coverage=cov,
        decomposition_specificity=spec,
        facets_total=len(gold.facets or []),
        facets_covered=covered,
        facets_specific=n_spec,
        retrieval_efficiency=eff,
        required_found=found,
        required_total=req_total,
        retrievals_used=int(trajectory.get("retrievals_used") or 0),
        critique_added_required=add_req,
        critique_added_any=add_any,
        critique_ran=bool(trajectory.get("critique_ran")),
        planner_failed=bool(trajectory.get("planner_failed")),
    )


def plan_outcome_coupling(
    scores: list[TrajectoryScore], outcomes: dict[str, float | None]
) -> dict[str, Any]:
    """Do trajectory measures predict `must_cite_recall`?

    The check that can invalidate the other three. If a plan that covers every facet answers
    no better than one that covers half, then either these measures are not measuring the
    plan or the plan does not matter -- and the plan says both are worth knowing, so this is
    reported whichever way it comes out.

    Spearman rather than Pearson: coverage is a bounded fraction over few distinct values and
    efficiency is a ratio with a small integer denominator, so monotone association is the
    most these can support.

    Queries where the planner failed are excluded. Those degraded to a single retrieval and
    have no plan to correlate; leaving them in would measure the fallback path.
    """
    from scipy.stats import spearmanr

    usable = [s for s in scores if not s.planner_failed]
    out: dict[str, Any] = {"n": len(usable), "excluded_planner_failed": len(scores) - len(usable)}
    for field in ("decomposition_specificity", "retrieval_efficiency"):
        pairs = [
            (getattr(s, field), outcomes[s.qid])
            for s in usable
            if getattr(s, field) is not None and outcomes.get(s.qid) is not None
        ]
        if len(pairs) < 3:
            out[field] = {"n": len(pairs), "rho": None, "p": None}
            continue
        xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
        # A constant column makes rho undefined; report that rather than a NaN that reads
        # like a measured zero.
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            out[field] = {"n": len(pairs), "rho": None, "p": None, "note": "constant input"}
            continue
        rho, p = spearmanr(xs, ys)
        out[field] = {"n": len(pairs), "rho": float(rho), "p": float(p)}
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize(scores: list[TrajectoryScore], coupling: dict[str, Any]) -> str:
    if not scores:
        return "no trajectories to score"
    lines = [f"Trajectory eval — {len(scores)} queries"]

    failed = sum(1 for s in scores if s.planner_failed)
    if failed:
        lines.append(f"  !! {failed} planner failures — those queries ran as single-retrieval")

    def block(rows: list[TrajectoryScore], title: str) -> None:
        cov = [s.decomposition_coverage for s in rows if s.decomposition_coverage is not None]
        eff = [s.retrieval_efficiency for s in rows if s.retrieval_efficiency is not None]
        ran = [s for s in rows if s.critique_ran]
        useful = sum(1 for s in ran if s.critique_added_required)
        moved = sum(1 for s in ran if s.critique_added_any)
        m_cov, m_eff = _mean(cov), _mean(eff)
        lines.append(f"\n{title}  (n={len(rows)})")
        spec = [s.decomposition_specificity for s in rows
                if s.decomposition_specificity is not None]
        m_spec = _mean(spec)
        cov_n = sum(s.facets_covered for s in rows)
        cov_d = sum(s.facets_total for s in rows)
        lines.append(
            "  decomposition coverage  "
            + (f"{m_cov:.3f}" if m_cov is not None else "—")
            + f"   ({cov_n}/{cov_d} facets)   !! NOT VALID — see below"
        )
        lines.append(
            "  decomposition specificity "
            + (f"{m_spec:.3f}" if m_spec is not None else "—")
            + f"   ({sum(s.facets_specific for s in rows)}/{cov_d} beat every rival plan)"
        )
        lines.append(
            "  retrieval efficiency    "
            + (f"{m_eff:.3f}" if m_eff is not None else "—")
            + "   required papers per retrieval call"
        )
        lines.append(
            f"  critique usefulness     {useful}/{len(ran)} added a REQUIRED paper; "
            f"{moved}/{len(ran)} changed the evidence at all"
        )

    block(scores, "overall")
    by_type: dict[str, list[TrajectoryScore]] = {}
    for s in scores:
        by_type.setdefault(s.query_type, []).append(s)
    for qt in sorted(by_type):
        block(by_type[qt], qt)

    lines.append(f"\nplan-outcome coupling vs must_cite_recall  (n={coupling.get('n')})")
    if coupling.get("excluded_planner_failed"):
        lines.append(f"  {coupling['excluded_planner_failed']} excluded: planner failed")
    for field in ("decomposition_specificity", "retrieval_efficiency"):
        c = coupling.get(field) or {}
        if c.get("rho") is None:
            why = c.get("note") or f"n={c.get('n')}"
            lines.append(f"  {field:24} — ({why})")
        else:
            flag = "" if c["p"] < 0.05 else "   (not significant)"
            lines.append(f"  {field:24} rho={c['rho']:+.2f} (p={c['p']:.3f}, n={c['n']}){flag}")
    lines.append(
        "\n  A trajectory measure that does not predict the outcome is either the wrong\n"
        "  measure or evidence the plan does not matter. Both are results."
    )
    return "\n".join(lines)