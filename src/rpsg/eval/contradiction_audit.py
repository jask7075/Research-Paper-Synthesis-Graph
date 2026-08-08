"""Sampling and scoring for the cross-paper contradiction audit.

The pass accepted 3,072 of 16,972 pairs (163 `refutes`, 2,909 `undercuts`). An eyeballed
sample of eight suggested roughly half the `refutes` were wrong, which is enough to
withhold the number but not enough to estimate a rate. This is the labelled version.

**The labeller never sees the model's verdict.** `extraction_audit` follows the same rule
for the same reason: shown a verdict, a human agrees with it. Hiding it turns the exercise
from confirmation into measurement, and it is what lets the result be reported as agreement
rather than as a review.

**Three strata, equal n, not proportional.** `undercuts` outnumbers `refutes` 18:1, so a
proportional sample of 60 would contain three `refutes` — the class the eyeball check
flagged. `neither` is sampled too, from the 13,900 rejected pairs, because the interesting
question is two-sided: over-acceptance inflates the graph with conflicts nobody asserts,
and under-acceptance leaves the 1-of-9 refutation result unimproved. Unlike the extraction
audit, where recall had no well-defined denominator, here it does — the candidate pool is
a fixed, known population.

What this cannot measure is contradictions that never became candidates, i.e. pairs below
the 0.90 similarity floor. That needs a lower floor and another pass.
"""

from __future__ import annotations

import random
from typing import Any, NamedTuple

VERDICTS = ("refutes", "undercuts", "neither")


class Sample(NamedTuple):
    pair_id: str
    a_text: str
    b_text: str
    a_evidence: list[str]
    b_evidence: list[str]
    a_paper: str | None
    b_paper: str | None
    similarity: float
    model_verdict: str


def sample_pairs(
    adjudicated: list[dict[str, Any]],
    *,
    per_verdict: int = 20,
    seed: int = 0,
) -> list[Sample]:
    """Draw `per_verdict` pairs from each of the three verdicts.

    Deterministic under `seed` so an audit can be extended without re-labelling what was
    already judged.
    """
    rng = random.Random(seed)
    by_verdict: dict[str, list[dict]] = {v: [] for v in VERDICTS}
    for row in adjudicated:
        if row.get("model_verdict") in by_verdict:
            by_verdict[row["model_verdict"]].append(row)

    out: list[Sample] = []
    for v in VERDICTS:
        pool = sorted(by_verdict[v], key=lambda r: r["pair_id"])
        for r in rng.sample(pool, min(len(pool), per_verdict)):
            out.append(Sample(**{k: r[k] for k in Sample._fields}))
    # Shuffle across strata before returning. Emitting them grouped leaks the very thing
    # the sheet withholds: a labeller who notices the sheet changes character at item 21
    # has recovered the model's verdict from position alone, and the audit degrades into
    # the confirmation exercise hiding the verdict was meant to prevent.
    rng.shuffle(out)
    return out


def _agrees(human: str, model: str) -> bool:
    return human == model


def precision(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement overall and per model verdict, from `{model_verdict, human}` rows.

    Rows without a human verdict are skipped rather than counted either way -- the rule the
    deterministic metrics, `repro_gold` and `extraction_audit` all follow.

    Reported per *model* verdict, because the two directions mean different things:
    disagreement on `refutes` / `undercuts` is a false edge that would enter the graph,
    disagreement on `neither` is a real contradiction the pass discarded.
    """
    judged = [r for r in labels if r.get("human") in VERDICTS]
    if not judged:
        return {"n": 0, "agreement": None, "by_verdict": {}, "skipped": len(labels),
                "confusion": {}}

    def _a(rows: list[dict]) -> dict[str, Any]:
        ok = sum(1 for r in rows if _agrees(r["human"], r["model_verdict"]))
        return {"n": len(rows), "agree": ok, "agreement": ok / len(rows) if rows else None}

    by_verdict = {v: _a([r for r in judged if r["model_verdict"] == v]) for v in VERDICTS}
    confusion: dict[str, dict[str, int]] = {v: dict.fromkeys(VERDICTS, 0) for v in VERDICTS}
    for r in judged:
        confusion[r["model_verdict"]][r["human"]] += 1
    return {
        **_a(judged),
        "by_verdict": {v: d for v, d in by_verdict.items() if d["n"]},
        "confusion": confusion,
        "skipped": len(labels) - len(judged),
    }


def edge_precision(result: dict[str, Any]) -> float | None:
    """Fraction of accepted edges a human would also accept, either label.

    A `refutes` the human calls `undercuts` is still a real disagreement and still a useful
    edge -- the type is wrong but the edge is not spurious. This is the number that says
    what fraction of the 3,072 belongs in the graph at all, and it is the one to apply to
    that total.
    """
    c = result.get("confusion") or {}
    accepted = [c.get(v, {}) for v in ("refutes", "undercuts")]
    total = sum(sum(row.values()) for row in accepted)
    if not total:
        return None
    real = sum(row.get("refutes", 0) + row.get("undercuts", 0) for row in accepted)
    return real / total


def summarize(result: dict[str, Any], *, accepted_total: int | None = None) -> str:
    if not result["n"]:
        return "nothing judged yet"
    lines = [
        f"exact agreement {result['agreement']:.1%}  ({result['agree']}/{result['n']} judged)"
    ]
    if result["skipped"]:
        lines.append(f"  {result['skipped']} unjudged, excluded")

    lines.append("\nby model verdict:")
    for v, d in result["by_verdict"].items():
        lines.append(f"  {v:12} {d['agreement']:>6.1%}  ({d['agree']}/{d['n']})")

    lines.append("\nconfusion (rows = model, cols = human):")
    lines.append(f"  {'':12}" + "".join(f"{v:>12}" for v in VERDICTS))
    for v in VERDICTS:
        row = result["confusion"].get(v, {})
        if sum(row.values()):
            lines.append(f"  {v:12}" + "".join(f"{row.get(h, 0):>12}" for h in VERDICTS))

    ep = edge_precision(result)
    if ep is not None:
        lines.append(f"\nedge precision {ep:.1%} -- accepted pairs a human also calls a "
                     f"disagreement,\n  counting refutes/undercuts as interchangeable")
        if accepted_total:
            lines.append(f"  -> of {accepted_total:,} accepted edges, "
                         f"~{round(ep * accepted_total):,} are real")

    miss = result["confusion"].get("neither", {})
    if sum(miss.values()):
        missed = miss.get("refutes", 0) + miss.get("undercuts", 0)
        lines.append(f"\nmissed: {missed}/{sum(miss.values())} sampled `neither` pairs are "
                     f"real disagreements the pass discarded")
    return "\n".join(lines)