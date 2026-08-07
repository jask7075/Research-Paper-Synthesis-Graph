"""Sampling and scoring for the extraction precision audit.

Every count reported about the graph so far is a *volume* — 26,879 nodes, 242 `Hardware` —
and none of it says what fraction is correct. That gap matters most when an arm
underperforms: without a precision number you cannot tell weak traversal from traversal
over bad nodes.

Precision only, deliberately. Recall would need "what *should* this paper have produced",
which is a granularity judgement rather than a fact, and matching gold names against
extracted names would report entity-resolution failure as extraction failure. Precision has
a well-defined denominator: the nodes that exist.

**What this can and cannot settle about the 0.65 gate.** Nodes below it were dropped at
extraction and are absent from `extractions.jsonl`, so an audit measures only what
survived: "of what we kept, X% is right", never "we are wrongly discarding Y% of good
nodes". Answering the second needs a re-extraction with the gate lowered.

What it *can* settle is whether confidence carries any signal. Rising precision across the
bands means the score is informative and the gate is doing work; flat precision means the
score is decorative and the threshold is arbitrary — itself a finding. Note the values are
coarse (36 distinct, clustering on 0.90/0.85/0.80/0.78), so 0.65 versus 0.70 may be
indistinguishable in principle.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, NamedTuple

#: Equal n per band, not proportional. Proportional sampling would put ~2 nodes in the
#: band nearest the gate, which is exactly where the threshold decision lives.
BANDS: tuple[tuple[float, float], ...] = ((0.65, 0.75), (0.75, 0.85), (0.85, 1.01))

#: The reproducibility types are newest and least confident — `Hardware` had 50.8% of its
#: nodes below 0.80 against ~36% corpus-wide — so their quality is least known. Reserving
#: a share of every band for them stops `Claim` (40% of the graph) from crowding them out.
REPRO_TYPES = frozenset({"Hardware", "Software", "Dataset", "ReproducibilityArtifact"})


class Sample(NamedTuple):
    node_id: str
    node_type: str
    name: str
    confidence: float
    band: str
    evidence: list[str]
    paper_id: str | None


def band_of(confidence: float) -> str | None:
    for lo, hi in BANDS:
        if lo <= confidence < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def _paper_of(node: dict[str, Any]) -> str | None:
    attrs = node.get("attrs") or {}
    return attrs.get("from_paper") if isinstance(attrs, dict) else None


def sample_nodes(
    nodes: list[dict[str, Any]],
    *,
    per_band: int = 20,
    repro_share: float = 1 / 3,
    seed: int = 0,
) -> list[Sample]:
    """Draw `per_band` nodes from each confidence band, reserving a share for repro types.

    Deterministic under `seed` so an audit can be reproduced or extended without
    re-labelling what was already judged.
    """
    rng = random.Random(seed)
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        b = band_of(float(n.get("confidence", 1.0)))
        if b:
            by_band[b].append(n)

    out: list[Sample] = []
    for lo, hi in BANDS:
        b = f"{lo:.2f}-{hi:.2f}"
        pool = by_band.get(b, [])
        repro = [n for n in pool if n["type"] in REPRO_TYPES]
        rest = [n for n in pool if n["type"] not in REPRO_TYPES]
        want_repro = min(len(repro), round(per_band * repro_share))
        picked = rng.sample(repro, want_repro) + rng.sample(
            rest, min(len(rest), per_band - want_repro)
        )
        for n in picked:
            out.append(
                Sample(
                    n["id"],
                    n["type"],
                    n["name"],
                    float(n.get("confidence", 1.0)),
                    b,
                    list(n.get("evidence") or []),
                    _paper_of(n),
                )
            )
    return out


def precision(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision overall, per band and per type, from `{band, node_type, correct}` rows.

    Rows without a `correct` verdict are skipped rather than counted either way — the
    same rule the deterministic metrics and `repro_gold` follow, so an unfinished audit
    reports what was judged instead of a flattering or punitive default.
    """
    judged = [r for r in labels if isinstance(r.get("correct"), bool)]
    if not judged:
        return {"n": 0, "precision": None, "by_band": {}, "by_type": {}, "skipped": len(labels)}

    def _p(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ok = sum(1 for r in rows if r["correct"])
        return {"n": len(rows), "correct": ok, "precision": ok / len(rows)}

    by_band = {b: _p([r for r in judged if r.get("band") == b]) for b in {r["band"] for r in judged}}
    by_type = {
        t: _p([r for r in judged if r.get("node_type") == t])
        for t in {r["node_type"] for r in judged}
    }
    return {
        **_p(judged),
        "by_band": dict(sorted(by_band.items())),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1]["n"])),
        "skipped": len(labels) - len(judged),
    }


def confidence_is_informative(result: dict[str, Any], *, min_spread: float = 0.10) -> bool | None:
    """Does precision actually rise with confidence?

    `None` when fewer than two bands were judged. A flat curve means the score carries no
    signal and the 0.65 gate is arbitrary — which is a result, not a failed audit.
    """
    bands = [v["precision"] for v in result.get("by_band", {}).values() if v["n"]]
    if len(bands) < 2:
        return None
    return (max(bands) - min(bands)) >= min_spread


def summarize(result: dict[str, Any]) -> str:
    if not result["n"]:
        return "nothing judged yet"
    lines = [f"precision {result['precision']:.1%}  ({result['correct']}/{result['n']} judged)"]
    if result["skipped"]:
        lines.append(f"  {result['skipped']} unjudged, excluded")
    lines.append("\nby confidence band:")
    for b, v in result["by_band"].items():
        lines.append(f"  {b:12} {v['precision']:>6.1%}  ({v['correct']}/{v['n']})")
    informative = confidence_is_informative(result)
    if informative is False:
        lines.append("  -> flat: confidence carries no signal, the 0.65 gate is arbitrary")
    elif informative:
        lines.append("  -> precision rises with confidence: the score is informative")
    lines.append("\nby node type:")
    for t, v in result["by_type"].items():
        lines.append(f"  {t:26} {v['precision']:>6.1%}  ({v['correct']}/{v['n']})")
    return "\n".join(lines)