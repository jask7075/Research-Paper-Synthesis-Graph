"""§3.5: compare arms across repeated runs, paired per query. No API keys, no network.

    python scripts/compare_arms.py --suffix queries.full34
    python scripts/compare_arms.py --suffix queries.full34 --metric key_claim_source_recall

Reads  every eval/runs/*<suffix>/ directory, grouped by the system that produced it
Writes eval/runs/comparison_<metric>.txt

**Paired, not a comparison of run means.** Every arm answers the same 34 questions, so the
comparison that matters is per query: on how many queries did A beat B, and by how much. A
Wilcoxon signed-rank over 34 paired differences is far more powerful than a t-test over three
run-level averages, which is what "3 repeats" would otherwise give.

**Repeats are averaged per query before pairing.** Each arm's score for a query is the mean
of its repeats, so the pairing is one number per query per arm and the repeat-to-repeat
spread is reported separately. Pooling repeats as independent observations would trebles the
apparent n while the queries stay the same 34, which inflates significance.

**The spread is reported beside every mean.** §4.1's n=10 figure and §6's whole calibration
table were each a single draw, and both turned out to be misleading. A mean without a spread
is the same mistake.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rpsg.config import get_settings

DETERMINISTIC = (
    "must_cite_recall",
    "citation_precision",
    "key_claim_source_recall",
    "refutation_surfaced",
)

#: The §3.5 breakdown: relational queries whose gold `key_claims` name a limitation or cost.
#: §4.5 predicts the agentic gain concentrates here. Pre-registered as underpowered — the
#: split is 5/9 — so a flat result reads as underpowered and a large one is not confirmation.
_LIMITATION = re.compile(
    r"limit|cost|overhead|constrain|fail|barrier|bottleneck|drawback|weak", re.I
)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _system_of(run: Path) -> str:
    """The arm that produced a run, from its trace rather than its directory name."""
    traces = run / "traces.jsonl"
    if traces.exists():
        rows = _jsonl(traces)
        if rows and rows[0].get("system"):
            return str(rows[0]["system"])
    return run.name.split("_", 1)[-1]


def load_runs(runs_dir: Path, suffix: str) -> dict[str, list[dict[str, dict]]]:
    """system -> [ {qid: score_row}, one per repeat ]."""
    out: dict[str, list[dict[str, dict]]] = defaultdict(list)
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or suffix not in run.name or not (run / "scores.jsonl").exists():
            continue
        out[_system_of(run)].append({r["qid"]: r for r in _jsonl(run / "scores.jsonl")})
    return dict(out)


def per_query(repeats: list[dict[str, dict]], metric: str) -> dict[str, float]:
    """Mean across repeats, per query. `None` scores are skipped, not coerced."""
    acc: dict[str, list[float]] = defaultdict(list)
    for rep in repeats:
        for qid, row in rep.items():
            if row.get(metric) is not None:
                acc[qid].append(float(row[metric]))
    return {q: sum(v) / len(v) for q, v in acc.items() if v}


def run_means(repeats: list[dict[str, dict]], metric: str) -> list[float]:
    """One mean per repeat, for the spread."""
    out = []
    for rep in repeats:
        vals = [float(r[metric]) for r in rep.values() if r.get(metric) is not None]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """Wilcoxon signed-rank over the queries both arms scored.

    Ties are counted and reported: on this gold set many queries score identically under two
    arms, and a test that silently drops them overstates how much evidence there is.
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
    if len(nonzero) < 6:
        res["p"] = None
        res["note"] = f"only {len(nonzero)} non-tied queries — too few to test"
        return res
    res["p"] = float(wilcoxon(nonzero).pvalue)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="queries.full34")
    ap.add_argument("--metric", default="must_cite_recall", choices=list(DETERMINISTIC))
    ap.add_argument("--baseline", default="vector_fulltext")
    ap.add_argument("--arm", default="agentic")
    args = ap.parse_args()

    settings = get_settings()
    runs = load_runs(settings.paths.eval_runs, args.suffix)
    if not runs:
        raise SystemExit(f"no runs matching {args.suffix!r} under {settings.paths.eval_runs}")

    gold = {
        json.loads(x)["qid"]: json.loads(x)
        for x in (settings.paths.eval_gold / "queries.full34.jsonl").read_text().splitlines()
        if x.strip()
    }

    lines: list[str] = [f"§3.5 — {args.metric}, {args.suffix}\n"]
    lines.append(f"{'arm':24}{'mean':>8}{'spread':>18}{'repeats':>9}")
    scores: dict[str, dict[str, float]] = {}
    for system, repeats in sorted(runs.items()):
        means = run_means(repeats, args.metric)
        scores[system] = per_query(repeats, args.metric)
        if not means:
            continue
        lo, hi = min(means), max(means)
        lines.append(
            f"  {system:22}{sum(means) / len(means):>8.3f}"
            f"{f'{lo:.3f} – {hi:.3f}':>18}{len(means):>9}"
        )

    # Paired comparisons against every other arm, not just the baseline: the plan's outcome
    # table distinguishes "beats static on relational" from "beats static uniformly", and
    # that needs the ablation and the graph arms too.
    if args.arm in scores:
        lines.append(f"\nPaired, per query — {args.arm} vs each arm")
        lines.append(f"  {'against':24}{'Δ mean':>9}{'wins':>7}{'losses':>8}{'ties':>6}{'p':>9}")
        for other in sorted(scores):
            if other == args.arm:
                continue
            r = paired(scores[args.arm], scores[other])
            p = "—" if r["p"] is None else f"{r['p']:.3f}"
            lines.append(
                f"  {other:22}{r['mean_diff']:>+9.3f}{r['a_wins']:>7}{r['b_wins']:>8}"
                f"{r['ties']:>6}{p:>9}"
            )
            if r.get("note"):
                lines.append(f"      ({r['note']})")

    # By query type, then the pre-registered relational split.
    lines.append("\nBy query type (mean over repeats)")
    types = sorted({g["query_type"] for g in gold.values()})
    header = f"  {'arm':22}" + "".join(f"{t[:11]:>13}" for t in types)
    lines.append(header)
    for system in sorted(scores):
        cells = []
        for t in types:
            qs = [q for q in scores[system] if gold.get(q, {}).get("query_type") == t]
            cells.append(
                f"{sum(scores[system][q] for q in qs) / len(qs):.3f}" if qs else "—"
            )
        lines.append(f"  {system:22}" + "".join(f"{c:>13}" for c in cells))

    lines.append(
        "\nRequired breakdown — relational, split on whether gold key_claims name a "
        "limitation/cost\n  (pre-registered as UNDERPOWERED: the split is 5/9)"
    )
    named, unnamed = [], []
    for q, g in gold.items():
        if g["query_type"] != "relational":
            continue
        (named if any(_LIMITATION.search(kc["text"]) for kc in g.get("key_claims", []))
         else unnamed).append(q)
    col_a = f"names limit (n={len(named)})"
    col_b = f"does not (n={len(unnamed)})"
    lines.append(f"  {'arm':22}{col_a:>22}{col_b:>22}")
    def avg(system: str, qs: list[str]) -> str:
        vals = [scores[system][q] for q in qs if q in scores[system]]
        return f"{sum(vals) / len(vals):.3f}" if vals else "—"

    for system in sorted(scores):
        lines.append(f"  {system:22}{avg(system, named):>22}{avg(system, unnamed):>22}")

    report = "\n".join(lines)
    print("\n" + report)
    out = settings.paths.eval_runs / f"comparison_{args.metric}.txt"
    out.write_text(report + "\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()