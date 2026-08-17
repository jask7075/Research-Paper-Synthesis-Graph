"""§3.5: compare arms across repeated runs, paired per query. No API keys, no network.

    python scripts/compare_arms.py --suffix queries.full34
    python scripts/compare_arms.py --suffix queries.full34 --metric key_claim_source_recall

Reads  every eval/runs/*<suffix>/ directory, grouped by the system that produced it
Writes eval/runs/comparison_<metric>.txt

The statistics live in `rpsg.eval.comparison` so they can be unit-tested; this script is the
file I/O, the query-type breakdowns and the report. See that module for why the comparison is
paired per query, why repeats are averaged before pairing, and why ties are surfaced.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.comparison import paired, per_query, run_means

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