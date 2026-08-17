"""Score an agentic run's plans, not its answers (§3.4). No API keys, no network.

    python scripts/score_trajectory.py <run_dir>
    python scripts/score_trajectory.py <run_dir> --sensitivity   # the facet threshold curve
    python scripts/score_trajectory.py <run_dir> --per-query

Reads  <run_dir>/traces.jsonl (the `trajectory` block 3.1 writes), <run_dir>/scores.jsonl,
       and the gold file covering the run's qids
Writes <run_dir>/trajectory.txt

Nothing here calls a model. The four measures are arithmetic over the trace and the gold
record, which is deliberate -- see `rpsg.eval.trajectory` for why a judged trajectory
criterion was avoided rather than calibrated.

Runs on the static arms too, and reports nothing for them: they have no `trajectory` block
because they have no plan. That is the correct output, not an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.gold_schema import resolve_gold
from rpsg.eval.trajectory import (
    DEFAULT_FACET_THRESHOLD,
    absolute_coverage_is_valid,
    coverage_sensitivity,
    plan_outcome_coupling,
    score_trajectory,
    summarize,
)
from rpsg.logging import get_logger

log = get_logger(__name__)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--threshold", type=float, default=DEFAULT_FACET_THRESHOLD)
    ap.add_argument("--sensitivity", action="store_true",
                    help="coverage across facet thresholds, so the constant is visible")
    ap.add_argument("--per-query", action="store_true")
    ap.add_argument("--hash-embed", action="store_true",
                    help="offline embedder; makes the numbers meaningless, for plumbing only")
    args = ap.parse_args()

    settings = get_settings()
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = settings.paths.eval_runs / args.run_dir

    traces = _jsonl(run / "traces.jsonl")
    trajectories = {t["qid"]: t["trajectory"] for t in traces if t.get("trajectory")}
    if not trajectories:
        raise SystemExit(
            f"{run.name} holds no trajectories — only the agentic arms write them, and a "
            "static arm has no plan to score"
        )

    _, gold = resolve_gold(set(trajectories), settings.paths.eval_gold, None)
    by_qid = {g.qid: g for g in gold}

    from rpsg.stores.embedder import HashEmbedder, SentenceTransformerEmbedder

    embedder = (
        HashEmbedder(dim=settings.embeddings.dim)
        if args.hash_embed
        else SentenceTransformerEmbedder(
            settings.embeddings.model_name,
            settings.embeddings.dim,
            settings.embeddings.batch_size,
        )
    )

    qids = [q for q in trajectories if q in by_qid]
    plans = {q: trajectories[q].get("sub_questions") or [] for q in qids}
    scores = [
        score_trajectory(
            trajectories[q], by_qid[q], embedder, args.threshold,
            # Every other query's plan, so specificity is a paired comparison. Absolute
            # coverage cannot separate a facet's own plan from a stranger's on this corpus.
            rival_plans=[plans[o] for o in qids if o != q and plans[o]],
        )
        for q in qids
    ]

    # Printed before the scores, not after: it decides whether the coverage line means
    # anything, and a validity check reported underneath a number has already lost.
    validity = absolute_coverage_is_valid(
        [by_qid[q].facets or [] for q in qids], [plans[q] for q in qids], embedder
    )
    print("\nIs a cosine threshold able to tell a facet's own plan from a stranger's?")
    print(f"  matched mean {validity['matched_mean']:.3f}   "
          f"null mean {validity['null_mean']:.3f}   "
          f"(n={validity['n_matched']} matched, {validity['n_null']} null)")
    for th, row in sorted(validity["by_threshold"].items()):
        print(f"    threshold {th:.2f}:  matched kept {row['matched_kept']:6.1%}   "
              f"null admitted {row['null_admitted']:6.1%}")
    print(f"  -> absolute decomposition coverage is "
          f"{'USABLE' if validity['valid'] else 'NOT VALID on this corpus'}; "
          f"{'' if validity['valid'] else 'read decomposition SPECIFICITY instead'}")

    outcomes = {
        r["qid"]: r.get("must_cite_recall")
        for r in _jsonl(run / "scores.jsonl")
    } if (run / "scores.jsonl").exists() else {}
    coupling = plan_outcome_coupling(scores, outcomes)

    report = summarize(scores, coupling)
    print(f"\n{report}\n\n  (facet threshold = {args.threshold})")

    if args.sensitivity:
        print("\nDecomposition coverage across facet thresholds:")
        curves = [
            coverage_sensitivity(
                trajectories[s.qid].get("sub_questions") or [],
                by_qid[s.qid].facets or [],
                embedder,
            )
            for s in scores
        ]
        curves = [c for c in curves if c]
        if curves:
            print("  " + "".join(f"{t:>8.2f}" for t in sorted(curves[0])))
            print("  " + "".join(
                f"{sum(c[t] for c in curves) / len(curves):>8.3f}" for t in sorted(curves[0])
            ))
            print("  the headline uses the column at the stated threshold; a measure that "
                  "moves\n  sharply across this row is a threshold artefact, not a finding")

    if args.per_query:
        print(f"\n{'qid':10}{'cov':>7}{'facets':>8}{'eff':>7}{'req':>8}{'used':>6}"
              f"{'crit+req':>10}{'crit+any':>10}")
        for s in sorted(scores, key=lambda x: x.qid):
            cov = f"{s.decomposition_coverage:.2f}" if s.decomposition_coverage is not None else "—"
            eff = f"{s.retrieval_efficiency:.2f}" if s.retrieval_efficiency is not None else "—"
            print(f"  {s.qid:10}{cov:>7}{f'{s.facets_covered}/{s.facets_total}':>8}{eff:>7}"
                  f"{f'{s.required_found}/{s.required_total}':>8}{s.retrievals_used:>6}"
                  f"{s.critique_added_required:>10}{s.critique_added_any:>10}")

    out = run / "trajectory.txt"
    out.write_text(report + f"\n\n(facet threshold = {args.threshold})\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()