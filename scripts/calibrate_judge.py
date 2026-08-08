"""Is the LLM judge trustworthy? Compare it against hand grades. No API keys, no network.

    python scripts/calibrate_judge.py <run_dir>
    python scripts/calibrate_judge.py <run_dir> --disagreements   # where the two diverge most

Reads  <run_dir>/human_grades.jsonl, <run_dir>/scores.jsonl
Writes <run_dir>/calibration.txt

`rpsg.eval.calibration` has computed quadratic-weighted kappa, Spearman and length bias
since Iteration 1, and `runner.write_calibration` has known how to persist it -- but
nothing ever read a `human_grades.jsonl` and joined it to the judge's scores, so the
numbers were produced by hand. This is that join.

Kappa is the headline rather than correlation because the criteria are ordinal 1-5 and
what matters is agreement on the level, not merely on the ranking. A judge that orders
answers exactly like the grader but sits a point lower has high rho and mediocre kappa,
and those two failures want different fixes: rescaling versus a prompt rewrite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.calibration import CalibrationReport, calibrate_criterion, length_bias
from rpsg.eval.runner import write_calibration

CRITERIA = ("coverage", "attribution", "hedging_accuracy", "refutation_handling", "synthesis")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-kappa", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--disagreements", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = settings.paths.eval_runs / run
    min_kappa = (
        args.min_kappa
        if args.min_kappa is not None
        else settings.eval.calibration.min_quadratic_kappa
    )

    human = {r["qid"]: r["grade"] for r in _jsonl(run / "human_grades.jsonl")}
    scores = {r["qid"]: r for r in _jsonl(run / "scores.jsonl")}
    answers = {r["qid"]: r["text"] for r in _jsonl(run / "answers.jsonl")}

    per_criterion, biases = [], []
    for c in CRITERIA:
        pairs = [
            (h, scores[q][f"judge_{c}"])
            for q, g in human.items()
            if (h := g.get(c)) is not None
            and q in scores
            and scores[q].get(f"judge_{c}") is not None
        ]
        if len(pairs) < 2:
            continue
        hs, js = [int(a) for a, _ in pairs], [int(b) for _, b in pairs]
        per_criterion.append(calibrate_criterion(hs, js, c, min_kappa))
        lens = [
            len(answers.get(q, ""))
            for q in human
            if q in scores and human[q].get(c) is not None
        ]
        biases.append(length_bias(lens[: len(js)], js, c, args.alpha))

    report = CalibrationReport(per_criterion=per_criterion, length_bias=biases)
    print(report.summary())
    print(f"\n  (min_kappa = {min_kappa})")
    write_calibration(report, run)
    print(f"  wrote {run / 'calibration.txt'}")

    if args.disagreements:
        print("\nlargest disagreements (human vs judge):")
        rows = []
        for q, g in human.items():
            for c in CRITERIA:
                h, j = g.get(c), scores.get(q, {}).get(f"judge_{c}")
                if h is not None and j is not None:
                    rows.append((abs(h - j), q, c, h, j))
        for d, q, c, h, j in sorted(rows, reverse=True)[:14]:
            arrow = "judge higher" if j > h else "judge lower"
            print(f"  {d}  {q:10} {c:20} human {h} judge {j}   ({arrow})")


if __name__ == "__main__":
    main()