"""Is the LLM judge trustworthy? Compare it against hand grades. No API keys, no network.

    python scripts/calibrate_judge.py <run_dir>                    # the active gold set
    python scripts/calibrate_judge.py <run_dir> --gold queries.full34.jsonl
    python scripts/calibrate_judge.py <run_dir> --all              # every qid in the run
    python scripts/calibrate_judge.py <run_dir> --disagreements    # where the two diverge most

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

**Calibration is reported on the ACTIVE gold set, and the run's full set is shown beside
it.** Which queries you calibrate on is a choice, not a property of the run, and on this
corpus it changes the answer rather than merely the confidence: over the same 34 stored
answers and one judge, `attribution` scores +0.79 on the active 10 and +0.30 on the other 24,
while `synthesis` runs the other way, +0.38 against +0.77. Only `coverage` is indifferent
(+0.76 vs +0.72). A single figure with no denominator beside it would let either subset be
reported as the truth, so both are always printed.

If the run holds `scores.sample*.jsonl` (see `rejudge.py --repeats`), every sample is
calibrated separately and the spread is reported. A criterion is then certified only if it
clears the bar on *all* of them. That rule exists because a single sample was enough to
certify `refutation_handling` at +0.65 in §6 and to put it at +0.43 on a second draw of the
same rubric over the same answers — one draw cannot tell a trustworthy criterion from a
lucky one.
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


def _report_stability(
    human: dict[str, dict], sample_files: list[Path], min_kappa: float
) -> None:
    """Per-criterion kappa on each judge sample, and the resulting certification.

    `worst` is the verdict, not `median`: certifying on the best or middle draw is what
    makes a lucky sample look like a trustworthy criterion. A criterion earns trust only by
    clearing the bar every time it is asked.
    """
    samples = [{r["qid"]: r for r in _jsonl(p)} for p in sample_files]
    print(f"\nStability across {len(samples)} judge samples (kappa per sample):")
    for c in CRITERIA:
        kappas = []
        for s in samples:
            pairs = [
                (h, s[q][f"judge_{c}"])
                for q, g in human.items()
                if (h := g.get(c)) is not None and q in s and s[q].get(f"judge_{c}") is not None
            ]
            if len(pairs) < 2:
                continue
            kappas.append(
                calibrate_criterion(
                    [int(a) for a, _ in pairs], [int(b) for _, b in pairs], c, min_kappa
                ).quadratic_kappa
            )
        if not kappas:
            continue
        worst, best = min(kappas), max(kappas)
        flag = "OK " if worst >= min_kappa else "!! "
        drew = " ".join(f"{k:+.2f}" for k in kappas)
        print(f"  {flag}{c:20s} worst={worst:+.2f} best={best:+.2f} "
              f"spread={best - worst:.2f}   [{drew}]")
    print("  (a criterion is certified only if its WORST sample clears the bar)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-kappa", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--disagreements", action="store_true")
    ap.add_argument("--gold", default="queries.jsonl",
                    help="restrict to this gold file's qids (default: the active set)")
    ap.add_argument("--all", action="store_true",
                    help="calibrate on every qid in the run instead of a gold subset")
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

    all_human = {r["qid"]: r["grade"] for r in _jsonl(run / "human_grades.jsonl")}
    scores = {r["qid"]: r for r in _jsonl(run / "scores.jsonl")}
    answers = {r["qid"]: r["text"] for r in _jsonl(run / "answers.jsonl")}

    label = "every qid in the run"
    human = all_human
    if not args.all:
        gold_path = settings.paths.eval_gold / args.gold
        keep = {json.loads(x)["qid"] for x in gold_path.read_text().splitlines() if x.strip()}
        human = {q: g for q, g in all_human.items() if q in keep}
        label = f"{gold_path.name} ({len(human)} of {len(all_human)} graded answers)"
        if not human:
            raise SystemExit(f"{gold_path.name} shares no qid with {run.name}")
    print(f"Calibrating on: {label}")

    sample_files = sorted(run.glob("scores.sample*.jsonl"))
    if len(sample_files) > 1:
        _report_stability(human, sample_files, min_kappa)

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

    # The queries NOT calibrated on. Reported because a subset that flatters a criterion and
    # a subset that punishes it look identical from inside the subset.
    rest = {q: g for q, g in all_human.items() if q not in human}
    if rest:
        print(f"\n  for comparison — the other {len(rest)} answers, and all {len(all_human)}:")
        print(f"  {'criterion':22}{'calibrated':>12}{'the rest':>10}{'all':>8}")
        for c in CRITERIA:
            cells = []
            for subset in (human, rest, all_human):
                pairs = [
                    (int(h), int(scores[q][f"judge_{c}"]))
                    for q, g in subset.items()
                    if (h := g.get(c)) is not None
                    and q in scores
                    and scores[q].get(f"judge_{c}") is not None
                ]
                if len(pairs) >= 2:
                    k = calibrate_criterion(
                        [a for a, _ in pairs], [b for _, b in pairs], c, min_kappa
                    ).quadratic_kappa
                    cells.append(f"{k:+.2f}")
                else:
                    cells.append(f"n={len(pairs)}")
            print(f"    {c:20}{cells[0]:>12}{cells[1]:>10}{cells[2]:>8}")
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