"""Re-score a finished run's stored answers with the judge. No retrieval, no synthesis.

    python scripts/rejudge.py 20260807T193557Z_vector_fulltext                  # v2 rubric
    python scripts/rejudge.py 20260807T193557Z_vector_fulltext --prompt-version v1

Reads  <run_dir>/{answers,traces,scores}.jsonl  + the gold file covering its qids
Writes eval/runs/<stamp>_<system>_judge-<version>/ — a full run dir, so
       `calibrate_judge.py` and `write_report` work on it unchanged

Why a separate script rather than re-running `06_run_eval.py`: changing a rubric must not
change the answers under test. Re-running the pipeline would resample synthesis too, and
any movement in kappa would then confound "the rubric reads attribution better" with "the
system wrote different answers this time". The answers are frozen on disk from the run that
was hand-graded; grading them again is the only comparison that isolates the rubric.

The source run is never modified. Iteration 2's reported numbers must stay reproducible
from the directory that produced them, so this writes a new run dir and records what it
came from in `rejudge.json`.

`--prompt-version v1` re-scores with the *old* rubric, which is the control: it measures how
far a criterion drifts on resampling alone. A v2 gain is only a gain to the extent it
exceeds that drift.

That control is why `--repeats` exists. Re-scoring the 34 hand-graded answers with the
unchanged v1 rubric did not reproduce §6: 13 of 34 attribution scores moved, and
`refutation_handling` went +0.65 -> +0.43, i.e. from certified-trustworthy to under the bar,
on nothing but a second draw. A kappa from one sample per answer is a draw, not a
measurement. `--repeats N` scores each answer N times so the spread is visible, and
`calibrate_judge.py` then certifies a criterion only if it clears the bar on *every* draw.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.gold_schema import GoldRecord
from rpsg.eval.gold_schema import resolve_gold as _resolve_gold
from rpsg.eval.judge import CRITERIA, PROMPT_VERSIONS, Judge
from rpsg.eval.metrics import Answer, deterministic_scores
from rpsg.eval.runner import write_report
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger

log = get_logger(__name__)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def resolve_gold(
    qids: set[str], gold_dir: Path, explicit: str | None
) -> tuple[Path, list[GoldRecord]]:
    """Thin wrapper turning the library's ValueError into a clean CLI exit."""
    try:
        return _resolve_gold(qids, gold_dir, explicit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _consensus(samples: list[list[dict]]) -> list[dict]:
    """One row per qid, judge criteria reduced across samples by median.

    Median rather than mean because the scale is ordinal 1-5: a mean of 2.5 is not a grade
    any rubric defines, and `quadratic_weighted_kappa` takes integer labels. With an even
    number of samples the lower of the two middle values is taken, so the consensus is never
    more generous than half the draws.

    Deterministic metrics are identical across samples — they are a pure function of the
    answer and the gold record — so the first sample's values carry through unchanged.
    """
    if len(samples) == 1:
        return samples[0]
    by_qid: dict[str, list[dict]] = {}
    for s in samples:
        for r in s:
            by_qid.setdefault(r["qid"], []).append(r)
    out = []
    for rs in by_qid.values():
        row = dict(rs[0])
        for c in CRITERIA:
            vals = sorted(r[f"judge_{c}"] for r in rs)
            row[f"judge_{c}"] = vals[(len(vals) - 1) // 2]
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a finished run directory (name under eval/runs, or a path)")
    ap.add_argument("--prompt-version", choices=list(PROMPT_VERSIONS), default=None,
                    help="judge rubric version; default is the judge's own default")
    ap.add_argument("--gold", default=None, help="gold file; default auto-selects by qid coverage")
    ap.add_argument("--model", default=None, help="judge model; default from settings")
    ap.add_argument("--out", default=None, help="output run dir; default is auto-named")
    ap.add_argument("--repeats", type=int, default=1,
                    help="score each answer N times so the sampling spread is visible")
    args = ap.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    settings = get_settings()
    src = Path(args.run_dir)
    if not src.is_absolute():
        src = settings.paths.eval_runs / args.run_dir
    if not (src / "answers.jsonl").exists():
        raise SystemExit(f"no answers.jsonl in {src}")

    answers = [Answer(**{k: v for k, v in r.items() if k in {"qid", "text", "cited_paper_ids"}})
               for r in _jsonl(src / "answers.jsonl")]
    # Evidence is what the judge reads to check attribution. `traces.jsonl` has persisted it
    # since Iteration 2 (§3.3); without it the judge grades whether claims *look* sourced,
    # which is the asymmetry that put attribution at kappa=+0.02 on the first calibrated run.
    traces = (
        {r["qid"]: r for r in _jsonl(src / "traces.jsonl")}
        if (src / "traces.jsonl").exists()
        else {}
    )
    if not traces:
        log.warning("no traces.jsonl in %s — judging without evidence, which is not comparable "
                    "to the source run's attribution scores", src)
    old_scores = (
        {r["qid"]: r for r in _jsonl(src / "scores.jsonl")}
        if (src / "scores.jsonl").exists()
        else {}
    )

    qids = {a.qid for a in answers}
    gold_path, gold = resolve_gold(qids, settings.paths.eval_gold, args.gold)
    by_qid = {g.qid: g for g in gold}
    log.info("re-judging %d answers from %s against %s", len(answers), src.name, gold_path.name)

    judge = Judge(model=args.model, prompt_version=args.prompt_version)
    # The trace records which system produced the answers; the directory name only encodes
    # it by convention, so prefer the recorded value and fall back to the name.
    system_name = next(iter(traces.values()), {}).get("system") or src.name.split("_", 1)[-1]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    default_out = f"{stamp}_{system_name}_judge-{judge.prompt_version}"
    out = Path(args.out or default_out)
    if not out.is_absolute():
        out = settings.paths.eval_runs / out
    out.mkdir(parents=True, exist_ok=True)

    samples: list[list[dict]] = []
    with (out / "judgements.jsonl").open("w") as jd_fh:
        for s in range(1, args.repeats + 1):
            rows_s: list[dict] = []
            with (out / f"scores.sample{s:02d}.jsonl").open("w") as sfh:
                for a in answers:
                    g = by_qid[a.qid]
                    evidence = traces.get(a.qid, {}).get("evidence", "")
                    js = judge.score(a, g, evidence=evidence)
                    row: dict = {"qid": a.qid, "query_type": g.query_type.value,
                                 **deterministic_scores(a, g),
                                 **{f"judge_{c}": js.scores[c] for c in CRITERIA}}
                    sfh.write(json.dumps(row) + "\n")
                    # Persisted because v1 discarded them, and "why did attribution move"
                    # is not answerable from a score alone. The v2 rubric's defect was
                    # found here and nowhere else.
                    jd_fh.write(json.dumps({"sample": s, **js.model_dump()}) + "\n")
                    rows_s.append(row)
                    log.info("re-judged %s (sample %d/%d)", a.qid, s, args.repeats)
            samples.append(rows_s)

    rows = _consensus(samples)
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    for name in ("answers.jsonl", "traces.jsonl", "human_grades.jsonl"):
        if (src / name).exists():
            (out / name).write_text((src / name).read_text())

    write_report(f"{system_name} (judge {judge.prompt_version})", rows, out / "report.md", [])
    (out / "rejudge.json").write_text(json.dumps({
        "source_run": src.name,
        "gold": gold_path.name,
        "prompt_version": judge.prompt_version,
        "judge_model": judge.model,
        "n": len(rows),
        "repeats": args.repeats,
        "stamp": stamp,
    }, indent=2) + "\n")

    print(f"\nrubric {judge.prompt_version} vs source run {src.name}  ({args.repeats} sample(s))")
    print(f"{'criterion':22} {'was':>6} {'now':>6} {'delta':>7}   moved   unstable")
    new = {r["qid"]: r for r in rows}
    for c in CRITERIA:
        k = f"judge_{c}"
        pairs = [(old_scores[q][k], new[q][k]) for q in new
                 if q in old_scores and old_scores[q].get(k) is not None]
        if not pairs:
            continue
        was = sum(o for o, _ in pairs) / len(pairs)
        now = sum(n for _, n in pairs) / len(pairs)
        moved = sum(1 for o, n in pairs if o != n)
        # How many answers this rubric could not score the same way twice. A criterion that
        # disagrees with itself cannot agree with a human any better than that ceiling.
        unstable = "—"
        if len(samples) > 1:
            per = [{r["qid"]: r[k] for r in s} for s in samples]
            n_un = sum(1 for q in new if len({p[q] for p in per}) > 1)
            unstable = f"{n_un}/{len(new)}"
        print(f"  {c:20} {was:6.2f} {now:6.2f} {now - was:+7.2f}   "
              f"{moved}/{len(pairs)}   {unstable}")
    print(f"\nwrote {out}")
    print(f"  next: python scripts/calibrate_judge.py {out.name}")
    print(USAGE.summary())


if __name__ == "__main__":
    main()