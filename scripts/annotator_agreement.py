"""Second pass over the hand-graded answers: draw a sample, grade it blind, score agreement.

    python scripts/annotator_agreement.py --sample <run_dir> > eval/gold/annotator_b.jsonl
    python scripts/annotator_agreement.py --show                    # the grading sheet
    python scripts/annotator_agreement.py --score <run_dir>         # once it is filled in

Reads  <run_dir>/{answers,human_grades,scores}.jsonl + the gold covering its qids
Writes eval/gold/annotator_b.jsonl  (one row per answer, `grade` to fill)

§10: every calibrated criterion is calibrated against one annotator, so §6's kappas say the
judge agrees with *that grader*, not that it grades well. This measures whether that
generalises -- and, more usefully, what the ceiling is. Two readers who agree at +0.45 on a
criterion put a 0.6 judge bar out of reach on that criterion by construction; see
`rpsg.eval.annotator_agreement.ceiling` for the three readings and what each implies.

WHO GRADES matters, and the tool cannot tell:

    --annotator second   a different person. The only pass that discharges §10.
    --annotator retest   the original grader, blind, later. Measures how stable the labels
                         are, which bounds every kappa in §6 -- but two passes by one reader
                         are not two readers.
    --annotator model    a different model, different prompt, no sight of the first grades.
                         Bounds rather than settles, exactly as §8.2's audit says of itself.

Record it honestly: the scoring output is labelled with whatever is passed, and a `model`
run reported as a second annotator would misdescribe the strongest claim in the section.

**The sheet withholds the first grades and the judge's scores.** Shown a prior label a
reader agrees with it, and the exercise becomes confirmation rather than measurement.

**The sheet also withholds the retrieved evidence**, because the first grader never saw it
(§3.1). Handing it to the second reader would measure that asymmetry instead of
disagreement between readers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.annotator_agreement import (
    CRITERIA,
    agreement,
    ceiling,
    gradeable_n,
    sample_answers,
    summarize,
)
from rpsg.eval.gold_schema import resolve_gold
from rpsg.logging import get_logger

log = get_logger(__name__)

ANNOTATOR_KINDS = ("second", "retest", "model")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _load_run(run: Path) -> tuple[dict, dict, dict]:
    answers = {r["qid"]: r["text"] for r in _jsonl(run / "answers.jsonl")}
    human = {r["qid"]: r["grade"] for r in _jsonl(run / "human_grades.jsonl")}
    scores = (
        {r["qid"]: r for r in _jsonl(run / "scores.jsonl")}
        if (run / "scores.jsonl").exists()
        else {}
    )
    return answers, human, scores


def _rows_for_sampling(run: Path, gold_dir: Path) -> list[dict]:
    """Join answers to their gold records, restricted to what was hand-graded.

    Only hand-graded answers are eligible: an answer the first pass never scored gives no
    pair to agree or disagree on, so sampling it would spend a reader's attention on a row
    that cannot enter any kappa.
    """
    answers, human, _ = _load_run(run)
    qids = set(answers) & set(human)
    try:
        _, gold = resolve_gold(qids, gold_dir, None)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    by_qid = {g.qid: g for g in gold}
    return [
        {
            "qid": q,
            "query": by_qid[q].query,
            "query_type": by_qid[q].query_type.value,
            "facets": by_qid[q].facets,
            "key_claims": [kc.model_dump() for kc in by_qid[q].key_claims],
            "known_refutations": [r.model_dump() for r in by_qid[q].known_refutations],
            "answer": answers[q],
        }
        for q in sorted(qids)
        if q in by_qid
    ]


def _print_sheet(rows: list[dict]) -> None:
    for i, s in enumerate(rows, 1):
        print(f"\n{'=' * 94}\n[{i}/{len(rows)}]  {s['qid']}")
        print(f"\n  QUESTION: {s['query']}")
        print("\n  A COMPLETE ANSWER MUST ADDRESS:")
        for f in s["facets"]:
            print(f"    - {f}")
        if s["key_claims"]:
            print("\n  KEY CLAIMS EXPECTED:")
            for kc in s["key_claims"]:
                print(f"    - {kc['text']}")
        if s["known_refutations"]:
            print("\n  KNOWN CONTRADICTIONS THE ANSWER SHOULD SURFACE:")
            for r in s["known_refutations"]:
                print(f"    - {r['a']}\n      vs {r['b']}")
        else:
            print("\n  (no contradiction in the gold -> leave refutation_handling null)")
        print(f"\n  ANSWER UNDER TEST:\n{'-' * 94}")
        print(s["answer"])
        print("-" * 94)
        print("\n  grade 1-5, or null where the gold gives the criterion nothing to measure:")
        for c in CRITERIA:
            print(f"    {c}: ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="a run with human_grades.jsonl")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--total", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--annotator", choices=list(ANNOTATOR_KINDS), default="second",
                    help="who produced the second pass; recorded in the output verbatim")
    ap.add_argument("--sheet", default=None, help="path to the second-pass file")
    args = ap.parse_args()

    settings = get_settings()
    sheet = Path(args.sheet) if args.sheet else settings.paths.eval_gold / "annotator_b.jsonl"
    if not sheet.is_absolute():
        sheet = settings.paths.eval_gold / sheet

    if args.show:
        if not sheet.exists():
            raise SystemExit(f"no sheet at {sheet} — run --sample first")
        _print_sheet(_jsonl(sheet))
        return

    if not args.run_dir:
        raise SystemExit("run_dir is required for --sample and --score")
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = settings.paths.eval_runs / args.run_dir

    if args.score:
        if not sheet.exists():
            raise SystemExit(f"no sheet at {sheet} — run --sample first")
        second = {r["qid"]: r.get("grade") for r in _jsonl(sheet)}
        kind = next(
            (r.get("annotator") for r in _jsonl(sheet) if r.get("annotator")), args.annotator
        )
        _, human, scores = _load_run(run)

        filled = {q: g for q, g in second.items() if g and any(v is not None for v in g.values())}
        if not filled:
            raise SystemExit(
                f"{sheet.name} has no grades yet — fill in `grade` on each row, then re-run"
            )
        if len(filled) < len(second):
            log.warning("%d of %d rows ungraded; they are excluded, not counted",
                        len(second) - len(filled), len(second))

        min_kappa = settings.eval.calibration.min_quadratic_kappa
        pairs_hh = [{"a": human[q], "b": filled[q]} for q in filled if q in human]
        # Both judge comparisons are restricted to the same sampled qids, so a difference
        # between them is disagreement between readers and not a difference in sample.
        jscores = {
            q: {c: scores[q].get(f"judge_{c}") for c in CRITERIA} for q in filled if q in scores
        }
        pairs_ja = [{"a": jscores[q], "b": human[q]} for q in jscores if q in human]
        pairs_jb = [{"a": jscores[q], "b": filled[q]} for q in jscores]

        hh = agreement(pairs_hh, min_kappa=min_kappa)
        ja = agreement(pairs_ja, min_kappa=min_kappa)
        jb = agreement(pairs_jb, min_kappa=min_kappa)

        noun = {"second": "second annotator", "retest": "same grader, re-graded",
                "model": "model pass (bounds only)"}[kind]
        print(f"\nSecond pass: {noun}   n={len(filled)} of {len(second)} sampled\n")
        print(summarize(hh, gradeable_n(pairs_hh), label=f"annotator A vs {noun}"))
        print()
        print(summarize(ja, gradeable_n(pairs_ja), label="judge vs annotator A"))
        print()
        print(summarize(jb, gradeable_n(pairs_jb), label=f"judge vs {noun}"))
        print(f"\n{ceiling(hh, ja, jb, min_kappa=min_kappa)}")
        print(f"\n  (min_kappa = {min_kappa})")
        if kind == "model":
            print("\n  CAVEAT: a model pass bounds the disagreement rate; it does not settle "
                  "§10.\n  Only a second person does that.")
        return

    rows = _rows_for_sampling(run, settings.paths.eval_gold)
    for s in sample_answers(rows, total=args.total, seed=args.seed):
        print(json.dumps({
            "qid": s.qid,
            "query": s.query,
            "query_type": s.query_type,
            "facets": s.facets,
            "key_claims": s.key_claims,
            "known_refutations": s.known_refutations,
            "answer": s.answer,
            "annotator": args.annotator,
            "grade": dict.fromkeys(CRITERIA),
            "note": None,
        }))


if __name__ == "__main__":
    main()