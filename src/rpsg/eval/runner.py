"""Run a system over the gold set and produce a scored, timestamped run.

Outputs (under eval/runs/<run_id>/):
    answers.jsonl     one Answer per query (text + cited paper ids)
    traces.jsonl      per-query trace (retrieved evidence, latency) for trajectory eval
    scores.jsonl      deterministic metrics + judge scores per query
    violations.jsonl  Level-1 well-formedness failures (see rpsg.eval.wellformed)
    report.md         aggregate + BY-QUERY-TYPE table (the headline)

Well-formedness is checked before the metrics are read, and reported at the top of
`report.md`, because a malformed answer makes its own scores meaningless — an empty answer
scores `must_cite_recall = 0.0` exactly like a careful wrong answer, and averaging the two
hides a bug inside a quality number.

`run_id` is passed in (scripts stamp it) because scripts must not call datetime at import
time in surprising ways — keep time at the edge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rpsg.config import get_settings
from rpsg.eval.calibration import CalibrationReport
from rpsg.eval.gold_schema import GoldRecord, QueryType
from rpsg.eval.judge import CRITERIA, Judge
from rpsg.eval.metrics import Answer, deterministic_scores
from rpsg.eval.wellformed import (
    Violation,
    check_answer,
    load_corpus_paper_ids,
    summarize,
)
from rpsg.logging import get_logger
from rpsg.retrieval.baselines import System

log = get_logger(__name__)

_CITE = re.compile(r"\[(paper:[^\]]+)\]")


def _cited_from_text(text: str, fallback: list[str]) -> list[str]:
    found = sorted(set(_CITE.findall(text)))
    return found or fallback


def run_system(
    system: System,
    gold: list[GoldRecord],
    run_dir: Path,
    *,
    use_judge: bool = True,
    corpus_ids: set[str] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    judge = Judge() if use_judge else None
    if corpus_ids is None:
        corpus_ids = load_corpus_paper_ids(
            get_settings().paths.data_external / "papers.jsonl"
        )
    if not corpus_ids:
        # Explicit, because silently skipping the hallucinated-citation check is exactly
        # the kind of gap that makes a clean report untrustworthy.
        log.warning("no corpus manifest found — the unknown_paper check will be skipped")

    answers_fh = (run_dir / "answers.jsonl").open("w")
    traces_fh = (run_dir / "traces.jsonl").open("w")
    scores_fh = (run_dir / "scores.jsonl").open("w")
    violations_fh = (run_dir / "violations.jsonl").open("w")

    all_scores: list[dict] = []
    all_violations: list[Violation] = []
    try:
        for g in gold:
            out = system.answer(g.query)
            cited = _cited_from_text(out.text, out.cited_paper_ids)
            answer = Answer(qid=g.qid, text=out.text, cited_paper_ids=cited)

            answers_fh.write(answer.model_dump_json() + "\n")
            # `evidence` itself, not just its length. The judge scores `attribution`
            # with the retrieved context in its prompt; a human grader reading only
            # the answer is judging whether claims *look* sourced while the judge
            # checks whether they *are*. Calibrating one against the other then
            # measures the asymmetry rather than the judge: on the first calibrated
            # run that criterion came back at kappa=+0.02, rho=+0.04, p=0.92 — no
            # relationship at all — while the other four correlated strongly.
            traces_fh.write(
                json.dumps(
                    {
                        "qid": g.qid,
                        "system": system.name,
                        "evidence": out.evidence,
                        "evidence_chars": len(out.evidence),
                    }
                )
                + "\n"
            )

            violations = check_answer(answer, corpus_ids=corpus_ids or None)
            for v in violations:
                violations_fh.write(v.model_dump_json() + "\n")
                log.warning("%s", v)
            all_violations.extend(violations)

            det = deterministic_scores(answer, g)
            row: dict = {"qid": g.qid, "query_type": g.query_type.value, **det}
            if judge is not None:
                js = judge.score(answer, g, evidence=out.evidence)
                row.update({f"judge_{c}": js.scores[c] for c in CRITERIA})
            scores_fh.write(json.dumps(row) + "\n")
            all_scores.append(row)
            log.info("scored %s (%s)", g.qid, g.query_type.value)
    finally:
        answers_fh.close()
        traces_fh.close()
        scores_fh.close()
        violations_fh.close()

    write_report(system.name, all_scores, run_dir / "report.md", all_violations)
    return run_dir


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _applicable(rows: list[dict], key: str) -> list[float]:
    """Values for `key`, dropping queries the metric does not apply to.

    `deterministic_scores` returns None where the gold gives a metric nothing to measure.
    Coercing those to 0 would punish the system for an unasked question and averaging them
    as 1.0 — the previous behaviour — credited it for one.
    """
    return [r[key] for r in rows if r.get(key) is not None]


def write_report(
    system_name: str,
    scores: list[dict],
    path: Path,
    violations: list[Violation] | None = None,
) -> None:
    """Aggregate + by-query-type table for a set of scored rows.

    Public because `scripts/rejudge.py` produces the same rows from stored answers and must
    render them identically — a re-judged run that reported its means differently would not
    be comparable to the run it re-scores.
    """
    if not scores:
        path.write_text(f"# {system_name}\n\nNo scores.\n")
        return
    metric_keys = [k for k in scores[0] if k not in ("qid", "query_type")]

    def table(rows: list[dict]) -> str:
        # `n` is shown per metric, not just per section: a mean over the 3 queries that
        # encode a contradiction must not read like a mean over all 10.
        header = "| metric | mean | n |\n|---|---|---|\n"
        body = ""
        for k in metric_keys:
            vals = _applicable(rows, k)
            mean = f"{_mean(vals):.3f}" if vals else "—"
            n = f"{len(vals)}" if len(vals) == len(rows) else f"{len(vals)} of {len(rows)}"
            body += f"| {k} | {mean} | {n} |\n"
        return header + body

    lines = [f"# Eval report — {system_name}\n", f"Queries: {len(scores)}\n"]
    # First, not last: every mean below is computed over these answers, so whether any of
    # them were malformed determines how much the means are worth.
    lines.append("## Well-formedness (Level 1)\n")
    lines.append(summarize(violations or []))
    lines.append("\n## Overall\n")
    lines.append(table(scores))
    lines.append("\n## By query type (the headline — relational is where the graph earns it)\n")
    for qt in QueryType:
        rows = [r for r in scores if r["query_type"] == qt.value]
        if rows:
            lines.append(f"\n### {qt.value}  (n={len(rows)})\n")
            lines.append(table(rows))
    path.write_text("\n".join(lines))
    log.info("wrote report -> %s", path)


def write_calibration(report: CalibrationReport, run_dir: Path) -> None:
    (run_dir / "calibration.txt").write_text(report.summary() + "\n")