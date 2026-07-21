"""Run a system over the gold set and produce a scored, timestamped run.

Outputs (under eval/runs/<run_id>/):
    answers.jsonl   one Answer per query (text + cited paper ids)
    traces.jsonl    per-query trace (retrieved evidence, latency) for trajectory eval
    scores.jsonl    deterministic metrics + judge scores per query
    report.md       aggregate + BY-QUERY-TYPE table (the headline)

`run_id` is passed in (scripts stamp it) because scripts must not call datetime at import
time in surprising ways — keep time at the edge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rpsg.eval.calibration import CalibrationReport
from rpsg.eval.gold_schema import GoldRecord, QueryType
from rpsg.eval.judge import CRITERIA, Judge
from rpsg.eval.metrics import Answer, deterministic_scores
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
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    judge = Judge() if use_judge else None

    answers_fh = (run_dir / "answers.jsonl").open("w")
    traces_fh = (run_dir / "traces.jsonl").open("w")
    scores_fh = (run_dir / "scores.jsonl").open("w")

    all_scores: list[dict] = []
    try:
        for g in gold:
            out = system.answer(g.query)
            cited = _cited_from_text(out.text, out.cited_paper_ids)
            answer = Answer(qid=g.qid, text=out.text, cited_paper_ids=cited)

            answers_fh.write(answer.model_dump_json() + "\n")
            traces_fh.write(
                json.dumps(
                    {"qid": g.qid, "system": system.name, "evidence_chars": len(out.evidence)}
                )
                + "\n"
            )

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

    _write_report(system.name, all_scores, run_dir / "report.md")
    return run_dir


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _write_report(system_name: str, scores: list[dict], path: Path) -> None:
    if not scores:
        path.write_text(f"# {system_name}\n\nNo scores.\n")
        return
    metric_keys = [k for k in scores[0] if k not in ("qid", "query_type")]

    def table(rows: list[dict]) -> str:
        header = "| metric | mean |\n|---|---|\n"
        body = "".join(
            f"| {k} | {_mean([r[k] for r in rows]):.3f} |\n" for k in metric_keys
        )
        return header + body

    lines = [f"# Eval report — {system_name}\n", f"Queries: {len(scores)}\n", "## Overall\n"]
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