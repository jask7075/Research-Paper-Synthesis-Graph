"""`run_system` end to end, with a stub system and no judge.

The runner produces the artifact the Iteration-1 exit criterion is read off, and had no
test. The one that matters most here is `test_traces_carry_the_evidence_text`: traces used
to record only `len(evidence)`, which silently made `attribution` uncalibratable — the
judge scores that criterion with the retrieved context in its prompt, so a human grading
from the answer alone is answering a different question. The first calibrated run put that
criterion at kappa=+0.02 (p=0.92) while the other four correlated strongly.
"""

from __future__ import annotations

import json
from pathlib import Path

from rpsg.eval.gold_schema import GoldRecord, KeyClaim, QueryType
from rpsg.eval.runner import run_system
from rpsg.retrieval.baselines import SystemOutput

EVIDENCE = "[P1] Barren plateaus arise in deep random circuits.\n[P2] Layerwise training helps."


class StubSystem:
    """Answers from a fixed script; records the queries it was asked."""

    name = "stub"

    def __init__(self, text: str = "A claim. [paper:aaa]", evidence: str = EVIDENCE) -> None:
        self._text = text
        self._evidence = evidence
        self.queries: list[str] = []

    def answer(self, query: str) -> SystemOutput:
        self.queries.append(query)
        return SystemOutput(text=self._text, cited_paper_ids=["paper:aaa"], evidence=self._evidence)


def _gold(qid: str = "q1", query_type: QueryType = QueryType.RELATIONAL) -> GoldRecord:
    return GoldRecord(
        qid=qid,
        query=f"question for {qid}?",
        query_type=query_type,
        facets=["a facet"],
        must_cite=["paper:aaa"],
        key_claims=[KeyClaim(text="a claim", papers=["paper:aaa"])],
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_traces_carry_the_evidence_text(tmp_path: Path):
    run_system(StubSystem(), [_gold()], tmp_path, use_judge=False, corpus_ids={"aaa"})
    (trace,) = _rows(tmp_path / "traces.jsonl")
    assert trace["evidence"] == EVIDENCE, "traces must persist the evidence, not just its length"


def test_trace_length_agrees_with_the_evidence_it_records(tmp_path: Path):
    run_system(StubSystem(), [_gold()], tmp_path, use_judge=False, corpus_ids={"aaa"})
    (trace,) = _rows(tmp_path / "traces.jsonl")
    assert trace["evidence_chars"] == len(trace["evidence"])


def test_trace_identifies_its_query_and_system(tmp_path: Path):
    run_system(StubSystem(), [_gold("q7")], tmp_path, use_judge=False, corpus_ids={"aaa"})
    (trace,) = _rows(tmp_path / "traces.jsonl")
    assert (trace["qid"], trace["system"]) == ("q7", "stub")


def test_every_expected_artifact_is_written(tmp_path: Path):
    run_system(StubSystem(), [_gold()], tmp_path, use_judge=False, corpus_ids={"aaa"})
    for name in ("answers.jsonl", "traces.jsonl", "scores.jsonl", "violations.jsonl", "report.md"):
        assert (tmp_path / name).exists(), f"{name} was not written"


def test_one_row_per_gold_query(tmp_path: Path):
    gold = [_gold("q1"), _gold("q2", QueryType.LOOKUP), _gold("q3", QueryType.REFUTATION)]
    run_system(StubSystem(), gold, tmp_path, use_judge=False, corpus_ids={"aaa"})
    for name in ("answers.jsonl", "traces.jsonl", "scores.jsonl"):
        assert [r["qid"] for r in _rows(tmp_path / name)] == ["q1", "q2", "q3"]


def test_citations_are_harvested_from_the_answer_text(tmp_path: Path):
    """The regex harvest is what `must_cite_recall` is computed from, so it is load-bearing."""
    system = StubSystem(text="Grounded. [paper:bbb] And more. [paper:aaa]")
    run_system(system, [_gold()], tmp_path, use_judge=False, corpus_ids={"aaa", "bbb"})
    (answer,) = _rows(tmp_path / "answers.jsonl")
    assert answer["cited_paper_ids"] == ["paper:aaa", "paper:bbb"]


def test_scores_are_written_without_judge_keys_when_the_judge_is_off(tmp_path: Path):
    run_system(StubSystem(), [_gold()], tmp_path, use_judge=False, corpus_ids={"aaa"})
    (row,) = _rows(tmp_path / "scores.jsonl")
    assert row["must_cite_recall"] == 1.0
    assert not [k for k in row if k.startswith("judge_")], "no judge ran; no judge scores"


def test_a_broken_answer_is_recorded_as_a_violation(tmp_path: Path):
    run_system(StubSystem(text=""), [_gold()], tmp_path, use_judge=False, corpus_ids={"aaa"})
    violations = _rows(tmp_path / "violations.jsonl")
    assert violations, "an empty answer must produce a violation"
    assert all(v["qid"] == "q1" for v in violations)