"""Level-1 well-formedness checks. No API keys, no network — these run on every commit."""

from __future__ import annotations

from pathlib import Path

from rpsg.eval.metrics import Answer
from rpsg.eval.wellformed import (
    check_answer,
    check_answers,
    load_corpus_paper_ids,
    summarize,
)

CORPUS = {"aaa111", "bbb222", "ccc333"}

GOOD = Answer(
    qid="q1",
    text=(
        "Hardware-efficient ansatze suffer from barren plateaus as qubit count grows "
        "[paper:aaa111], which layerwise training partially mitigates [paper:bbb222]."
    ),
    cited_paper_ids=["paper:aaa111", "paper:bbb222"],
)


def _codes(answer: Answer, **kw) -> set[str]:
    return {v.code for v in check_answer(answer, **kw)}


def test_wellformed_answer_has_no_violations():
    assert check_answer(GOOD, corpus_ids=CORPUS) == []


def test_empty_and_whitespace_text_are_caught():
    for text in ("", "   \n\t "):
        answer = Answer(qid="q1", text=text, cited_paper_ids=["paper:aaa111"])
        assert "empty_text" in _codes(answer, corpus_ids=CORPUS)


def test_empty_text_is_not_also_reported_as_short():
    """One failure should produce one violation, or the counts stop meaning anything."""
    answer = Answer(qid="q1", text="", cited_paper_ids=["paper:aaa111"])
    codes = _codes(answer, corpus_ids=CORPUS)
    assert codes == {"empty_text"}


def test_stub_answer_is_caught_as_too_short():
    answer = Answer(qid="q1", text="Not enough evidence.", cited_paper_ids=["paper:aaa111"])
    assert "text_too_short" in _codes(answer, corpus_ids=CORPUS)


def test_uncited_answer_is_caught():
    answer = Answer(qid="q1", text=GOOD.text, cited_paper_ids=[])
    assert "no_citations" in _codes(answer, corpus_ids=CORPUS)


def test_hallucinated_paper_id_is_caught():
    """The headline check: a citation that looks authoritative and points at nothing."""
    answer = Answer(
        qid="q1",
        text=GOOD.text + " Deep circuits are always trainable [paper:deadbeef].",
        cited_paper_ids=["paper:aaa111", "paper:deadbeef"],
    )
    violations = check_answer(answer, corpus_ids=CORPUS)
    assert [v.code for v in violations] == ["unknown_paper"]
    assert "deadbeef" in violations[0].detail


def test_unknown_paper_check_is_skipped_without_a_corpus():
    """A missing manifest must not stop the other checks from running."""
    answer = Answer(qid="q1", text=GOOD.text, cited_paper_ids=["paper:deadbeef"])
    assert _codes(answer, corpus_ids=None) == set()
    assert _codes(answer, corpus_ids=CORPUS) == {"unknown_paper"}


def test_dangling_synthesis_handle_is_caught():
    """`[P1]` in the final text means handle resolution failed; metrics cannot read it."""
    answer = Answer(
        qid="q1",
        text="Layerwise training mitigates barren plateaus [P2], as several works show.",
        cited_paper_ids=["paper:aaa111"],
    )
    assert "dangling_handle" in _codes(answer, corpus_ids=CORPUS)


def test_comma_separated_dangling_handles_are_caught():
    """`_HANDLE` in baselines tolerates `[P1, P3]`, so the check must too."""
    answer = Answer(
        qid="q1", text=GOOD.text + " Both agree [P1, P3].", cited_paper_ids=["paper:aaa111"]
    )
    assert "dangling_handle" in _codes(answer, corpus_ids=CORPUS)


def test_resolved_citations_are_not_mistaken_for_handles():
    """`[paper:...]` is the resolved form and must never trip the handle check."""
    assert "dangling_handle" not in _codes(GOOD, corpus_ids=CORPUS)


def test_violations_carry_the_qid():
    answer = Answer(qid="rel-007", text="", cited_paper_ids=[])
    assert all(v.qid == "rel-007" for v in check_answer(answer, corpus_ids=CORPUS))


def test_check_answers_aggregates_across_a_run():
    bad = Answer(qid="q2", text="", cited_paper_ids=[])
    violations = check_answers([GOOD, bad], corpus_ids=CORPUS)
    assert {v.qid for v in violations} == {"q2"}
    assert {v.code for v in violations} == {"empty_text", "no_citations"}


def test_summarize_is_explicit_when_clean():
    assert "No well-formedness violations" in summarize([])


def test_summarize_groups_by_code():
    violations = check_answers(
        [
            Answer(qid="q1", text="", cited_paper_ids=[]),
            Answer(qid="q2", text="", cited_paper_ids=[]),
        ],
        corpus_ids=CORPUS,
    )
    out = summarize(violations)
    assert "`empty_text` × 2" in out
    assert "q1" in out and "q2" in out


def test_uncited_answer_is_not_credited_with_the_retrieved_papers():
    """Regression guard spanning baselines -> runner -> metrics -> checks.

    `VectorRAGSystem.answer` used to fall back to every retrieved paper when the model
    cited nothing. That scored `must_cite_recall = 1.0` for an answer with no citations
    at all (whenever retrieval had found the required papers) and hid the failure from
    `check_answer`, because the ids were backfilled before the check ran. Both symptoms
    are asserted here: the metric must report the failure, and the check must see it.
    """
    from rpsg.eval.gold_schema import GoldRecord, QueryType
    from rpsg.eval.metrics import deterministic_scores
    from rpsg.eval.runner import _cited_from_text
    from rpsg.retrieval.baselines import VectorRAGSystem

    gold = GoldRecord(
        qid="rel-001",
        query="q",
        query_type=QueryType.RELATIONAL,
        facets=["a"],
        must_cite=["paper:aaa111", "paper:bbb222"],
    )
    # Retrieval succeeded — both required papers were offered to the synthesizer.
    handles = {"P1": "paper:aaa111", "P2": "paper:bbb222", "P3": "paper:ccc333"}
    uncited = "Barren plateaus arise as depth grows, and mitigations have been proposed."

    _, cited = VectorRAGSystem._resolve_handles(uncited, handles)
    assert cited == [], "an answer with no handles must report no citations"

    answer = Answer(qid="rel-001", text=uncited, cited_paper_ids=_cited_from_text(uncited, cited))
    assert deterministic_scores(answer, gold)["must_cite_recall"] == 0.0
    assert "no_citations" in _codes(answer, corpus_ids=CORPUS)


def test_load_corpus_paper_ids(tmp_path: Path):
    p = tmp_path / "papers.jsonl"
    p.write_text('{"paperId": "aaa111"}\n\n{"paperId": "bbb222"}\n{"title": "no id"}\n')
    assert load_corpus_paper_ids(p) == {"aaa111", "bbb222"}


def test_load_corpus_paper_ids_tolerates_a_missing_file(tmp_path: Path):
    assert load_corpus_paper_ids(tmp_path / "absent.jsonl") == set()
