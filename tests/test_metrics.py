"""Deterministic metrics — the numbers you can defend without a calibrated judge."""

from __future__ import annotations

from rpsg.eval.gold_schema import GoldRecord, KeyClaim, QueryType, RefutationPair
from rpsg.eval.metrics import (
    Answer,
    citation_precision,
    key_claim_source_recall,
    must_cite_recall,
    refutation_surfaced,
)


def _gold(**kw) -> GoldRecord:
    base = dict(qid="q1", query="?", query_type=QueryType.RELATIONAL, facets=["f1"])
    base.update(kw)
    return GoldRecord(**base)


def test_must_cite_recall_partial():
    gold = _gold(must_cite=["paper:a", "paper:b"])
    ans = Answer(qid="q1", text="", cited_paper_ids=["paper:a"])
    assert must_cite_recall(ans, gold) == 0.5


def test_must_cite_recall_empty_requirement_is_one():
    gold = _gold(must_cite=[])
    ans = Answer(qid="q1", text="", cited_paper_ids=[])
    assert must_cite_recall(ans, gold) == 1.0


def test_citation_precision_penalizes_spray():
    gold = _gold(must_cite=["paper:a"])
    ans = Answer(qid="q1", text="", cited_paper_ids=["paper:a", "paper:x", "paper:y", "paper:z"])
    assert citation_precision(ans, gold) == 0.25


def test_key_claim_source_recall():
    gold = _gold(
        key_claims=[
            KeyClaim(text="c1", papers=["paper:a"]),
            KeyClaim(text="c2", papers=["paper:b"]),
        ]
    )
    ans = Answer(qid="q1", text="", cited_paper_ids=["paper:a"])
    assert key_claim_source_recall(ans, gold) == 0.5


def test_refutation_surfaced_requires_both_sides():
    gold = _gold(
        known_refutations=[
            RefutationPair(a="paper:a claims X", b="paper:b shows not-X"),
        ]
    )
    only_one = Answer(qid="q1", text="", cited_paper_ids=["paper:a"])
    both = Answer(qid="q1", text="", cited_paper_ids=["paper:a", "paper:b"])
    assert refutation_surfaced(only_one, gold) == 0.0
    assert refutation_surfaced(both, gold) == 1.0


def test_refutation_surfaced_no_refutations_is_one():
    gold = _gold(known_refutations=[])
    ans = Answer(qid="q1", text="", cited_paper_ids=[])
    assert refutation_surfaced(ans, gold) == 1.0