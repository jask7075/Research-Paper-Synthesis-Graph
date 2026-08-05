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


def test_must_cite_recall_is_none_when_the_gold_requires_nothing():
    """Inapplicable, not perfect. A default of 1.0 credits the system for a question the
    gold never asked, and those credits are averaged into the headline number."""
    gold = _gold(must_cite=[])
    ans = Answer(qid="q1", text="", cited_paper_ids=[])
    assert must_cite_recall(ans, gold) is None


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


def test_refutation_surfaced_is_none_without_a_known_contradiction():
    """7 of 10 gold queries encode no contradiction. Defaulting them to 1.0 reported an
    aggregate of 0.700 while all three queries that did encode one scored 0.00."""
    gold = _gold(known_refutations=[])
    ans = Answer(qid="q1", text="", cited_paper_ids=[])
    assert refutation_surfaced(ans, gold) is None