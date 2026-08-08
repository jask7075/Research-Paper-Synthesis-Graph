"""Cross-paper contradiction detection.

The rules under test are the ones that decide whether an edge means anything: that a
contradiction is only ever claimed between *different* papers, that only proposition-shaped
nodes are compared, and above all that no failure path can invent a disagreement.
"""

from __future__ import annotations

from rpsg.extraction.contradiction import (
    CLAIM_TYPES,
    Contradiction,
    adjudicate,
    candidate_pairs,
    summarize,
    to_edges,
)


class FakeEmbedder:
    """Maps each distinct text to a fixed unit vector; identical texts collide exactly."""

    def __init__(self, groups: dict[str, int]) -> None:
        self.groups = groups

    def encode(self, texts: list[str]) -> list[list[float]]:
        dim = max(self.groups.values()) + 1
        out = []
        for t in texts:
            v = [0.0] * dim
            v[self.groups[t]] = 1.0
            out.append(v)
        return out


def node(i: int, name: str, paper: str, ntype: str = "Claim") -> dict:
    return {"id": f"claim:{i}", "type": ntype, "name": name,
            "evidence": ["quote"], "attrs": {"from_paper": paper}}


def _c(verdict: str, a_paper: str = "p1", b_paper: str = "p2") -> Contradiction:
    return Contradiction("a", "b", "A", "B", a_paper, b_paper, 0.95, verdict, "why")


# --- candidate generation -------------------------------------------------------------

def test_same_paper_pairs_are_excluded():
    """A paper's internal contradictions are what per-paper extraction already emits;
    re-deriving them would double-count and spend adjudication re-finding known edges."""
    nodes = [node(1, "x", "p1"), node(2, "x", "p1")]
    assert candidate_pairs(nodes, FakeEmbedder({"x": 0}), floor=0.5) == []


def test_cross_paper_pairs_are_kept():
    nodes = [node(1, "x", "p1"), node(2, "x", "p2")]
    pairs = candidate_pairs(nodes, FakeEmbedder({"x": 0}), floor=0.5)
    assert [(a, b) for a, b, _ in pairs] == [("claim:1", "claim:2")]


def test_only_proposition_nodes_are_compared():
    """A Method or Dataset is a thing, not an assertion, so it cannot contradict anything
    and adjudicating it would spend money on pairs with no possible verdict."""
    assert "Claim" in CLAIM_TYPES and "Limitation" in CLAIM_TYPES
    assert not ({"Method", "Dataset", "Hardware", "Software"} & CLAIM_TYPES)
    nodes = [node(1, "x", "p1", "Method"), node(2, "x", "p2", "Method")]
    assert candidate_pairs(nodes, FakeEmbedder({"x": 0}), floor=0.5) == []


def test_dissimilar_claims_are_not_candidates():
    nodes = [node(1, "x", "p1"), node(2, "y", "p2")]
    assert candidate_pairs(nodes, FakeEmbedder({"x": 0, "y": 1}), floor=0.5) == []


def test_a_node_without_a_paper_is_skipped():
    """`from_paper` is what makes a pair cross-paper; without it the exclusion above
    cannot be enforced, so the pair must not be proposed at all."""
    a = node(1, "x", "p1")
    b = node(2, "x", "p2")
    b["attrs"] = {}
    assert candidate_pairs([a, b], FakeEmbedder({"x": 0}), floor=0.5) == []


# --- adjudication ---------------------------------------------------------------------

class BrokenClient:
    def text(self, **kw):
        raise RuntimeError("api down")


class GarbageClient:
    def text(self, **kw):
        return "I think perhaps they might disagree somewhat"


def _adjudicate_with(client, monkeypatch):
    import rpsg.extraction.contradiction as mod
    monkeypatch.setattr(mod, "get_chat_client", lambda model: client)
    nodes = {"claim:1": node(1, "x", "p1"), "claim:2": node(2, "x", "p2")}
    return adjudicate([("claim:1", "claim:2", 0.95)], nodes, model="m", workers=1)


def test_an_api_failure_does_not_invent_a_disagreement(monkeypatch):
    """A false `refutes` edge is worse than a missing one: it routes a refutation query to
    a contradiction no paper asserts. Every failure path must fail closed."""
    (out,) = _adjudicate_with(BrokenClient(), monkeypatch)
    assert out.verdict == "neither" and not out.is_edge


def test_an_unparseable_response_does_not_invent_a_disagreement(monkeypatch):
    (out,) = _adjudicate_with(GarbageClient(), monkeypatch)
    assert out.verdict == "neither" and not out.is_edge


# --- edges ----------------------------------------------------------------------------

def test_only_accepted_verdicts_become_edges():
    edges = to_edges([_c("refutes"), _c("undercuts"), _c("neither")])
    assert sorted(e["type"] for e in edges) == ["refutes", "undercuts"]


def test_edge_confidence_carries_similarity_not_a_fabricated_one():
    """The model made a binary call and has no calibrated probability to offer; the pair's
    similarity is a real measured quantity, and it is what typed_graph orders by when
    max_nodes truncates."""
    (e,) = to_edges([_c("refutes")])
    assert e["confidence"] == 0.95


def test_the_reason_is_carried_as_evidence():
    (e,) = to_edges([_c("undercuts")])
    assert e["evidence"] == ["why"]


def test_summary_reports_the_acceptance_rate_and_paper_span():
    out = summarize([_c("refutes"), _c("neither"), _c("undercuts", "p3", "p4")])
    assert "2 edges" in out and "66.7% accepted" in out
    assert "spanning 4 papers" in out


def test_summary_says_so_when_nothing_was_adjudicated():
    assert "no pairs adjudicated" in summarize([])