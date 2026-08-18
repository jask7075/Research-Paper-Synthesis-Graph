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
    can_conflict,
    is_comparable,
    summarize,
    to_edges,
)


def node(i: int, name: str, paper: str, ntype: str = "Claim") -> dict:
    return {"id": f"claim:{i}", "type": ntype, "name": name,
            "evidence": ["quote"], "attrs": {"from_paper": paper}}


def _c(verdict: str, a_paper: str = "p1", b_paper: str = "p2") -> Contradiction:
    return Contradiction("a", "b", "A", "B", a_paper, b_paper, 0.95, verdict, "why")


# --- eligibility ----------------------------------------------------------------------
#
# These are the rules `candidate_pairs` applies before it searches. They are tested through
# the predicates rather than through `candidate_pairs` itself so they run on a `[dev]`
# install: faiss lives in the optional `vector` extras, and a test that skips without it
# would leave these rules unverified in CI, which is the only place verification counts.

def test_only_proposition_nodes_are_comparable():
    """A Method or Dataset is a thing, not an assertion, so nothing it pairs with has a
    possible verdict and adjudicating it would spend money for no answer."""
    assert "Claim" in CLAIM_TYPES and "Limitation" in CLAIM_TYPES
    assert not ({"Method", "Dataset", "Hardware", "Software"} & CLAIM_TYPES)
    assert is_comparable(node(1, "x", "p1", "Claim"))
    assert not is_comparable(node(2, "x", "p1", "Method"))


def test_an_unnamed_node_is_not_comparable():
    assert not is_comparable({"type": "Claim", "name": "   ", "attrs": {}})


def test_same_paper_pairs_are_excluded():
    """A paper's internal contradictions are what per-paper extraction already emits;
    re-deriving them would double-count and spend adjudication re-finding known edges."""
    assert not can_conflict(node(1, "x", "p1"), node(2, "y", "p1"))


def test_cross_paper_pairs_are_kept():
    assert can_conflict(node(1, "x", "p1"), node(2, "y", "p2"))


def test_a_node_cannot_conflict_with_itself():
    assert not can_conflict(node(1, "x", "p1"), node(1, "x", "p2"))


def test_a_node_without_a_paper_is_skipped():
    """`from_paper` is what makes a pair cross-paper. Missing it, the exclusion above
    cannot be enforced -- and assuming cross-paper is exactly what that exclusion exists
    to prevent."""
    orphan = node(2, "x", "p2")
    orphan["attrs"] = {}
    assert not can_conflict(node(1, "x", "p1"), orphan)


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

def test_an_unapproved_edge_set_is_not_applied_to_the_graph(tmp_path, monkeypatch, caplog):
    """An adjudicated set is a proposal until audited. The first pass scored 32.5% edge
    precision and was rejected; applying on file presence alone would have put ~2,000
    fabricated disagreements into the graph on the next rebuild."""
    import importlib.util
    import json
    import logging

    # Located from the repository root rather than the working directory: a CWD-relative
    # path passes under `pytest` from the root and fails anywhere else, making the test a
    # property of where it was run. `rpsg.config.PROJECT_ROOT` is derived from __file__.
    from rpsg.config import PROJECT_ROOT

    stage05 = PROJECT_ROOT / "scripts" / "05_build_stores.py"
    spec = importlib.util.spec_from_file_location("stage05", stage05)
    stage05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage05)

    class Paths:
        data_processed = tmp_path

    class Settings:
        paths = Paths()

    edge = {"src": "claim:1", "dst": "claim:2", "type": "refutes",
            "confidence": 0.9, "evidence": []}

    (tmp_path / "contradictions.json").write_text(
        json.dumps({"approved": False, "edges": [edge]}))
    with caplog.at_level(logging.WARNING):
        assert stage05._contradiction_edges(Settings()) == []
    assert "not approved" in caplog.text

    (tmp_path / "contradictions.json").write_text(
        json.dumps({"approved": True, "edges": [edge]}))
    assert len(stage05._contradiction_edges(Settings())) == 1
