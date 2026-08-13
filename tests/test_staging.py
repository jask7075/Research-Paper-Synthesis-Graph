"""Query-time STAGED writes (§3.2). Deterministic: no API key, no network.

The acceptance criterion is an equality -- a CURATED-only run produces identical numbers
with and without STAGED present -- so the tests here are mostly about what must NOT happen.

The invariant these guard was documented in `stores/base.py` since Iteration 1 and enforced
nowhere: neither retrieval arm filtered on `source_layer`, and the promise held only because
nothing had ever written a STAGED node. 3.2 writes them, so the guards have to be real.
"""

from __future__ import annotations

import pytest

from rpsg.extraction.schema import Node, NodeType, SourceLayer
from rpsg.retrieval.staging import (
    STAGED_CONFIDENCE,
    decomposition_nodes,
    write_staged,
)


def _traj(**kw) -> dict:
    base = {
        "query": "which methods, and what limits each",
        "sub_questions": ["which methods", "what limits each method"],
        "per_sub_question": [
            {"sub_question": "which methods", "papers": ["pA", "pB"]},
            {"sub_question": "what limits each method", "papers": ["pC"]},
        ],
        "planner_failed": False,
    }
    base.update(kw)
    return base


class _Store:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list = []

    def upsert_nodes(self, nodes) -> None:  # noqa: ANN001
        self.nodes.extend(nodes)

    def upsert_edges(self, edges) -> None:  # noqa: ANN001
        self.edges.extend(edges)


# ---- what gets staged ---------------------------------------------------------------

def test_every_staged_node_is_marked_staged() -> None:
    """The whole safety story rests on this one field."""
    nodes, edges = decomposition_nodes(_traj(), qid="q1", system="agentic", model="m")
    assert nodes and edges
    assert all(n.source_layer is SourceLayer.STAGED for n in nodes)
    assert all(e.source_layer is SourceLayer.STAGED for e in edges)


def test_staged_nodes_carry_provenance() -> None:
    """Without provenance a staged node is an unattributable assertion, and the review path
    `promote_staged` implements would have nothing to review."""
    nodes, _ = decomposition_nodes(_traj(), qid="q1", system="agentic", model="nano")
    for n in nodes:
        assert n.attrs["derived_by"] == "agentic"
        assert n.attrs["planner_model"] == "nano"
        assert n.attrs["qid"] == "q1"


def test_a_sub_question_records_the_papers_it_reached() -> None:
    """The part worth keeping: which papers jointly bear on one part of a question.
    Per-paper extraction cannot observe that, because it never holds two papers at once."""
    nodes, _ = decomposition_nodes(_traj(), qid="q1", system="agentic", model="m")
    sub = next(n for n in nodes if n.attrs.get("staged_kind") == "sub_question")
    assert sub.attrs["papers_reached"] in ("pA,pB", "pC")


def test_a_failed_planner_stages_nothing() -> None:
    """The fallback is one retrieval on the original query. Staging that would record the
    absence of a plan as a plan."""
    failed = decomposition_nodes(_traj(planner_failed=True), qid="q", system="s", model="m")
    empty = decomposition_nodes(_traj(sub_questions=[]), qid="q", system="s", model="m")
    assert failed == ([], [])
    assert empty == ([], [])


def test_the_same_sub_question_staged_twice_is_one_node() -> None:
    """Ids are content-addressed, so re-running a query must not grow the graph."""
    a, _ = decomposition_nodes(_traj(), qid="q1", system="s", model="m")
    b, _ = decomposition_nodes(_traj(), qid="q2", system="s", model="m")
    assert [n.id for n in a] == [n.id for n in b]


def test_staged_confidence_cannot_outrank_a_curated_node() -> None:
    """`typed_graph` orders by confidence when `max_nodes` truncates. The source_layer
    filter should already make this unreachable; this is the second lock."""
    assert STAGED_CONFIDENCE < 1.0
    nodes, _ = decomposition_nodes(_traj(), qid="q", system="s", model="m")
    assert all(n.confidence == STAGED_CONFIDENCE for n in nodes)


# ---- what must be refused -----------------------------------------------------------

def test_write_staged_refuses_a_curated_node() -> None:
    """A CURATED node written down this path enters the layer the metrics read, with no
    audit and no review — the one thing the separation exists to prevent."""
    store = _Store()
    bad = Node(id="x", type=NodeType.PROBLEM, name="n", source_layer=SourceLayer.CURATED)
    with pytest.raises(ValueError, match="non-STAGED"):
        write_staged(store, [bad], [])
    assert store.nodes == []


def test_write_staged_writes_nothing_when_there_is_nothing_to_write() -> None:
    store = _Store()
    assert write_staged(store, [], []) == 0
    assert store.nodes == []


def test_promotion_refuses_without_explicit_approval() -> None:
    """§8.2's precedent: a 3,072-edge proposal would have entered the graph on file presence
    alone, stopped only by an approval flag, and was later measured at 32.5% precision."""
    from rpsg.stores.graph_store import KuzuGraphStore

    store = KuzuGraphStore.__new__(KuzuGraphStore)  # no db needed; the guard is first
    with pytest.raises(PermissionError, match="approved=True"):
        store.promote_staged()


# ---- the invariant the retrieval arms must honour ------------------------------------

def test_both_retrieval_arms_filter_on_source_layer() -> None:
    """The defect 3.2 found. `stores/base.py` has promised CURATED-only since Iteration 1
    and neither arm enforced it; the promise held only because nothing wrote STAGED. An
    unfiltered traversal would let an agent raise its own score by writing to the graph."""
    from pathlib import Path

    for module in ("typed_graph.py", "citation_graph.py"):
        src = Path("src/rpsg/retrieval") / module
        text = src.read_text()
        assert "source_layer = $curated" in text, f"{module} does not filter on source_layer"


def test_the_agentic_arm_does_not_stage_by_default() -> None:
    """3.5's scored run must not write to the graph it is measured against. The equality is
    demonstrated by running both ways, not assumed — so the default is off."""
    import inspect

    from rpsg.retrieval.agentic import AgenticSystem

    sig = inspect.signature(AgenticSystem.__init__)
    assert sig.parameters["stage_writes"].default is False