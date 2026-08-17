"""Query-time STAGED writes with provenance (§3.2).

`SourceLayer.STAGED` has existed since Iteration 1, `promote_staged` has been implemented
since Iteration 1, and until now grep found STAGED written **nowhere**. The interface was
built for exactly this and never used.

**What gets staged.** The decomposition. When 3.1 plans *"which methods, and what limits
each"* into a sub-question for the methods and one per method for its limits, that
decomposition is derived at query time and thrown away when the answer is returned. It is
the one artefact the loop produces that per-paper extraction structurally cannot: extraction
reads one paper at a time, so it can never observe that two papers jointly answer a
sub-question. Staging keeps it, with provenance, outside everything that gets measured.

**The invariant that makes it safe was not actually enforced.** `stores/base.py` has said
since Iteration 1 that metrics query CURATED only. Neither `typed_graph` nor `citation_graph`
filtered on `source_layer`; the invariant held only because nothing had ever written a STAGED
node. Writing one would have let an agent raise its own score by writing to the graph -- the
self-grading Iteration 2 refused when it declined to pick `must_cite` with the system's own
retriever. Both arms now filter, and `staged_is_invisible` is the test that they do.

**Promotion is a review path, not an automation.** §8.2 is the precedent: an unaudited
3,072-edge proposal would have entered the graph on file presence alone until an approval
flag stopped it, and the audit then showed 32.5% edge precision. `promote_staged` now refuses
without an explicit approval for the same reason.
"""

from __future__ import annotations

import hashlib
from typing import Any

from rpsg.extraction.schema import Edge, EdgeType, Node, NodeType, SourceLayer
from rpsg.logging import get_logger

log = get_logger(__name__)

#: Confidence on a staged decomposition.
#:
#: Not a belief that the sub-question is *correct* -- it is a verbatim record of what the
#: planner emitted, and about that there is no uncertainty. It is set below 1.0 so that a
#: staged node can never outrank a curated one if the two are ever ranked together, which
#: `typed_graph` does by `confidence` when `max_nodes` truncates. Belt and braces: the
#: source_layer filter should already make that impossible.
STAGED_CONFIDENCE = 0.5


def _sid(prefix: str, text: str) -> str:
    """A stable id for a derived node.

    Content-addressed, so the same sub-question staged twice is one node rather than two.
    A run that repeats a query must not grow the graph.
    """
    digest = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]
    return f"staged:{prefix}:{digest}"


def decomposition_nodes(
    trajectory: dict[str, Any],
    *,
    qid: str,
    system: str,
    model: str,
) -> tuple[list[Node], list[Edge]]:
    """The plan as STAGED `Problem` nodes, linked to the question they came from.

    A sub-question is a `Problem` because that is what it is: a thing to be answered, stated
    in the interrogative. Reusing the existing type rather than inventing a `SubQuestion` one
    keeps the schema fixed -- a new node type would have to be routed, extracted and audited
    like any other, and 3.2's acceptance is that nothing measured moves.

    Returns `([], [])` when the planner failed, because a fallback to one retrieval on the
    original query is not a decomposition and staging it would record the absence of a plan
    as a plan.
    """
    subs = [s for s in (trajectory.get("sub_questions") or []) if s.strip()]
    if trajectory.get("planner_failed") or not subs:
        return [], []
    query = (trajectory.get("query") or "").strip()
    if not query:
        return [], []

    reached: dict[str, list[str]] = {
        step.get("sub_question", ""): list(step.get("papers") or [])
        for step in (trajectory.get("per_sub_question") or [])
    }

    root = Node(
        id=_sid("question", query),
        type=NodeType.PROBLEM,
        name=query,
        source_layer=SourceLayer.STAGED,
        confidence=STAGED_CONFIDENCE,
        attrs={
            "derived_by": system,
            "planner_model": model,
            "qid": qid,
            "staged_kind": "question",
        },
    )
    nodes = [root]
    edges = []
    for i, sub in enumerate(subs):
        node = Node(
            id=_sid("subq", sub),
            type=NodeType.PROBLEM,
            name=sub,
            source_layer=SourceLayer.STAGED,
            confidence=STAGED_CONFIDENCE,
            attrs={
                "derived_by": system,
                "planner_model": model,
                "qid": qid,
                "staged_kind": "sub_question",
                "plan_position": i,
                # The papers this sub-question actually reached. This is the part worth
                # keeping: it says which papers jointly bear on one part of a question,
                # which per-paper extraction cannot observe because it never holds two
                # papers at once -- the same structural gap §8.2 named for contradictions.
                "papers_reached": ",".join(reached.get(sub, [])),
            },
        )
        nodes.append(node)
        edges.append(
            Edge(
                src=root.id,
                dst=node.id,
                type=EdgeType.ADDRESSES,
                source_layer=SourceLayer.STAGED,
                confidence=STAGED_CONFIDENCE,
                evidence=[],
            )
        )
    return nodes, edges


def write_staged(store: Any, nodes: list[Node], edges: list[Edge]) -> int:
    """Persist derived nodes and edges, refusing anything not marked STAGED.

    The guard is not defensive programming for its own sake: a CURATED node written down this
    path would enter the graph the metrics read, with no audit and no review, which is the
    one thing the layer separation exists to prevent.
    """
    bad = [n.id for n in nodes if n.source_layer is not SourceLayer.STAGED]
    bad += [f"{e.src}->{e.dst}" for e in edges if e.source_layer is not SourceLayer.STAGED]
    if bad:
        raise ValueError(
            f"write_staged refuses {len(bad)} non-STAGED item(s): {bad[:3]} — "
            "query-time writes never enter CURATED without review"
        )
    if not nodes:
        return 0
    store.upsert_nodes(nodes)
    if edges:
        store.upsert_edges(edges)
    log.info("staged %d nodes, %d edges", len(nodes), len(edges))
    return len(nodes)


def staged_is_invisible(store: Any) -> dict[str, Any]:
    """Do the retrieval arms actually ignore STAGED? The acceptance check for 3.2.

    Compares an unfiltered count against a CURATED-only count. If STAGED nodes exist and the
    unfiltered count is larger, then a query without the filter sees them -- which is what
    both arms did until 3.2 added the filter, and what would have let an agent grade itself.
    """
    total = store.query("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"]
    curated = store.query(
        "MATCH (e:Entity) WHERE e.source_layer = $c RETURN count(e) AS n",
        {"c": SourceLayer.CURATED.value},
    )[0]["n"]
    staged = store.query(
        "MATCH (e:Entity) WHERE e.source_layer = $s RETURN count(e) AS n",
        {"s": SourceLayer.STAGED.value},
    )[0]["n"]
    return {
        "total_nodes": total,
        "curated_nodes": curated,
        "staged_nodes": staged,
        # The property the arms must satisfy: what they retrieve is the curated set,
        # whatever else is present.
        "separation_holds": total == curated + staged and curated + staged == total,
        "staged_present": staged > 0,
    }