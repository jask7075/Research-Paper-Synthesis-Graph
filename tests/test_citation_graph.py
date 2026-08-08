"""The citation-graph ablation.

What is tested here is not "does traversal work" but the things that would silently
invalidate the comparison: traversing the wrong edges, reaching more evidence than the arm
it is being compared against, or walking in one direction only.
"""

from __future__ import annotations

from rpsg.retrieval.citation_graph import CitationGraphSystem, _bare


class FakeStore:
    """Records the Cypher it is asked for; returns canned neighbours."""

    def __init__(self, neighbours: dict[str, list[str]] | None = None) -> None:
        self.neighbours = neighbours or {}
        self.queries: list[str] = []

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        self.queries.append(cypher)
        if "e.type = 'Paper'" in cypher and "RETURN e.id AS id, e.name AS name" in cypher:
            return [{"id": f"paper:{i}", "name": n} for i, n in enumerate(("alpha", "beta"))]
        out, seen = [], set()
        for src in (params or {}).get("ids", []):
            for dst in self.neighbours.get(src, []):
                if dst not in seen:
                    seen.add(dst)
                    out.append({"id": dst})
        return out


def _sys(store, **kw) -> CitationGraphSystem:
    return CitationGraphSystem("citation_graph", embedder=None, store=store,
                               vector_store=None, **kw)


def test_paper_ids_lose_their_prefix_for_chunk_lookup():
    """Chunks are keyed by bare 40-hex; a `paper:` prefix would match nothing and the arm
    would silently retrieve no evidence at all."""
    assert _bare("paper:abc123") == "abc123"
    assert _bare("abc123") == "abc123"


def test_only_cites_edges_are_traversed():
    """The whole point of the ablation is that no extracted edge is used. If `addresses`
    or `builds_on` leaked in, this would measure a hybrid and answer nothing."""
    store = FakeStore()
    _sys(store)._expand(["paper:a"])
    walk = [q for q in store.queries if "MATCH (a:Entity)-[r:REL]-" in q]
    assert walk and all("r.type = 'cites'" in q for q in walk)
    assert all("b.type = 'Paper'" in q for q in walk)


def test_the_walk_is_undirected():
    """A paper's references and the papers citing it are both useful neighbours. A
    directed walk would answer only half the relational questions."""
    store = FakeStore()
    _sys(store)._expand(["paper:a"])
    walk = next(q for q in store.queries if "MATCH (a:Entity)-[r:REL]-" in q)
    assert "-[r:REL]-(b" in walk and "-[r:REL]->(b" not in walk


def test_hops_bound_the_walk():
    chain = {"paper:a": ["paper:b"], "paper:b": ["paper:c"], "paper:c": ["paper:d"]}
    assert set(_sys(FakeStore(chain), hops=1)._expand(["paper:a"])) == {"paper:a", "paper:b"}
    assert set(_sys(FakeStore(chain), hops=2)._expand(["paper:a"])) == {
        "paper:a", "paper:b", "paper:c"}


def test_the_cap_counts_papers_not_entities():
    """The typed arm's 150 caps *entities*, which route to ~10 papers. Copying 150 here
    would cap *papers*, giving this arm ~5x the evidence and letting it win on volume
    rather than structure."""
    store = FakeStore({"paper:a": [f"paper:{i}" for i in range(50)]})
    assert len(_sys(store, max_nodes=30)._expand(["paper:a"])) == 30


def test_seeds_are_included_in_what_is_reached():
    """A seed paper is evidence in its own right, not merely a starting point."""
    assert "paper:a" in _sys(FakeStore())._expand(["paper:a"])


def test_a_paper_reached_twice_is_only_counted_once():
    store = FakeStore({"paper:a": ["paper:b", "paper:c"], "paper:b": ["paper:c"]})
    reached = _sys(store, hops=2)._expand(["paper:a"])
    assert sorted(reached) == ["paper:a", "paper:b", "paper:c"]


def test_an_isolated_seed_yields_only_itself():
    """163 of 354 papers are cited by nothing in-corpus; the arm must degrade to the seed
    rather than fail."""
    assert _sys(FakeStore())._expand(["paper:lonely"]) == ["paper:lonely"]