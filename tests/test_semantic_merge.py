"""Semantic merging, where the model decides and the code must not over-reach.

The rules under test are about restraint: only entity types, only within a type, errors
resolve to `different`, and merges compose without depending on verdict order.
"""

from __future__ import annotations

import json
from pathlib import Path

from rpsg.extraction.semantic_merge import (
    ENTITY_TYPES,
    Verdict,
    adjudicate,
    merge_map,
)


def _v(a: str, b: str, same: bool) -> Verdict:
    return Verdict(a, b, a, b, 0.99, same, "")


# --- what may be merged at all -----------------------------------------------------

def test_propositions_are_out_of_scope():
    """Merging two Claims asserts they are the same statement — a stronger and different
    claim than "these name the same method"."""
    assert "Claim" not in ENTITY_TYPES
    assert "Limitation" not in ENTITY_TYPES
    assert {"Method", "Problem", "Dataset", "Software", "Hardware"} == set(ENTITY_TYPES)


# --- map construction ---------------------------------------------------------------

def test_a_rejected_pair_merges_nothing():
    assert merge_map([_v("m:a", "m:b", same=False)]) == {}


def test_an_accepted_pair_points_at_the_smaller_id():
    assert merge_map([_v("m:b", "m:a", same=True)]) == {"m:b": "m:a"}


def test_merges_compose_across_pairs():
    """A~B and B~C accepted means all three are one entity; leaving C on B would make the
    result depend on which pair was seen first."""
    m = merge_map([_v("m:b", "m:c", True), _v("m:a", "m:b", True)])
    assert m["m:b"] == "m:a" and m["m:c"] == "m:a"


def test_the_result_does_not_depend_on_verdict_order():
    pairs = [_v("m:a", "m:b", True), _v("m:b", "m:c", True), _v("m:c", "m:d", True)]
    assert merge_map(pairs) == merge_map(list(reversed(pairs)))


def test_no_merge_chains_survive():
    m = merge_map([_v("m:a", "m:b", True), _v("m:b", "m:c", True)])
    assert all(dst not in m for dst in m.values())


def test_rejections_do_not_leak_into_an_accepted_component():
    m = merge_map([_v("m:a", "m:b", True), _v("m:b", "m:z", False)])
    assert "m:z" not in m


# --- adjudication -------------------------------------------------------------------

class _StubClient:
    """Returns a scripted verdict; counts calls so caching can be checked."""

    def __init__(self, same: bool = True, boom: bool = False) -> None:
        self.same, self.boom, self.calls = same, boom, 0

    def json(self, **_: object) -> dict:
        self.calls += 1
        if self.boom:
            raise RuntimeError("upstream exploded")
        return {"same": self.same, "reason": "stub"}


def test_a_failed_call_resolves_to_different(monkeypatch):
    """Failing closed matters: a transport error must never become a merge."""
    stub = _StubClient(boom=True)
    monkeypatch.setattr("rpsg.extraction.semantic_merge.get_chat_client", lambda _m: stub)
    (v,) = adjudicate([("m:a", "m:b", 0.99)], {"m:a": "A", "m:b": "B"}, model="x")
    assert v.same is False
    assert merge_map([v]) == {}


def test_verdicts_are_cached_by_name_pair(tmp_path: Path, monkeypatch):
    stub = _StubClient(same=True)
    monkeypatch.setattr("rpsg.extraction.semantic_merge.get_chat_client", lambda _m: stub)
    cache = tmp_path / "verdicts.json"
    names = {"m:a": "A", "m:b": "B"}
    adjudicate([("m:a", "m:b", 0.99)], names, model="x", cache_path=cache)
    adjudicate([("m:a", "m:b", 0.99)], names, model="x", cache_path=cache)
    assert stub.calls == 1, "the second run must be served from cache"
    assert json.loads(cache.read_text())


def test_the_cache_is_symmetric_in_the_pair(tmp_path: Path, monkeypatch):
    """(A, B) and (B, A) are the same question and must not be billed twice."""
    stub = _StubClient(same=True)
    monkeypatch.setattr("rpsg.extraction.semantic_merge.get_chat_client", lambda _m: stub)
    cache = tmp_path / "verdicts.json"
    adjudicate([("m:a", "m:b", 0.99)], {"m:a": "A", "m:b": "B"}, model="x", cache_path=cache)
    adjudicate([("m:b", "m:a", 0.99)], {"m:a": "A", "m:b": "B"}, model="x", cache_path=cache)
    assert stub.calls == 1


def test_a_verdict_carries_both_names_for_review(monkeypatch):
    stub = _StubClient(same=True)
    monkeypatch.setattr("rpsg.extraction.semantic_merge.get_chat_client", lambda _m: stub)
    (v,) = adjudicate([("m:a", "m:b", 0.97)], {"m:a": "VQE", "m:b": "vqe"}, model="x")
    assert (v.a_name, v.b_name, v.similarity) == ("VQE", "vqe", 0.97)