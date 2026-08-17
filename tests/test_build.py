"""`build_system` dispatch. Stores are faked, so this needs no index, graph or API key.

These pin the configuration each arm name produces, because two entrypoints now share this
function and a silent change here would alter what `06_run_eval.py` measures — the script
that produced the reported results.
"""

from __future__ import annotations

import pytest

from rpsg.retrieval import build
from rpsg.retrieval.agentic import AgenticSystem
from rpsg.retrieval.baselines import VectorRAGSystem
from rpsg.retrieval.build import CORPUS, SYSTEMS, build_system
from rpsg.retrieval.citation_graph import CitationGraphSystem
from rpsg.retrieval.typed_graph import TypedGraphSystem
from rpsg.stores.base import Embedder


class _Embedder(Embedder):
    dim = 8

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


class _FakeVectorStore:
    """Records that `load()` was called, since a store that is built but never loaded
    silently retrieves nothing."""

    def __init__(self, *_a, **_kw) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True


class _FakeGraphStore:
    def __init__(self, *_a, **_kw) -> None:
        pass


@pytest.fixture(autouse=True)
def _fake_stores(monkeypatch):
    monkeypatch.setattr(build, "FaissVectorStore", _FakeVectorStore)
    monkeypatch.setattr(build, "KuzuGraphStore", _FakeGraphStore)


def _build(name: str, **kw):
    return build_system(name, embedder=_Embedder(), **kw)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("vector_fulltext", VectorRAGSystem),
        ("vector_abstract", VectorRAGSystem),
        ("typed_graph", TypedGraphSystem),
        ("typed_graph_chunks", TypedGraphSystem),
        ("citation_graph", CitationGraphSystem),
        ("citation_graph_seeded", CitationGraphSystem),
        ("agentic", AgenticSystem),
        ("agentic_no_critique", AgenticSystem),
    ],
)
def test_every_arm_builds_the_right_class(name, expected):
    system = _build(name)
    assert isinstance(system, expected)
    assert system.name == name


def test_all_declared_systems_are_buildable():
    """`SYSTEMS` is what both CLIs offer as `--system` choices, so an unbuildable entry
    would be an option that always crashes."""
    for name in SYSTEMS:
        assert _build(name) is not None


def test_unknown_arm_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown system"):
        _build("vector_middletext")


def test_vector_arms_read_the_corpus_they_are_named_for():
    for name, corpus in CORPUS.items():
        assert _build(name)._corpus == corpus


def test_top_k_reaches_the_vector_arm():
    assert _build("vector_fulltext", top_k=7)._top_k == 7


def test_top_k_does_not_override_the_agentic_arms_measured_default():
    """The agentic `top_k` was set to 20 so several retrievals stay within one static
    retrieval's evidence budget. A CLI default of 60 must not silently replace it."""
    default = _build("agentic")._top_k
    assert _build("agentic", top_k=60)._top_k == default


def test_no_critique_suffix_disables_the_critique_and_nothing_else():
    assert _build("agentic")._critique is True
    assert _build("agentic_no_critique")._critique is False


def test_seeded_suffix_switches_the_citation_arm_to_chunk_seeding():
    assert _build("citation_graph")._seed_from == "title"
    assert _build("citation_graph_seeded")._seed_from == "chunks"


def test_only_the_router_variant_of_typed_graph_gets_a_vector_store():
    assert _build("typed_graph")._vector_store is None
    assert _build("typed_graph_chunks")._vector_store is not None


def test_graph_hints_can_be_withheld_from_the_agentic_arm():
    assert _build("agentic", graph_hints=False)._graph_store is None
    assert _build("agentic", graph_hints=True)._graph_store is not None


def test_agentic_flags_default_to_the_measured_configuration():
    """§3.5's scored run: anchor on, staging off. Staging must default off because the
    deliverable must not write to the graph it is measured against."""
    system = _build("agentic")
    assert system._anchor is True
    assert system._stage_writes is False


def test_agentic_flags_are_settable():
    system = _build("agentic", anchor=False, stage_writes=True, max_retrievals=3)
    assert system._anchor is False
    assert system._stage_writes is True
    assert system._max_retrievals == 3


def test_vector_stores_are_loaded_not_merely_constructed():
    for name in ("vector_fulltext", "typed_graph_chunks", "citation_graph", "agentic"):
        system = _build(name)
        store = getattr(system, "_store", None) or getattr(system, "_vector_store", None)
        candidates = [
            s
            for s in (getattr(system, "_store", None), getattr(system, "_vector_store", None))
            if isinstance(s, _FakeVectorStore)
        ]
        assert candidates, f"{name} built no vector store"
        assert all(s.loaded for s in candidates), f"{name} left a vector store unloaded"
        assert store is not None


def test_arms_that_need_no_vector_index_do_not_load_one():
    """`typed_graph` reads the graph only. Loading the index would make a graph-only arm
    fail on a machine that has never built one."""
    system = _build("typed_graph")
    assert not isinstance(getattr(system, "_store", None), _FakeVectorStore)
    assert system._vector_store is None
