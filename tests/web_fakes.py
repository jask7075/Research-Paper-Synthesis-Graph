"""Fake arms for the web UI's tests, shared by `test_web` and `test_web_http`.

Not a test module. It lives here rather than in a `conftest.py` because the two web test
files are the only consumers, and a `conftest` fixture would be offered to all twenty-odd
other test modules that have no use for it.

The vector fake subclasses the real `VectorRAGSystem` on purpose: `AskService` decides
whether an arm has a pre-synthesis chunk list with `isinstance`, so a duck-typed stub would
take the wrong branch and the tests would pass while the service was wrong.
"""

from __future__ import annotations

import pytest

from rpsg.llm.usage import USAGE
from rpsg.retrieval.baselines import SystemOutput, VectorRAGSystem
from rpsg.stores.base import Chunk, Embedder, SearchHit
from rpsg.web import service as svc_mod
from rpsg.web.service import AskService


class FakeEmbedder(Embedder):
    dim = 8

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


def hit(paper_id: str, score: float = 0.5) -> SearchHit:
    return SearchHit(
        chunk=Chunk(
            id=f"{paper_id}:c0",
            paper_id=paper_id,
            # Deliberately ragged whitespace and long: the service collapses and truncates.
            text="  Barren   plateaus  flatten the   landscape. " * 40,
            section_title="Results",
            section_type="results",
            char_start=0,
            char_end=10,
        ),
        score=score,
    )


class FakeStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    def search(self, *_a, **_kw) -> list[SearchHit]:
        return self._hits


class FakeVectorArm(VectorRAGSystem):
    """A real `VectorRAGSystem` with only the LLM call replaced."""

    def __init__(self, name: str = "vector_fulltext", hits: list[SearchHit] | None = None) -> None:
        super().__init__(
            name=name,
            embedder=FakeEmbedder(),
            store=FakeStore(hits if hits is not None else [hit("p1"), hit("p2", 0.4)]),
            corpus="fulltext",
        )
        self.answered = 0

    def answer(self, query: str) -> SystemOutput:
        self.answered += 1
        USAGE.record("fake-model", input_tokens=1000, output_tokens=200)
        return SystemOutput(
            text="Overparameterization helps [paper:p1].",
            cited_paper_ids=["paper:p1"],
            evidence="[P1] (results) barren plateaus …",
        )


class FakeGraphArm:
    """Satisfies the `System` protocol without being a `VectorRAGSystem`, which is how the
    service tells "has a pre-synthesis chunk list" from "does not"."""

    name = "typed_graph"

    def answer(self, query: str) -> SystemOutput:
        return SystemOutput(text="traversed", cited_paper_ids=[], evidence="", trace={})


def make_service(monkeypatch: pytest.MonkeyPatch) -> AskService:
    """An `AskService` whose arms are fakes and whose stores are always present.

    The built arm is attached as `.arm` so a test can count synthesis calls or replace
    `answer` to simulate a provider failure.
    """
    arm = FakeVectorArm()
    graph_arm = FakeGraphArm()

    def _build(name, **_kw):
        return graph_arm if name.startswith("typed_graph") else arm

    monkeypatch.setattr(svc_mod, "build_system", _build)
    monkeypatch.setattr(svc_mod, "missing_store", lambda _name: None)
    service = AskService(hash_embed=True)
    service.arm = arm  # type: ignore[attr-defined]
    return service
