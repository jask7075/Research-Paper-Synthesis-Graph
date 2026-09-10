"""`AskService` — everything the web UI decides, with no web framework involved.

No index, no graph, no API key: the arms are fakes from `web_fakes`. FastAPI is not
imported here on purpose, so these run in CI, which installs the dev extras only. The HTTP
translation is tested separately in `test_web_http.py`, which skips without it.

The refusals get most of the attention. A UI that answers a question the user asked *not*
to pay for, or that silently substitutes its own `top_k` for a swept one, is wrong in a way
that produces a plausible-looking answer instead of an error.
"""

from __future__ import annotations

import pytest

from rpsg.llm.usage import USAGE
from rpsg.web import service as svc_mod
from rpsg.web.service import AskService, ServiceError
from tests.web_fakes import FakeVectorArm, make_service


@pytest.fixture
def service(monkeypatch):
    return make_service(monkeypatch)


# --- refusals ---------------------------------------------------------------------


def test_a_blank_question_is_refused_before_anything_is_built(service):
    with pytest.raises(ServiceError, match="Ask a question"):
        service.ask("   ")


def test_unknown_arm_names_the_alternatives(service):
    with pytest.raises(ServiceError, match="unknown arm"):
        service.ask("q", system="vector_middletext")


def test_retrieval_only_is_refused_on_arms_that_plan_before_they_retrieve(service):
    """Refused, not downgraded. Answering in full would spend money the user declined to
    spend by ticking the box."""
    with pytest.raises(ServiceError, match="retrieval-only is not available"):
        service.ask("q", system="typed_graph", retrieval_only=True)


def test_top_k_is_refused_rather_than_ignored_on_the_graph_arms(service):
    """`ask.py` warns and continues; a form that shows a filled-in box has to say the
    number will not be used, or the user reads the answer as having used it."""
    with pytest.raises(ServiceError, match="vector arms only"):
        service.ask("q", system="typed_graph", top_k=5)


def test_a_missing_store_surfaces_the_message_that_says_how_to_build_it(service, monkeypatch):
    monkeypatch.setattr(svc_mod, "missing_store", lambda _n: "no vector index at data/x.faiss")
    with pytest.raises(ServiceError, match="no vector index"):
        service.ask("q")


def test_a_missing_api_key_reaches_the_user_as_a_refusal(service, monkeypatch):
    """The adapters raise `RuntimeError` for an unset key. That is configuration the user
    can fix, so it must not be reported as an internal failure."""

    def _boom(_query):
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example).")

    monkeypatch.setattr(service.arm, "answer", _boom)
    with pytest.raises(ServiceError, match="OPENAI_API_KEY"):
        service.ask("q")


def test_an_empty_retrieval_is_reported_instead_of_being_synthesized(monkeypatch):
    arm = FakeVectorArm(hits=[])
    monkeypatch.setattr(svc_mod, "build_system", lambda name, **_kw: arm)
    monkeypatch.setattr(svc_mod, "missing_store", lambda _n: None)
    with pytest.raises(ServiceError, match="Nothing retrieved"):
        AskService(hash_embed=True).ask("q")
    assert arm.answered == 0, "synthesized on no evidence"


# --- answering --------------------------------------------------------------------


def test_a_vector_answer_carries_the_retrieval_it_was_grounded_on(service):
    result = service.ask("what mitigates barren plateaus?")
    assert result.answer is not None
    assert result.cited_paper_ids == ["paper:p1"]
    assert [h.paper_id for h in result.hits] == ["p1", "p2"]
    assert result.retrieval_only is False
    assert result.elapsed_s >= 0.0


def test_excerpts_are_whitespace_collapsed_and_capped(service):
    excerpt = service.ask("q").hits[0].excerpt
    assert "  " not in excerpt
    assert len(excerpt) <= svc_mod.EXCERPT_CHARS


def test_retrieval_only_shows_the_chunks_and_makes_no_llm_call(service):
    result = service.ask("q", retrieval_only=True)
    assert result.hits and result.answer is None
    assert result.usage == []
    assert service.arm.answered == 0


def test_evidence_is_withheld_unless_asked_for(service):
    assert service.ask("q").evidence is None
    assert service.ask("q", show_evidence=True).evidence is not None


def test_graph_arms_answer_without_a_pre_synthesis_chunk_list(service):
    result = service.ask("q", system="typed_graph")
    assert result.answer == "traversed"
    assert result.hits == []


# --- accounting -------------------------------------------------------------------


def test_usage_is_the_delta_for_this_question_not_the_process_total(service):
    """`USAGE` accumulates for the life of the server, so a second question must not
    report the first one's tokens as well."""
    service.ask("q")
    second = service.ask("q")
    row = next(r for r in second.usage if r.model == "fake-model")
    assert (row.calls, row.input_tokens, row.output_tokens) == (1, 1000, 200)


def test_an_unpriced_model_reports_no_cost_rather_than_zero(service):
    row = next(r for r in service.ask("q").usage if r.model == "fake-model")
    assert row.cost_usd is None


def test_models_untouched_by_this_question_are_left_out(service):
    USAGE.record("some-other-model", input_tokens=5, output_tokens=5)
    assert [r.model for r in service.ask("q").usage] == ["fake-model"]


# --- arm cache --------------------------------------------------------------------


def test_arms_are_built_once_and_reused(service, monkeypatch):
    built = []
    monkeypatch.setattr(
        svc_mod, "build_system", lambda name, **_kw: built.append(name) or FakeVectorArm()
    )
    service.ask("q")
    service.ask("q")
    assert built == ["vector_fulltext"]


def test_a_different_top_k_builds_a_different_arm(service, monkeypatch):
    """Reusing the cached arm would answer the second question at the first one's
    breadth while the page reports the number the user typed."""
    built = []
    monkeypatch.setattr(
        svc_mod, "build_system", lambda name, **kw: built.append(kw["top_k"]) or FakeVectorArm()
    )
    service.ask("q", top_k=5)
    service.ask("q", top_k=9)
    assert built == [5, 9]


def test_the_ui_never_asks_for_staged_writes(service, monkeypatch):
    """The browser is read-only: a mistyped question must not persist STAGED nodes into
    the graph the eval measures."""
    seen = {}
    monkeypatch.setattr(
        svc_mod, "build_system", lambda name, **kw: seen.update(kw) or FakeVectorArm()
    )
    service.ask("q")
    assert seen["stage_writes"] is False


def test_arms_reports_every_declared_system_with_its_capabilities(service):
    arms = {a.name: a for a in service.arms()}
    assert arms["vector_fulltext"].supports_retrieval_only is True
    assert arms["typed_graph"].supports_retrieval_only is False
    assert arms["typed_graph"].needs_graph is True
    assert all(a.ready for a in arms.values())


def test_an_arm_whose_store_is_missing_is_reported_unready_with_the_reason(service, monkeypatch):
    monkeypatch.setattr(
        svc_mod,
        "missing_store",
        lambda n: "no graph at data/processed/rpsg.kuzu" if n == "typed_graph" else None,
    )
    arms = {a.name: a for a in service.arms()}
    assert arms["typed_graph"].ready is False
    assert "no graph" in arms["typed_graph"].blocked_on
    assert arms["vector_fulltext"].ready is True
