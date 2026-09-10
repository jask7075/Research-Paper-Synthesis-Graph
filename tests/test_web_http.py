"""The FastAPI layer: status codes, and what does and does not reach the browser.

Separate from `test_web.py` because the skip below is module-wide. Folding these into that
file would skip the service tests too on any machine without the optional `web` extra — CI
included — and a suite that silently tests nothing is worse than one that fails.

Behaviour belongs in `test_web.py`. What is asserted here is only the translation: a
refusal becomes a 409 carrying its message, an unexpected failure becomes a 500 carrying
none of it.
"""

from __future__ import annotations

import pytest

from tests.web_fakes import make_service

pytest.importorskip("fastapi", reason="the web extra is optional; CI installs dev only")


@pytest.fixture
def service(monkeypatch):
    return make_service(monkeypatch)


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    from rpsg.web.app import create_app

    return TestClient(create_app(service))


def test_arms_endpoint_feeds_the_picker(client):
    body = client.get("/api/arms").json()
    assert body["default"] == "vector_fulltext"
    assert {a["name"] for a in body["arms"]} >= {"vector_fulltext", "typed_graph"}
    assert body["synthesis_model"]


def test_ask_endpoint_returns_the_answer_and_the_retrieval(client):
    body = client.post("/api/ask", json={"question": "why?"}).json()
    assert body["cited_paper_ids"] == ["paper:p1"]
    assert len(body["hits"]) == 2
    assert body["usage"][0]["model"] == "fake-model"


def test_a_refusal_is_a_409_carrying_the_message_verbatim(client):
    res = client.post("/api/ask", json={"question": "q", "system": "nope"})
    assert res.status_code == 409
    assert "unknown arm" in res.json()["detail"]


def test_an_unexpected_failure_does_not_leak_the_exception_text(client, service, monkeypatch):
    """A provider SDK's exception can carry request internals. The log gets it; the
    browser gets the type and nothing else."""

    def _boom(_query):
        raise ValueError("sk-secret-in-the-message")

    monkeypatch.setattr(service.arm, "answer", _boom)
    res = client.post("/api/ask", json={"question": "q"})
    assert res.status_code == 500
    assert "sk-secret" not in res.json()["detail"]
    assert "ValueError" in res.json()["detail"]


def test_a_blank_question_is_rejected_by_the_schema(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_the_page_is_served_at_the_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Ask the corpus" in res.text
