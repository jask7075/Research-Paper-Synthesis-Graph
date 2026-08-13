"""Local-inference routing (§3.3 plumbing). Deterministic: no server, no network.

The run 3.3 specifies is deferred -- `Qwen/Qwen2.5-14B-Instruct-AWQ` needs ~8.5GB in 4-bit
and CUDA-only AWQ kernels, against an 8GB M2 whose Metal working set caps near 5.3GB. The
plumbing is landed anyway because it is required whichever hardware eventually serves the
model, and because it touches nothing 3.5 measures.
"""

from __future__ import annotations

import pytest

from rpsg.llm import get_chat_client, get_local_chat_client
from rpsg.llm.openai_client import OpenAIChatClient


def test_a_base_url_forces_the_openai_branch() -> None:
    """A local server's model ids carry no recognisable prefix, so `infer_provider` would
    refuse them. Supplying a base_url has to bypass inference rather than fail on it."""
    client = get_chat_client("Qwen/Qwen2.5-14B-Instruct-AWQ", base_url="http://x:8000/v1")
    assert isinstance(client, OpenAIChatClient)
    assert client.base_url == "http://x:8000/v1"


def test_the_hosted_path_is_unchanged() -> None:
    assert get_chat_client("gpt-5.4-mini").base_url is None


def test_a_local_server_does_not_need_a_real_api_key(monkeypatch) -> None:
    """vLLM, Ollama and llama.cpp authenticate nothing. Requiring a hosted key would make
    §3.3 impossible to run without one."""
    from rpsg.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(type(settings), "openai_api_key", None, raising=False)
    assert OpenAIChatClient("m", base_url="http://x:8000/v1").base_url is not None


def test_an_unset_base_url_raises_rather_than_falling_back(monkeypatch) -> None:
    """A --local run that quietly used the hosted model would report a cost and latency
    delta of zero and read as a success. It has to fail instead."""
    from rpsg.config import get_settings

    monkeypatch.setattr(get_settings().models, "local_inference_base_url", None)
    with pytest.raises(RuntimeError, match="local_inference_base_url is not set"):
        get_local_chat_client()


def test_the_configured_local_model_is_the_one_3_3_specifies() -> None:
    """Pins the acceptance criterion's subject. Substituting a smaller model that fits this
    machine would report a different experiment under §3.3's name."""
    from rpsg.config import get_settings

    assert get_settings().models.local_inference_model == "Qwen/Qwen2.5-14B-Instruct-AWQ"
