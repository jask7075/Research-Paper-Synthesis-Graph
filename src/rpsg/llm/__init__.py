"""Chat-model provider selection — the one place the provider branch lives.

Routing is inferred from the model id, so switching providers means editing
`models:` in configs/settings.yaml and nothing else:

    extraction_model: gpt-5.4-nano      -> OpenAI
    extraction_model: claude-haiku-4-5  -> Anthropic

Set `models.provider` in settings.yaml to override the inference (e.g. pointing
an OpenAI-compatible gateway at a model id this wouldn't recognise).
"""

from __future__ import annotations

from rpsg.config import get_settings
from rpsg.llm.base import ChatClient

#: Model-id prefixes that route to OpenAI. Extend when you add a family.
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")
_ANTHROPIC_PREFIXES = ("claude-",)


def infer_provider(model: str) -> str:
    """Map a model id to a provider name."""
    if model.startswith(_ANTHROPIC_PREFIXES):
        return "anthropic"
    if model.startswith(_OPENAI_PREFIXES):
        return "openai"
    raise ValueError(
        f"Cannot infer a provider for model {model!r}. "
        "Set models.provider in configs/settings.yaml, or add the prefix to "
        "rpsg.llm._OPENAI_PREFIXES / _ANTHROPIC_PREFIXES."
    )


def get_chat_client(
    model: str, provider: str | None = None, base_url: str | None = None
) -> ChatClient:
    """Build the adapter for `model`. This is the provider if/else.

    `base_url` points the OpenAI adapter at any OpenAI-compatible server -- vLLM, Ollama,
    llama.cpp. Supplying one also forces the OpenAI branch, because a local server's model
    ids ("Qwen/Qwen2.5-14B-Instruct-AWQ") carry no recognisable prefix and `infer_provider`
    would refuse them.
    """
    if base_url:
        from rpsg.llm.openai_client import OpenAIChatClient

        return OpenAIChatClient(model, base_url=base_url)
    provider = (provider or get_settings().models.provider or infer_provider(model)).lower()

    if provider == "openai":
        from rpsg.llm.openai_client import OpenAIChatClient

        return OpenAIChatClient(model)
    if provider == "anthropic":
        from rpsg.llm.anthropic_client import AnthropicChatClient

        return AnthropicChatClient(model)
    raise ValueError(f"Unknown provider {provider!r} (expected 'openai' or 'anthropic').")


def get_local_chat_client() -> ChatClient:
    """The configured local model, on the configured local server (§3.3).

    Raises rather than silently falling back to the hosted provider: a `--local` run that
    quietly used the hosted model would report a cost and latency delta of zero and look
    like a success.
    """
    settings = get_settings()
    base_url = settings.models.local_inference_base_url
    if not base_url:
        raise RuntimeError(
            "models.local_inference_base_url is not set. Start an OpenAI-compatible server "
            "(e.g. `vllm serve " + settings.models.local_inference_model + "`) and set the "
            "URL in configs/settings.yaml, or drop --local."
        )
    return get_chat_client(settings.models.local_inference_model, base_url=base_url)


__all__ = [
    "ChatClient",
    "get_chat_client",
    "get_local_chat_client",
    "infer_provider",
]