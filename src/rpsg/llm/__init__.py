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


def get_chat_client(model: str, provider: str | None = None) -> ChatClient:
    """Build the adapter for `model`. This is the provider if/else."""
    provider = (provider or get_settings().models.provider or infer_provider(model)).lower()

    if provider == "openai":
        from rpsg.llm.openai_client import OpenAIChatClient

        return OpenAIChatClient(model)
    if provider == "anthropic":
        from rpsg.llm.anthropic_client import AnthropicChatClient

        return AnthropicChatClient(model)
    raise ValueError(f"Unknown provider {provider!r} (expected 'openai' or 'anthropic').")


__all__ = ["ChatClient", "get_chat_client", "infer_provider"]