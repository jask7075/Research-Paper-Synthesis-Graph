"""Anthropic adapter.

Kept alongside the OpenAI one so the provider swap stays a config change and the
portability claim in the README is demonstrable rather than aspirational. Only
constructed if a `claude-*` model id is configured, so the `anthropic` package is
imported lazily and stays an optional install.
"""

from __future__ import annotations

import json

from rpsg.config import get_settings
from rpsg.llm.base import ChatClient
from rpsg.llm.usage import USAGE
from rpsg.logging import get_logger

log = get_logger(__name__)


class AnthropicChatClient(ChatClient):
    def __init__(self, model: str) -> None:
        import anthropic

        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model

    def _first_text(self, resp) -> str:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            USAGE.record(
                self.model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
        return next((b.text for b in resp.content if b.type == "text"), "")

    def json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str = "output",  # noqa: ARG002 - Anthropic infers the name
        max_tokens: int = 4096,
    ) -> dict:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return json.loads(self._first_text(resp) or "{}")

    def text(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return self._first_text(resp)