"""OpenAI adapter.

Structured output uses `response_format={"type": "json_schema", ..., "strict": True}`,
which guarantees a schema-valid object — but strict mode constrains the schema
itself: every object needs `additionalProperties: false`, every property must
appear in `required`, and free-form objects (`{"type": "object"}` with no
declared properties) are rejected. `rpsg.extraction.schema` is written to satisfy
those rules; see the note there about `attrs` crossing the wire as a JSON string.

Token budgets use `max_completion_tokens`. On reasoning-capable models that
budget covers reasoning *and* the visible answer, so a too-small value truncates
the JSON — we surface that as an explicit error instead of a confusing
JSONDecodeError.
"""

from __future__ import annotations

import json

from rpsg.config import get_settings
from rpsg.llm.base import ChatClient
from rpsg.logging import get_logger

log = get_logger(__name__)


class OpenAIChatClient(ChatClient):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example).")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.model = model

    def _complete(self, system: str, user: str, max_tokens: int, response_format=None):
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Newer OpenAI models reject `max_tokens`; if you swap in a 4o-class
            # model the API error will name the parameter it wants instead.
            "max_completion_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise RuntimeError(f"{self.model} refused the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"{self.model} hit the {max_tokens}-token completion budget before finishing. "
                "Raise max_tokens (on reasoning models this budget also covers reasoning)."
            )
        return choice.message.content or ""

    def json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str = "output",
        max_tokens: int = 4096,
    ) -> dict:
        text = self._complete(
            system,
            user,
            max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        return json.loads(text)

    def text(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        return self._complete(system, user, max_tokens)