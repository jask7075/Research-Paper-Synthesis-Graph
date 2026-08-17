"""Provider-neutral chat interface.

The system asks a chat model for exactly two things:

    json()  schema-constrained output  — extraction (Tier B/C), the eval judge
    text()  free-form prose            — answer synthesis

Keeping the surface this small is what makes the provider a config change rather
than a rewrite (README design principle 5). Adapters implement it per provider;
`rpsg.llm.get_chat_client` selects one from the model id.

`temperature` is optional and defaults to None, meaning "whatever the provider defaults
to" — which is what every call in Iteration 1 and 2 used. It is a parameter rather than a
global setting because the right value differs by task: a grader should be reproducible,
while forcing a synthesiser to be deterministic is a separate decision with its own
evidence. See `rpsg.eval.judge` for why the judge now pins it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ChatClient(ABC):
    """A chat model behind a two-method contract."""

    model: str

    @abstractmethod
    def json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        schema_name: str = "output",
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> dict:
        """Return a parsed object guaranteed to satisfy `schema`.

        `schema_name` is required by some providers (OpenAI) and ignored by
        others (Anthropic). Implementations raise on refusal or truncation
        rather than returning a partial object.
        """

    @abstractmethod
    def text(
        self, *, system: str, user: str, max_tokens: int = 2048, temperature: float | None = None
    ) -> str:
        """Return the model's prose response."""