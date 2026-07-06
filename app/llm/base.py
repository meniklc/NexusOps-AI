"""Provider-agnostic LLM + embedding interfaces.

Everything downstream (graph nodes, tools, eval) depends only on these
abstractions, never on OpenAI directly — so the provider is swappable via the
`LLM_PROVIDER` setting. This module has **no third-party dependencies** so it
stays importable and unit-testable offline.

A tiny convention lets the offline Fake provider behave sensibly: every system
prompt begins with a machine-readable tag line::

    LLM-TASK: <name>

Real models simply treat it as one more instruction line; the Fake provider
uses it to dispatch to a deterministic responder. See `app/graph/prompts.py`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


def extract_task(messages: Sequence[ChatMessage]) -> str | None:
    """Return the `LLM-TASK: <name>` tag from the first system message, if any."""
    for msg in messages:
        if msg.role == "system":
            first_line = msg.content.lstrip().splitlines()[0] if msg.content.strip() else ""
            if first_line.upper().startswith("LLM-TASK:"):
                return first_line.split(":", 1)[1].strip().lower()
            return None
    return None


class LLMProvider(abc.ABC):
    """A chat-completion provider."""

    name: str = "base"

    @abc.abstractmethod
    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> str:
        """Return the full completion for a list of messages."""

    @abc.abstractmethod
    def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> Iterator[str]:
        """Yield the completion incrementally as text deltas."""

    # Convenience helpers ---------------------------------------------------
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        return self.chat(
            [ChatMessage("system", system), ChatMessage("user", user)],
            temperature=temperature,
        )


class EmbeddingProvider(abc.ABC):
    """A text-embedding provider."""

    name: str = "base"
    dim: int = 1536

    @abc.abstractmethod
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
