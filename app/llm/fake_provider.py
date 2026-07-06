"""Deterministic, offline LLM + embedding providers.

No network, no API key, fully reproducible — ideal for local dev, CI, and the
eval harness. The chat provider dispatches on the `LLM-TASK:` tag (see
`base.extract_task`) so the graph's SQL / synthesis / critique nodes all get
sensible, deterministic outputs.

These are *stand-ins*, not real models. They demonstrate the plumbing and let
the whole system run for free; swap `LLM_PROVIDER=openai` for real quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Iterable, Iterator, Sequence

from app.llm.base import ChatMessage, EmbeddingProvider, LLMProvider, extract_task
from app.tools.nl_sql import heuristic_nl_to_sql

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbeddingProvider(EmbeddingProvider):
    """Hashing bag-of-words embeddings.

    Deterministic and dependency-free, yet carries a real lexical-similarity
    signal: identical text -> identical vector, and overlapping vocabulary ->
    higher cosine similarity. Good enough to make hybrid retrieval demonstrably
    work offline.
    """

    name = "fake"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Avoid a zero vector (undefined cosine distance).
            vec[0] = 1.0
            norm = 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def _numbered_context(user_content: str) -> list[tuple[int, str]]:
    """Parse `[n] ...text...` context blocks out of a synthesis prompt."""
    entries: list[tuple[int, str]] = []
    for match in re.finditer(r"\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", user_content, re.DOTALL):
        entries.append((int(match.group(1)), " ".join(match.group(2).split())))
    return entries


def _trim(text: str, limit: int = 220) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


class FakeLLMProvider(LLMProvider):
    """Deterministic chat provider that dispatches on the LLM-TASK tag."""

    name = "fake"

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> str:
        task = extract_task(messages)
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if task == "sql":
            question = self._extract_field(user, "Question") or user
            return heuristic_nl_to_sql(question)

        if task == "critique":
            # A pass-through verdict. The real model actually checks the draft
            # against the context; the offline stub trusts the draft.
            return json.dumps(
                {"supported": True, "issues": [], "revised_answer": None}
            )

        if task == "classify":
            return self._classify(user)

        # Default: grounded synthesis with citations.
        return self._synthesize(user)

    def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> Iterator[str]:
        text = self.chat(messages, temperature=temperature)
        # Emit whitespace-preserving tokens so the client reassembles verbatim.
        for token in re.findall(r"\S+\s*", text):
            yield token

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _extract_field(text: str, label: str) -> str | None:
        m = re.search(rf"{label}:\s*(.+)", text)
        return m.group(1).strip() if m else None

    def _synthesize(self, user: str) -> str:
        question = self._extract_field(user, "Question")
        entries = _numbered_context(user)
        if not entries:
            base = "I don't have retrieved context to answer that precisely."
            return f"{base} (offline fake provider)" if not question else base
        top = entries[:2]
        parts = [f"{_trim(text)} [{n}]" for n, text in top]
        lead = "Based on the retrieved sources, "
        return lead + " ".join(parts)

    @staticmethod
    def _classify(user: str) -> str:
        q = user.lower()
        if any(w in q for w in ("create a ticket", "open a ticket", "file a ticket", "raise a ticket")):
            return "action"
        if any(w in q for w in ("how many", "revenue", "orders", "sql", "average", "sum ")):
            return "sql"
        return "qa"
