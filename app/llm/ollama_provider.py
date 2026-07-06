"""Ollama-backed LLM + embedding providers.

Ollama exposes an OpenAI-compatible REST API under `/v1`, so we reuse the same
`openai` client library — only the `base_url` differs, and no real API key is
required. Imported lazily by the factory so the app never needs a running Ollama
server unless a provider is set to ``ollama``.

Run a local Ollama first, e.g.::

    ollama pull llama3
    ollama pull nomic-embed-text

Inside Docker the host machine's Ollama is reachable at
``http://host.docker.internal:11434/v1`` (see OLLAMA_BASE_URL).
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from openai import OpenAI

from app.llm.base import ChatMessage, EmbeddingProvider, LLMProvider

# Ollama ignores the key, but the OpenAI client refuses an empty one, so we pass
# a harmless placeholder.
_DUMMY_KEY = "ollama"


class OllamaLLMProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=_DUMMY_KEY)
        self._model = model

    def _payload(self, messages: Sequence[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=self._payload(messages),
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def stream(
        self, messages: Sequence[ChatMessage], *, temperature: float = 0.0
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._payload(messages),
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaEmbeddingProvider(EmbeddingProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, dim: int) -> None:
        self._client = OpenAI(base_url=base_url, api_key=_DUMMY_KEY)
        self._model = model
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        resp = self._client.embeddings.create(model=self._model, input=items)
        return [d.embedding for d in resp.data]
