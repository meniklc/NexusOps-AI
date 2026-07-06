"""OpenAI-backed LLM + embedding providers.

Imported lazily by the factory so the app never requires the `openai` package
(or an API key) when running with `LLM_PROVIDER=fake`.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from openai import OpenAI

from app.llm.base import ChatMessage, EmbeddingProvider, LLMProvider


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Set it in .env or use LLM_PROVIDER=fake."
            )
        self._client = OpenAI(api_key=api_key)
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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Set it in .env or use LLM_PROVIDER=fake."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        resp = self._client.embeddings.create(model=self._model, input=items)
        return [d.embedding for d in resp.data]
