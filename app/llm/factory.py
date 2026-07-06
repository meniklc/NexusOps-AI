"""Provider factory: resolves the configured LLM + embedding providers.

    from app.llm.factory import get_llm, get_embedder

The rest of the codebase depends only on the `LLMProvider` / `EmbeddingProvider`
interfaces, so switching providers is a one-line `.env` change.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.fake_provider import FakeEmbeddingProvider, FakeLLMProvider


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(settings.openai_api_key, settings.openai_chat_model)
    if settings.llm_provider == "ollama":
        from app.llm.ollama_provider import OllamaLLMProvider

        return OllamaLLMProvider(settings.ollama_base_url, settings.ollama_llm_model)
    return FakeLLMProvider()


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    provider = settings.resolved_embedding_provider
    if provider == "openai":
        from app.llm.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            settings.openai_api_key,
            settings.openai_embedding_model,
            settings.embedding_dim,
        )
    if provider == "ollama":
        from app.llm.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(
            settings.ollama_base_url,
            settings.ollama_embedding_model,
            settings.ollama_embedding_dim,
        )
    return FakeEmbeddingProvider(dim=settings.embedding_dim)
