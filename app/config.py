"""Centralised configuration.

All settings come from environment variables (loaded from a gitignored `.env`
via pydantic-settings). Import the singleton `settings` everywhere else:

    from app.config import settings
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider ---
    llm_provider: Literal["openai", "fake", "ollama"] = "fake"
    # Embedding provider is independent so you can mix (e.g. OpenAI chat +
    # Ollama embeddings). "inherit" (default) reuses whatever LLM_PROVIDER is.
    embedding_provider: Literal["inherit", "openai", "fake", "ollama"] = "inherit"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Ollama (OpenAI-compatible endpoint) ---
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    ollama_llm_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"
    # nomic-embed-text emits 768-dim vectors (differs from OpenAI's 1536).
    # The pgvector column in db/init.sql must match the ACTIVE embedder's dim.
    ollama_embedding_dim: int = 768

    # --- Database ---
    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "ragdb"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- Retrieval tuning ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    rrf_k: int = 60

    # --- Web search ---
    tavily_api_key: str = ""

    # --- LangSmith / observability ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "enterprise-rag-agent"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    backend_url: str = "http://localhost:8000"

    @property
    def resolved_embedding_provider(self) -> str:
        """Effective embedding provider, resolving the "inherit" sentinel."""
        if self.embedding_provider == "inherit":
            return self.llm_provider
        return self.embedding_provider

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy/psycopg connection URL."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def psycopg_dsn(self) -> str:
        """Plain libpq DSN for direct psycopg connections."""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    def apply_langsmith_env(self) -> None:
        """Export LangSmith env vars so LangChain/LangGraph auto-trace runs.

        Call once at process startup. When tracing is disabled this is a no-op.
        """
        import os

        os.environ["LANGCHAIN_TRACING_V2"] = "true" if self.langchain_tracing_v2 else "false"
        if self.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
