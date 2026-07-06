"""Web search tool.

Uses Tavily when `TAVILY_API_KEY` is set; otherwise returns a deterministic
offline stub so the tool is always callable. Stdlib-only at import time so it
stays offline-testable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def as_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _stub_results(query: str, k: int) -> list[SearchResult]:
    """Deterministic placeholder results (no network)."""
    return [
        SearchResult(
            title=f"[stub] Result {i + 1} for: {query}",
            url=f"https://example.com/search?q={query.replace(' ', '+')}&r={i + 1}",
            snippet=(
                f"Offline stub snippet {i + 1}. Set TAVILY_API_KEY in .env to enable "
                f"real web search for '{query}'."
            ),
        )
        for i in range(k)
    ]


def _tavily_results(query: str, k: int, api_key: str) -> list[SearchResult]:
    from tavily import TavilyClient  # lazy import

    client = TavilyClient(api_key=api_key)
    resp = client.search(query=query, max_results=k)
    return [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        )
        for item in resp.get("results", [])
    ]


def web_search(query: str, k: int = 3) -> list[SearchResult]:
    # Import settings lazily so the stub path works without pydantic installed.
    try:
        from app.config import settings

        api_key = settings.tavily_api_key
    except Exception:
        api_key = ""

    if api_key:
        try:
            return _tavily_results(query, k, api_key)
        except Exception:
            # Fail soft to the stub rather than breaking the graph.
            return _stub_results(query, k)
    return _stub_results(query, k)
