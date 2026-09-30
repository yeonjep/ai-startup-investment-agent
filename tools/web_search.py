import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from agents.common import configure_runtime


@lru_cache(maxsize=10)
def _get_tavily_search(max_results: int = 5) -> TavilySearch:
    configure_runtime()
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is missing; configure it in the project .env file.")
    return TavilySearch(
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )


@tool("web_search")
def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web with Tavily and return results with citation metadata."""
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")
    if not query.strip():
        raise ValueError("query must not be empty")

    response = _get_tavily_search(max_results).invoke({"query": query})
    if not isinstance(response, dict):
        raise TypeError("Tavily returned an unexpected response format")

    if response.get("error"):
        raise RuntimeError("Tavily returned a provider error")

    normalized_results = []
    for result in response.get("results", []):
        if not isinstance(result, dict):
            continue
        normalized_results.append(
            {
                "title": result.get("title"),
                "source": urlparse(result.get("url") or "").hostname or "",
                "published_date": result.get("published_date") or result.get("date"),
                "url": result.get("url"),
                "date": (
                    result.get("published_date")
                    or result.get("date")
                    or result.get("publishedDate")
                ),
                "content": result.get("content"),
                "score": result.get("score"),
            }
        )
    return normalized_results[:max_results]
