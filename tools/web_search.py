# 설계 확정 후 교체할 부분: Tavily 검색 범위와 결과 수는 최종 Agent 설계에 맞춰 조정합니다.

import os
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from agents.common import configure_runtime


@lru_cache(maxsize=1)
def _get_tavily_search() -> TavilySearch:
    configure_runtime()
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is missing; configure it in the project .env file.")
    return TavilySearch(
        max_results=5,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )


@tool("web_search")
def web_search(query: str) -> list[dict[str, Any]]:
    """Search the web with Tavily and return results with citation metadata."""
    if not query.strip():
        raise ValueError("query must not be empty")

    response = _get_tavily_search().invoke({"query": query})
    if not isinstance(response, dict):
        raise TypeError("Tavily returned an unexpected response format")

    normalized_results = []
    for result in response.get("results", []):
        if not isinstance(result, dict):
            continue
        normalized_results.append(
            {
                "title": result.get("title"),
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
    return normalized_results