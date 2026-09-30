# 설계 확정 후 교체할 부분: Tavily 검색 범위와 결과 수는 최종 Agent 설계에 맞춰 조정합니다.

import os
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from agents.common import configure_runtime


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
    if not query.strip():
        raise ValueError("query must not be empty")

    response = _get_tavily_search(max_results=max_results).invoke({"query": query})
    if not isinstance(response, dict):
        raise TypeError("Tavily returned an unexpected response format")

    normalized_results = []
    for result in response.get("results", []):
        if not isinstance(result, dict):
            continue
        published = (
            result.get("published_date")
            or result.get("date")
            or result.get("publishedDate")
            or ""
        )
        normalized_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "published_date": published,
                "retrieved_date": "2026-09-30",
                "score": result.get("score"),
            }
        )
    return normalized_results


class MetricItem(BaseModel):
    metric: str = Field(description="지표명 (예: TAM, CAGR, TOPS, 개발단계, 투자액, 인력수 등)")
    value: str | int | float = Field(description="추출된 수치 또는 상태값")
    unit: str = Field(default="", description="단위 (예: %, 억원, TOPS/W 등)")
    source_url: str = Field(default="", description="출처 URL 또는 문서 식별자")


class SummarizeSourcesOutput(BaseModel):
    summary: str = Field(description="주제에 대한 핵심 분석 및 요약 내용")
    metrics: list[MetricItem] = Field(default_factory=list, description="문서에서 추출된 정량/정성 지표 목록")
    sources: list[str] = Field(default_factory=list, description="인용된 출처 URL 또는 문서명 목록")


def summarize_sources(topic: str, documents: list[dict[str, Any]]) -> SummarizeSourcesOutput:
    """Summarize search results or documents on a topic and extract structured metrics."""
    if not documents:
        return SummarizeSourcesOutput(
            summary=f"'{topic}'에 대한 분석 가능한 문서가 없습니다.",
            metrics=[],
            sources=[],
        )

    from agents.common import create_llm

    llm = create_llm()
    structured_llm = llm.with_structured_output(SummarizeSourcesOutput)

    doc_contexts = []
    for idx, doc in enumerate(documents, start=1):
        title = doc.get("title", f"Doc {idx}")
        url = doc.get("url") or doc.get("doc_id", "")
        content = doc.get("content", "")
        doc_contexts.append(f"[{idx}] {title} ({url})\n{content}\n")

    prompt = (
        f"당신은 AI 반도체 스타트업 투자 평가 분석가입니다.\n"
        f"주제: {topic}\n\n"
        f"제공된 다음 문서들을 바탕으로 핵심 내용을 요약하고 관련 지표(metrics)와 출처(sources)를 추출하세요.\n\n"
        f"--- 문서 목록 ---\n"
        + "\n".join(doc_contexts)
    )

    try:
        result = structured_llm.invoke(prompt)
        if isinstance(result, SummarizeSourcesOutput):
            return result
        elif isinstance(result, dict):
            return SummarizeSourcesOutput(**result)
    except Exception as e:
        pass

    # Fallback if structured output fails
    return SummarizeSourcesOutput(
        summary=f"{topic} 요약 생성 중 오류 발생 또는 간이 요약",
        metrics=[],
        sources=[d.get("url", "") for d in documents if d.get("url")],
    )