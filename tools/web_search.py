# 설계 확정 후 교체할 부분: Tavily 검색 범위와 결과 수는 최종 Agent 설계에 맞춰 조정합니다.

import os
import re
from datetime import date
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


_DATE_PATTERNS = (
    (r"(?<!\d)(20\d{2})[-./](\d{1,2})[-./](\d{1,2})(?!\d)", (1, 2, 3)),
    (r"(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", (1, 2, 3)),
    (r"/(20\d{2})/(\d{2})/(\d{2})(?:/|$|\?)", (1, 2, 3)),
    (r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", (1, 2, 3)),
    (r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{4,}", (1, 2, 3)),  # 기사 ID 접두 YYYYMMDDhhmm...
)


def _infer_date(url: str, title: str, content: str) -> str | None:
    """Tavily가 공개일을 주지 않을 때 URL·제목·본문에 명시된 날짜를 찾는다. 찾지 못하면 None."""
    for text in (url or "", title or "", (content or "")[:600]):
        for pattern, groups in _DATE_PATTERNS:
            match = re.search(pattern, text)
            if not match:
                continue
            year, month, day = (int(match.group(g)) for g in groups)
            try:
                found = date(year, month, day)
            except ValueError:
                continue
            if found <= date.today():
                return found.isoformat()
    return None


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
            or None
        )
        normalized_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "date": (
                    str(published)[:10]
                    if published
                    else _infer_date(result.get("url", ""), result.get("title", ""), result.get("content", ""))
                ),
                "accessed_at": date.today().isoformat(),
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