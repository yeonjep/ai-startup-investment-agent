# Evidence 생성·병합 공통 헬퍼 (docs/DESIGN.md D-1 근거 보존 규칙, D-2 Evidence 스키마).

import hashlib
import re
from datetime import date
from typing import Any

from agents.state import Evidence


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "-", text).strip("-").lower() or "unknown"


def _host(url: str | None) -> str | None:
    match = re.match(r"https?://(?:www\.)?([^/]+)", url or "")
    return match.group(1) if match else None


def source_lookup(search_docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """URL -> 검색 결과. LLM이 반환한 URL이 실제 검색 결과에 있는지 검증하는 데 쓴다."""
    return {doc["url"]: doc for doc in search_docs if doc.get("url")}


def make_evidence(
    *,
    company: str,
    category: str,
    metric: str,
    value: Any,
    unit: str | None,
    url: str,
    quote: str,
    sources: dict[str, dict[str, Any]],
    scope: str | None = None,
) -> Evidence | None:
    """검색 결과에 실제로 있는 URL과 원문 근거(quote)가 있을 때만 Evidence를 만든다."""
    source = sources.get(url)
    if source is None or not str(quote or "").strip() or value in (None, ""):
        return None
    digest = hashlib.sha1(f"{url}|{metric}|{value}".encode()).hexdigest()[:8]
    return {
        "evidence_id": f"{_slug(company)}:{category}:{metric}:{digest}",
        "company": company,
        "category": category,  # type: ignore[typeddict-item]
        "metric": metric,
        "value": value,
        "unit": unit or None,
        "source_type": "web",
        "title": source.get("title"),
        "source": _host(url),
        "date": source.get("date"),
        "accessed_at": source.get("accessed_at") or date.today().isoformat(),
        "url": url,
        "doc_id": None,
        "page": None,
        "chunk_id": None,
        "quote": str(quote).strip(),
        "scope": scope or None,
    }


def merge_evidence(existing: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    """evidence_id 기준 병합. 병합된 전체 목록을 반환한다."""
    merged = {item["evidence_id"]: item for item in existing}
    for item in new:
        merged[item["evidence_id"]] = item
    return list(merged.values())


def format_search_docs(search_docs: list[dict[str, Any]], limit: int = 8) -> str:
    return "\n\n".join(
        f"[{i + 1}] {doc.get('title', '')} ({doc.get('url', '')}) 공개일: {doc.get('date') or '미확인'}\n"
        f"{doc.get('content', '')}"
        for i, doc in enumerate(search_docs[:limit])
    )
