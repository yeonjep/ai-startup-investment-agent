"""Competitor Agent: docs/DESIGN.md B-1/B-2(Competitor), C-1, D-1/D-2.

동일 유형·고객 문제의 국내외 경쟁사를 비교하고 Evidence만 만든다(탈락 판정 없음).
자기 Evidence 필드(competitor_evidence)만 반환한다.
"""

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.evidence import format_search_docs, make_evidence, merge_evidence, source_lookup
from agents.state import Evidence, InvestmentState
from tools.web_search import web_search

BASIS = {
    "CHIP": "동일 워크로드·정밀도·배치·전력 조건의 칩과 SW 이식·지원 수준",
    "DESIGN_AI": "동일 회로·제약에서 기존 EDA·수동 설계 대비 개선 효과, 도구 연동·재현성",
    "PROCESS_AI": "동일 장비·제품·라인·기간에서 기존 공정 제어·검사 대비 효과, 현장 안정성",
}
BARRIER = {
    "CHIP": "칩 IP·특허 / 자체 컴파일러·SDK·제품 스택 / 파운드리·고객 등 핵심 파트너십",
    "DESIGN_AI": "설계·알고리즘 IP / 사용 권리가 확인된 설계 데이터·모델·제품 기술 / EDA·PDK·고객 워크플로 연동·협력",
    "PROCESS_AI": "공정·검사 IP / 사용 권리가 확인된 현장 데이터·모델·운영 기술 / 장비·생산 시스템 연동·고객 협력",
}

KNOWN_COMPETITOR_COUNTRIES = {
    "퓨리오사ai": "대한민국",
    "furiosaai": "대한민국",
    "furiosa": "대한민국",
    "리벨리온": "대한민국",
    "rebellions": "대한민국",
}


def _normalized_competitor_country(name: str, country: str) -> str:
    key = "".join(character.lower() for character in name if character.isalnum())
    return KNOWN_COMPETITOR_COUNTRIES.get(key, country)


class CompetitorProfile(BaseModel):
    name: str = Field(description="경쟁 기업 또는 기존(비AI) 방식 이름")
    country: str = Field(default="", description="소재 국가. 확인되지 않으면 빈 문자열")
    product: str = Field(description="경쟁 제품 또는 기존 방식")
    axis1_performance: str = Field(description="성능·개선 효과. 비교 조건이 다르면 명시")
    axis2_ecosystem: str = Field(description="기존 도구·현장 연동, SW 생태계")
    axis3_commercialization: str = Field(description="사업화 단계")
    comparison_summary: str = Field(description="대상 기업 대비 강점·열위")
    quote: str = Field(description="검색 결과에서 그대로 발췌한 근거 문장")
    source_url: str = Field(description="검색 결과에 있는 출처 URL")


class BarrierItem(BaseModel):
    category: str = Field(description="진입장벽 범주(유형별 3개 범주 중 하나)")
    status: Literal["confirmed", "absent_confirmed"] = Field(description="근거로 확인된 존재/부재만 기재. 자료가 없으면 항목을 만들지 않는다")
    quote: str
    source_url: str


class CompetitorAnalysisOutput(BaseModel):
    peer_group: list[str] = Field(description="선정한 경쟁 3~5개 이름")
    competitors: list[CompetitorProfile]
    differentiation_summary: str = Field(description="차별성 요약. 주장만 있으면 그렇게 명시")
    differentiation_quote: str = Field(default="", description="차별성 근거 원문 발췌")
    differentiation_source_url: str = Field(default="")
    barriers_to_entry: str = Field(description="진입장벽 요약. 확인되지 않은 범주는 '확인되지 않음'")
    barrier_items: list[BarrierItem] = Field(default_factory=list)
    overall_analysis: str


def _analyze(company: str, company_type: str | None, product: str | None, tech_summary: dict[str, Any], docs: list[dict[str, Any]]) -> CompetitorAnalysisOutput:
    prompt = (
        f"대상 기업: {company}\n유형: {company_type or '미확정'}\n평가 제품: {product or '미확인'}\n"
        f"기술 요약: {tech_summary.get('summary', '')}\n"
        f"비교 기준: {BASIS.get(company_type or '', '동일 고객 문제를 푸는 제품·방법')}\n"
        f"진입장벽 범주: {BARRIER.get(company_type or '', 'IP / 제품·데이터 / 연동·파트너십')}\n\n"
        "같은 유형·고객 문제를 다루는 국내외 경쟁 3~5개(기존 비AI 방식 포함)를 검색 결과에서 골라 비교하세요.\n"
        "- 검색 결과에 없는 사실은 쓰지 않는다. 자료가 없는 진입장벽 범주는 '없음 확인'과 구분해 항목을 만들지 않는다.\n"
        "- 조건이 다른 수치를 우위 근거로 쓰지 않고, 기업 주장만 있으면 그렇게 명시한다.\n"
        "- 모든 비교에는 검색 결과의 원문 quote와 source_url이 있어야 한다. 탈락 판정은 하지 않는다.\n\n"
        f"검색 결과:\n{format_search_docs(docs)}"
    )
    result = create_llm(temperature=0).with_structured_output(CompetitorAnalysisOutput).invoke(prompt)
    if not isinstance(result, CompetitorAnalysisOutput):
        raise TypeError("unexpected structured output")
    return result


def competitor_agent(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    company = startup.get("name", "대상 기업")
    company_type = startup.get("company_type")
    product = startup.get("evaluation_product")
    tech_summary = state.get("tech_summary") or {}

    queries = [
        f"{company} {product or 'AI 반도체'} 경쟁사 비교 차별성",
        f"{company} competitors comparison alternative benchmark",
    ]
    if state.get("evidence_retry", 0) > 0:  # evidence_refresh 재호출: 결측 보완용 추가 질의
        queries.append(f"{company} patent SDK partnership 특허 파트너십")
    docs: list[dict[str, Any]] = []
    for query in queries:
        try:
            docs.extend(web_search.invoke({"query": query, "max_results": 6}))
        except Exception:  # noqa: BLE001 - 검색 오류는 근거 없음으로 기록 (D-3)
            continue

    error = None
    analysis = None
    if docs:
        try:
            analysis = _analyze(company, company_type, product, tech_summary, docs)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = "웹 검색 결과 없음"

    sources = source_lookup(docs)
    new_evidence: list[Evidence] = []

    def add(metric: str, value: str, url: str, quote: str, scope: str | None = None) -> None:
        item = make_evidence(company=company, category="competitor", metric=metric, value=value,
                             unit=None, url=url, quote=quote, sources=sources, scope=scope)
        if item:
            new_evidence.append(item)

    if analysis:
        add("competitive_difference", analysis.differentiation_summary,
            analysis.differentiation_source_url, analysis.differentiation_quote)
        for peer in analysis.competitors:
            add("competitive_difference", f"{peer.name}({peer.product}): {peer.comparison_summary}",
                peer.source_url, peer.quote, scope=peer.axis1_performance)
        for barrier in analysis.barrier_items:
            add("entry_barrier", f"{barrier.category}: {'확인' if barrier.status == 'confirmed' else '없음 확인'}",
                barrier.source_url, barrier.quote)
        competitor_analysis: dict[str, Any] = {
            "summary": analysis.overall_analysis,
            "peer_group": analysis.peer_group,
            "competitors": [
                {
                    **peer.model_dump(),
                    "country": _normalized_competitor_country(peer.name, peer.country),
                    "comparison": peer.comparison_summary,
                }
                for peer in analysis.competitors
            ],
            "differentiation_summary": analysis.differentiation_summary,
            "barriers_to_entry": analysis.barriers_to_entry,
        }
    else:  # 보완 재호출이 실패해도 기존 분석은 유지한다
        competitor_analysis = state.get("competitor_analysis") or {
            "summary": "경쟁 근거를 확인하지 못했다.", "competitors": [], "error": error}
    return {
        "competitor_analysis": competitor_analysis,
        "competitor_evidence": merge_evidence(list(state.get("competitor_evidence", [])), new_evidence),
    }
