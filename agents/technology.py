"""Technology Agent: docs/DESIGN.md B-1/B-2(Technology), C-1 유형별 기술 평가 기준, D-1/D-2.

분석과 Evidence만 만들고 탈락 판정은 하지 않는다. 자기 Evidence 필드(tech_evidence)만 반환한다.
"""

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.evidence import format_search_docs, make_evidence, merge_evidence, source_lookup
from agents.state import Evidence, InvestmentState
from tools.web_search import web_search

STAGE_GUIDE = {
    "CHIP": "양산 / 테이프아웃 / FPGA·시험 칩 시제품 / 설계 / 구상 중 하나의 표현을 그대로 사용",
    "DESIGN_AI": "상용 운영 / 고객 설계 PoC 완료 / 프로토타입 / 개발 중 / 구상 중 하나의 표현을 그대로 사용",
    "PROCESS_AI": "상용 운영 / 현장 PoC 완료 / 프로토타입 / 개발 중 / 구상 중 하나의 표현을 그대로 사용",
}
TYPE_GUIDE = {
    "CHIP": "칩 성능(TOPS)·전력효율(TOPS/W)·지연, 제조·검증 단계, SW 생태계. 워크로드·정밀도·배치·전력 조건을 함께 기록",
    "DESIGN_AI": "설계 시간 단축·PPA 개선·제약 충족. 대상 회로·설계 제약·비교 도구·검증 데이터를 함께 기록",
    "PROCESS_AI": "수율·불량률·검사 성능·공정 시간 개선. 장비·라인·기간·제품·데이터 분할을 함께 기록(%와 %p 구분)",
}


class TechMetric(BaseModel):
    metric: Literal["development_stage", "technical_originality", "risk_mitigation"] = Field(
        description="development_stage=개발 단계, technical_originality=독창성·검증된 효과, risk_mitigation=기술 위험과 완화"
    )
    value: str = Field(description="development_stage는 단계 표현 그대로. 그 외는 한 문장 요약")
    unit: str = Field(default="", description="수치가 있으면 단위")
    scope: str = Field(default="", description="검증 조건(워크로드·정밀도·대상 회로·라인 등). 기업 주장만이면 '기업 주장'")
    quote: str = Field(description="검색 결과에서 그대로 발췌한 근거 문장")
    source_url: str = Field(description="검색 결과에 있는 출처 URL")


class TechnologyAnalysis(BaseModel):
    development_stage: str = Field(description="현재 개발 단계와 근거. 확인되지 않으면 '확인되지 않음'")
    architecture_overview: str = Field(description="핵심 기술·아키텍처. 확인되지 않으면 '확인되지 않음'")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    summary: str = Field(description="장단점·개발 단계를 포함한 요약. 기업 주장과 검증된 효과를 구분")
    metrics: list[TechMetric] = Field(default_factory=list)


def _analyze(company: str, company_type: str | None, product: str | None, docs: list[dict[str, Any]]) -> TechnologyAnalysis:
    type_text = TYPE_GUIDE.get(company_type or "", "AI 반도체 관련 기술 지표와 개발 단계")
    stage_text = STAGE_GUIDE.get(company_type or "", "확인되는 개발 단계 표현을 그대로 사용")
    prompt = (
        f"기업명: {company}\n기업 유형: {company_type or '미확정'}\n평가 제품: {product or '미확인'}\n"
        f"기술 평가 핵심: {type_text}\n개발 단계 표현: {stage_text}\n\n"
        "검색 결과만 근거로 기술 특징·개발 단계·장단점·핵심 지표를 분석하세요.\n"
        "- 검색 결과에 없는 사실은 쓰지 않고 '확인되지 않음'으로 적는다.\n"
        "- 조건이 다르거나 기업 주장뿐인 수치는 검증된 우위로 쓰지 않고 scope에 '기업 주장'으로 표시한다.\n"
        "- 테이프아웃은 성능 검증 완료가 아니며, 내부 시연은 고객 PoC가 아니다.\n"
        "- 모든 metric에는 검색 결과의 원문 quote와 source_url이 있어야 한다. 탈락 여부는 판정하지 않는다.\n\n"
        f"검색 결과:\n{format_search_docs(docs)}"
    )
    result = create_llm(temperature=0).with_structured_output(TechnologyAnalysis).invoke(prompt)
    if not isinstance(result, TechnologyAnalysis):
        raise TypeError("unexpected structured output")
    return result


def technology_agent(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    company = startup.get("name", "대상 기업")
    company_type = startup.get("company_type")
    product = startup.get("evaluation_product")

    queries = [f"{company} {product or 'AI 반도체'} 기술 성능 개발 단계 아키텍처"]
    if state.get("evidence_retry", 0) > 0:  # evidence_refresh 재호출: 결측 보완용 추가 질의
        queries.append(f"{company} 벤치마크 검증 결과 고객 PoC 테이프아웃 양산")
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
            analysis = _analyze(company, company_type, product, docs)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = "웹 검색 결과 없음"

    sources = source_lookup(docs)
    new_evidence: list[Evidence] = []
    for metric in analysis.metrics if analysis else []:
        item = make_evidence(
            company=company, category="tech", metric=metric.metric, value=metric.value,
            unit=metric.unit, url=metric.source_url, quote=metric.quote, sources=sources, scope=metric.scope,
        )
        if item:
            new_evidence.append(item)

    if analysis:
        tech_summary: dict[str, Any] = {
            "summary": analysis.summary,
            "development_stage": analysis.development_stage,
            "architecture_overview": analysis.architecture_overview,
            "strengths": analysis.strengths,
            "weaknesses": analysis.weaknesses,
        }
    else:  # 보완 재호출이 실패해도 기존 분석은 유지한다
        tech_summary = state.get("tech_summary") or {"summary": "기술 근거를 확인하지 못했다.", "error": error}
    return {
        "tech_summary": tech_summary,
        "tech_evidence": merge_evidence(list(state.get("tech_evidence", [])), new_evidence),
    }
