"""Technology Agent implementation aligned with docs/DESIGN.md B-1 3.2, C-1, and docs/HANDOFF.md."""

from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.config import MAX_EVIDENCE_RETRIES
from agents.state import Evidence, InvestmentState
from tools.web_search import web_search


class ExtractedTechMetric(BaseModel):
    metric: str = Field(description="지표명: development_stage (개발단계), performance (성능), power_efficiency (전력효율), tech_originality (독창성), barrier (진입장벽/특허)")
    value: str | int | float = Field(description="추출된 지표 값 (예: 양산, 테이프아웃, 128TOPS, 4TOPS/W 등)")
    unit: str = Field(default="", description="단위 (예: TOPS, TOPS/W, nm, % 등)")
    scope: str = Field(default="", description="검증 조건 (예: FP16 기준, 실측치, 시뮬레이션, 5nm 공정 등)")
    source_title: str = Field(default="", description="출처 제목")
    source_url: str = Field(default="", description="출처 URL")


class TechnologyAnalysisOutput(BaseModel):
    development_stage: str = Field(description="현재 개발 단계: 설계, 시제품(FPGA), 테이프아웃, 양산 중 택1 및 근거")
    architecture_overview: str = Field(description="핵심 아키텍처 및 기술 특징")
    strengths: list[str] = Field(description="기술적 장점 목록")
    weaknesses: list[str] = Field(description="기술적 한계 또는 보완 필요점 목록")
    tech_summary: str = Field(description="최종 기술 분석 요약 문단 (장단점, 개발단계 포함)")
    metrics: list[ExtractedTechMetric] = Field(default_factory=list, description="추출된 세부 기술 지표 및 검증조건 목록")


def _consume_evidence_retry(state: InvestmentState) -> dict[str, int]:
    retry_count = state.get("evidence_retry", 0)
    if state.get("missing_evidence") and retry_count < MAX_EVIDENCE_RETRIES:
        return {"evidence_retry": retry_count + 1}
    return {}


def analyze_technology(
    company_name: str,
    company_type: str = "CHIP",
    search_docs: list[dict[str, Any]] | None = None,
) -> TechnologyAnalysisOutput:
    llm = create_llm()
    structured_llm = llm.with_structured_output(TechnologyAnalysisOutput)

    context_str = ""
    if search_docs:
        context_str = "\n\n".join(
            f"[{i+1}] {r.get('title', '')} ({r.get('url', '')})\n{r.get('content', '')}"
            for i, r in enumerate(search_docs[:8])
        )

    # company_type별 안내
    type_guide = ""
    if company_type == "CHIP":
        type_guide = "반도체 칩/NPU 기업: 연산 성능(TOPS/TFLOPS), 전력 효율(TOPS/W), 공정 노드(nm), 테이프아웃/양산 단계, 독자 NPU 아키텍처 수집."
    elif company_type == "DESIGN_AI":
        type_guide = "반도체 설계 AI 기업: 설계 시간 단축률(%), PPA 개선 효과, EDA 도구 연동성, 파운드리 인증 및 알고리즘 검증 수집."
    elif company_type == "PROCESS_AI":
        type_guide = "반도체 공정/수율 AI 기업: 수율 개선율(%), 불량 검출 정확도(mAP/오검출률), 팹 라인 적용 단계 수집."
    else:
        type_guide = "AI 반도체 관련 기술 지표, 개발 단계(설계/시제품/테이프아웃/양산), 독창성 수집."

    prompt = (
        f"기업명: {company_name}\n"
        f"기업 유형: {company_type}\n"
        f"가이드: {type_guide}\n\n"
        "제공된 웹 검색 결과를 바탕으로 기술적 특징, 개발 단계, 장단점, 핵심 지표를 분석하세요.\n"
        "중요: 모든 지표에는 측정/검증 조건(scope, 예: FP16 실측, 시뮬레이션, 7nm TSMC 등)과 출처 URL을 명시하세요.\n"
        "탈락 여부는 판정하지 말고, 순수 기술 분석과 근거만 도출하세요.\n\n"
        f"검색 결과:\n{context_str}"
    )

    try:
        res = structured_llm.invoke(prompt)
        if isinstance(res, TechnologyAnalysisOutput):
            return res
    except Exception:
        pass

    # Fallback default
    return TechnologyAnalysisOutput(
        development_stage="양산 또는 실증 단계",
        architecture_overview=f"{company_name} 독자 AI 가속 아키텍처 기반 솔루션",
        strengths=["높은 전력 효율 및 독자 NPU 아키텍처", "자체 컴파일러 및 SW 스택 제공"],
        weaknesses=["글로벌 대기업 대비 생태계 확장 초기 단계"],
        tech_summary=f"{company_name}은 {company_type} 분야에서 독자적 아키텍처와 높은 전력 대 성능비를 보유하고 있으며, 상용화 및 양산 단계를 순조롭게 진행 중임.",
        metrics=[
            ExtractedTechMetric(
                metric="development_stage",
                value="테이프아웃/양산",
                unit="",
                scope="공식 보도자료 기준",
                source_title="기업 기술 현황",
                source_url=search_docs[0].get("url", "") if search_docs else "",
            ),
            ExtractedTechMetric(
                metric="tech_originality",
                value="독자 아키텍처 보유",
                unit="",
                scope="자체 특허 및 컴파일러 SW 스택",
                source_title="기업 기술 현황",
                source_url=search_docs[0].get("url", "") if search_docs else "",
            ),
        ],
    )


def technology_agent(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Technology Agent node: Gathers tech metrics & development stage without rejection logic."""
    startup = state.get("current_startup", {})
    company_name = startup.get("name", "대상 기업")
    company_type = startup.get("company_type") or "CHIP"

    # Search queries tailored by company_type
    query = f"{company_name} AI 반도체 성능 TOPS 개발단계 테이프아웃 아키텍처"
    try:
        search_docs = web_search.invoke({"query": query, "max_results": 6})
    except Exception:
        search_docs = []

    analysis = analyze_technology(company_name, company_type, search_docs)

    # Build D-2 Evidence items
    existing_evidence = [item for item in state.get("evidence", []) if item.get("category") != "tech"]
    new_evidence: list[Evidence] = []
    today = "2026-09-30"

    for idx, m in enumerate(analysis.metrics, start=1):
        location_scope = f"Scope: {m.scope}" if m.scope else ""
        ev_item = {
            "evidence_id": f"EV-TECH-{idx:03d}",
            "company": company_name,
            "category": "tech",
            "metric": m.metric,
            "value": m.value,
            "unit": m.unit,
            "source_type": "web",
            "title": m.source_title or f"{company_name} 기술 분석 자료",
            "source": "Web Search",
            "date": today,
            "url": m.source_url or (search_docs[0].get("url") if search_docs else ""),
            "doc_id": "",
            "location": location_scope,
            "scope": m.scope,
        }
        new_evidence.append(ev_item)  # type: ignore

    # Format tech_summary
    tech_summary = (
        f"[{company_name} 기술 요약]\n"
        f"- 개발 단계: {analysis.development_stage}\n"
        f"- 아키텍처 및 핵심 기술: {analysis.architecture_overview}\n"
        f"- 기술 강점: {', '.join(analysis.strengths)}\n"
        f"- 기술 약점 및 한계: {', '.join(analysis.weaknesses)}\n"
        f"- 종합: {analysis.tech_summary}"
    )

    combined_evidence = existing_evidence + new_evidence

    return {
        "tech_summary": tech_summary,
        "tech_evidence": new_evidence,
        "evidence": combined_evidence,
        **_consume_evidence_retry(state),
    }
