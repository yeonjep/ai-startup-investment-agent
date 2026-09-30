"""Startup Agent implementation aligned with docs/DESIGN.md A-3, B-1 3.1, and docs/HANDOFF.md."""

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.config import (
    MAX_CANDIDATES,
    MAX_CANDIDATE_POOL,
    MAX_SELECTION_RETRIES,
)
from agents.evidence import format_search_docs, make_evidence, merge_evidence, source_lookup
from agents.state import Evidence, InvestmentState
from tools.web_search import web_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# --- Fuzzy String Matching Fallback ---
try:
    from rapidfuzz import fuzz

    def _string_similarity(a: str, b: str) -> float:
        return float(fuzz.ratio(a, b))
except ImportError:
    from difflib import SequenceMatcher

    def _string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0


def _normalize_company_name(name: str) -> str:
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", name)
    cleaned = re.sub(r"(주식회사|\(주\)|주\.|Inc\.|Corp\.|Ltd\.|Co\.)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", "", cleaned).strip().lower()
    return cleaned


# --- Pydantic Schemas for Structured Output ---

class DiscoveredStartup(BaseModel):
    name: str = Field(description="공식 기업명 (예: 리벨리온, 퓨리오사AI, 딥엑스)")
    aliases: list[str] = Field(default_factory=list, description="영문명 또는 기타 표기")
    country: str = Field(default="한국", description="본사 소재 국가")
    domain_url: str = Field(default="", description="공식 웹사이트 URL")
    founded_year: str = Field(default="", description="설립연도")
    ceo: str = Field(default="", description="대표자명")
    recent_round: str = Field(default="", description="최근 투자 라운드 (Seed, Series A, Series B 등)")
    total_funding: str = Field(default="", description="누적 투자 유치액")
    description: str = Field(default="", description="기업 및 주력 제품/솔루션 개요")


class DiscoveryResult(BaseModel):
    startups: list[DiscoveredStartup] = Field(default_factory=list)


class CriterionEvaluation(BaseModel):
    criterion_name: str = Field(description="기준명 (domain_relevance, ai_core, unlisted, funding_stage, no_exit, info_sufficiency)")
    status: Literal["PASS", "FAIL", "REVIEW"] = Field(description="판정 결과")
    reason: str = Field(description="판정 근거 문장")
    source_url: str = Field(default="", description="출처 URL")


class SelectionEvaluation(BaseModel):
    company_type: Literal["CHIP", "DESIGN_AI", "PROCESS_AI"] | None = Field(
        default=None,
        description="AI 반도체 유형: CHIP (팹리스/NPU), DESIGN_AI (설계 솔루션), PROCESS_AI (공정/수율 솔루션). 불명확하면 None",
    )
    criteria: list[CriterionEvaluation] = Field(description="6개 기준 판정 목록")


class ProfileMetric(BaseModel):
    metric: Literal[
        "team_experience", "technical_headcount", "funding_total",
        "customer_traction", "revenue_stage", "risk_mitigation",
    ] = Field(description=(
        "team_experience=핵심 인력 반도체 경력(문장), technical_headcount=기술 인력 수(숫자, 전체 직원 수로 대체 금지), "
        "funding_total=누적 투자액(숫자+통화), customer_traction=고객·PoC·계약 수(중복 제외, 숫자), "
        "revenue_stage=매출 단계, risk_mitigation=기술·규제·공급망 위험과 완화 현황(문장)"
    ))
    text: str = Field(default="", description="문장형 지표의 값. revenue_stage는 '반복 매출' / '초기 매출' / '매출 없음' 중 하나")
    number: float | None = Field(default=None, description="숫자형 지표의 값. funding_total은 통화 전체 단위 값(예: 1,200억 원 = 120000000000)")
    unit: str = Field(default="", description="technical_headcount='명', customer_traction='건', funding_total='KRW' 또는 'USD'")
    quote: str = Field(description="검색 결과에서 그대로 발췌한 근거 문장")
    source_url: str = Field(description="검색 결과에 있는 출처 URL")


class ExtractedProfileMetrics(BaseModel):
    metrics: list[ProfileMetric] = Field(default_factory=list, description="근거가 확인된 지표만 포함. 없는 지표는 만들지 않는다")


# --- Step 1 & 2: Candidate Discovery & Normalization ---

def discover_candidates(domain: str = "AI 반도체") -> list[dict[str, Any]]:
    queries = [
        "국내 AI 반도체 팹리스 스타트업 NPU 가속기 투자 유치 비상장",
        "한국 AI 반도체 설계 공정 스타트업 시리즈 투자",
        "Korea AI semiconductor startup NPU fabless funding",
        "AI chip startup raises Series A B NPU accelerator funding",
        "AI chip design EDA semiconductor manufacturing yield AI startup raises funding",
    ]

    all_search_results: list[dict[str, Any]] = []
    for q in queries:
        try:
            results = web_search.invoke({"query": q, "max_results": 5})
            all_search_results.extend(results)
        except Exception:
            continue

    if not all_search_results:
        return []  # 검색 결과 없음: 후보를 만들어 내지 않는다 (D-3)

    llm = create_llm()
    structured_llm = llm.with_structured_output(DiscoveryResult)

    docs_text = "\n\n".join(
        f"[{i+1}] {r.get('title', '')} ({r.get('url', '')})\n{r.get('content', '')}"
        for i, r in enumerate(all_search_results[:12])
    )

    prompt = (
        f"다음 검색 결과를 바탕으로 {domain} 분야의 비상장 스타트업 후보 목록을 추출하세요.\n"
        "상장사(예: 대기업, 코스닥 상장사)는 제외하고, Seed ~ Series C 단계의 비상장 스타트업만 추출하세요.\n\n"
        f"{docs_text}"
    )

    try:
        res = structured_llm.invoke(prompt)
        discovered = res.startups if isinstance(res, DiscoveryResult) else []
    except Exception:
        discovered = []

    # Dedup using rapidfuzz / SequenceMatcher
    unique_candidates: list[dict[str, Any]] = []
    for item in discovered:
        norm_name = _normalize_company_name(item.name)
        if not norm_name:
            continue
        is_dup = False
        for existing in unique_candidates:
            existing_norm = _normalize_company_name(existing["name"])
            if (
                norm_name in existing_norm
                or existing_norm in norm_name
                or _string_similarity(norm_name, existing_norm) >= 80
            ):
                is_dup = True
                break
        if not is_dup:
            candidate_dict = item.model_dump()
            candidate_dict["discovery_done"] = True
            unique_candidates.append(candidate_dict)

    # A-3: 국적에 따른 가점·감점 없음. 최신 공개일 정보가 없으므로 투자 단계가 확인된 후보를 앞에 두고 검색 순서를 유지한다.
    unique_candidates.sort(key=lambda c: 0 if c.get("recent_round") else 1)
    return unique_candidates[:MAX_CANDIDATE_POOL]


# --- Step 4: Selection Criteria Evaluation (A-3) ---

CRITERIA_NAMES = [
    "domain_relevance",   # 1. 도메인 해당 (주력 제품이 AI 반도체/솔루션)
    "ai_core",            # 2. AI 핵심성 (AI가 제품의 핵심 가치)
    "unlisted",           # 3. 비상장 (코스피/코스닥/코넥스 미상장)
    "funding_stage",      # 4. 투자 단계 (최근 라운드 Seed~Series C)
    "no_exit",            # 5. Exit 미완료 (피인수·합병·상장 이력 없음)
    "info_sufficiency",   # 6. 정보 충분성 (공개 자료 홈페이지 + 기사 2건 이상)
]


def _evaluate_candidate_criteria(
    company_name: str,
    search_context: list[dict[str, Any]],
    discovery_hint: dict[str, Any] | None = None,
) -> SelectionEvaluation:
    llm = create_llm()
    structured_llm = llm.with_structured_output(SelectionEvaluation)

    context_str = "\n\n".join(
        f"[{i+1}] {r.get('title', '')} ({r.get('url', '')})\n{r.get('content', '')}"
        for i, r in enumerate(search_context[:8])
    )

    prompt = (
        f"대상 기업: {company_name}\n"
        f"발굴 단계에서 확인된 정보(검색 자료와 충돌하면 REVIEW 또는 FAIL 근거로 삼는다): "
        f"최근 라운드={(discovery_hint or {}).get('recent_round') or '미확인'}, "
        f"누적 투자={(discovery_hint or {}).get('total_funding') or '미확인'}\n"
        "다음 검색 자료를 바탕으로 DESIGN.md A-3의 6개 스타트업 선정 기준을 평가하고, company_type을 분류하세요.\n\n"
        "기준 1: domain_relevance (도메인 해당) - 주력 제품이 AI 반도체(NPU, AI가속기, AI설계/공정솔루션)이면 PASS, 무관하면 FAIL, 일부 한정이면 REVIEW\n"
        "기준 2: ai_core (AI 핵심성) - AI가 제품 핵심가치면 PASS, 마케팅 문구면 FAIL, 불명확하면 REVIEW\n"
        "기준 3: unlisted (비상장) - 코스피/코스닥/코넥스 미상장이면 PASS, 상장완료 FAIL, 상장심사/IPO주관사 공식발표 REVIEW\n"
        "기준 4: funding_stage (투자 단계) - 최근 라운드 Seed~Series C면 PASS, Series D 이상/Pre-IPO FAIL, 비공개 REVIEW\n"
        "기준 5: no_exit (Exit 미완료) - 피인수/합병/상장 이력 없으면 PASS, 인수/소멸 FAIL, 합병/매각진행중 REVIEW\n"
        "기준 6: info_sufficiency (정보 충분성) - 홈페이지+기사 2건 이상이면 PASS, 거의 없으면 FAIL, 1건 수준 REVIEW\n\n"
        "company_type 분류: CHIP (NPU/가속기 칩 팹리스), DESIGN_AI (설계 자동화 AI), PROCESS_AI (제조/수율/검사 AI). 불명확하면 None.\n\n"
        f"검색 컨텍스트:\n{context_str}"
    )

    try:
        res = structured_llm.invoke(prompt)
        if isinstance(res, SelectionEvaluation):
            return res
    except Exception:
        pass

    # 평가 실패: PASS를 만들어 내지 않고 모든 기준을 REVIEW(확인 불가)로 둔다
    return SelectionEvaluation(
        company_type=None,
        criteria=[
            CriterionEvaluation(criterion_name=name, status="REVIEW", reason="선정 기준 평가 호출 실패", source_url="")
            for name in CRITERIA_NAMES
        ],
    )


def determine_selection_status(criteria_list: list[CriterionEvaluation]) -> tuple[str, bool]:
    """코드 판정 규칙:

    - FAIL이 1개 이상이면 FAIL
    - 모든 기준이 PASS이면 PASS
    - FAIL 없이 REVIEW가 있으면 REVIEW
    """
    statuses = [c.status for c in criteria_list]
    if "FAIL" in statuses:
        return "FAIL", False
    if all(s == "PASS" for s in statuses):
        return "PASS", False
    return "REVIEW", False


# --- Step 5: Profile Evidence Collection for PASS Candidate ---

def collect_profile_evidence(company_name: str, search_docs: list[dict[str, Any]]) -> list[Evidence]:
    """D-2 Evidence로 팀·인력·투자·고객·매출 근거를 수집한다. 실제 검색 결과와 연결되지 않으면 버린다."""
    if not search_docs:
        return []
    prompt = (
        f"기업명: {company_name}\n"
        "검색 결과만 근거로 창업자·핵심 인력 경력, 기술 인력 수, 누적 투자액, 고객·PoC·계약 수, 매출 단계, 주요 위험을 추출하세요.\n"
        "- 검색 결과에 없는 지표는 만들지 않는다. 협력 발표를 유료 계약으로 해석하지 않고, 같은 고객·프로젝트는 한 번만 센다.\n"
        "- 모든 지표에는 검색 결과의 원문 quote와 source_url이 있어야 한다.\n\n"
        f"검색 결과:\n{format_search_docs(search_docs)}"
    )
    try:
        result = create_llm(temperature=0).with_structured_output(ExtractedProfileMetrics).invoke(prompt)
        metrics = result.metrics if isinstance(result, ExtractedProfileMetrics) else []
    except Exception:  # noqa: BLE001 - 추출 실패는 근거 없음으로 기록 (D-3)
        metrics = []

    sources = source_lookup(search_docs)
    evidence: list[Evidence] = []
    for metric in metrics:
        numeric = metric.metric in {"technical_headcount", "funding_total", "customer_traction"}
        value: Any = metric.number if numeric else metric.text
        item = make_evidence(
            company=company_name, category="profile", metric=metric.metric, value=value,
            unit=metric.unit, url=metric.source_url, quote=metric.quote, sources=sources,
        )
        if item:
            evidence.append(item)
    return merge_evidence([], evidence)


# --- Main Startup Agent Node ---

def startup_agent(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Startup Agent node: docs/DESIGN.md A-3, D-3 (discovery_done, selection_retry 규칙)."""
    domain = state.get("domain", "AI 반도체")
    candidates = list(state.get("candidates", []))
    discovery_done = state.get("discovery_done", False)

    # 후보 수집은 discovery_done=False일 때만 (D-3)
    if not discovery_done:
        configured_candidates = (config or {}).get("configurable", {}).get("demo_candidates")
        candidates = list(configured_candidates) if configured_candidates else discover_candidates(domain=domain)
        discovery_done = True

    current_idx = state.get("current_idx", 0)
    if not candidates or current_idx >= len(candidates):
        return {
            "candidates": candidates,
            "discovery_done": discovery_done,
            "current_startup": None,
            "selection_status": None,
        }

    candidate = dict(candidates[current_idx])
    company_name = candidate.get("name", f"후보_{current_idx + 1}")
    selection_retry = state.get("selection_retry", 0)

    # REVIEW로 돌아온 재호출이면 추가 웹검색 1회 (selection_retry는 첫 REVIEW 반환 시 이미 1로 증가)
    is_retry = state.get("selection_status") == "REVIEW" and selection_retry >= 1
    queries = [f"{company_name} AI 반도체 투자 설립 대표 기술"]
    if is_retry:
        queries.append(f"{company_name} 반도체 투자 비상장 상장 인수 매출 대표 실적 funding IPO acquisition")

    search_docs: list[dict[str, Any]] = []
    for query in queries:
        try:
            search_docs.extend(web_search.invoke({"query": query, "max_results": 5}))
        except Exception:  # noqa: BLE001 - 검색 오류는 근거 없음으로 기록 (D-3)
            continue

    eval_result = _evaluate_candidate_criteria(company_name, search_docs, candidate)
    candidate["company_type"] = eval_result.company_type
    candidate["criteria_results"] = [c.model_dump() for c in eval_result.criteria]
    candidate.setdefault("evaluation_product", candidate.get("description"))

    final_status, _ = determine_selection_status(eval_result.criteria)
    uncertain = False

    scenario = (config or {}).get("configurable", {}).get("scenario")
    if scenario == "zero_pass":
        final_status = "FAIL"

    if final_status == "REVIEW" and not is_retry:
        candidate["selection_status"] = "REVIEW"
        candidates[current_idx] = candidate
        return {
            "candidates": candidates,
            "discovery_done": discovery_done,
            "current_startup": None,
            "selection_status": "REVIEW",
            "selection_retry": 1,  # 검색 직전이 아니라 재검색 요청 시점에 증가 (그래프가 startup_agent로 복귀)
            "uncertain": False,
        }

    if final_status == "REVIEW":  # 재검색 후에도 REVIEW: 불확실 기준을 남기고 분석 진행, 최종 판정은 보류
        uncertain = True
    if final_status != "FAIL" and eval_result.company_type is None:
        uncertain = True  # 기업 유형 미확정 (C-2: 유형 의존 지표는 결측)

    candidate["selection_status"] = final_status
    candidate["uncertain"] = uncertain
    if uncertain:
        candidate["uncertain_criteria"] = [
            c.criterion_name for c in eval_result.criteria if c.status == "REVIEW"
        ] or ["company_type"]

    if final_status == "FAIL":
        failed = [c for c in eval_result.criteria if c.status == "FAIL"]
        candidate["failed_criterion"] = failed[0].criterion_name if failed else None
        candidate["exclusion_reason"] = failed[0].reason if failed else None
        candidate["sources"] = [
            {"title": d.get("title"), "url": d.get("url"), "date": d.get("date")} for d in search_docs[:1]
        ]
        current_startup = None
    else:
        candidate["profile_evidence"] = collect_profile_evidence(company_name, search_docs)
        current_startup = candidate

    candidates[current_idx] = candidate

    # Save candidates to outputs/candidates.json
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (OUTPUTS_DIR / "candidates.json").write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "candidates": candidates,
        "discovery_done": discovery_done,
        "current_startup": current_startup,
        "selection_status": final_status,
        "selection_retry": selection_retry,
        "uncertain": uncertain,
    }


def run_startup_evaluation():
    """Run discovery and selection evaluation for all candidates and print markdown table."""
    print("=== [Startup Agent] AI 반도체 스타트업 발굴 및 A-3 기준 평가 시작 ===")
    candidates = discover_candidates("AI 반도체")
    print(f"총 {len(candidates)}개 후보 수집 완료.")

    eval_results = []
    state: InvestmentState = {
        "domain": "AI 반도체",
        "candidates": candidates,
        "discovery_done": True,
        "current_idx": 0,
    }

    for idx in range(len(candidates)):
        state["current_idx"] = idx
        result = startup_agent(state)
        state["candidates"] = result["candidates"]
        cand = result["candidates"][idx]
        eval_results.append(cand)

    print("\n| 후보 기업 | 국가 | 유형 | 1.도메인 | 2.AI핵심 | 3.비상장 | 4.투자단계 | 5.Exit미완료 | 6.정보충분성 | 최종 판정 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in eval_results:
        c_type = c.get("company_type") or "None"
        cr_map = {item["criterion_name"]: item["status"] for item in c.get("criteria_results", [])}
        print(
            f"| **{c.get('name')}** | {c.get('country', '한국')} | {c_type} | "
            f"{cr_map.get('domain_relevance', '-')} | {cr_map.get('ai_core', '-')} | "
            f"{cr_map.get('unlisted', '-')} | {cr_map.get('funding_stage', '-')} | "
            f"{cr_map.get('no_exit', '-')} | {cr_map.get('info_sufficiency', '-')} | "
            f"**{c.get('selection_status', '-')}** |"
        )
    print(f"\n평가 결과가 {OUTPUTS_DIR / 'candidates.json'}에 저장되었습니다.")


if __name__ == "__main__":
    run_startup_evaluation()
