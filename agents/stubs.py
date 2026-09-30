# DESIGN.md D-2 nodes; internal agent logic remains deterministic demo stubs.

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.config import (
    MAX_CANDIDATES,
    MAX_CANDIDATE_POOL,
    MAX_EVIDENCE_RETRIES,
    MAX_RAG_RETRIES,
    MAX_SELECTION_RETRIES,
)
from agents.state import InvestmentState


def _scenario(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("scenario", "all_hold")


def _consume_evidence_retry(state: InvestmentState) -> dict[str, int]:
    retry_count = state.get("evidence_retry", 0)
    if state.get("missing_evidence") and retry_count < MAX_EVIDENCE_RETRIES:
        return {"evidence_retry": retry_count + 1}
    return {}


def initialize_state(state: InvestmentState) -> dict[str, Any]:
    return {
        "domain": state.get("domain", "AI 반도체"),
        "max_candidates": MAX_CANDIDATES,
        "candidates": [],
        "current_idx": 0,
        "current_startup": {},
        "selection_status": "",
        "selection_retry": 0,
        "uncertain": False,
        "tech_summary": "",
        "market_analysis": "",
        "rag_retry": 0,
        "competitor_analysis": "",
        "evidence": [],
        "evidence_retry": 0,
        "scores": {},
        "total_score": 0.0,
        "missing_evidence": [],
        "decision": "",
        "evaluated": [],
        "final_report": "",
    }


def startup_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, Any]:
    scenario = _scenario(config)
    candidates = list(state.get("candidates", []))
    if not candidates:
        configured = config.get("configurable", {}).get("demo_candidates")
        candidates = list(configured or [{"name": "[샘플 후보 A]"}, {"name": "[샘플 후보 B]"}])
    candidates = candidates[:MAX_CANDIDATE_POOL]

    current_idx = state.get("current_idx", 0)
    if current_idx >= len(candidates):
        return {
            "candidates": candidates,
            "current_startup": {},
            "selection_status": "FAIL",
            "selection_retry": 0,
            "uncertain": False,
        }

    candidate = candidates[current_idx]
    selection_status = (
        "FAIL" if scenario == "zero_pass" else candidate.get("selection_status", "PASS")
    )
    selection_retry = state.get("selection_retry", 0)
    if selection_status == "REVIEW" and selection_retry < MAX_SELECTION_RETRIES:
        return {
            "candidates": candidates,
            "current_startup": {},
            "selection_status": "REVIEW",
            "selection_retry": selection_retry + 1,
            "uncertain": False,
        }

    return {
        "candidates": candidates,
        "current_startup": candidate if selection_status != "FAIL" else {},
        "selection_status": selection_status,
        "selection_retry": selection_retry,
        "uncertain": selection_status == "REVIEW",
    }


def technology_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, Any]:
    startup = state.get("current_startup", {})
    evidence = [item for item in state.get("evidence", []) if item.get("category") != "tech"]
    return {
        "tech_summary": f"[stub] {startup.get('name', '후보')} 기술 분석 결과",
        "evidence": evidence,
        **_consume_evidence_retry(state),
    }


def market_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, Any]:
    startup = state.get("current_startup", {})
    evidence = [item for item in state.get("evidence", []) if item.get("category") != "market"]
    rag_retry = state.get("rag_retry", 0)
    if _scenario(config) == "evidence_retry" and rag_retry < MAX_RAG_RETRIES:
        return {
            "market_analysis": "[stub] 관련 청크 부족, RAG 재검색 필요",
            "rag_retry": rag_retry + 1,
            "evidence": evidence,
        }
    return {
        "market_analysis": f"[stub] {startup.get('name', '후보')} 시장 분석 결과",
        "evidence": evidence,
        **_consume_evidence_retry(state),
    }


def competitor_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, Any]:
    startup = state.get("current_startup", {})
    evidence = [item for item in state.get("evidence", []) if item.get("category") != "competitor"]
    return {
        "competitor_analysis": f"[stub] {startup.get('name', '후보')} 경쟁사 비교 결과",
        "evidence": evidence,
        **_consume_evidence_retry(state),
    }


def evaluator_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, Any]:
    scenario = _scenario(config)
    if (
        scenario == "evidence_retry"
        and MAX_EVIDENCE_RETRIES > 0
        and state.get("evidence_retry", 0) == 0
    ):
        return {
            "scores": {"market": 3.0},
            "total_score": 65.0,
            "missing_evidence": ["market_cagr"],
            "evidence_retry": 0,
        }

    total_score = 75.0 if scenario in {"invest", "evidence_retry"} else 60.0
    return {
        "scores": {"stub_score": total_score},
        "total_score": total_score,
        "missing_evidence": [],
    }


def decision_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, Any]:
    total_score = state.get("total_score", 0.0)
    missing_evidence = state.get("missing_evidence", [])
    decision = "투자" if total_score >= 70 and len(missing_evidence) < 5 else "보류"
    startup = state.get("current_startup", {})
    record = {
        "company": startup.get("name", "후보"),
        "scores": state.get("scores", {}),
        "total_score": total_score,
        "decision": decision,
        "reason": "[stub] 분기 검증용 점수이며 C-1 점수 산출은 아직 미구현",
    }
    return {"decision": decision, "evaluated": [record]}


def next_candidate(state: InvestmentState) -> dict[str, Any]:
    return {
        "current_idx": state.get("current_idx", 0) + 1,
        "current_startup": {},
        "selection_status": "",
        "selection_retry": 0,
        "uncertain": False,
        "tech_summary": "",
        "market_analysis": "",
        "rag_retry": 0,
        "competitor_analysis": "",
        "evidence": [],
        "evidence_retry": 0,
        "scores": {},
        "total_score": 0.0,
        "missing_evidence": [],
        "decision": "",
    }


def report_agent(state: InvestmentState, config: RunnableConfig) -> dict[str, str]:
    evaluated = state.get("evaluated", [])
    candidate_lines = [
        f"- {item.get('company', '후보')}: {item.get('decision', '미정')} "
        f"({item.get('total_score', 0):.1f}점)"
        for item in evaluated
    ] or ["- 분석 대상으로 확정된 후보가 없습니다."]
    report = "\n".join(
        [
            "# SUMMARY",
            "",
            "그래프 분기 확인용 더미 보고서입니다. 에이전트 분석은 구현 후 교체합니다.",
            "",
            "## 후보 처리 결과",
            *candidate_lines,
            "",
            "# REFERENCE",
            "",
            "실제 활용 자료 없음 (stub).",
        ]
    )
    return {"final_report": report}