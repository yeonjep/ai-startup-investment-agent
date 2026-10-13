# Supervisor routing. Workers return here; they do not call each other.
# Routing state is the control fields in docs/DESIGN.md D-1a.

from typing import Any

from agents.config import (
    MAX_CANDIDATES,
    MAX_EVIDENCE_RETRIES,
    MAX_QUALITY_RETRIES,
    MAX_RAG_RETRIES,
    MAX_SUPERVISOR_STEPS,
)
from agents.state import InvestmentState

AREA_TO_NODE = {
    "tech": "technology_agent",
    "market": "market_agent",
    "competitor": "competitor_agent",
}
NODE_TO_AREA = {node: area for area, node in AREA_TO_NODE.items()}
REFRESH_AREAS = {
    "development_stage": {"tech"},
    "technical_originality": {"tech"},
    "market_tam": {"market"},
    "market_cagr": {"market"},
    "demand_clarity": {"market"},
    "competitive_difference": {"competitor"},
    "entry_barrier": {"competitor"},
    "risk_mitigation": {"tech", "market"},
}
AREA_ORDER = ("tech", "market", "competitor")


def merge_vote(state: dict[str, Any], worker: str, sufficient: bool, reason: str) -> dict[str, Any]:
    votes = dict(state.get("sufficiency_votes") or {})
    votes[worker] = {"vote": "sufficient" if sufficient else "insufficient", "reason": reason}
    return votes


def pending_refresh_areas(state: dict[str, Any]) -> list[str]:
    needed: set[str] = set()
    for metric in state.get("missing_evidence") or []:
        needed |= REFRESH_AREAS.get(metric, set())
    status = state.get("node_status") or {}
    return [
        area
        for area in AREA_ORDER
        if area in needed and status.get(f"refresh:{area}") != "done"
    ]


def choose_route(state: dict[str, Any]) -> tuple[str, str]:
    """Return the next node name and the latest routing reason."""
    if int(state.get("step_count") or 0) >= MAX_SUPERVISOR_STEPS:
        return _capped(state)

    if not state.get("discovery_done"):
        return "startup_agent", "후보 수집 전"

    candidates = list(state.get("candidates") or [])
    index = int(state.get("current_idx") or 0)
    if not candidates or index >= len(candidates):
        return _finish(state)

    status = state.get("selection_status")
    if status is None:
        return "startup_agent", "현재 후보의 선정 결과가 없음"
    if status == "REVIEW" and not state.get("uncertain") and not state.get("current_startup"):
        return "startup_agent", "선정 REVIEW 재검색"
    if status == "FAIL":
        return "next_candidate", "선정 FAIL"

    if not state.get("tech_summary"):
        return "technology_agent", "기술 요약 없음"

    market = state.get("market_analysis") or {}
    if not market:
        return "market_agent", "시장 분석 없음"
    topics = list(market.get("insufficient_topics") or [])
    retries = state.get("rag_retry") or {}
    market_vote = ((state.get("sufficiency_votes") or {}).get("market_agent") or {}).get("vote")
    if market_vote == "insufficient" and any(
        int(retries.get(topic) or 0) < MAX_RAG_RETRIES for topic in topics
    ):
        return "market_agent", "시장 topic 재검색 예산이 남음"

    if not state.get("competitor_analysis"):
        return "competitor_agent", "경쟁 분석 없음"

    if state.get("scores") and state.get("decision") is None:
        pending = pending_refresh_areas(state)
        if int(state.get("evidence_retry") or 0) < MAX_EVIDENCE_RETRIES and pending:
            area = pending[0]
            return AREA_TO_NODE[area], f"결측 재배정: {area}"
        if (state.get("node_status") or {}).get("rescore") == "needed":
            return "evaluator_agent", "결측 재배정 후 재채점"

    if not state.get("scores"):
        return "evaluator_agent", "점수 없음"
    if state.get("decision") is None:
        return "decision_agent", "판정 없음"
    if state.get("decision") == "보류" and _can_continue(state):
        return "next_candidate", "보류 후 다음 후보"
    return _finish(state)


def supervisor(state: InvestmentState) -> dict[str, Any]:
    status = dict(state.get("node_status") or {})
    active = state.get("active_refresh_area")
    if active:
        status[f"refresh:{active}"] = "done"
    previous = state.get("next_route")
    error = _worker_error(state, previous)
    if previous and previous != "end":
        status[previous] = "error" if error else "done"

    virtual = {**state, "node_status": status, "active_refresh_area": None}
    route, reason = choose_route(virtual)
    update: dict[str, Any] = {
        "node_status": status,
        "active_refresh_area": None,
        "next_route": route,
        "route_reason": reason,
        "last_error": error,
    }
    at_cap = int(state.get("step_count") or 0) >= MAX_SUPERVISOR_STEPS
    if route != "end" and not at_cap:
        update["step_count"] = int(state.get("step_count") or 0) + 1
        status[route] = "running"

    quality = state.get("quality_result") or {}
    if (
        route == "report_agent"
        and quality
        and not quality.get("passed")
        and int(state.get("quality_retry") or 0) < MAX_QUALITY_RETRIES
    ):
        update["quality_feedback"] = _quality_feedback(quality)
        update["quality_retry"] = int(state.get("quality_retry") or 0) + 1
        update["quality_result"] = None
        update["final_report"] = ""

    if route == "evaluator_agent":
        status["rescore"] = "done"

    if (
        route in NODE_TO_AREA
        and virtual.get("scores")
        and virtual.get("decision") is None
        and int(virtual.get("evidence_retry") or 0) < MAX_EVIDENCE_RETRIES
    ):
        area = NODE_TO_AREA[route]
        pending = pending_refresh_areas(virtual)
        if area in pending:
            update["active_refresh_area"] = area
            status["rescore"] = "needed"
            status[f"refresh:{area}"] = "running"
            if not [item for item in pending if item != area]:
                update["evidence_retry"] = 1

    update["node_status"] = status
    return update


def _finish(state: dict[str, Any]) -> tuple[str, str]:
    if not str(state.get("final_report") or "").strip():
        return "report_agent", "보고서 작성"
    quality = state.get("quality_result")
    if quality is None:
        return "quality_agent", "보고서 품질 평가"
    if not quality.get("passed") and int(state.get("quality_retry") or 0) < MAX_QUALITY_RETRIES:
        return "report_agent", "품질 미달로 보고서 1회 재작성"
    if quality.get("passed"):
        return "end", "품질 평가 통과"
    return "end", "품질 재작성 상한 도달"


def _capped(state: dict[str, Any]) -> tuple[str, str]:
    if not str(state.get("final_report") or "").strip():
        return "report_agent", "배정 상한 도달, 보고서 작성"
    if state.get("quality_result") is None:
        return "quality_agent", "배정 상한 도달, 품질 평가"
    return "end", "배정 상한 도달"


def _can_continue(state: dict[str, Any]) -> bool:
    nxt = int(state.get("current_idx") or 0) + 1
    candidates = list(state.get("candidates") or [])
    evaluated = list(state.get("evaluated") or [])
    limit = int(state.get("max_candidates") or MAX_CANDIDATES)
    return nxt < len(candidates) and len(evaluated) < limit


def _worker_error(state: dict[str, Any], previous: str | None) -> str | None:
    if previous == "technology_agent":
        error = (state.get("tech_summary") or {}).get("error")
        return str(error) if error else None
    if previous == "market_agent":
        for topic in ((state.get("market_analysis") or {}).get("topics") or {}).values():
            if topic.get("error"):
                return str(topic["error"])
        error = (state.get("market_analysis") or {}).get("analysis_error")
        return str(error) if error else None
    if previous == "competitor_agent":
        error = (state.get("competitor_analysis") or {}).get("error")
        return str(error) if error else None
    return None


def _quality_feedback(result: dict[str, Any]) -> str:
    parts = []
    for name, check in (result.get("checks") or {}).items():
        if not check.get("passed"):
            parts.append(f"{name}: {check.get('reason')}")
    return " ".join(parts)
