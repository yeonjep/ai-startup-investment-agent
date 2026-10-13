# Supervisor handles only exceptional rework: missing-evidence analysis and
# report revision after a failed quality gate.

from typing import Any

from agents.config import (
    MAX_EVIDENCE_RETRIES,
    MAX_QUALITY_RETRIES,
    MAX_SUPERVISOR_STEPS,
)
from agents.state import InvestmentState

AREA_TO_NODE = {
    "tech": "technology_agent",
    "market": "market_agent",
    "competitor": "competitor_agent",
}
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
    del state
    return {
        worker: {
            "vote": "sufficient" if sufficient else "insufficient",
            "reason": reason,
        }
    }


def pending_refresh_areas(state: dict[str, Any]) -> list[str]:
    needed: set[str] = set()
    for metric in state.get("missing_evidence") or []:
        needed |= REFRESH_AREAS.get(metric, set())
    return [area for area in AREA_ORDER if area in needed]


def choose_routes(state: dict[str, Any]) -> tuple[list[str], str]:
    """Choose only exceptional rework destinations."""
    if int(state.get("step_count") or 0) >= MAX_SUPERVISOR_STEPS:
        return ["report_agent"], "재작업 상한 도달, 현재 결과로 보고서 작성"

    quality = state.get("quality_result")
    if quality and not quality.get("passed"):
        if int(state.get("quality_retry") or 0) < MAX_QUALITY_RETRIES:
            return ["report_agent"], "품질 미달로 보고서 수정"
        return [], "품질 재작성 상한 도달"

    areas = pending_refresh_areas(state)
    if (
        state.get("scores")
        and state.get("decision") is None
        and areas
        and int(state.get("evidence_retry") or 0) < MAX_EVIDENCE_RETRIES
    ):
        nodes = [AREA_TO_NODE[area] for area in areas]
        return nodes, "결측 영역 추가 분석: " + ", ".join(areas)

    return ["report_agent"], "추가 분석 대상 없음, 보고서 수정"


def supervisor(state: InvestmentState) -> dict[str, Any]:
    routes, reason = choose_routes(state)
    update: dict[str, Any] = {
        "next_routes": routes,
        "route_reason": reason,
        "refresh_areas": [],
    }
    if routes:
        update["step_count"] = int(state.get("step_count") or 0) + 1

    quality = state.get("quality_result") or {}
    if (
        routes == ["report_agent"]
        and quality
        and not quality.get("passed")
        and int(state.get("quality_retry") or 0) < MAX_QUALITY_RETRIES
    ):
        update["quality_feedback"] = _quality_feedback(quality)
        update["quality_retry"] = int(state.get("quality_retry") or 0) + 1
        update["quality_result"] = None
        update["final_report"] = ""

    analysis_routes = [route for route in routes if route in AREA_TO_NODE.values()]
    if analysis_routes:
        areas = [
            area for area in AREA_ORDER if AREA_TO_NODE[area] in analysis_routes
        ]
        update["refresh_areas"] = areas
        update["evidence_retry"] = min(
            MAX_EVIDENCE_RETRIES,
            int(state.get("evidence_retry") or 0) + 1,
        )
        update["scores"] = {}
        update["total_score"] = None
        update["decision"] = None

    return update


def _quality_feedback(result: dict[str, Any]) -> str:
    parts = []
    for name, check in (result.get("checks") or {}).items():
        if not check.get("passed"):
            parts.append(f"{name}: {check.get('reason')}")
    return " ".join(parts)
