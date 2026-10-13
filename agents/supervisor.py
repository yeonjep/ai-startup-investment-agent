# Supervisor is the only router. Every worker returns here, and the next
# worker is chosen from State by code rules.

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.config import (
    MAX_CANDIDATES,
    MAX_EVIDENCE_RETRIES,
    MAX_QUALITY_RETRIES,
    get_max_step_count,
)
from agents.route_log import append_route_log
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
ANALYSIS_NODES = (
    "technology_agent",
    "market_agent",
    "competitor_agent",
)
WORKER_TO_AREA = {
    "technology_agent": "tech",
    "market_agent": "market",
    "competitor_agent": "competitor",
}
ADVANCE_REASONS = {"보류 후 다음 후보", "선정 제외 후 다음 후보"}


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
    votes = state.get("sufficiency_votes") or {}
    for worker, area in WORKER_TO_AREA.items():
        if (votes.get(worker) or {}).get("vote") != "sufficient":
            needed.add(area)
    return [area for area in AREA_ORDER if area in needed]


def _step_cap(state: dict[str, Any]) -> int:
    if state.get("max_step_count") is not None:
        return int(state["max_step_count"])
    return get_max_step_count()


def _analysis_missing(state: dict[str, Any]) -> list[str]:
    votes = state.get("sufficiency_votes") or {}
    missing = []
    for node in ANALYSIS_NODES:
        vote = (votes.get(node) or {}).get("vote")
        if vote not in {"sufficient", "insufficient"}:
            missing.append(node)
    return missing


def _has_next_candidate(state: dict[str, Any]) -> bool:
    next_index = int(state.get("current_idx") or 0) + 1
    has_next = next_index < len(state.get("candidates") or [])
    below_limit = len(state.get("evaluated") or []) < int(
        state.get("max_candidates") or MAX_CANDIDATES
    )
    return has_next and below_limit


def _has_later_index(state: dict[str, Any]) -> bool:
    return int(state.get("current_idx") or 0) + 1 < len(state.get("candidates") or [])


def choose_routes(state: dict[str, Any]) -> tuple[list[str], str]:
    """Pick the next workers from the current state. Empty means END."""
    if int(state.get("step_count") or 0) >= _step_cap(state):
        if not state.get("final_report"):
            return ["report_agent"], "step 상한 도달, 보고서 작성 후 종료"
        return [], "step 상한 도달, 종료"

    quality = state.get("quality_result")
    if quality:
        if quality.get("passed"):
            return [], "품질 평가 통과"
        if int(state.get("quality_retry") or 0) < MAX_QUALITY_RETRIES:
            return ["report_agent"], "품질 미달로 보고서 수정"
        return [], "품질 재작성 상한 도달"

    failed = [
        name
        for name, status in (state.get("node_status") or {}).items()
        if status == "failed"
    ]
    if failed:
        if state.get("final_report"):
            return [], "노드 실패, 종료"
        return ["report_agent"], "노드 실패, 보고서 작성 후 종료"

    if state.get("final_report") and quality is None:
        return ["quality_agent"], "보고서 작성 후 품질 평가"

    if state.get("decision"):
        if state.get("decision") == "투자" or not _has_next_candidate(state):
            return ["report_agent"], "투자 판정 또는 후보 소진으로 보고서 작성"
        return ["startup_agent"], "보류 후 다음 후보"

    if state.get("scores") and state.get("decision") is None:
        areas = pending_refresh_areas(state)
        if areas and int(state.get("evidence_retry") or 0) < MAX_EVIDENCE_RETRIES:
            nodes = [AREA_TO_NODE[area] for area in areas]
            return nodes, "결측 영역 추가 분석: " + ", ".join(areas)
        if areas:
            return ["decision_agent"], "근거 보완 상한 도달, 결측을 유지하고 판정"
        if state.get("missing_evidence"):
            return ["decision_agent"], "재분석 대상 외 결측을 유지하고 판정"
        return ["decision_agent"], "근거 충분성 확인 완료"

    if (
        state.get("discovery_done")
        and state.get("current_startup")
        and state.get("selection_status") in {"PASS", "REVIEW"}
    ):
        missing = _analysis_missing(state)
        if missing:
            return missing, "선정 통과 후보의 부족한 분석 배정: " + ", ".join(missing)
        return ["evaluator_agent"], "분석 결과 수집 완료, 채점"

    if not state.get("discovery_done"):
        return ["startup_agent"], "후보 탐색"

    if state.get("selection_status") == "REVIEW" and not state.get("current_startup"):
        return ["startup_agent"], "선정 REVIEW 보완"

    if state.get("selection_status") == "FAIL" and _has_later_index(state):
        return ["startup_agent"], "선정 제외 후 다음 후보"

    candidates = state.get("candidates") or []
    index = int(state.get("current_idx") or 0)
    if state.get("discovery_done") and candidates and index < len(candidates) and not state.get("selection_status"):
        return ["startup_agent"], "현재 후보 선정"

    return ["report_agent"], "평가 가능한 후보 없음"


def supervisor(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    del config
    step = int(state.get("step_count") or 0) + 1
    view = dict(state)
    view["step_count"] = step
    routes, reason = choose_routes(view)
    update: dict[str, Any] = {
        "step_count": step,
        "next_routes": routes,
        "route_reason": reason,
        "refresh_areas": [],
        "node_status": {"supervisor": "done", **{route: "running" for route in routes}},
    }

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
    rework = bool(analysis_routes and state.get("scores") and state.get("decision") is None)
    if rework:
        areas = [area for area in AREA_ORDER if AREA_TO_NODE[area] in analysis_routes]
        update["refresh_areas"] = areas
        update["evidence_retry"] = min(
            MAX_EVIDENCE_RETRIES,
            int(state.get("evidence_retry") or 0) + 1,
        )
        update["scores"] = {}
        update["total_score"] = None
        update["decision"] = None
        update["evidence_gate_status"] = "pending"
    elif routes == ["decision_agent"]:
        update["evidence_gate_status"] = (
            "insufficient_accepted"
            if pending_refresh_areas(state) or state.get("missing_evidence")
            else "sufficient"
        )

    if reason in ADVANCE_REASONS:
        from agents.stubs import next_candidate

        advanced = next_candidate(state)
        advanced.pop("next_routes", None)
        advanced.pop("node_status", None)
        update.update(advanced)
        update["next_routes"] = routes
        update["route_reason"] = reason

    append_route_log(
        state.get("trace_id"),
        "supervisor",
        ",".join(routes) if routes else "END",
        reason,
    )
    return update


def _quality_feedback(result: dict[str, Any]) -> str:
    parts = []
    for name, check in (result.get("checks") or {}).items():
        if check.get("passed"):
            continue
        reasons = check.get("reasons") or [check.get("reason") or ""]
        section = check.get("target_section") or name
        detail = "; ".join(str(reason) for reason in reasons if reason)
        parts.append(f"[{name}] 섹션={section}: {detail}")
    return "\n".join(parts)
