# Candidate gate -> parallel analysis -> evaluation/decision -> report/quality.
# Supervisor is used only for exceptional analysis or report rework.

from typing import Literal, Sequence

from langgraph.graph import END, START, StateGraph

from agents.config import (
    MAX_CANDIDATES,
    MAX_EVIDENCE_RETRIES,
    MAX_QUALITY_RETRIES,
    MAX_SUPERVISOR_STEPS,
)
from agents.quality import quality_agent
from agents.report import report_agent
from agents.state import InvestmentState
from agents.stubs import (
    competitor_agent,
    decision_agent,
    evaluator_agent,
    initialize_state,
    market_agent,
    next_candidate,
    startup_agent,
    technology_agent,
)
from agents.supervisor import supervisor

ANALYSIS_NODES = (
    "technology_agent",
    "market_agent",
    "competitor_agent",
)
SUPERVISOR_DESTINATIONS = (
    "technology_agent",
    "market_agent",
    "competitor_agent",
    "report_agent",
)


def route_after_startup(
    state: InvestmentState,
) -> str | Sequence[str]:
    candidates = list(state.get("candidates") or [])
    index = int(state.get("current_idx") or 0)
    if not candidates or index >= len(candidates):
        return "report_agent"
    status = state.get("selection_status")
    if status == "REVIEW" and not state.get("current_startup"):
        return "startup_agent"
    if status == "FAIL":
        return "next_candidate"
    if status in {"PASS", "REVIEW"} and state.get("current_startup"):
        return list(ANALYSIS_NODES)
    return "report_agent"


def route_after_evaluator(
    state: InvestmentState,
) -> Literal["supervisor", "decision_agent"]:
    refreshable = any(
        metric
        in {
            "development_stage",
            "technical_originality",
            "market_tam",
            "market_cagr",
            "demand_clarity",
            "competitive_difference",
            "entry_barrier",
            "risk_mitigation",
        }
        for metric in state.get("missing_evidence") or []
    )
    if refreshable and int(state.get("evidence_retry") or 0) < MAX_EVIDENCE_RETRIES:
        return "supervisor"
    return "decision_agent"


def route_after_decision(
    state: InvestmentState,
) -> Literal["next_candidate", "report_agent"]:
    if state.get("decision") == "투자":
        return "report_agent"
    next_index = int(state.get("current_idx") or 0) + 1
    has_next = next_index < len(state.get("candidates") or [])
    below_limit = len(state.get("evaluated") or []) < int(
        state.get("max_candidates") or MAX_CANDIDATES
    )
    return "next_candidate" if has_next and below_limit else "report_agent"


def route_after_quality(
    state: InvestmentState,
) -> Literal["supervisor", "end"]:
    quality = state.get("quality_result") or {}
    if quality.get("passed"):
        return "end"
    if int(state.get("quality_retry") or 0) < MAX_QUALITY_RETRIES:
        return "supervisor"
    return "end"


def route_from_supervisor(state: InvestmentState) -> str | Sequence[str]:
    routes = [
        route
        for route in state.get("next_routes") or []
        if route in SUPERVISOR_DESTINATIONS
    ]
    return routes or "report_agent"


def build_graph():
    builder = StateGraph(InvestmentState)
    builder.add_node("initialize_state", initialize_state)
    builder.add_node("supervisor", supervisor)
    builder.add_node("startup_agent", startup_agent)
    builder.add_node("technology_agent", technology_agent)
    builder.add_node("market_agent", market_agent)
    builder.add_node("competitor_agent", competitor_agent)
    builder.add_node("evaluator_agent", evaluator_agent)
    builder.add_node("decision_agent", decision_agent)
    builder.add_node("next_candidate", next_candidate)
    builder.add_node("report_agent", report_agent)
    builder.add_node("quality_agent", quality_agent)

    builder.add_edge(START, "initialize_state")
    builder.add_edge("initialize_state", "startup_agent")
    builder.add_conditional_edges(
        "startup_agent",
        route_after_startup,
        [
            "startup_agent",
            "technology_agent",
            "market_agent",
            "competitor_agent",
            "next_candidate",
            "report_agent",
        ],
    )
    for node_name in ANALYSIS_NODES:
        builder.add_edge(node_name, "evaluator_agent")
    builder.add_conditional_edges(
        "evaluator_agent",
        route_after_evaluator,
        {
            "supervisor": "supervisor",
            "decision_agent": "decision_agent",
        },
    )
    builder.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {
            "next_candidate": "next_candidate",
            "report_agent": "report_agent",
        },
    )
    builder.add_edge("next_candidate", "startup_agent")
    builder.add_edge("report_agent", "quality_agent")
    builder.add_conditional_edges(
        "quality_agent",
        route_after_quality,
        {
            "supervisor": "supervisor",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        list(SUPERVISOR_DESTINATIONS),
    )
    return builder.compile()


def recursion_limit() -> int:
    return MAX_SUPERVISOR_STEPS * 2 + 20
