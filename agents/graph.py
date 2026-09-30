# Graph topology follows docs/DESIGN.md D-2, D-4, D-9, and D-10.

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.config import MAX_CANDIDATES, MAX_EVIDENCE_RETRIES, MAX_RAG_RETRIES
from agents.state import InvestmentState
from agents.stubs import (
    competitor_agent,
    decision_agent,
    evaluator_agent,
    initialize_state,
    market_agent,
    next_candidate,
    report_agent,
    startup_agent,
    technology_agent,
)


def route_after_startup(
    state: InvestmentState,
) -> Literal["technology_agent", "startup_agent", "next_candidate"]:
    selection_status = state.get("selection_status", "FAIL")
    if selection_status == "PASS":
        return "technology_agent"
    if selection_status == "REVIEW":
        if state.get("uncertain", False):
            return "technology_agent"
        return "startup_agent"
    return "next_candidate"


def route_after_market(
    state: InvestmentState,
) -> Literal["market_agent", "competitor_agent"]:
    if (
        "관련 청크 부족" in state.get("market_analysis", "")
        and state.get("rag_retry", 0) < MAX_RAG_RETRIES
    ):
        return "market_agent"
    return "competitor_agent"


def _missing_evidence_area(state: InvestmentState) -> str:
    counts = {"technology_agent": 0, "market_agent": 0, "competitor_agent": 0}
    prefixes = {
        "tech": "technology_agent",
        "technology": "technology_agent",
        "market": "market_agent",
        "competitor": "competitor_agent",
    }
    for missing_item in state.get("missing_evidence", []):
        prefix = missing_item.split("_", maxsplit=1)[0].lower()
        area = prefixes.get(prefix)
        if area:
            counts[area] += 1
    if not any(counts.values()):
        return "decision_agent"
    return max(counts, key=counts.get)


def route_after_evaluator(
    state: InvestmentState,
) -> Literal["technology_agent", "market_agent", "competitor_agent", "decision_agent"]:
    if (
        state.get("missing_evidence")
        and state.get("evidence_retry", 0) < MAX_EVIDENCE_RETRIES
    ):
        return _missing_evidence_area(state)
    return "decision_agent"


def route_after_decision(state: InvestmentState) -> Literal["report_agent", "next_candidate"]:
    return "report_agent" if state.get("decision") == "투자" else "next_candidate"


def route_after_next_candidate(
    state: InvestmentState,
) -> Literal["startup_agent", "report_agent"]:
    has_candidates = state.get("current_idx", 0) < len(state.get("candidates", []))
    under_evaluation_limit = len(state.get("evaluated", [])) < state.get(
        "max_candidates", MAX_CANDIDATES
    )
    return "startup_agent" if has_candidates and under_evaluation_limit else "report_agent"


def build_graph():
    builder = StateGraph(InvestmentState)
    builder.add_node("initialize_state", initialize_state)
    builder.add_node("startup_agent", startup_agent)
    builder.add_node("technology_agent", technology_agent)
    builder.add_node("market_agent", market_agent)
    builder.add_node("competitor_agent", competitor_agent)
    builder.add_node("evaluator_agent", evaluator_agent)
    builder.add_node("decision_agent", decision_agent)
    builder.add_node("next_candidate", next_candidate)
    builder.add_node("report_agent", report_agent)

    builder.add_edge(START, "initialize_state")
    builder.add_edge("initialize_state", "startup_agent")
    builder.add_conditional_edges(
        "startup_agent",
        route_after_startup,
        {
            "technology_agent": "technology_agent",
            "startup_agent": "startup_agent",
            "next_candidate": "next_candidate",
        },
    )
    builder.add_edge("technology_agent", "market_agent")
    builder.add_conditional_edges(
        "market_agent",
        route_after_market,
        {
            "market_agent": "market_agent",
            "competitor_agent": "competitor_agent",
        },
    )
    builder.add_edge("competitor_agent", "evaluator_agent")
    builder.add_conditional_edges(
        "evaluator_agent",
        route_after_evaluator,
        {
            "technology_agent": "technology_agent",
            "market_agent": "market_agent",
            "competitor_agent": "competitor_agent",
            "decision_agent": "decision_agent",
        },
    )
    builder.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {
            "report_agent": "report_agent",
            "next_candidate": "next_candidate",
        },
    )
    builder.add_conditional_edges(
        "next_candidate",
        route_after_next_candidate,
        {
            "startup_agent": "startup_agent",
            "report_agent": "report_agent",
        },
    )
    builder.add_edge("report_agent", END)
    return builder.compile()