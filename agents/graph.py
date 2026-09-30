# Graph topology follows docs/DESIGN.md D-3 and D-4.

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.config import (
    INVESTMENT_SCORE_THRESHOLD,
    MAX_CANDIDATES,
    MAX_EVIDENCE_RETRIES,
    MAX_MISSING_EVIDENCE_FOR_INVESTMENT,
    MAX_RAG_RETRIES,
)
from agents.state import InvestmentState
from agents.report import report_agent
from agents.report import report_agent
from agents.stubs import (
    competitor_agent,
    decision_agent,
    evaluator_agent,
    evidence_refresh,
    initialize_state,
    market_agent,
    next_candidate,
    startup_agent,
    technology_agent,
)

EVIDENCE_REFRESH_METRICS = {
    "development_stage",
    "technical_originality",
    "market_tam",
    "market_cagr",
    "demand_clarity",
    "competitive_difference",
    "entry_barrier",
    "risk_mitigation",
}


def route_after_startup(
    state: InvestmentState,
) -> Literal["technology_agent", "startup_agent", "next_candidate", "report_agent"]:
    if not state.get("candidates"):
        return "report_agent"
    status = state.get("selection_status")
    if status == "PASS" or (status == "REVIEW" and state.get("uncertain", False)):
        return "technology_agent"
    if status == "REVIEW":
        return "startup_agent"
    if status == "FAIL":
        return "next_candidate"
    return "report_agent"


def route_after_market(
    state: InvestmentState,
) -> Literal["market_agent", "competitor_agent"]:
    analysis = state.get("market_analysis", {})
    insufficient_topics = analysis.get("insufficient_topics", [])
    retries = state.get("rag_retry", {})
    if any(retries.get(topic, 0) < MAX_RAG_RETRIES for topic in insufficient_topics):
        return "market_agent"
    return "competitor_agent"


def route_after_evaluator(
    state: InvestmentState,
) -> Literal["evidence_refresh", "decision_agent"]:
    has_refreshable_missing = any(
        metric in EVIDENCE_REFRESH_METRICS
        for metric in state.get("missing_evidence", [])
    )
    if (
        has_refreshable_missing
        and state.get("evidence_retry", 0) < MAX_EVIDENCE_RETRIES
    ):
        return "evidence_refresh"
    return "decision_agent"


def route_after_decision(state: InvestmentState) -> Literal["report_agent", "next_candidate"]:
    total_score = state.get("total_score")
    eligible = (
        total_score is not None
        and total_score >= INVESTMENT_SCORE_THRESHOLD
        and len(state.get("missing_evidence", [])) < MAX_MISSING_EVIDENCE_FOR_INVESTMENT
        and not state.get("uncertain", False)
    )
    return "report_agent" if eligible else "next_candidate"


def route_after_next_candidate(
    state: InvestmentState,
) -> Literal["startup_agent", "report_agent"]:
    has_candidates = state.get("current_idx", 0) < len(state.get("candidates", []))
    below_evaluation_limit = len(state.get("evaluated", [])) < state.get(
        "max_candidates", MAX_CANDIDATES
    )
    return "startup_agent" if has_candidates and below_evaluation_limit else "report_agent"


def build_graph():
    builder = StateGraph(InvestmentState)
    builder.add_node("initialize_state", initialize_state)
    builder.add_node("startup_agent", startup_agent)
    builder.add_node("technology_agent", technology_agent)
    builder.add_node("market_agent", market_agent)
    builder.add_node("competitor_agent", competitor_agent)
    builder.add_node("evaluator_agent", evaluator_agent)
    builder.add_node("evidence_refresh", evidence_refresh)
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
            "report_agent": "report_agent",
        },
    )
    builder.add_edge("technology_agent", "market_agent")
    builder.add_conditional_edges(
        "market_agent",
        route_after_market,
        {"market_agent": "market_agent", "competitor_agent": "competitor_agent"},
    )
    builder.add_edge("competitor_agent", "evaluator_agent")
    builder.add_conditional_edges(
        "evaluator_agent",
        route_after_evaluator,
        {"evidence_refresh": "evidence_refresh", "decision_agent": "decision_agent"},
    )
    builder.add_edge("evidence_refresh", "evaluator_agent")
    builder.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {"report_agent": "report_agent", "next_candidate": "next_candidate"},
    )
    builder.add_conditional_edges(
        "next_candidate",
        route_after_next_candidate,
        {"startup_agent": "startup_agent", "report_agent": "report_agent"},
    )
    builder.add_edge("report_agent", END)
    return builder.compile()