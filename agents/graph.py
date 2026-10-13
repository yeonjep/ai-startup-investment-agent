# Supervisor hub. Workers return to supervisor. See docs/DESIGN.md D-3 and D-4.

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.config import MAX_SUPERVISOR_STEPS
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

WORKER_NODES = (
    "startup_agent",
    "technology_agent",
    "market_agent",
    "competitor_agent",
    "evaluator_agent",
    "decision_agent",
    "next_candidate",
    "report_agent",
    "quality_agent",
)
RouteName = Literal[
    "startup_agent",
    "technology_agent",
    "market_agent",
    "competitor_agent",
    "evaluator_agent",
    "decision_agent",
    "next_candidate",
    "report_agent",
    "quality_agent",
    "end",
]


def route_from_supervisor(state: InvestmentState) -> RouteName:
    route = state.get("next_route") or "end"
    if route not in {*WORKER_NODES, "end"}:
        return "end"
    return route  # type: ignore[return-value]


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
    builder.add_edge("initialize_state", "supervisor")
    for node_name in WORKER_NODES:
        builder.add_edge(node_name, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {node_name: node_name for node_name in WORKER_NODES} | {"end": END},
    )
    return builder.compile()


def recursion_limit() -> int:
    return MAX_SUPERVISOR_STEPS * 2 + 20
