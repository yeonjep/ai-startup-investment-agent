# 설계 확정 후 교체할 부분: 가이드의 6개 노드만 연결한 그래프 뼈대입니다.

from langgraph.graph import END, START, StateGraph

from agents.config import MAX_ITERATIONS
from agents.state import InvestmentState
from agents.stubs import (
    competitor_comparison,
    investment_decision,
    market_evaluation,
    report_generation,
    startup_search,
    technology_summary,
)


def route_after_startup(state: InvestmentState) -> str:
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    evaluated_count = len(state.get("evaluated", []))
    if evaluated_count >= max_iterations:
        return "report_generation"
    return "technology_summary" if state.get("current_startup") else "report_generation"


def route_after_decision(state: InvestmentState) -> str:
    if state.get("decision") != "보류":
        return "report_generation"

    candidates = state.get("candidates", [])
    current_idx = state.get("current_idx", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATIONS)
    evaluated_count = len(state.get("evaluated", []))
    if current_idx < len(candidates) and evaluated_count < max_iterations:
        return "startup_search"
    return "report_generation"


def build_graph():
    builder = StateGraph(InvestmentState)
    builder.add_node("startup_search", startup_search)
    builder.add_node("technology_summary", technology_summary)
    builder.add_node("market_evaluation", market_evaluation)
    builder.add_node("competitor_comparison", competitor_comparison)
    builder.add_node("investment_decision", investment_decision)
    builder.add_node("report_generation", report_generation)

    builder.add_edge(START, "startup_search")
    builder.add_conditional_edges(
        "startup_search",
        route_after_startup,
        {
            "technology_summary": "technology_summary",
            "report_generation": "report_generation",
        },
    )
    builder.add_edge("technology_summary", "market_evaluation")
    builder.add_edge("market_evaluation", "competitor_comparison")
    builder.add_edge("competitor_comparison", "investment_decision")
    builder.add_conditional_edges(
        "investment_decision",
        route_after_decision,
        {
            "startup_search": "startup_search",
            "report_generation": "report_generation",
        },
    )
    builder.add_edge("report_generation", END)
    return builder.compile()