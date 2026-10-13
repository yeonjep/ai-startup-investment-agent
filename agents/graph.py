# Hub-and-spoke: every worker returns to supervisor, which chooses the next one.

import inspect
from collections.abc import Sequence
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.config import (
    MAX_CANDIDATE_POOL,
    MAX_EVIDENCE_RETRIES,
    MAX_QUALITY_RETRIES,
    MAX_SELECTION_RETRIES,
)
from agents.quality import quality_agent
from agents.report import report_agent
from agents.route_log import append_route_log
from agents.state import InvestmentState
from agents.stubs import (
    competitor_agent,
    decision_agent,
    evaluator_agent,
    initialize_state,
    market_agent,
    startup_agent,
    technology_agent,
)
from agents.supervisor import ANALYSIS_NODES, supervisor

WORKERS = (
    "startup_agent",
    "technology_agent",
    "market_agent",
    "competitor_agent",
    "evaluator_agent",
    "decision_agent",
    "report_agent",
    "quality_agent",
)
WORKER_FUNCTIONS = {
    "startup_agent": startup_agent,
    "technology_agent": technology_agent,
    "market_agent": market_agent,
    "competitor_agent": competitor_agent,
    "evaluator_agent": evaluator_agent,
    "decision_agent": decision_agent,
    "report_agent": report_agent,
    "quality_agent": quality_agent,
}


def _accepts_config(fn: Callable[..., Any]) -> bool:
    return "config" in inspect.signature(fn).parameters


def _track(name: str, fn: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
    """Record running/done/failed without storing the route log in State."""

    def wrapped(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        try:
            result = fn(state, config) if _accepts_config(fn) else fn(state)
        except Exception as exc:  # noqa: BLE001 - failure is recorded, then supervisor stops the loop
            append_route_log(state.get("trace_id"), name, "failed", str(exc))
            return {
                "node_status": {name: "failed"},
                "last_error": f"{name}: {exc}",
            }
        if not isinstance(result, dict):
            result = {}
        result["node_status"] = {**(result.get("node_status") or {}), name: "done"}
        append_route_log(state.get("trace_id"), name, "done", "completed")
        return result

    wrapped.__name__ = name
    return wrapped


def route_from_supervisor(state: InvestmentState) -> str | Sequence[str]:
    routes = [route for route in state.get("next_routes") or [] if route in WORKERS]
    if not routes:
        return "end"
    if len(routes) == 1:
        return routes[0]
    return routes


def build_graph(checkpointer: Any = None):
    builder = StateGraph(InvestmentState)
    builder.add_node("initialize_state", initialize_state)
    builder.add_node("supervisor", supervisor)
    for name, fn in WORKER_FUNCTIONS.items():
        builder.add_node(name, _track(name, fn))

    builder.add_edge(START, "initialize_state")
    builder.add_edge("initialize_state", "supervisor")
    for name in WORKERS:
        builder.add_edge(name, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {name: name for name in WORKERS} | {"end": END},
    )
    return builder.compile(checkpointer=MemorySaver() if checkpointer is None else checkpointer)


def worker_successors(graph) -> dict[str, set[str]]:
    """Compiled edge targets for each worker. Hub-and-spoke means only supervisor."""
    successors = {name: set() for name in WORKERS}
    for edge in graph.get_graph().edges:
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if source in successors and target:
            successors[source].add(target)
    return successors


def recursion_limit() -> int:
    # Engine guard. Business stop is the step cap plus candidate and retry budgets.
    per_candidate = 24 + MAX_SELECTION_RETRIES + (6 * MAX_EVIDENCE_RETRIES)
    quality_steps = 6 * (1 + MAX_QUALITY_RETRIES)
    return MAX_CANDIDATE_POOL * per_candidate + quality_steps + 20
