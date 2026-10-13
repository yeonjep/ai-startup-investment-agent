# 기본 실행(real)은 실제 에이전트로 보고서를 생성한다. 나머지 시나리오는 그래프 분기 검증용 stub 후보를 사용한다.

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Literal

from langchain_core.tracers.context import collect_runs

from agents.common import configure_runtime
from agents.config import (
    get_as_of_date,
    validate_max_candidates,
)
from agents.graph import build_graph, recursion_limit
from agents.report import REPORT_BASENAME
from agents.state import InvestmentState


PROJECT_ROOT = Path(__file__).resolve().parent
Scenario = Literal[
    "real",
    "invest",
    "all_hold",
    "zero_pass",
    "evidence_retry",
    "no_candidates",
    "uncertain",
]
DEMO_SCENARIOS = {
    "real": None,  # stub 후보 없이 Startup Agent가 직접 후보를 탐색한다
    "invest": [{"name": "[샘플 투자 후보]"}],
    "all_hold": [{"name": "[샘플 후보 A]"}, {"name": "[샘플 후보 B]"}],
    "zero_pass": [{"name": "[부적합 후보 A]"}, {"name": "[부적합 후보 B]"}],
    "evidence_retry": [{"name": "[근거 보완 후보]"}],
    "no_candidates": [],
    "uncertain": [{"name": "[불확실 후보]"}],
}


def main(
    scenario: Scenario = "real",
    from_state: str | None = None,
    from_node: str | None = None,
) -> dict[str, Any]:
    configure_runtime()
    trace_id = uuid.uuid4().hex
    initial_state: InvestmentState = {"domain": "AI 반도체", "trace_id": trace_id}
    if from_state:
        loaded = json.loads(Path(from_state).read_text(encoding="utf-8"))
        loaded["trace_id"] = loaded.get("trace_id") or trace_id
        trace_id = str(loaded["trace_id"])
        initial_state = loaded
    max_candidates = validate_max_candidates()
    graph = build_graph()
    report_basename = REPORT_BASENAME if scenario == "real" else f"stub-{scenario}"
    graph_config = {
        "recursion_limit": recursion_limit(),
        "run_name": f"investment-{trace_id}",
        "metadata": {"trace_id": trace_id},
        "tags": ["investment-eval", trace_id],
        "configurable": {
            "thread_id": trace_id,
            "trace_id": trace_id,
            "scenario": scenario,
            "report_basename": report_basename,
            "demo_candidates": DEMO_SCENARIOS[scenario],
            "max_candidates": max_candidates,
            "as_of_date": get_as_of_date(),
        },
    }

    result: dict[str, Any] = dict(initial_state)
    node_visits: list[str] = []
    if from_state and from_node:
        result, node_visits = _run_from_node(result, graph_config, from_node)
        _save_last_state(result)
        _print_run(scenario, trace_id, node_visits, result, report_basename, None)
        return result

    with collect_runs() as traced:
        try:
            for update in graph.stream(
                initial_state,
                config=graph_config,
                stream_mode="updates",
            ):
                for node_name, node_update in update.items():
                    if node_name == "supervisor":
                        routes = node_update.get("next_routes") or []
                        node_visits.append(
                            f"supervisor[{','.join(routes) or 'END'}: "
                            f"{node_update.get('route_reason')}]"
                        )
                    else:
                        node_visits.append(node_name)
                    if "evaluated" in node_update:
                        result["evaluated"] = [
                            *result.get("evaluated", []),
                            *node_update["evaluated"],
                        ]
                    result.update({key: value for key, value in node_update.items() if key != "evaluated"})
                    _save_last_state(result)
        except Exception:
            _save_last_state(result)
            print(f"Stopped after: {' -> '.join(node_visits)}")
            raise

    run_url = _langsmith_url(traced.traced_runs)
    _save_last_state(result)
    _print_run(scenario, trace_id, node_visits, result, report_basename, run_url)
    return result


def _run_from_node(state: dict[str, Any], config: dict[str, Any], start: str) -> tuple[dict[str, Any], list[str]]:
    """Replay from a saved node through Quality. Earlier search nodes are not called."""
    from agents.quality import quality_agent
    from agents.report import report_agent
    from agents.stubs import decision_agent, evaluator_agent

    pipeline = (
        ("evaluator_agent", evaluator_agent),
        ("decision_agent", decision_agent),
        ("report_agent", report_agent),
        ("quality_agent", quality_agent),
    )
    aliases = {
        "evaluator": "evaluator_agent",
        "decision": "decision_agent",
        "report": "report_agent",
        "quality": "quality_agent",
    }
    start_name = aliases.get(start, start)
    names = [name for name, _ in pipeline]
    if start_name not in names:
        raise ValueError(f"resume node must be one of: {', '.join(names)}")
    if start_name == "evaluator_agent":
        company = (state.get("current_startup") or {}).get("name")
        if company:
            state["evaluated"] = [
                record
                for record in state.get("evaluated") or []
                if (record.get("company_profile") or {}).get("name") != company
            ]
        state["scores"] = {}
        state["decision"] = None
        state["final_report"] = ""
        state["quality_result"] = None
    elif start_name == "decision_agent":
        state["decision"] = None
        state["final_report"] = ""
        state["quality_result"] = None
    elif start_name == "report_agent":
        state["final_report"] = ""
        state["quality_result"] = None
    else:
        state["quality_result"] = None
    visits: list[str] = []
    for name, fn in pipeline[names.index(start_name):]:
        update = fn(state, config)
        visits.append(name)
        if "evaluated" in update:
            state["evaluated"] = [*state.get("evaluated", []), *update["evaluated"]]
        state.update({key: value for key, value in update.items() if key != "evaluated"})
        _save_last_state(state)
    return state, visits


def _save_last_state(result: dict[str, Any]) -> None:
    path = PROJECT_ROOT / "outputs" / "last_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    skip = {"node_visits", "report_path", "langsmith_url", "route_log"}
    payload = {key: value for key, value in result.items() if key not in skip}
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _print_run(
    scenario: str,
    trace_id: str,
    node_visits: list[str],
    result: dict[str, Any],
    report_basename: str,
    run_url: str | None,
) -> None:
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"{trace_id}.jsonl"
    print(f"Scenario: {scenario}")
    print(f"Trace ID: {trace_id}")
    print(f"Node visits: {' -> '.join(node_visits)}")
    print(f"Candidates processed: {len(result.get('evaluated', []))}")
    print(f"Report markdown: {PROJECT_ROOT / 'outputs' / (report_basename + '.md')}")
    print(f"Report PDF: {PROJECT_ROOT / 'outputs' / (report_basename + '.pdf')}")
    print(f"LangSmith run: {run_url or 'unavailable'}")
    print(f"Route log: {log_path}")
    print(f"Saved state: {PROJECT_ROOT / 'outputs' / 'last_state.json'}")
    result["node_visits"] = node_visits
    result["trace_id"] = trace_id
    result["langsmith_url"] = run_url
    result["route_log"] = str(log_path)
    result["report_path"] = str(PROJECT_ROOT / "outputs" / f"{report_basename}.pdf")


def _langsmith_url(traced_runs: list[Any]) -> str | None:
    if not traced_runs:
        return None
    run = traced_runs[0]
    url = getattr(run, "url", None)
    if url:
        return str(url)
    run_id = getattr(run, "id", None)
    if run_id is None:
        return None
    try:
        from langsmith import Client

        return str(Client().read_run(str(run_id)).url)
    except Exception:  # noqa: BLE001 - tracing is optional if the API is unavailable
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the investment evaluation graph (default: real agents; other scenarios use stub candidates).")
    parser.add_argument(
        "--scenario",
        choices=("real", "invest", "all_hold", "zero_pass", "evidence_retry", "no_candidates", "uncertain"),
        default="real",
        help="real = 실제 후보 탐색·평가·보고서 생성. 나머지는 분기 검증용 stub 후보.",
    )
    parser.add_argument("--from-state", default=None, help="outputs/last_state.json 같은 저장 State")
    parser.add_argument("--from-node", default=None, help="evaluator면 검색 없이 Evaluator→Decision→Report→Quality")
    args = parser.parse_args()
    main(args.scenario, from_state=args.from_state, from_node=args.from_node)