# 기본 실행(real)은 실제 에이전트로 보고서를 생성한다. 나머지 시나리오는 그래프 분기 검증용 stub 후보를 사용한다.

import argparse
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


def main(scenario: Scenario = "real") -> dict[str, Any]:
    configure_runtime()
    trace_id = uuid.uuid4().hex
    initial_state: InvestmentState = {"domain": "AI 반도체", "trace_id": trace_id}
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
    with collect_runs() as traced:
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

    run_url = _langsmith_url(traced.traced_runs)
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"{trace_id}.jsonl"
    print(f"Scenario: {scenario}")
    print(f"Trace ID: {trace_id}")
    print(f"Node visits: {' -> '.join(node_visits)}")
    print(f"Candidates processed: {len(result.get('evaluated', []))}")
    print(f"Report markdown: {PROJECT_ROOT / 'outputs' / (report_basename + '.md')}")
    print(f"Report PDF: {PROJECT_ROOT / 'outputs' / (report_basename + '.pdf')}")
    print(f"LangSmith run: {run_url or 'unavailable'}")
    print(f"Route log: {log_path}")
    result["node_visits"] = node_visits
    result["trace_id"] = trace_id
    result["langsmith_url"] = run_url
    result["route_log"] = str(log_path)
    result["report_path"] = str(PROJECT_ROOT / "outputs" / f"{report_basename}.pdf")
    return result


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
    main(parser.parse_args().scenario)