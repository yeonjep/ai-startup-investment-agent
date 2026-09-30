# 기본 실행(real)은 실제 에이전트로 보고서를 생성한다. 나머지 시나리오는 그래프 분기 검증용 stub 후보를 사용한다.

import argparse
from pathlib import Path
from typing import Any, Literal

from agents.common import configure_runtime
from agents.config import (
    MAX_CANDIDATE_POOL,
    MAX_EVIDENCE_RETRIES,
    MAX_RAG_RETRIES,
    MAX_SELECTION_RETRIES,
    get_as_of_date,
    validate_max_candidates,
)
from agents.graph import build_graph
from agents.report import REPORT_BASENAME
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
    initial_state: InvestmentState = {"domain": "AI 반도체"}  # This line is unchanged
    max_candidates = validate_max_candidates()
    graph = build_graph()
    max_steps = (
        MAX_CANDIDATE_POOL * (MAX_SELECTION_RETRIES + 2)
        + max_candidates * (12 + MAX_RAG_RETRIES * len(("market_size", "market_growth", "demand_risk")) + MAX_EVIDENCE_RETRIES * 2)
        + 10
    )
    # stub 시나리오가 제출용 보고서를 덮어쓰지 않도록 파일명을 분리한다
    report_basename = REPORT_BASENAME if scenario == "real" else f"stub-{scenario}"
    graph_config = {
        "recursion_limit": max(25, max_steps),
        "configurable": {
            "scenario": scenario,
            "report_basename": report_basename,
            "demo_candidates": DEMO_SCENARIOS[scenario],
            "max_candidates": max_candidates,
            "as_of_date": get_as_of_date(),
        },
    }

    result: dict[str, Any] = dict(initial_state)
    node_visits: list[str] = []
    for update in graph.stream(
        initial_state,
        config=graph_config,
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            node_visits.append(node_name)
            if "evaluated" in node_update:
                result["evaluated"] = [
                    *result.get("evaluated", []),
                    *node_update["evaluated"],
                ]
            result.update({key: value for key, value in node_update.items() if key != "evaluated"})

    print(f"Scenario: {scenario}")
    print(f"Node visits: {' -> '.join(node_visits)}")
    print(f"Candidates processed: {len(result.get('evaluated', []))}")
    print(f"Report markdown: {PROJECT_ROOT / 'outputs' / (report_basename + '.md')}")
    print(f"Report PDF: {PROJECT_ROOT / 'outputs' / (report_basename + '.pdf')}")
    result["node_visits"] = node_visits
    result["report_path"] = str(PROJECT_ROOT / "outputs" / f"{report_basename}.pdf")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the investment evaluation graph (default: real agents; other scenarios use stub candidates).")
    parser.add_argument(
        "--scenario",
        choices=("real", "invest", "all_hold", "zero_pass", "evidence_retry", "no_candidates", "uncertain"),
        default="real",
        help="real = 실제 후보 탐색·평가·보고서 생성. 나머지는 분기 검증용 stub 후보.",
    )
    main(parser.parse_args().scenario)