# 설계 확정 후 교체할 부분: 후보와 분석 값은 그래프 분기 검증용 stub fixture입니다.

import argparse
from pathlib import Path
from typing import Any, Literal

from agents.common import configure_runtime
from agents.config import (
    MAX_CANDIDATES,
    MAX_CANDIDATE_POOL,
    MAX_EVIDENCE_RETRIES,
    MAX_RAG_RETRIES,
    MAX_SELECTION_RETRIES,
)
from agents.graph import build_graph
from agents.state import InvestmentState


PROJECT_ROOT = Path(__file__).resolve().parent
Scenario = Literal["invest", "all_hold", "zero_pass", "evidence_retry"]
DEMO_CANDIDATES = [
    {"name": "[샘플 후보 A]"},
    {"name": "[샘플 후보 B]"},
]


def main(scenario: Scenario = "all_hold") -> dict[str, Any]:
    configure_runtime()
    initial_state: InvestmentState = {"domain": "AI 반도체"}
    graph = build_graph()
    max_steps = (
        MAX_CANDIDATE_POOL * (MAX_SELECTION_RETRIES + 2)
        + MAX_CANDIDATES * (10 + MAX_RAG_RETRIES + MAX_EVIDENCE_RETRIES)
        + 10
    )
    graph_config = {
        "recursion_limit": max(25, max_steps),
        "configurable": {
            "scenario": scenario,
            "demo_candidates": DEMO_CANDIDATES,
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
            for key, value in node_update.items():
                if key == "evaluated":
                    result[key] = [*result.get(key, []), *value]
                else:
                    result[key] = value

    report_path = PROJECT_ROOT / "outputs" / f"placeholder_report_{scenario}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result["final_report"], encoding="utf-8")
    print(f"Scenario: {scenario}")
    print(f"Node visits: {' -> '.join(node_visits)}")
    print(f"Candidates processed: {len(result.get('evaluated', []))}")
    print(f"Dummy report written: {report_path}")
    result["node_visits"] = node_visits
    result["report_path"] = str(report_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DESIGNED investment graph with stubs.")
    parser.add_argument(
        "--scenario",
        choices=("invest", "all_hold", "zero_pass", "evidence_retry"),
        default="all_hold",
        help="Stub branch to exercise.",
    )
    main(parser.parse_args().scenario)