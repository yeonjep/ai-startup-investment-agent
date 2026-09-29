# 설계 확정 후 교체할 부분: 후보 입력과 그래프 초기 State는 실행 확인용입니다.

from pathlib import Path
import argparse
from typing import Any

from agents.common import configure_runtime
from agents.config import MAX_ITERATIONS
from agents.graph import build_graph
from agents.state import InvestmentState


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_CANDIDATES = [
    {"name": "[샘플 후보 A]"},
    {"name": "[샘플 후보 B]"},
]


def main(scenario: str = "all_hold") -> dict[str, Any]:
	if scenario not in {"invest", "all_hold"}:
		raise ValueError("scenario must be 'invest' or 'all_hold'")

	configure_runtime()
	initial_state: InvestmentState = {
		"domain": "AI 반도체",
		"candidates": DEMO_CANDIDATES,
		"current_idx": 0,
		"max_iterations": MAX_ITERATIONS,
		"scenario": scenario,
		"evaluated": [],
		"sources": [],
	}
	graph = build_graph()
	recursion_limit = max(25, min(len(DEMO_CANDIDATES), MAX_ITERATIONS) * 6 + 3)
	result = dict(initial_state)
	node_visits = []
	for update in graph.stream(
		initial_state,
		config={"recursion_limit": recursion_limit},
		stream_mode="updates",
	):
		for node_name, node_update in update.items():
			node_visits.append(node_name)
			result.update(node_update)

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
	parser = argparse.ArgumentParser(description="Run the investment-agent skeleton.")
	parser.add_argument(
		"--scenario",
		choices=("invest", "all_hold"),
		default="all_hold",
		help="Stub branch to exercise: immediate invest or hold all candidates.",
	)
	main(parser.parse_args().scenario)
