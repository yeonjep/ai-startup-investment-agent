import unittest

from agents.graph import (
    ANALYSIS_NODES,
    route_after_decision,
    route_after_evaluator,
    route_after_quality,
    route_after_startup,
    route_from_supervisor,
)
from agents.quality import check_bias, check_groundedness
from agents.state import merge_dicts
from agents.supervisor import choose_routes, pending_refresh_areas, supervisor


def _open_candidate(**extra):
    state = {
        "discovery_done": True,
        "candidates": [{"name": "에이"}],
        "current_idx": 0,
        "selection_status": "PASS",
        "current_startup": {"name": "에이"},
        "tech_summary": {"summary": "기술"},
        "market_analysis": {"insufficient_topics": [], "topics": {}},
        "competitor_analysis": {"summary": "경쟁"},
        "sufficiency_votes": {"market_agent": {"vote": "sufficient", "reason": "충분"}},
        "scores": {},
        "decision": None,
        "evidence_retry": 0,
        "missing_evidence": [],
        "node_status": {},
        "evaluated": [],
        "max_candidates": 5,
        "step_count": 0,
        "final_report": "",
        "quality_result": None,
        "quality_retry": 0,
    }
    state.update(extra)
    return state


class SupervisorRouteTests(unittest.TestCase):
    def test_pass_candidate_fans_out_to_three_analysis_workers(self):
        routes = route_after_startup(_open_candidate())
        self.assertEqual(routes, list(ANALYSIS_NODES))

    def test_fail_goes_to_next_candidate(self):
        route = route_after_startup(
            _open_candidate(selection_status="FAIL", current_startup=None)
        )
        self.assertEqual(route, "next_candidate")

    def test_empty_candidates_go_directly_to_report(self):
        route = route_after_startup(
            _open_candidate(candidates=[], current_startup=None, selection_status=None)
        )
        self.assertEqual(route, "report_agent")

    def test_evaluator_delegates_refreshable_missing_to_supervisor(self):
        state = _open_candidate(
            scores={"technical_originality": {"status": "missing"}},
            missing_evidence=["technical_originality", "market_tam"],
        )
        self.assertEqual(route_after_evaluator(state), "supervisor")
        routes, reason = choose_routes(state)
        self.assertEqual(routes, ["technology_agent", "market_agent"])
        self.assertIn("tech, market", reason)

        dispatched = supervisor(state)
        self.assertEqual(
            dispatched["next_routes"],
            ["technology_agent", "market_agent"],
        )
        self.assertEqual(dispatched["refresh_areas"], ["tech", "market"])
        self.assertEqual(dispatched["evidence_retry"], 1)
        self.assertEqual(dispatched["scores"], {})

    def test_evaluator_moves_to_decision_after_retry_budget(self):
        state = _open_candidate(
            scores={"technical_originality": {"status": "missing"}},
            missing_evidence=["technical_originality"],
            evidence_retry=1,
        )
        self.assertEqual(route_after_evaluator(state), "decision_agent")

    def test_hold_with_room_continues_and_invest_goes_to_report(self):
        hold = _open_candidate(
            scores={"team_experience": {"score": 3}},
            decision="보류",
            candidates=[{"name": "에이"}, {"name": "비"}],
            evaluated=[{"decision": "보류"}],
        )
        self.assertEqual(route_after_decision(hold), "next_candidate")
        invest = _open_candidate(
            scores={"team_experience": {"score": 5}},
            decision="투자",
            evaluated=[{"decision": "투자"}],
        )
        self.assertEqual(route_after_decision(invest), "report_agent")

    def test_quality_failure_rewrites_once_then_stops(self):
        failed = _open_candidate(
            scores={"team_experience": {"score": 5}},
            decision="투자",
            evaluated=[{"decision": "투자"}],
            final_report="# SUMMARY\n\n판정 투자\n",
            quality_result={"passed": False, "checks": {"groundedness": {"passed": False, "reason": "수치 없음"}}},
        )
        self.assertEqual(route_after_quality(failed), "supervisor")
        update = supervisor(failed)
        self.assertEqual(update["quality_retry"], 1)
        self.assertEqual(update["final_report"], "")
        self.assertIn("groundedness", update["quality_feedback"])
        self.assertEqual(route_from_supervisor(update), ["report_agent"])

        again = _open_candidate(
            scores={"team_experience": {"score": 5}},
            decision="투자",
            evaluated=[{"decision": "투자"}],
            final_report="# SUMMARY\n\n판정 투자\n",
            quality_result={"passed": False, "checks": {}},
            quality_retry=1,
        )
        self.assertEqual(route_after_quality(again), "end")

    def test_pending_areas_follow_tech_market_competitor_order(self):
        areas = pending_refresh_areas(
            {"missing_evidence": ["entry_barrier", "market_tam", "technical_originality"], "node_status": {}}
        )
        self.assertEqual(areas, ["tech", "market", "competitor"])

    def test_parallel_vote_reducer_merges_worker_keys(self):
        merged = merge_dicts(
            {"technology_agent": {"vote": "sufficient", "reason": "기술"}},
            {"market_agent": {"vote": "insufficient", "reason": "시장"}},
        )
        self.assertEqual(
            set(merged),
            {"technology_agent", "market_agent"},
        )
        self.assertEqual(merge_dicts(merged, {}), {})


class QualityRuleTests(unittest.TestCase):
    def test_groundedness_accepts_cited_quantity_and_score(self):
        state = {
            "evaluated": [
                {
                    "total_score": 72.5,
                    "missing_evidence": [],
                    "scores": {"funding_total": {"score": 3, "raw_value": "550억원"}},
                    "evidence": [{"value": "550억원", "quote": "누적 투자 550억원", "scope": ""}],
                }
            ]
        }
        ok, _reason = check_groundedness("# SUMMARY\n\n누적 투자 550억원 [1]\n\n총점 72.50\n", state)
        self.assertTrue(ok)
        failed, reason = check_groundedness("# SUMMARY\n\n시장 규모 999억 달러.\n", state)
        self.assertFalse(failed)
        self.assertIn("999", reason)

    def test_bias_fails_when_one_source_exceeds_seventy_percent(self):
        report = (
            "본문 [1] [1] [1] [2]\n\n# REFERENCE\n\n"
            "1. 기관(2024). *보고서*. 문서 ID gov_2025_strategy.\n"
            "2. 언론(2024-01-01). *기사*. 사이트, https://news.example/a.\n"
        )
        ok, reason = check_bias(report)
        self.assertFalse(ok)
        self.assertIn("70%", reason)
        spread = (
            "본문 [1] [2]\n\n# REFERENCE\n\n"
            "1. 기관(2024). *보고서*. 문서 ID gov_2025_strategy.\n"
            "2. 언론(2024-01-01). *기사*. 사이트, https://news.example/a.\n"
        )
        self.assertTrue(check_bias(spread)[0])


if __name__ == "__main__":
    unittest.main()
