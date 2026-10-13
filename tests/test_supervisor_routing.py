import unittest

from agents.quality import check_bias, check_groundedness
from agents.supervisor import choose_route, pending_refresh_areas, supervisor


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
    def test_discovery_goes_to_startup(self):
        route, reason = choose_route({"discovery_done": False, "step_count": 0})
        self.assertEqual(route, "startup_agent")
        self.assertTrue(reason)

    def test_fail_goes_to_next_candidate(self):
        route, _reason = choose_route(_open_candidate(selection_status="FAIL", current_startup=None))
        self.assertEqual(route, "next_candidate")

    def test_missing_tech_goes_to_technology_before_market(self):
        route, _reason = choose_route(_open_candidate(tech_summary={}, market_analysis={}))
        self.assertEqual(route, "technology_agent")

    def test_refresh_dispatches_one_area_then_rescores(self):
        state = _open_candidate(
            scores={"technical_originality": {"status": "missing"}},
            missing_evidence=["technical_originality", "market_tam"],
        )
        route, reason = choose_route(state)
        self.assertEqual(route, "technology_agent")
        self.assertIn("tech", reason)

        dispatched = supervisor({**state, "next_route": "evaluator_agent"})
        self.assertEqual(dispatched["active_refresh_area"], "tech")
        self.assertEqual(dispatched["next_route"], "technology_agent")
        self.assertNotIn("evidence_retry", dispatched)

        after_tech = {
            **state,
            **dispatched,
            "tech_summary": {"summary": "보완"},
        }
        second = supervisor(after_tech)
        self.assertEqual(second["next_route"], "market_agent")
        self.assertEqual(second["evidence_retry"], 1)
        self.assertEqual(second["node_status"]["refresh:tech"], "done")

        after_market = {**after_tech, **second, "market_analysis": state["market_analysis"]}
        third = supervisor(after_market)
        self.assertEqual(third["next_route"], "evaluator_agent")
        self.assertEqual(third["route_reason"], "결측 재배정 후 재채점")

    def test_hold_with_room_continues_and_invest_goes_to_report(self):
        hold = _open_candidate(
            scores={"team_experience": {"score": 3}},
            decision="보류",
            candidates=[{"name": "에이"}, {"name": "비"}],
            evaluated=[{"decision": "보류"}],
        )
        self.assertEqual(choose_route(hold)[0], "next_candidate")
        invest = _open_candidate(scores={"team_experience": {"score": 5}}, decision="투자", evaluated=[{"decision": "투자"}])
        self.assertEqual(choose_route(invest)[0], "report_agent")

    def test_quality_failure_rewrites_once_then_stops(self):
        failed = _open_candidate(
            scores={"team_experience": {"score": 5}},
            decision="투자",
            evaluated=[{"decision": "투자"}],
            final_report="# SUMMARY\n\n판정 투자\n",
            quality_result={"passed": False, "checks": {"groundedness": {"passed": False, "reason": "수치 없음"}}},
        )
        self.assertEqual(choose_route(failed)[0], "report_agent")
        update = supervisor(failed)
        self.assertEqual(update["quality_retry"], 1)
        self.assertEqual(update["final_report"], "")
        self.assertIn("groundedness", update["quality_feedback"])

        again = _open_candidate(
            scores={"team_experience": {"score": 5}},
            decision="투자",
            evaluated=[{"decision": "투자"}],
            final_report="# SUMMARY\n\n판정 투자\n",
            quality_result={"passed": False, "checks": {}},
            quality_retry=1,
        )
        self.assertEqual(choose_route(again)[0], "end")

    def test_step_cap_ends_after_quality(self):
        state = _open_candidate(
            step_count=120,
            final_report="# SUMMARY\n",
            quality_result={"passed": True, "checks": {}},
        )
        self.assertEqual(choose_route(state)[0], "end")

    def test_pending_areas_follow_tech_market_competitor_order(self):
        areas = pending_refresh_areas(
            {"missing_evidence": ["entry_barrier", "market_tam", "technical_originality"], "node_status": {}}
        )
        self.assertEqual(areas, ["tech", "market", "competitor"])


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
