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
        "sufficiency_votes": {
            worker: {"vote": "sufficient", "reason": "충분"}
            for worker in ("technology_agent", "market_agent", "competitor_agent")
        },
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

    def test_review_retries_once_then_enters_parallel_analysis(self):
        retry = _open_candidate(
            selection_status="REVIEW",
            current_startup=None,
            selection_retry=1,
        )
        self.assertEqual(route_after_startup(retry), "startup_agent")
        resolved = _open_candidate(
            selection_status="REVIEW",
            current_startup={"name": "에이"},
            selection_retry=1,
            uncertain=True,
        )
        self.assertEqual(route_after_startup(resolved), list(ANALYSIS_NODES))

    def test_empty_candidates_go_directly_to_report(self):
        route = route_after_startup(
            _open_candidate(candidates=[], current_startup=None, selection_status=None)
        )
        self.assertEqual(route, "report_agent")

    def test_evaluator_always_delegates_to_evidence_supervisor(self):
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

    def test_supervisor_moves_to_decision_after_retry_budget(self):
        state = _open_candidate(
            scores={"technical_originality": {"status": "missing"}},
            missing_evidence=["technical_originality"],
            evidence_retry=1,
        )
        self.assertEqual(route_after_evaluator(state), "supervisor")
        self.assertEqual(choose_routes(state)[0], ["decision_agent"])
        update = supervisor(state)
        self.assertEqual(update["evidence_gate_status"], "insufficient_accepted")

    def test_insufficient_vote_triggers_reanalysis_without_missing_metric(self):
        state = _open_candidate(
            scores={"team_experience": {"status": "observed"}},
            sufficiency_votes={
                "technology_agent": {"vote": "sufficient", "reason": "충분"},
                "market_agent": {"vote": "insufficient", "reason": "시장 부족"},
                "competitor_agent": {"vote": "sufficient", "reason": "충분"},
            },
        )
        self.assertEqual(choose_routes(state)[0], ["market_agent"])

    def test_step_count_never_forces_incomplete_report(self):
        state = _open_candidate(
            scores={"team_experience": {"status": "observed"}},
            step_count=120,
        )
        self.assertEqual(choose_routes(state)[0], ["decision_agent"])
        self.assertEqual(supervisor(state)["evidence_gate_status"], "sufficient")

    def test_non_refreshable_missing_is_explicitly_accepted(self):
        state = _open_candidate(
            scores={"customer_traction": {"status": "missing"}},
            missing_evidence=["customer_traction"],
        )
        routes, reason = choose_routes(state)
        self.assertEqual(routes, ["decision_agent"])
        self.assertIn("재분석 대상 외 결측", reason)
        self.assertEqual(
            supervisor(state)["evidence_gate_status"],
            "insufficient_accepted",
        )

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

    def test_groundedness_rejects_wrong_number_even_with_citation(self):
        state = {
            "evaluated": [
                {
                    "total_score": 60.0,
                    "missing_evidence": [],
                    "scores": {},
                    "evidence": [
                        {
                            "source_type": "web",
                            "url": "https://market.example/report",
                            "value": 550,
                            "unit": "억원",
                            "quote": "누적 투자 550억원",
                            "scope": "누적 투자",
                        }
                    ],
                }
            ]
        }
        report = (
            "# SUMMARY\n\n시장 규모 999억원 [1]\n\n# REFERENCE\n\n"
            "1. 기관(2026-01-01). *자료*. 기관, https://market.example/report.\n"
        )
        ok, reason = check_groundedness(report, state)
        self.assertFalse(ok)
        self.assertIn("999", reason)

    def test_groundedness_rejects_invalid_catalog_tag(self):
        state = {
            "evaluated": [
                {
                    "total_score": 60.0,
                    "missing_evidence": [],
                    "scores": {},
                    "evidence": [],
                }
            ]
        }
        ok, reason = check_groundedness("주장 [INVALID:E9]", state)
        self.assertFalse(ok)
        self.assertIn("없는 인용", reason)

    def test_groundedness_matches_rag_reference_by_official_url(self):
        state = {
            "evaluated": [
                {
                    "total_score": 60.0,
                    "missing_evidence": [],
                    "scores": {},
                    "evidence": [
                        {
                            "source_type": "rag",
                            "doc_id": "gov_strategy",
                            "url": "https://gov.example/strategy",
                            "value": 8124,
                            "unit": "억원",
                            "quote": "AI 반도체 분야 8,124억원 투자",
                            "scope": "2020~2025년",
                        }
                    ],
                }
            ]
        }
        report = (
            "# SUMMARY\n\n정부는 8,124억원을 투자했다 [1].\n\n"
            "# REFERENCE\n\n1. 정부(2025). *전략*. "
            "[원문 링크](https://gov.example/strategy).\n"
        )
        ok, reason = check_groundedness(report, state)
        self.assertTrue(ok, reason)

    def test_groundedness_accepts_formatted_krw_value(self):
        state = {
            "evaluated": [
                {
                    "total_score": 60.0,
                    "missing_evidence": [],
                    "scores": {},
                    "evidence": [
                        {
                            "source_type": "web",
                            "url": "https://funding.example/round",
                            "value": 111_000_000_000,
                            "unit": "KRW",
                            "quote": "누적 투자액이 증가했다",
                            "scope": "누적 투자",
                        }
                    ],
                }
            ]
        }
        report = (
            "# SUMMARY\n\n누적 투자액은 1,110억원이다 [1].\n\n"
            "# REFERENCE\n\n1. 기관(2026). *투자*. "
            "https://funding.example/round.\n"
        )
        ok, reason = check_groundedness(report, state)
        self.assertTrue(ok, reason)

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
