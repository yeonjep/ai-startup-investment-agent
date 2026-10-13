import unittest
from unittest.mock import patch

from agents.quality import check_groundedness, _has_unmarked_conflict
from agents.report import _conflict_lines, _regulation_text, _same_company
from agents.stubs import (
    _extract_market_span,
    _has_measured_third_party,
    _market_size_storage,
    _reference_figures,
)
from agents.supervisor import choose_routes, supervisor


class ReviewFixTests(unittest.TestCase):
    def test_percent_is_not_stored_as_market_size(self):
        stored = _market_size_storage(36.3, "%")
        self.assertIsNone(stored["value"])
        self.assertEqual(stored["validation"], "미확인")
        self.assertEqual(_market_size_storage(None, None)["validation"], "미확인")
        self.assertEqual(_market_size_storage(100, "USD")["value"], 100)

    def test_reference_period_comes_only_from_quote(self):
        figures = _reference_figures(
            {
                "market_size": {
                    "value": 10,
                    "unit": "USD",
                    "validation": "타깃 시장과 범위가 다름",
                    "evidence_ids": ["e1"],
                    "scope": "2020",
                }
            },
            [{"evidence_id": "e1", "quote": "글로벌 HBM 시장은 2022-2027년 10달러다.", "source_type": "rag"}],
        )
        self.assertEqual(figures[0]["period"], "2022–2027")
        self.assertIn("시장", figures[0]["market_scope"])
        self.assertEqual(figures[0]["note"], "참고(범위 상이)")
        self.assertEqual(_extract_market_span("금액만 적혀 있다")[1], "")

    def test_unpublished_benchmark_is_not_verified(self):
        evidence = [{"evidence_id": "a", "quote": "독립 벤치마크 미공개", "scope": "", "value": ""}]
        self.assertFalse(_has_measured_third_party(evidence, ["a"]))

    def test_only_aligned_amounts_conflict(self):
        prose = {
            "company_profile": {"name": "에이"},
            "scores": {},
            "evidence": [
                {"metric": "technical_originality", "value": "독자 회로 설계", "unit": None},
                {"metric": "technical_originality", "value": "저전력 구조", "unit": None},
            ],
        }
        self.assertFalse(_has_unmarked_conflict("본문", {"evaluated": [prose]}))
        money = {
            "company_profile": {"name": "에이"},
            "scores": {},
            "evidence": [
                {"metric": "funding_total", "value": 520, "unit": "억원", "source": "A", "date": "2026-01-01"},
                {"metric": "funding_total", "value": 120, "unit": "억원", "source": "B", "date": "2026-01-01"},
            ],
        }
        self.assertTrue(_has_unmarked_conflict("본문", {"evaluated": [money]}))
        line = _conflict_lines([money])[0]
        self.assertIn("520", line)
        self.assertIn("120", line)
        self.assertIn("미확인", line)

    def test_parenthetical_alias_matches_short_name(self):
        self.assertTrue(_same_company("아이에이치더블유 (iHW)", "iHW"))

    def test_regulation_sentence_is_not_clipped(self):
        class _Tags:
            def tag(self, evidence_ids):
                return "[1]"

        policy = [{
            "evidence": [{"metric": "demand_risk", "quote": "AI 반도체 이니셔티브 전략을", "evidence_id": "p"}]
        }]
        self.assertEqual(_regulation_text(policy, "", _Tags()), "공개 근거 미확인")
        risk = [{
            "evidence": [{"metric": "demand_risk", "quote": "이 기업은 수출 통제 대상이다.", "evidence_id": "r"}]
        }]
        with patch("agents.report._complete_regulation_summary", return_value="수출 통제로 공급이 제한될 수 있다."):
            text = _regulation_text(risk, "", _Tags())
        self.assertNotIn("…", text)
        self.assertFalse(text.replace("[1]", "").strip().endswith(("을", "를", "은", "는", "이", "가")))

    def test_groundedness_rejects_mismatched_quantity(self):
        state = {
            "evaluated": [{
                "total_score": 70,
                "missing_evidence": [],
                "scores": {},
                "evidence": [{"value": 50, "unit": "명", "quote": "기술 인력 50명", "scope": ""}],
            }]
        }
        reference = "# REFERENCE\n\n1. 기관(2026). *자료*. https://example.com/a.\n"
        growth, reason = check_groundedness(f"# SUMMARY\n\n성장률 50% [1]\n\n{reference}", state)
        self.assertFalse(growth)
        self.assertIn("50", reason)
        invented, invented_reason = check_groundedness(f"# SUMMARY\n\n성장률 999% [1]\n\n{reference}", state)
        self.assertFalse(invented)
        self.assertIn("999", invented_reason)
        table, table_reason = check_groundedness(
            f"# SUMMARY\n\n| 창업자/팀 | 기술 인력 수 | 999% | 3 | 근거 |\n\n{reference}",
            state,
        )
        self.assertFalse(table)
        self.assertIn("999", table_reason)

    def test_step_cap_still_reaches_quality(self):
        capped = {
            "step_count": 80,
            "max_step_count": 80,
            "final_report": "있음",
            "quality_result": None,
            "quality_retry": 0,
        }
        self.assertEqual(choose_routes(capped)[0], ["quality_agent"])
        failed = {
            **capped,
            "quality_result": {"passed": False, "checks": {"groundedness": {"passed": False, "reason": "불일치"}}},
        }
        self.assertEqual(choose_routes(failed)[0], ["report_agent"])
        update = supervisor(failed)
        self.assertIn("groundedness", update["quality_feedback"])


if __name__ == "__main__":
    unittest.main()
