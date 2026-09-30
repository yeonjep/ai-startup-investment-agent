import unittest

from agents.competitor import _normalized_competitor_country
from agents.report import _compact
from agents.stubs import _scope_matches_target_market


class OutputConsistencyTests(unittest.TestCase):
    def test_component_market_is_not_target_market_proxy(self):
        state = {
            "domain": "AI 반도체",
            "current_startup": {
                "company_type": "CHIP",
                "evaluation_product": "LLM 추론용 LPU 가속기",
            },
        }
        self.assertFalse(
            _scope_matches_target_market(
                state,
                "글로벌, 2022-2027년, AI 반도체 시장, HBM 매출 기준",
            )
        )
        self.assertTrue(
            _scope_matches_target_market(
                state,
                "글로벌, 2024-2030년, AI 추론 가속기 시장",
            )
        )

    def test_known_korean_competitor_country_is_corrected(self):
        self.assertEqual(
            _normalized_competitor_country("퓨리오사AI", "미국"),
            "대한민국",
        )
        self.assertEqual(
            _normalized_competitor_country("리벨리온", "미국"),
            "대한민국",
        )

    def test_report_uses_verified_funding_instead_of_discovery_hint(self):
        record = {
            "company_profile": {
                "name": "하이퍼엑셀",
                "recent_round": "Series B",
                "total_funding": "약 2,060억원",
                "evaluation_product": "Series B를 마감했다",
            },
            "scores": {
                "funding_total": {
                    "status": "observed",
                    "raw_value": {"value": 55_000_000_000, "unit": "KRW"},
                    "score": 3,
                    "reason": "검증됨",
                }
            },
            "evidence": [
                {
                    "metric": "funding_total",
                    "title": "550억원 규모 시리즈A 투자유치",
                    "quote": "약 550억 규모의 Series A 투자를 마무리했다.",
                }
            ],
            "tech_summary": {},
            "market_analysis": {},
            "competitor_analysis": {},
            "total_score": 70,
            "missing_evidence": [],
            "decision": "투자",
            "reason": "조건 충족",
        }
        profile = _compact(record)["company_profile"]
        self.assertEqual(profile["verified_funding_total"], "약 550억 원")
        self.assertEqual(profile["verified_funding_round"], "Series A")
        self.assertNotIn("total_funding", profile)
        self.assertNotIn("evaluation_product", profile)


if __name__ == "__main__":
    unittest.main()
