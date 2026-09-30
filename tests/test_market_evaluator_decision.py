import unittest
from unittest.mock import patch

from agents.stubs import (
    MarketFinding,
    MarketFindings,
    QualitativeScore,
    QualitativeScores,
    decision_agent,
    evaluator_agent,
    market_agent,
)


def evidence(evidence_id, category, metric, value, unit=None, quote="확인된 근거"):
    return {
        "evidence_id": evidence_id,
        "company": "테스트AI",
        "category": category,
        "metric": metric,
        "value": value,
        "unit": unit,
        "source_type": "web",
        "title": "테스트 자료",
        "source": "테스트 기관",
        "date": "2026-01-01",
        "accessed_at": "2026-09-30",
        "url": "https://example.com",
        "doc_id": None,
        "page": None,
        "chunk_id": None,
        "quote": quote,
        "scope": None,
    }


class FakeStructuredLLM:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, _messages):
        if self.schema is MarketFindings:
            return MarketFindings(
                findings=[
                    MarketFinding(
                        topic="market_size",
                        summary="글로벌 시장 규모",
                        value=100,
                        unit="billion USD",
                        scope="글로벌, 2026년",
                        evidence_ids=["테스트ai-market-market_size-market-size"],
                    ),
                    MarketFinding(
                        topic="market_growth",
                        summary="시장 성장률",
                        value=20,
                        unit="%",
                        scope="글로벌, 2026~2030년",
                        evidence_ids=["테스트ai-market-market_growth-market-growth"],
                    ),
                    MarketFinding(
                        topic="demand_risk",
                        summary="데이터센터 수요와 규제 위험",
                        value="데이터센터 고객 수요 확인",
                        scope="글로벌",
                        evidence_ids=["테스트ai-market-demand_risk-demand-risk"],
                    ),
                ]
            )
        return QualitativeScores(
            scores=[
                QualitativeScore(
                    metric=metric,
                    score=5,
                    reason="5점 근거 확인",
                    evidence_ids=[evidence_id],
                )
                for metric, evidence_id in {
                    "team_experience": "team",
                    "demand_clarity": "demand",
                    "technical_originality": "originality",
                    "competitive_difference": "difference",
                    "entry_barrier": "barrier",
                    "risk_mitigation": "risk",
                }.items()
            ]
        )


class FakeLLM:
    def with_structured_output(self, schema):
        return FakeStructuredLLM(schema)


class AgentImplementationTests(unittest.TestCase):
    def test_market_agent_extracts_structured_findings(self):
        def fake_search(query, **_kwargs):
            if "시장 규모" in query:
                chunk_id = "market-size"
            elif "CAGR" in query:
                chunk_id = "market-growth"
            else:
                chunk_id = "demand-risk"
            return [
                {
                    "content": (
                        "2026년 글로벌 AI 반도체 시장은 100 billion USD이며 CAGR 20%로 "
                        "성장한다. 데이터센터 고객 수요가 있으나 수출 규제 위험이 있다."
                    ),
                    "doc_id": "deloitte_2026_outlook",
                    "title": "시장 보고서",
                    "page": 1,
                    "chunk_id": chunk_id,
                    "rank": 1,
                }
            ]

        def fake_grade(query, chunks):
            return [
                {
                    "chunk_id": chunks[0]["chunk_id"],
                    "relevance": "yes",
                    "reason": "질의에 필요한 수치와 범위 포함",
                    "answerable_metrics": [query],
                }
            ]

        state = {
            "current_startup": {
                "name": "테스트AI",
                "company_type": "CHIP",
                "evaluation_product": "AI 가속기",
            },
            "as_of_date": "2026-09-30",
            "rag_retry": {"market_size": 0, "market_growth": 0, "demand_risk": 0},
            "market_evidence": [],
        }
        with (
            patch("agents.stubs.rag_search", fake_search),
            patch("agents.stubs.grade_relevance", fake_grade),
            patch("agents.stubs.create_llm", return_value=FakeLLM()),
        ):
            result = market_agent(state, {})

        self.assertEqual(result["market_analysis"]["insufficient_topics"], [])
        self.assertEqual(result["market_analysis"]["findings"]["market_size"]["value"], 100)
        self.assertEqual(len(result["market_evidence"]), 3)

    def test_market_agent_stops_at_retry_limit_when_external_tools_fail(self):
        state = {
            "current_startup": {
                "name": "오류테스트AI",
                "company_type": "CHIP",
                "evaluation_product": "AI 가속기",
            },
            "as_of_date": "2026-09-30",
            "rag_retry": {"market_size": 0, "market_growth": 0, "demand_risk": 0},
            "market_evidence": [],
        }

        def fail(**_kwargs):
            raise ConnectionError("외부 도구 연결 실패")

        with (
            patch("agents.stubs.rag_search", fail),
            patch("agents.stubs.web_search", fail),
        ):
            result = market_agent(state, {})

        self.assertEqual(
            result["rag_retry"],
            {"market_size": 2, "market_growth": 2, "demand_risk": 2},
        )
        self.assertEqual(
            result["market_analysis"]["insufficient_topics"],
            ["market_size", "market_growth", "demand_risk"],
        )
        self.assertTrue(
            all(
                topic["web_searched"]
                for topic in result["market_analysis"]["topics"].values()
            )
        )

    def test_evaluator_calculates_all_thirteen_metrics_and_weighted_total(self):
        profile = [
            evidence("team", "profile", "team_experience", "상용화 리드 이력"),
            evidence("headcount", "profile", "technical_headcount", 100, "명"),
            evidence("traction", "profile", "customer_traction", 5, "건"),
            evidence("revenue", "profile", "revenue_stage", "반복 매출"),
            evidence("funding", "profile", "funding_total", 3000, "억원"),
        ]
        state = {
            "current_startup": {
                "name": "테스트AI",
                "company_type": "CHIP",
                "profile_evidence": profile,
            },
            "tech_evidence": [
                evidence("stage", "tech", "development_stage", "양산"),
                evidence("originality", "tech", "technical_originality", "독자 기술과 실측 효과"),
                evidence("risk", "tech", "risk_mitigation", "주요 위험 완화"),
            ],
            "market_evidence": [
                evidence("tam", "market", "market_size", None),
                evidence("cagr", "market", "market_growth", None),
                evidence("demand", "market", "demand_risk", "구체적 고객 수요"),
            ],
            "competitor_evidence": [
                evidence("difference", "competitor", "competitive_difference", "동일 조건 우위"),
                evidence("barrier", "competitor", "entry_barrier", "세 범주 확인"),
            ],
            "market_analysis": {
                "findings": {
                    "market_size": {
                        "value": 100,
                        "unit": "billion USD",
                        "evidence_ids": ["tam"],
                        "conflict": False,
                    },
                    "market_growth": {
                        "value": 20,
                        "unit": "%",
                        "evidence_ids": ["cagr"],
                        "conflict": False,
                    },
                    "demand_risk": {
                        "value": "구체적 고객 수요",
                        "evidence_ids": ["demand"],
                        "conflict": False,
                    },
                }
            },
        }
        with patch("agents.stubs.create_llm", return_value=FakeLLM()):
            result = evaluator_agent(state, {})

        self.assertEqual(len(result["scores"]), 13)
        self.assertEqual(result["missing_evidence"], [])
        self.assertAlmostEqual(result["total_score"], 100.0)

        decision_state = {**state, **result, "uncertain": False}
        decision = decision_agent(decision_state, {})
        self.assertEqual(decision["decision"], "투자")
        self.assertIn("세 가지 투자 조건을 모두 충족", decision["evaluated"][0]["reason"])

    def test_missing_evidence_uses_neutral_scores_and_forces_hold(self):
        state = {
            "current_startup": {"name": "근거부족AI", "company_type": "CHIP"},
            "tech_evidence": [],
            "market_evidence": [],
            "competitor_evidence": [],
            "market_analysis": {},
            "uncertain": False,
        }
        result = evaluator_agent(state, {})
        self.assertEqual(len(result["missing_evidence"]), 13)
        self.assertAlmostEqual(result["total_score"], 60.0)

        decision = decision_agent({**state, **result}, {})
        self.assertEqual(decision["decision"], "보류")
        self.assertIn("결측 13개", decision["evaluated"][0]["reason"])

    def test_decision_policy_uses_exact_thresholds_and_uncertainty(self):
        base = {
            "current_startup": {"name": "판정AI", "company_type": "CHIP"},
            "tech_summary": {},
            "market_analysis": {},
            "competitor_analysis": {},
            "tech_evidence": [],
            "market_evidence": [],
            "competitor_evidence": [],
            "scores": {},
            "total_score": 70.0,
        }
        cases = (
            ({**base, "missing_evidence": ["a", "b", "c", "d"], "uncertain": False}, "투자"),
            ({**base, "missing_evidence": ["a", "b", "c", "d", "e"], "uncertain": False}, "보류"),
            ({**base, "missing_evidence": [], "uncertain": True}, "보류"),
        )
        for state, expected in cases:
            with self.subTest(expected=expected, state=state):
                self.assertEqual(decision_agent(state, {})["decision"], expected)


if __name__ == "__main__":
    unittest.main()
