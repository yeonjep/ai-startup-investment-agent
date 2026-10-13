import unittest

from agents.report import (
    Catalog,
    _cite_numeric_sentences,
    _finalize_citations,
    _limitations,
    _market_table,
    _report_market_analysis,
    report_type,
)


def _record(decision: str = "보류") -> dict:
    return {
        "company_profile": {"name": "완료AI", "country": "한국", "company_type": "CHIP"},
        "scores": {
            "market_tam": {
                "metric": "market_tam",
                "raw_value": None,
                "score": 3.0,
                "status": "missing",
                "reason": "공개 근거 미확인",
                "evidence_ids": [],
            }
        },
        "total_score": 60.0,
        "missing_evidence": ["market_tam"],
        "decision": decision,
        "evidence": [],
    }


class ReportContractTests(unittest.TestCase):
    def test_no_target_requires_no_unfinished_eligible_candidate(self):
        self.assertEqual(
            report_type(
                {
                    "candidates": [
                        {"name": "제외AI", "selection_status": "FAIL"}
                    ],
                    "evaluated": [],
                }
            ),
            "none",
        )
        with self.assertRaisesRegex(ValueError, "판정 스냅샷"):
            report_type(
                {
                    "candidates": [
                        {"name": "미완료AI", "selection_status": "PASS"}
                    ],
                    "current_startup": {"name": "미완료AI"},
                    "selection_status": "PASS",
                    "evaluated": [],
                }
            )

    def test_completed_records_choose_invest_or_all_hold(self):
        hold = _record()
        invest = _record("투자")
        self.assertEqual(report_type({"evaluated": [hold]}), "all_hold")
        self.assertEqual(report_type({"evaluated": [hold, invest]}), "invest")

    def test_incomplete_evaluation_record_is_rejected(self):
        incomplete = {**_record(), "scores": {}, "total_score": None}
        with self.assertRaisesRegex(ValueError, "채점 결과"):
            report_type({"evaluated": [incomplete]})

    def test_missing_market_metric_is_shown_as_neutral_three(self):
        record = _record()
        table = _market_table([record], Catalog([record]))
        self.assertIn("중립 대입·미확인 (3점)", table)

    def test_limitations_distinguish_invest_from_all_hold(self):
        state = {"candidates": [{"name": "완료AI"}], "evaluated": [_record()]}
        self.assertIn("최초 투자 판정", _limitations(state, "invest"))
        self.assertNotIn("최초 투자 판정", _limitations(state, "all_hold"))

    def test_catalog_keeps_all_metric_evidence_tags(self):
        record = {
            "evidence": [
                {
                    "evidence_id": "a",
                    "category": "market",
                    "source_type": "web",
                },
                {
                    "evidence_id": "b",
                    "category": "market",
                    "source_type": "web",
                },
            ]
        }
        self.assertEqual(Catalog([record]).tag(["a", "b"]), "[E1][E2]")

    def test_numeric_sentence_uses_only_matching_evidence(self):
        record = {
            "evidence": [
                {
                    "evidence_id": "tam",
                    "category": "market",
                    "value": 100,
                    "unit": "billion USD",
                    "quote": "시장 규모는 100 billion USD",
                    "scope": "글로벌 2026",
                    "source_type": "web",
                },
                {
                    "evidence_id": "other",
                    "category": "market",
                    "value": 20,
                    "unit": "%",
                    "quote": "CAGR 20%",
                    "scope": "글로벌",
                    "source_type": "web",
                },
            ]
        }
        catalog = Catalog([record])
        cited = _cite_numeric_sentences(
            "시장 규모는 100 billion USD다.",
            catalog,
            ("market",),
        )
        self.assertIn("[E1]", cited)
        self.assertNotIn("[E2]", cited)

    def test_invalid_catalog_tag_remains_visible_for_quality_failure(self):
        finalized = _finalize_citations("근거 없는 주장 [E99]", Catalog([]))
        self.assertIn("[INVALID:E99]", finalized)

    def test_invalid_market_finding_is_hidden_from_report_narrative(self):
        record = _record()
        record["market_analysis"] = {
            "summary": "HBM CAGR 36.3%",
            "findings": {
                "market_growth": {
                    "value": 36.3,
                    "validation": "타깃 시장 불일치",
                }
            },
        }
        sanitized = _report_market_analysis(record)
        self.assertEqual(sanitized["findings"], {})
        self.assertNotIn("36.3", sanitized["summary"])


if __name__ == "__main__":
    unittest.main()
