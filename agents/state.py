# 설계 확정 후 교체할 부분: 이 State는 CLAUDE.md 4.2의 [초안]입니다.

from typing import Any, TypedDict


class Candidate(TypedDict, total=False):
    name: str


class InvestmentState(TypedDict, total=False):
    domain: str
    candidates: list[Candidate]
    current_idx: int
    current_startup: Candidate | None
    max_iterations: int
    tech_summary: str
    market_analysis: str
    competitor_analysis: str
    scores: dict[str, Any]
    total_score: float | None
    missing_evidence: list[str]
    decision: str
    scenario: str
    evaluated: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    final_report: str