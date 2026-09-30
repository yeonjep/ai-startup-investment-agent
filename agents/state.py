# State 계약: docs/DESIGN.md D-6 및 D-7.

import operator
from typing import Annotated, Any, TypedDict


class Evidence(TypedDict):
    evidence_id: str
    company: str
    category: str
    metric: str
    value: str | int | float
    unit: str
    source_type: str
    title: str
    source: str
    date: str
    url: str
    doc_id: str
    location: str


class InvestmentState(TypedDict, total=False):
    domain: str
    max_candidates: int
    candidates: list[dict[str, Any]]
    current_idx: int
    current_startup: dict[str, Any]
    selection_status: str
    selection_retry: int
    uncertain: bool
    tech_summary: str
    market_analysis: str
    rag_retry: int
    competitor_analysis: str
    evidence: list[Evidence]
    evidence_retry: int
    scores: dict[str, Any]
    total_score: float
    missing_evidence: list[str]
    decision: str
    evaluated: Annotated[list[dict[str, Any]], operator.add]
    final_report: str