# State contract: docs/DESIGN.md D-1 and D-2.

import operator
from typing import Annotated, Any, Literal, TypedDict


class Evidence(TypedDict):
    evidence_id: str
    company: str
    category: Literal["profile", "tech", "market", "competitor"]
    metric: str
    value: str | int | float | None
    unit: str | None
    source_type: Literal["rag", "web"]
    title: str | None
    source: str | None
    date: str | None
    accessed_at: str | None
    url: str | None
    doc_id: str | None
    page: int | None
    chunk_id: str | None
    quote: str
    scope: str | None


class ScoreEntry(TypedDict):
    metric: str
    raw_value: Any
    score: float
    status: Literal["observed", "missing"]
    reason: str
    evidence_ids: list[str]


class EvaluationRecord(TypedDict):
    company_profile: dict[str, Any]
    selection_uncertain: bool
    tech_summary: dict[str, Any]
    market_analysis: dict[str, Any]
    competitor_analysis: dict[str, Any]
    evidence: list[Evidence]
    scores: dict[str, ScoreEntry]
    total_score: float | None
    missing_evidence: list[str]
    decision: Literal["투자", "보류"]
    reason: str


class InvestmentState(TypedDict, total=False):
    domain: str
    as_of_date: str
    max_candidates: int
    candidates: list[dict[str, Any]]
    discovery_done: bool
    current_idx: int
    current_startup: dict[str, Any] | None
    selection_status: Literal["PASS", "FAIL", "REVIEW"] | None
    selection_retry: int
    uncertain: bool
    tech_summary: dict[str, Any]
    market_analysis: dict[str, Any]
    competitor_analysis: dict[str, Any]
    tech_evidence: list[Evidence]
    market_evidence: list[Evidence]
    competitor_evidence: list[Evidence]
    rag_retry: dict[str, int]
    evidence_retry: int
    scores: dict[str, ScoreEntry]
    total_score: float | None
    missing_evidence: list[str]
    decision: Literal["투자", "보류"] | None
    evaluated: Annotated[list[EvaluationRecord], operator.add]
    final_report: str