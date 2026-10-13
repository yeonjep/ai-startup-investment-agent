# State contract: docs/DESIGN.md D-1, D-1a, and D-2.
# Payload fields hold the investment evaluation. Control fields are only what
# graph gates and the rework supervisor need. The guide's perspective schema is not used.

import operator
from typing import Annotated, Any, Literal, TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge parallel worker keys; an explicit empty update resets a round."""
    if right == {}:
        return {}
    return {**(left or {}), **(right or {})}


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


class SufficiencyVote(TypedDict):
    vote: Literal["sufficient", "insufficient"]
    reason: str


class QualityCheck(TypedDict):
    passed: bool
    reason: str


class QualityResult(TypedDict):
    passed: bool
    checks: dict[str, QualityCheck]


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
    sufficiency_votes: Annotated[dict[str, SufficiencyVote], merge_dicts]
    quality_result: QualityResult | None
    quality_retry: int
    quality_feedback: str
    trace_id: str
    step_count: int
    node_status: dict[str, str]
    last_error: str | None
    next_routes: list[str]
    route_reason: str
    refresh_areas: list[Literal["tech", "market", "competitor"]]