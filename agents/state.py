from operator import add
from typing import Annotated, Any, TypedDict


class InvestmentState(TypedDict, total=False):
    domain: str
    max_candidates: int
    candidates: list[dict]
    current_idx: int
    current_startup: dict | None
    selection_status: str
    selection_retry: int
    uncertain: bool
    tech_summary: str
    market_analysis: str
    competitor_analysis: str
    rag_retry: int
    evidence: list[dict]
    evidence_retry: int
    retry_target: str
    scores: dict[str, Any]
    total_score: float
    missing_evidence: list[str]
    decision: str
    evaluated: Annotated[list[dict], add]
    final_report: str
    report_path: str
    output_dir: str
    warnings: Annotated[list[str], add]
    diagnostics: Annotated[list[dict], add]
    quality: dict
    demo: bool


def candidate_defaults() -> dict:
    return dict(current_startup=None, selection_status='', selection_retry=0,
                uncertain=False, tech_summary='', market_analysis='',
                competitor_analysis='', rag_retry=0, evidence=[], evidence_retry=0,
                retry_target='', scores={}, total_score=0.0, missing_evidence=[], decision='')


def initialize_state(state: InvestmentState) -> dict:
    limit = state.get('max_candidates', 5)
    if not 1 <= limit <= 10:
        raise ValueError('max_candidates must be between 1 and 10')
    return {**candidate_defaults(), 'domain': state.get('domain', 'AI 반도체'),
            'max_candidates': limit, 'current_idx': 0,
            'candidates': state.get('candidates', [])[:10]}


def next_candidate(state: InvestmentState) -> dict:
    return {**candidate_defaults(), 'current_idx': state['current_idx'] + 1}
