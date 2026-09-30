"""LLM은 사실 추출/정성 해석만 수행하며 최종 분기는 코드가 결정한다."""
from typing import Literal
from pydantic import BaseModel, Field

CriterionName = Literal['domain', 'ai_core', 'unlisted', 'funding_stage', 'no_exit', 'information']


class Candidate(BaseModel):
    name: str
    founded_year: int | None = None
    ceo: str | None = None
    latest_round: str | None = None
    total_funding: str | None = None
    urls: list[str] = Field(default_factory=list)


class Candidates(BaseModel):
    candidates: list[Candidate] = Field(max_length=10)


class Criterion(BaseModel):
    criterion: CriterionName
    status: Literal['PASS', 'FAIL', 'REVIEW']
    reason: str
    source_ids: list[str]


class Selection(BaseModel):
    criteria: list[Criterion]


class Metric(BaseModel):
    metric: Literal['team_experience','technical_headcount','customer_count','revenue_stage','funding_total','development_stage','technology_originality','market_tam','market_cagr','market_demand','risk','competitive_edge','entry_barriers']
    value: float | str | None = None
    unit: str = ''
    quote: str
    source_id: str = Field(pattern=r'^S[1-9][0-9]*$')
    raw_value: float | None = None
    raw_unit: str = ''
    value_kind: Literal['actual', 'cumulative', 'round', 'forecast', 'target', 'unknown'] = 'actual'
    observation_date: str = ''
    scope: str = ''
    period: str = ''


class Finding(BaseModel):
    topic: str
    statement: str
    quote: str
    source_id: str = Field(pattern=r'^S[1-9][0-9]*$')


class MissingMetric(BaseModel):
    metric: Literal['team_experience','technical_headcount','customer_count','revenue_stage','funding_total','development_stage','technology_originality','market_tam','market_cagr','market_demand','risk','competitive_edge','entry_barriers']
    reason: str


class Analysis(BaseModel):
    summary: str
    metrics: list[Metric]
    findings: list[Finding] = Field(default_factory=list)
    missing: list[MissingMetric] = Field(default_factory=list)


class Queries(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3)


class Relevance(BaseModel):
    relevant_source_ids: list[str]


class Rewrite(BaseModel):
    query: str


class Judgment(BaseModel):
    metric: Literal['team_experience','technical_headcount','customer_count','revenue_stage','funding_total','development_stage','technology_originality','market_tam','market_cagr','market_demand','risk','competitive_edge','entry_barriers']
    score: Literal[1, 2, 3, 4, 5]
    reason: str
    evidence_ids: list[str]


class Judgments(BaseModel):
    judgments: list[Judgment]


class ReportText(BaseModel):
    summary: str = Field(max_length=450)
    overview: str = Field(max_length=750)
    technology: str = Field(max_length=1000)
    market: str = Field(max_length=1000)
    risks: str = Field(max_length=900)


class Peer(BaseModel):
    name: str
    product: str
    workload: str
    source_id: str = Field(pattern=r'^S[1-9][0-9]*$')
    quote: str


class PeerGroup(BaseModel):
    peers: list[Peer] = Field(max_length=5)
