# Node internals are deterministic branch-test stubs; replace agent logic only.

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.config import (
    INVESTMENT_SCORE_THRESHOLD,
    MAX_CANDIDATE_POOL,
    MAX_EVIDENCE_RETRIES,
    MAX_RAG_RETRIES,
    MAX_SELECTION_RETRIES,
    MAX_MISSING_EVIDENCE_FOR_INVESTMENT,
    get_as_of_date,
    validate_max_candidates,
)
from agents.state import EvaluationRecord, Evidence, InvestmentState

TOPICS = ("market_size", "market_growth", "demand_risk")


def _scenario(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("scenario", "all_hold")


def initialize_state(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    configurable = config.get("configurable", {})
    max_candidates = validate_max_candidates(
        configurable.get("max_candidates", state.get("max_candidates"))
    )
    return {
        "domain": state.get("domain", "AI 반도체"),
        "as_of_date": configurable.get("as_of_date", get_as_of_date()),
        "max_candidates": max_candidates,
        "candidates": list(state.get("candidates", [])),
        "discovery_done": state.get("discovery_done", False),
        "current_idx": state.get("current_idx", 0),
        "current_startup": state.get("current_startup"),
        "selection_status": state.get("selection_status"),
        "selection_retry": state.get("selection_retry", 0),
        "uncertain": state.get("uncertain", False),
        "tech_summary": state.get("tech_summary", {}),
        "market_analysis": state.get("market_analysis", {}),
        "competitor_analysis": state.get("competitor_analysis", {}),
        "tech_evidence": state.get("tech_evidence", []),
        "market_evidence": state.get("market_evidence", []),
        "competitor_evidence": state.get("competitor_evidence", []),
        "rag_retry": state.get("rag_retry", {topic: 0 for topic in TOPICS}),
        "evidence_retry": state.get("evidence_retry", 0),
        "scores": state.get("scores", {}),
        "total_score": state.get("total_score"),
        "missing_evidence": state.get("missing_evidence", []),
        "decision": state.get("decision"),
        "evaluated": state.get("evaluated", []),
        "final_report": state.get("final_report", ""),
    }


from agents.startup import startup_agent
from agents.technology import technology_agent
from agents.competitor import competitor_agent


def market_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    retries = dict(state.get("rag_retry", {topic: 0 for topic in TOPICS}))
    insufficient_topics: list[str] = []

    previous_analysis = state.get("market_analysis", {})
    initial_search_done = previous_analysis.get("initial_search_done", False)
    if _scenario(config) == "evidence_retry":
        topic = "market_growth"
        if not initial_search_done:
            insufficient_topics = [topic]
        elif retries[topic] < MAX_RAG_RETRIES:
            retries[topic] += 1
            insufficient_topics = [topic]

    return {
        "market_analysis": {
            "summary": f"[stub] {startup.get('name', '기업')} 시장 분석",
            "initial_search_done": True,
            "insufficient_topics": insufficient_topics,
        },
        "rag_retry": retries,
        "market_evidence": list(state.get("market_evidence", [])),
    }


def evaluator_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    scenario = _scenario(config)
    if scenario == "evidence_retry" and state.get("evidence_retry", 0) == 0:
        return {
            "scores": {},
            "total_score": 65.0,
            "missing_evidence": ["market_cagr"],
        }

    total_score = 75.0 if scenario in {"invest", "uncertain", "evidence_retry"} else 60.0
    return {
        "scores": {},
        "total_score": total_score,
        "missing_evidence": [],
    }


def _refresh_area_for_metric(metric: str) -> set[str]:
    mapping = {
        "development_stage": {"tech"},
        "technical_originality": {"tech"},
        "market_tam": {"market"},
        "market_cagr": {"market"},
        "demand_clarity": {"market"},
        "competitive_difference": {"competitor"},
        "entry_barrier": {"competitor"},
        "risk_mitigation": {"tech", "market"},
    }
    return mapping.get(metric, set())


def evidence_refresh(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    missing = state.get("missing_evidence", [])
    needed_areas = set().union(*(_refresh_area_for_metric(metric) for metric in missing))
    refresh_state = {**state, "evidence_retry": min(MAX_EVIDENCE_RETRIES, state.get("evidence_retry", 0) + 1)}
    update: dict[str, Any] = {"evidence_retry": refresh_state["evidence_retry"]}

    if "tech" in needed_areas:
        update.update(technology_agent(refresh_state, config))
    if "market" in needed_areas:
        update.update(market_agent(refresh_state, config))
    if "competitor" in needed_areas:
        update.update(competitor_agent(refresh_state, config))

        # update["missing_evidence"] = []  # This line is removed to preserve D-3 missing evidence rule
    return update


def decision_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    from agents.config import INVESTMENT_SCORE_THRESHOLD, MAX_MISSING_EVIDENCE_FOR_INVESTMENT

    total_score = state.get("total_score")
    missing_evidence = list(state.get("missing_evidence", []))
    decision = (
        "투자"
        if total_score is not None
        and total_score >= INVESTMENT_SCORE_THRESHOLD
        and len(missing_evidence) < MAX_MISSING_EVIDENCE_FOR_INVESTMENT
        and not state.get("uncertain", False)
        else "보류"
    )
    startup = state.get("current_startup") or {}
    evidence = [
        *state.get("tech_evidence", []),
        *state.get("market_evidence", []),
        *state.get("competitor_evidence", []),
        *startup.get("profile_evidence", []),
    ]
    record: EvaluationRecord = {
        "company_profile": dict(startup),
        "selection_uncertain": state.get("uncertain", False),
        "tech_summary": dict(state.get("tech_summary", {})),
        "market_analysis": dict(state.get("market_analysis", {})),
        "competitor_analysis": dict(state.get("competitor_analysis", {})),
        "evidence": evidence,
        "scores": dict(state.get("scores", {})),
        "total_score": total_score,
        "missing_evidence": missing_evidence,
        "decision": decision,
        "reason": "[stub] D-1/C-3 deterministic decision path",
    }
    return {"decision": decision, "evaluated": [record]}


def next_candidate(state: InvestmentState) -> dict[str, Any]:
    return {
        "current_idx": state.get("current_idx", 0) + 1,
        "current_startup": None,
        "selection_status": None,
        "selection_retry": 0,
        "uncertain": False,
        "tech_summary": {},
        "market_analysis": {},
        "competitor_analysis": {},
        "tech_evidence": [],
        "market_evidence": [],
        "competitor_evidence": [],
        "rag_retry": {topic: 0 for topic in TOPICS},
        "evidence_retry": 0,
        "scores": {},
        "total_score": None,
        "missing_evidence": [],
        "decision": None,
    }


def report_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, str]:
    evaluated = state.get("evaluated", [])
    records = [
        f"- {record['company_profile'].get('name', '기업')}: {record['decision']} "
        f"(점수: {record['total_score']})"
        for record in evaluated
    ] or ["- 분석 대상 기업이 없습니다."]
    return {
        "final_report": "\n".join(
            [
                "# SUMMARY",
                "",
                "D-4 graph 분기 확인용 더미 보고서입니다.",
                f"평가 기준일: {state.get('as_of_date', '미설정')}",
                "",
                *records,
                "",
                "# REFERENCE",
                "",
                "실제 활용 자료 없음 (stub).",
            ]
        )
    }