# Node internals are deterministic branch-test stubs; replace agent logic only.

import re
from datetime import date
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
from rag.retriever import grade_relevance, rag_search
from tools.web_search import web_search

TOPICS = ("market_size", "market_growth", "demand_risk")

_TOPIC_LABELS = {
    "market_size": "시장 규모",
    "market_growth": "시장 성장률",
    "demand_risk": "수요처와 시장·규제 리스크",
}


def _invoke_tool(tool: Any, **kwargs: Any) -> Any:
    """Invoke either a LangChain tool or a plain callable (useful in tests)."""
    invoke = getattr(tool, "invoke", None)
    return invoke(kwargs) if callable(invoke) else tool(**kwargs)


def _market_scope(state: InvestmentState) -> str:
    startup = state.get("current_startup") or {}
    company_type = startup.get("company_type")
    type_scope = {
        "CHIP": "AI 반도체 칩",
        "DESIGN_AI": "AI 반도체 설계 자동화 EDA",
        "PROCESS_AI": "반도체 공정 AI 수율 불량 예측",
    }.get(company_type, state.get("domain", "AI 반도체"))
    return str(startup.get("target_market") or startup.get("evaluation_product") or type_scope)


def _market_query(state: InvestmentState, topic: str, attempt: int) -> str:
    startup = state.get("current_startup") or {}
    company = startup.get("name", "대상 기업")
    scope = _market_scope(state)
    as_of = state.get("as_of_date", get_as_of_date())
    focus = {
        "market_size": "TAM 시장 규모 금액 기준연도 지역 제품 범위",
        "market_growth": "CAGR 성장률 예측기간 기준연도 지역 제품 범위",
        "demand_risk": "주요 수요처 고객군 도입 요인 시장 위험 규제 위험",
    }[topic]
    refinements = (
        "기관 보고서와 원문 통계를 우선",
        "수치의 단위·기준연도·예측기간·지역을 명시",
        "시장 범위를 좁혀 상충 수치를 구분하고 수요와 리스크를 각각 확인",
    )
    return f"{company} 타깃 {scope} {focus} {as_of} 기준 {refinements[min(attempt, 2)]}"


def _is_yes(grade: dict[str, Any]) -> bool:
    value = grade.get("relevance", grade.get("relevant", grade.get("grade", "")))
    return value is True or str(value).strip().lower() in {"yes", "y", "true", "relevant", "1"}


def _graded_chunks(
    chunks: list[dict[str, Any]], grades: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {
        str(item.get("chunk_id")): item
        for item in grades
        if item.get("chunk_id") is not None
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        grade = by_id.get(str(chunk.get("chunk_id")))
        if grade is None and index < len(grades):
            grade = grades[index]
        if grade and _is_yes(grade):
            pairs.append((chunk, grade))
    return pairs


def _grade_signals(grade: dict[str, Any]) -> str:
    values = (
        grade.get("answerable_metrics")
        or grade.get("metrics")
        or grade.get("indicators")
        or []
    )
    if isinstance(values, str):
        values = [values]
    return " ".join(str(value) for value in values)


def _topic_sufficient(
    topic: str, relevant: list[tuple[dict[str, Any], dict[str, Any]]]
) -> bool:
    if not relevant:
        return False
    text = " ".join(
        f"{chunk.get('content', '')} {_grade_signals(grade)} {grade.get('reason', '')}"
        for chunk, grade in relevant
    ).lower()
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b|기준\s*연도|예측\s*기간", text))
    has_scope = bool(
        re.search(r"글로벌|세계|한국|국내|아시아|북미|유럽|시장|market|global|region", text)
    )
    if topic == "market_size":
        has_value = bool(
            re.search(
                r"(?:[$₩€£]|usd|krw|달러|원|억원|조원|million|billion|trillion)\s*[\d,.]+|"
                r"[\d,.]+\s*(?:usd|krw|달러|원|억원|조원|million|billion|trillion)",
                text,
            )
        )
        return has_value and has_year and has_scope
    if topic == "market_growth":
        has_rate = bool(re.search(r"cagr|연평균|성장률|\d+(?:\.\d+)?\s*%", text))
        return has_rate and has_year and has_scope
    has_demand = bool(re.search(r"수요|고객|수요처|도입|채택|구매|demand|customer|adoption", text))
    has_risk = bool(re.search(r"리스크|위험|규제|제약|장벽|risk|regulat|barrier", text))
    return has_demand and has_risk


def _evidence_id(company: str, topic: str, source_key: str) -> str:
    safe_company = re.sub(r"[^0-9A-Za-z가-힣]+", "-", company).strip("-").lower()
    safe_source = re.sub(r"[^0-9A-Za-z가-힣]+", "-", source_key).strip("-").lower()
    return f"{safe_company or 'company'}-market-{topic}-{safe_source or 'source'}"


def _rag_evidence(
    company: str,
    topic: str,
    relevant: list[tuple[dict[str, Any], dict[str, Any]]],
    accessed_at: str,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for chunk, _grade in relevant:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("rank") or len(evidence) + 1)
        content = str(chunk.get("content") or "").strip()
        evidence.append(
            {
                "evidence_id": _evidence_id(company, topic, chunk_id),
                "company": company,
                "category": "market",
                "metric": topic,
                "value": None,
                "unit": None,
                "source_type": "rag",
                "title": chunk.get("title"),
                "source": chunk.get("source") or chunk.get("title"),
                "date": chunk.get("date") or chunk.get("published_at"),
                "accessed_at": accessed_at,
                "url": None,
                "doc_id": chunk.get("doc_id"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "quote": content[:500],
                "scope": chunk.get("scope"),
            }
        )
    return evidence


def _web_evidence(
    company: str, topic: str, results: list[dict[str, Any]], accessed_at: str
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for index, result in enumerate(results):
        url = str(result.get("url") or "")
        source_key = url or str(result.get("title") or index + 1)
        evidence.append(
            {
                "evidence_id": _evidence_id(company, topic, source_key),
                "company": company,
                "category": "market",
                "metric": topic,
                "value": None,
                "unit": None,
                "source_type": "web",
                "title": result.get("title"),
                "source": result.get("source") or result.get("title"),
                "date": result.get("published_at") or result.get("date"),
                "accessed_at": result.get("accessed_at") or accessed_at,
                "url": url or None,
                "doc_id": None,
                "page": None,
                "chunk_id": None,
                "quote": str(result.get("content") or "").strip()[:500],
                "scope": result.get("scope"),
            }
        )
    return evidence


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


def startup_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    scenario = _scenario(config)
    candidates = list(state.get("candidates", []))
    discovery_done = state.get("discovery_done", False)

    if not discovery_done:
        configured_candidates = config.get("configurable", {}).get("demo_candidates")
        if configured_candidates is None:
            configured_candidates = candidates or [
                {"name": "[샘플 후보 A]"},
                {"name": "[샘플 후보 B]"},
            ]
        candidates = list(configured_candidates)[:MAX_CANDIDATE_POOL]
        discovery_done = True

    current_idx = state.get("current_idx", 0)
    if not candidates or current_idx >= len(candidates):
        return {
            "candidates": candidates,
            "discovery_done": discovery_done,
            "current_startup": None,
            "selection_status": None,
        }

    candidate = candidates[current_idx]
    if scenario == "zero_pass":
        status = "FAIL"
    elif scenario == "uncertain":
        status = "REVIEW"
    else:
        status = candidate.get("selection_status", "PASS")

    selection_retry = state.get("selection_retry", 0)
    if status == "REVIEW" and selection_retry < MAX_SELECTION_RETRIES:
        selection_retry += 1
        return {
            "candidates": candidates,
            "discovery_done": discovery_done,
            "current_startup": None,
            "selection_status": "REVIEW",
            "selection_retry": selection_retry,
            "uncertain": False,
        }

    uncertain = status == "REVIEW"
    selected_startup = dict(candidate)
    selected_startup.setdefault("country", "KR")
    selected_startup.setdefault("company_type", "CHIP")
    selected_startup.setdefault("evaluation_product", selected_startup.get("name", ""))
    return {
        "candidates": candidates,
        "discovery_done": discovery_done,
        "current_startup": selected_startup if status != "FAIL" else None,
        "selection_status": status,
        "selection_retry": selection_retry,
        "uncertain": uncertain,
    }


def technology_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    return {
        "tech_summary": {"summary": f"[stub] {startup.get('name', '기업')} 기술 분석"},
        "tech_evidence": list(state.get("tech_evidence", [])),
    }


def market_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    company = str(startup.get("name") or "대상 기업")
    retries = dict(state.get("rag_retry", {topic: 0 for topic in TOPICS}))
    previous_analysis = dict(state.get("market_analysis", {}))
    previous_topics = dict(previous_analysis.get("topics", {}))
    accessed_at = str(state.get("as_of_date") or date.today().isoformat())
    evidence_by_id = {
        item["evidence_id"]: item
        for item in state.get("market_evidence", [])
        if item.get("evidence_id")
    }
    topic_results: dict[str, dict[str, Any]] = {}

    for topic in TOPICS:
        prior = dict(previous_topics.get(topic, {}))
        if prior.get("sufficient"):
            topic_results[topic] = prior
            continue
        if prior.get("web_searched") and retries.get(topic, 0) >= MAX_RAG_RETRIES:
            topic_results[topic] = prior
            continue

        relevant: list[tuple[dict[str, Any], dict[str, Any]]] = []
        query_history = list(prior.get("queries", []))
        error: str | None = None
        initial_done = bool(prior.get("initial_search_done"))

        while True:
            attempt = retries.get(topic, 0) if initial_done else 0
            query = _market_query(state, topic, attempt)
            if initial_done:
                if retries.get(topic, 0) >= MAX_RAG_RETRIES:
                    break
                retries[topic] = retries.get(topic, 0) + 1
                query = _market_query(state, topic, retries[topic])
            initial_done = True
            query_history.append(query)
            try:
                chunks = list(_invoke_tool(rag_search, query=query, k=5, filters=None) or [])
                grades = list(_invoke_tool(grade_relevance, query=query, chunks=chunks) or [])
                relevant = _graded_chunks(chunks, grades)
                error = None
            except Exception as exc:  # External model/index failures become explicit missing data.
                relevant = []
                error = f"{type(exc).__name__}: {exc}"
            sufficient = _topic_sufficient(topic, relevant)
            print(
                f"[market] topic={topic} query={query!r} "
                f"relevant_yes={len(relevant)} retry={retries.get(topic, 0)}"
            )
            for item in _rag_evidence(company, topic, relevant, accessed_at):
                evidence_by_id[item["evidence_id"]] = item
            if sufficient or retries.get(topic, 0) >= MAX_RAG_RETRIES:
                break

        web_searched = bool(prior.get("web_searched"))
        if not _topic_sufficient(topic, relevant) and retries.get(topic, 0) >= MAX_RAG_RETRIES and not web_searched:
            web_searched = True
            web_query = _market_query(state, topic, MAX_RAG_RETRIES)
            try:
                web_results = list(_invoke_tool(web_search, query=web_query) or [])
                for item in _web_evidence(company, topic, web_results, accessed_at):
                    evidence_by_id[item["evidence_id"]] = item
                web_text = " ".join(str(item.get("content") or "") for item in web_results)
                web_probe = [({"content": web_text}, {"relevance": "yes"})] if web_text else []
                sufficient = _topic_sufficient(topic, relevant + web_probe)
            except Exception as exc:
                sufficient = False
                error = f"{type(exc).__name__}: {exc}"
        else:
            sufficient = _topic_sufficient(topic, relevant) or bool(prior.get("sufficient"))

        topic_results[topic] = {
            "label": _TOPIC_LABELS[topic],
            "queries": query_history,
            "initial_search_done": initial_done,
            "relevant_count": len(relevant),
            "sufficient": sufficient,
            "web_searched": web_searched,
            "error": error,
        }

    insufficient_topics = [topic for topic in TOPICS if not topic_results[topic]["sufficient"]]
    summaries = [
        f"{_TOPIC_LABELS[topic]}: {'근거 확보' if topic_results[topic]['sufficient'] else '결측'}"
        for topic in TOPICS
    ]

    return {
        "market_analysis": {
            "summary": f"{company} 시장 분석 — " + "; ".join(summaries),
            "initial_search_done": True,
            "target_market_scope": _market_scope(state),
            "topics": topic_results,
            "insufficient_topics": insufficient_topics,
            "missing_metrics": insufficient_topics,
        },
        "rag_retry": retries,
        "market_evidence": list(evidence_by_id.values()),
    }


def competitor_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    return {
        "competitor_analysis": {"summary": f"[stub] {startup.get('name', '기업')} 경쟁 분석"},
        "competitor_evidence": list(state.get("competitor_evidence", [])),
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
