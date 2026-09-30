# Node internals are deterministic branch-test stubs; replace agent logic only.

import re
from datetime import date
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.config import (
    INVESTMENT_SCORE_THRESHOLD,
    MAX_CANDIDATE_POOL,
    MAX_EVIDENCE_RETRIES,
    MAX_RAG_RETRIES,
    MAX_SELECTION_RETRIES,
    MAX_MISSING_EVIDENCE_FOR_INVESTMENT,
    SCORE_WEIGHTS,
    USD_KRW_FIXED_RATE,
    get_as_of_date,
    validate_max_candidates,
)
from agents.state import EvaluationRecord, Evidence, InvestmentState, ScoreEntry
from rag.doc_meta import DOC_META
from rag.retriever import grade_relevance, rag_search
from tools.web_search import web_search

TOPICS = ("market_size", "market_growth", "demand_risk")

_TOPIC_LABELS = {
    "market_size": "시장 규모",
    "market_growth": "시장 성장률",
    "demand_risk": "수요처와 시장·규제 리스크",
}

_TOPIC_TO_METRICS = {
    "market_size": ("market_tam",),
    "market_growth": ("market_cagr",),
    "demand_risk": ("demand_clarity", "risk_mitigation"),
}

_METRIC_GROUPS = {
    "team": ("team_experience", "technical_headcount"),
    "market": ("market_tam", "market_cagr", "demand_clarity"),
    "product_technology": ("development_stage", "technical_originality"),
    "competitive_advantage": ("competitive_difference", "entry_barrier"),
    "traction": ("customer_traction", "revenue_stage"),
    "funding_risk": ("funding_total", "risk_mitigation"),
}

_QUALITATIVE_METRICS = {
    "team_experience",
    "demand_clarity",
    "technical_originality",
    "competitive_difference",
    "entry_barrier",
    "risk_mitigation",
}


class MarketFinding(BaseModel):
    topic: Literal["market_size", "market_growth", "demand_risk"]
    summary: str
    value: float | str | None = None
    unit: str | None = None
    scope: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    conflict: bool = False


class MarketFindings(BaseModel):
    findings: list[MarketFinding]


class QualitativeScore(BaseModel):
    metric: Literal[
        "team_experience",
        "demand_clarity",
        "technical_originality",
        "competitive_difference",
        "entry_barrier",
        "risk_mitigation",
    ]
    score: Literal[1, 3, 5]
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)


class QualitativeScores(BaseModel):
    scores: list[QualitativeScore]


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
    return str(startup.get("target_market") or type_scope)


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
    metadata_by_id = {str(item["doc_id"]): item for item in DOC_META.values()}
    for chunk, _grade in relevant:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("rank") or len(evidence) + 1)
        content = str(chunk.get("content") or "").strip()
        document_metadata = metadata_by_id.get(str(chunk.get("doc_id")), {})
        evidence.append(
            {
                "evidence_id": _evidence_id(company, topic, chunk_id),
                "company": company,
                "category": "market",
                "metric": topic,
                "value": None,
                "unit": None,
                "source_type": "rag",
                "title": chunk.get("title") or document_metadata.get("title"),
                "source": chunk.get("source") or document_metadata.get("source"),
                "date": chunk.get("date") or document_metadata.get("date"),
                "accessed_at": accessed_at,
                "url": document_metadata.get("url"),
                "doc_id": chunk.get("doc_id"),
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "quote": content[:500],
                "scope": chunk.get("scope"),
            }
        )
    return evidence


def _market_findings(
    state: InvestmentState,
    topic_results: dict[str, dict[str, Any]],
    evidence: list[Evidence],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    evidence_by_topic = {
        topic: [item for item in evidence if item.get("metric") == topic]
        for topic in TOPICS
    }
    topics_with_evidence = [topic for topic, items in evidence_by_topic.items() if items]
    if not topics_with_evidence:
        return {}, None

    allowed_ids = {
        topic: {item["evidence_id"] for item in items}
        for topic, items in evidence_by_topic.items()
    }
    evidence_payload = {
        topic: [
            {
                "evidence_id": item["evidence_id"],
                "quote": item["quote"],
                "source": item.get("source"),
                "date": item.get("date"),
                "scope": item.get("scope"),
            }
            for item in items
        ]
        for topic, items in evidence_by_topic.items()
        if items
    }
    try:
        response = create_llm(temperature=0).with_structured_output(MarketFindings).invoke(
            [
                (
                    "system",
                    "You extract investment-market facts only from supplied evidence. "
                    "Return one finding per supplied topic. Write summary in Korean. For market_size, keep the numeric "
                    "value exactly as written in the evidence and its original currency/scale unit. For market_growth, return CAGR "
                    "as a percentage number. For demand_risk, summarize concrete target customers, "
                    "demand drivers, and market/regulatory risks; value may be text. Include scope "
                    "(region, base year/forecast period, product definition). Cite only exact supplied "
                    "evidence_ids. If figures have incompatible scope or unresolved conflict, set "
                    "conflict=true and do not choose or combine a figure. Do not infer missing facts.",
                ),
                (
                    "human",
                    f"Company: {(state.get('current_startup') or {}).get('name')}\n"
                    f"Target market: {_market_scope(state)}\n"
                    f"As-of date: {state.get('as_of_date')}\n"
                    f"Topic retrieval status: {topic_results}\n"
                    f"Evidence: {evidence_payload}",
                ),
            ]
        )
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"

    findings: dict[str, dict[str, Any]] = {}
    for finding in response.findings:
        if finding.topic not in topics_with_evidence:
            continue
        valid_ids = [
            evidence_id
            for evidence_id in finding.evidence_ids
            if evidence_id in allowed_ids[finding.topic]
        ]
        if not valid_ids:
            continue
        item = finding.model_dump()
        item["evidence_ids"] = valid_ids
        quotes = " ".join(
            str(e.get("quote") or "") for e in evidence_by_topic[finding.topic] if e["evidence_id"] in valid_ids
        )
        numeric_topic = finding.topic in {"market_size", "market_growth"}
        if numeric_topic and finding.value is not None:
            if not _number_in_text(finding.value, quotes):
                item["conflict"] = True  # 인용 원문에 없는 수치는 채택하지 않는다
                item["validation"] = "수치가 인용 원문에 없음"
            elif not finding.scope or (finding.topic == "market_size" and not finding.unit):
                item["validation"] = "시장 범위·기준연도·단위 미확인"
        item["sufficient"] = bool(
            topic_results[finding.topic].get("sufficient")
            and not item["conflict"]
            and finding.value is not None
            and not item.get("validation")
        )
        findings[finding.topic] = item
    return findings, None


def _number_in_text(value: Any, text: str) -> bool:
    number = _number(value)
    if number is None:
        return False
    found = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    return any(abs(number - n) <= 1e-9 * max(1.0, abs(n)) for n in found)


def _all_evidence(state: InvestmentState) -> list[Evidence]:
    startup = state.get("current_startup") or {}
    evidence = [
        *startup.get("profile_evidence", []),
        *state.get("tech_evidence", []),
        *state.get("market_evidence", []),
        *state.get("competitor_evidence", []),
    ]
    by_id: dict[str, Evidence] = {}
    for item in evidence:
        evidence_id = item.get("evidence_id")
        if evidence_id:
            by_id[str(evidence_id)] = item
    return list(by_id.values())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    return float(match.group(0).replace(",", "")) if match else None


def _scaled_money(value: Any, unit: Any, target_currency: Literal["USD", "KRW"]) -> float | None:
    amount = _number(value)
    if amount is None:
        return None
    # 값에 "1,228억"처럼 배율이 붙어 오는 경우도 있으므로 값 문자열과 단위를 함께 본다
    scale_source = str(value) if isinstance(value, str) else ""
    unit_text = f"{scale_source}{unit or ''}".lower().replace(" ", "")
    multiplier = 1.0
    if any(token in unit_text for token in ("trillion", "조달러")):
        multiplier = 1_000_000_000_000.0
    elif any(token in unit_text for token in ("billion", "십억달러")):
        multiplier = 1_000_000_000.0
    elif "억달러" in unit_text:
        multiplier = 100_000_000.0
    elif any(token in unit_text for token in ("million", "백만달러")):
        multiplier = 1_000_000.0
    elif "조원" in unit_text:
        multiplier = 1_000_000_000_000.0
    elif "억원" in unit_text:
        multiplier = 100_000_000.0

    amount *= multiplier
    is_usd = any(token in unit_text for token in ("usd", "$", "달러"))
    is_krw = any(token in unit_text for token in ("krw", "₩", "원"))
    if not is_usd and not is_krw:
        return None
    if target_currency == "USD":
        return amount if is_usd else amount / USD_KRW_FIXED_RATE
    return amount if is_krw else amount * USD_KRW_FIXED_RATE


def _missing_score(metric: str, reason: str = "평가 가능한 공개 근거가 확인되지 않음") -> ScoreEntry:
    return {
        "metric": metric,
        "raw_value": None,
        "score": 3.0,
        "status": "missing",
        "reason": reason,
        "evidence_ids": [],
    }


def _observed_score(
    metric: str,
    raw_value: Any,
    score: float,
    reason: str,
    evidence_ids: list[str],
) -> ScoreEntry:
    return {
        "metric": metric,
        "raw_value": raw_value,
        "score": float(score),
        "status": "observed",
        "reason": reason,
        "evidence_ids": evidence_ids,
    }


def _threshold_score(value: float, boundaries: tuple[float, float, float, float]) -> int:
    if value >= boundaries[3]:
        return 5
    if value >= boundaries[2]:
        return 4
    if value >= boundaries[1]:
        return 3
    if value >= boundaries[0]:
        return 2
    return 1


def _evidence_for_metric(
    metric: str,
    evidence: list[Evidence],
    market_analysis: dict[str, Any],
) -> tuple[list[Evidence], dict[str, Any] | None]:
    direct = [item for item in evidence if item.get("metric") == metric]
    topic = {
        "market_tam": "market_size",
        "market_cagr": "market_growth",
        "demand_clarity": "demand_risk",
        "risk_mitigation": "demand_risk",
    }.get(metric)
    finding = dict(market_analysis.get("findings", {}).get(topic, {})) if topic else None
    if finding:
        referenced = set(finding.get("evidence_ids", []))
        direct.extend(item for item in evidence if item.get("evidence_id") in referenced)
    deduplicated = {item["evidence_id"]: item for item in direct if item.get("evidence_id")}
    return list(deduplicated.values()), finding


def _development_stage_score(company_type: Any, value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    stage_terms = {
        "CHIP": (
            (5, ("양산", "mass production")),
            (4, ("테이프아웃", "tape-out", "tapeout")),
            (3, ("fpga", "시험 칩", "시험칩", "prototype", "시제품")),
            (2, ("설계", "design")),
            (1, ("구상", "concept")),
        ),
        "DESIGN_AI": (
            (5, ("상용 운영", "commercial operation", "production use")),
            (4, ("고객 설계", "workflow poc", "customer poc", "poc 완료")),
            (3, ("프로토타입", "prototype")),
            (2, ("개발 중", "in development")),
            (1, ("구상", "concept")),
        ),
        "PROCESS_AI": (
            (5, ("상용 운영", "commercial operation", "production use")),
            (4, ("현장 poc", "생산 데이터", "customer poc", "poc 완료")),
            (3, ("프로토타입", "오프라인 데이터", "prototype")),
            (2, ("개발 중", "in development")),
            (1, ("구상", "concept")),
        ),
    }
    for score, terms in stage_terms.get(str(company_type), ()):
        if any(term in text for term in terms):
            return score
    return None


def _revenue_stage_score(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if any(term in text for term in ("반복 매출", "recurring", "상용 매출", "commercial revenue")):
        return 5
    if any(term in text for term in ("초기 매출", "일회성", "initial revenue", "one-time")):
        return 3
    if any(term in text for term in ("매출 없음", "매출 미발생", "no revenue", "pre-revenue")):
        return 1
    return None


def _score_deterministic_metric(
    metric: str,
    company_type: Any,
    evidence: list[Evidence],
    finding: dict[str, Any] | None,
) -> ScoreEntry:
    evidence_ids = [item["evidence_id"] for item in evidence]
    if finding and finding.get("conflict"):
        return _missing_score(metric, "범위가 다른 수치 또는 상충 근거가 해소되지 않음")
    if not finding:
        observed_values = {
            (str(item.get("value")), str(item.get("unit")))
            for item in evidence
            if item.get("value") is not None
        }
        if len(observed_values) > 1:
            return _missing_score(metric, "서로 다른 관측값이 있어 하나의 값으로 확정할 수 없음")
    raw_value = finding.get("value") if finding else (evidence[0].get("value") if evidence else None)
    unit = finding.get("unit") if finding else (evidence[0].get("unit") if evidence else None)
    if not evidence_ids:
        return _missing_score(metric)

    score: int | None = None
    normalized: Any = raw_value
    if metric == "technical_headcount":
        value = _number(raw_value)
        if value is not None:
            normalized = value
            score = _threshold_score(value, (10, 20, 50, 100))
    elif metric == "market_tam":
        value = _scaled_money(raw_value, unit, "USD")
        if value is not None:
            normalized = {"value": value, "unit": "USD", "source_value": raw_value, "source_unit": unit}
            score = _threshold_score(value, (1e9, 1e10, 5e10, 1e11))
    elif metric == "market_cagr":
        value = _number(raw_value)
        if value is not None:
            normalized = {"value": value, "unit": "%"}
            score = _threshold_score(value, (5, 10, 15, 20))
    elif metric == "development_stage":
        score = _development_stage_score(company_type, raw_value)
    elif metric == "customer_traction":
        value = _number(raw_value)
        if value is not None:
            normalized = value
            score = _threshold_score(value, (1, 2, 3, 5))
    elif metric == "revenue_stage":
        score = _revenue_stage_score(raw_value)
    elif metric == "funding_total":
        value = _scaled_money(raw_value, unit, "KRW")
        if value is not None:
            normalized = {"value": value, "unit": "KRW", "source_value": raw_value, "source_unit": unit}
            score = _threshold_score(value, (1e10, 3e10, 1e11, 3e11))

    if score is None:
        return _missing_score(metric, "근거는 있으나 설계 기준에 맞는 값·단위·범주를 확정할 수 없음")
    return _observed_score(
        metric,
        normalized,
        score,
        f"설계된 정량·범주 구간을 코드로 적용하여 {score}점 산출",
        evidence_ids,
    )


def _score_qualitative_metrics(
    state: InvestmentState,
    evidence: list[Evidence],
) -> dict[str, ScoreEntry]:
    market_analysis = state.get("market_analysis", {})
    payload: dict[str, list[dict[str, Any]]] = {}
    allowed_ids: dict[str, set[str]] = {}
    raw_values: dict[str, list[Any]] = {}
    results: dict[str, ScoreEntry] = {}
    for metric in sorted(_QUALITATIVE_METRICS):
        metric_evidence, _ = _evidence_for_metric(metric, evidence, market_analysis)
        allowed_ids[metric] = {item["evidence_id"] for item in metric_evidence}
        if not metric_evidence:
            results[metric] = _missing_score(metric)
            continue
        payload[metric] = [
            {
                "evidence_id": item["evidence_id"],
                "value": item.get("value"),
                "unit": item.get("unit"),
                "quote": item.get("quote"),
                "scope": item.get("scope"),
            }
            for item in metric_evidence
        ]
        raw_values[metric] = [item.get("value") for item in metric_evidence]

    if not payload:
        return results

    company_type = (state.get("current_startup") or {}).get("company_type")
    try:
        response = create_llm(temperature=0).with_structured_output(QualitativeScores).invoke(
            [
                (
                    "system",
                    "Score only the requested qualitative investment metrics from supplied evidence. "
                    "Use only 1, 3, or 5 and cite exact evidence_ids. Apply these rubrics: "
                    "team_experience: commercialization/customer-site lead experience=5, related "
                    "experience=3, confirmed no related experience=1; demand_clarity: concrete target "
                    "and demand evidence=5, target only=3, confirmed unclear=1; technical_originality: "
                    "proprietary method plus comparable validated effect=5, claim only=3, confirmed no "
                    "difference=1; competitive_difference: same-condition advantage proven=5, claim "
                    "only=3, confirmed none=1; entry_barrier: all three type-specific IP/product-data/"
                    "integration-partnership categories=5, one or two=3, confirmed none=1; "
                    "risk_mitigation: major risks mitigated=5, partially mitigated=3, fatal unresolved "
                    "risk=1. Missing information is not negative evidence. Do not invent facts. "
                    "Write every reason in Korean, one or two sentences.",
                ),
                (
                    "human",
                    f"Company type: {company_type}\nMetrics and evidence: {payload}",
                ),
            ]
        )
    except Exception as exc:
        reason = f"정성 채점 모델 호출 실패: {type(exc).__name__}"
        for metric in payload:
            results[metric] = _missing_score(metric, reason)
        return results

    returned: set[str] = set()
    for item in response.scores:
        metric = item.metric
        if metric not in payload or metric in returned:
            continue
        valid_ids = [evidence_id for evidence_id in item.evidence_ids if evidence_id in allowed_ids[metric]]
        if not valid_ids:
            results[metric] = _missing_score(metric, "정성 점수가 실제 Evidence와 연결되지 않음")
            continue
        results[metric] = _observed_score(
            metric,
            raw_values[metric],
            item.score,
            item.reason,
            valid_ids,
        )
        returned.add(metric)
    for metric in payload:
        if metric not in returned:
            results[metric] = _missing_score(metric, "정성 채점 결과에서 해당 지표가 누락됨")
    return results


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
                "source": re.sub(r"^https?://(?:www\.)?([^/]+).*$", r"\1", url) if url else result.get("title"),
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


from agents.startup import startup_agent
from agents.technology import technology_agent
from agents.competitor import competitor_agent


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

    market_evidence = list(evidence_by_id.values())
    findings, analysis_error = _market_findings(state, topic_results, market_evidence)
    for finding in findings.values():  # D-2: 근거의 scope에 시장 범위·기준연도·기간을 기록
        if finding.get("scope"):
            for item in market_evidence:
                if item["evidence_id"] in finding.get("evidence_ids", []):
                    item["scope"] = finding["scope"]
    for topic in TOPICS:
        finding = findings.get(topic)
        topic_results[topic]["finding"] = finding
        if finding is not None:
            topic_results[topic]["sufficient"] = bool(finding.get("sufficient"))
        elif topic_results[topic]["sufficient"]:
            topic_results[topic]["sufficient"] = False

    insufficient_topics = [topic for topic in TOPICS if not topic_results[topic]["sufficient"]]
    missing_metrics = [
        metric
        for topic in insufficient_topics
        for metric in _TOPIC_TO_METRICS[topic]
    ]
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
            "findings": findings,
            "insufficient_topics": insufficient_topics,
            "missing_metrics": missing_metrics,
            "analysis_error": analysis_error,
        },
        "rag_retry": retries,
        "market_evidence": market_evidence,
    }


def evaluator_agent(
    state: InvestmentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    del config
    evidence = _all_evidence(state)
    market_analysis = state.get("market_analysis", {})
    company_type = (state.get("current_startup") or {}).get("company_type")
    scores = _score_qualitative_metrics(state, evidence)

    deterministic_metrics = {
        "technical_headcount",
        "market_tam",
        "market_cagr",
        "development_stage",
        "customer_traction",
        "revenue_stage",
        "funding_total",
    }
    for metric in deterministic_metrics:
        metric_evidence, finding = _evidence_for_metric(metric, evidence, market_analysis)
        scores[metric] = _score_deterministic_metric(
            metric,
            company_type,
            metric_evidence,
            finding,
        )

    missing_evidence = [
        metric
        for metrics in _METRIC_GROUPS.values()
        for metric in metrics
        if scores[metric]["status"] == "missing"
    ]
    total_score = sum(
        (
            sum(scores[metric]["score"] for metric in metrics)
            / len(metrics)
            / 5
            * SCORE_WEIGHTS[group]
        )
        for group, metrics in _METRIC_GROUPS.items()
    )
    return {
        "scores": scores,
        "total_score": total_score,
        "missing_evidence": missing_evidence,
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
    del config
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
    conditions = {
        "score": total_score is not None and total_score >= INVESTMENT_SCORE_THRESHOLD,
        "missing": len(missing_evidence) < MAX_MISSING_EVIDENCE_FOR_INVESTMENT,
        "certainty": not state.get("uncertain", False),
    }
    failed_conditions = [
        label
        for key, label in (
            ("score", f"총점 {total_score if total_score is not None else '미산출'} < {INVESTMENT_SCORE_THRESHOLD}"),
            (
                "missing",
                f"결측 {len(missing_evidence)}개 >= {MAX_MISSING_EVIDENCE_FOR_INVESTMENT}개",
            ),
            ("certainty", "선정 기준 불확실성 존재"),
        )
        if not conditions[key]
    ]
    if decision == "투자":
        reason = (
            f"총점 {total_score:.2f}점, 결측 {len(missing_evidence)}개, 선정 불확실성 없음으로 "
            "세 가지 투자 조건을 모두 충족"
        )
    else:
        reason = "보류 조건: " + "; ".join(failed_conditions)
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
        "reason": reason,
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
