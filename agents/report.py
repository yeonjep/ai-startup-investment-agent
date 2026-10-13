# Report Agent: docs/DESIGN.md E장(목차·유형·REFERENCE), B-3 export_report_pdf.
# 표·점수·판정·REFERENCE는 코드가 만들고, 서술만 LLM이 쓴다. 근거는 [E번호]로만 인용한다.

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agents.common import create_llm, load_prompt
from tools.charts import group_score_chart, total_score_chart
from tools.pdf_export import render_markdown_pdf
from tools.web_meta import fetch_published_date

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_BASENAME = "RAG-Output_울산캠퍼스-4반_박연제+이정인+정예지+박태준"
MAX_PAGES = 10
MAX_REGENERATIONS = 2
# 재생성 회차별 서술 분량 배율 (최초, 재생성 1회, 재생성 2회)
LENGTH_SCALES = (1.0, 0.7, 0.5)
BASE_LIMITS = {"summary": 600, "overview": 700, "team_tech": 1500, "market_competition": 1500, "risk": 260}

METRIC_INFO = {
    "team_experience": ("창업자/팀", "핵심 인력 반도체 경력"),
    "technical_headcount": ("창업자/팀", "기술 인력 수"),
    "market_tam": ("시장성", "목표 시장 규모"),
    "market_cagr": ("시장성", "목표 시장 CAGR"),
    "demand_clarity": ("시장성", "수요처 명확성"),
    "development_stage": ("제품/기술력", "개발 단계"),
    "technical_originality": ("제품/기술력", "기술 독창성"),
    "competitive_difference": ("경쟁 우위", "동종 대비 차별성"),
    "entry_barrier": ("경쟁 우위", "진입장벽"),
    "customer_traction": ("실적", "고객·PoC·계약 수"),
    "revenue_stage": ("실적", "매출 발생"),
    "funding_total": ("자금조달/리스크", "누적 투자 유치액"),
    "risk_mitigation": ("자금조달/리스크", "위험 완화"),
}
TYPE_LABEL = {"CHIP": "AI 칩", "DESIGN_AI": "AI 설계·EDA", "PROCESS_AI": "AI 공정"}
NEUTRAL = "중립 대입·미확인"


class ReportSections(BaseModel):
    summary: str
    overview: str
    team_tech: str
    stakeholders: str
    domain_application: str
    market_competition: str
    risk_market: str
    risk_tech: str
    risk_regulation: str
    risk_competition: str


# ---------------------------------------------------------------- helpers
def _report_readiness_error(state: dict[str, Any]) -> str | None:
    """Return why a final report cannot yet be written, or None when ready."""
    evaluated = list(state.get("evaluated") or [])
    evaluated_names = {
        str((record.get("company_profile") or {}).get("name") or "")
        for record in evaluated
    }
    unfinished = {
        str(candidate.get("name") or "")
        for candidate in state.get("candidates") or []
        if candidate.get("selection_status") in {"PASS", "REVIEW"}
        and str(candidate.get("name") or "") not in evaluated_names
    }
    current = state.get("current_startup") or {}
    if (
        state.get("selection_status") in {"PASS", "REVIEW"}
        and str(current.get("name") or "") not in evaluated_names
    ):
        unfinished.add(str(current.get("name") or "현재 후보"))
    unfinished.discard("")
    if unfinished:
        return "평가 가능한 후보의 판정 스냅샷이 없음: " + ", ".join(sorted(unfinished))

    for record in evaluated:
        name = str((record.get("company_profile") or {}).get("name") or "기업")
        if record.get("decision") not in {"투자", "보류"}:
            return f"{name}의 투자 판정이 완료되지 않음"
        if not record.get("scores") or record.get("total_score") is None:
            return f"{name}의 채점 결과가 완료되지 않음"
    return None


def report_type(state: dict[str, Any]) -> str:
    readiness_error = _report_readiness_error(state)
    if readiness_error:
        raise ValueError(f"보고서 작성 불가: {readiness_error}")
    evaluated = state.get("evaluated", [])
    if any(record.get("decision") == "투자" for record in evaluated):
        return "invest"
    return "all_hold" if evaluated else "none"


def _cell(value: Any, limit: int | None = None) -> str:
    """Keep the full cell text so the PDF wraps it instead of cutting it off."""
    _ = limit
    return "-" if value in (None, "") else str(value).replace("|", "/").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _name(record: dict[str, Any]) -> str:
    return record["company_profile"].get("name", "기업")


def _shared_status_line(records: list[dict[str, Any]]) -> str:
    for record in records:
        for entry in (record.get("scores") or {}).values():
            verification = (entry or {}).get("verification_status")
            sdk = (entry or {}).get("sdk_status")
            if verification or sdk:
                return f"성능 근거 상태는 {verification or 'undisclosed'}이다. SDK 상태는 {sdk or 'undisclosed'}이다."
    return ""


def _name_keys(raw: Any) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    keys = {text.casefold()}
    bare = re.sub(r"\s*\([^)]*\)", "", text).strip()
    if bare:
        keys.add(bare.casefold())
    keys.update(part.strip().casefold() for part in re.findall(r"\(([^)]+)\)", text) if part.strip())
    return keys


def _same_company(left: Any, right: Any) -> bool:
    return bool(_name_keys(left) & _name_keys(right))


def _fmt_raw(value: Any) -> str:
    """ScoreEntry.raw_value(스칼라·dict·list)를 표에 넣을 짧은 문자열로 바꾼다."""
    if isinstance(value, dict):
        amount, unit = value.get("value"), value.get("unit")
        if isinstance(amount, (int, float)):
            if unit == "KRW":
                return f"약 {amount / 1e8:,.0f}억 원"
            if unit == "USD":
                return f"약 ${amount / 1e9:,.1f}B" if amount >= 1e9 else f"약 ${amount / 1e6:,.0f}M"
            return f"{amount:g}{unit or ''}"
        value = amount
    if isinstance(value, (list, tuple)):
        value = next((v for v in value if v not in (None, "")), None)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return "-" if value in (None, "") else str(value)


def _raw(record: dict[str, Any], metric: str) -> Any:
    entry = record.get("scores", {}).get(metric)
    return None if not entry or entry.get("status") == "missing" else _fmt_raw(entry.get("raw_value"))


def _report_market_analysis(record: dict[str, Any]) -> dict[str, Any]:
    """Expose only market findings that survived Evaluator validation."""
    analysis = record.get("market_analysis") or {}
    metric_by_topic = {
        "market_size": "market_tam",
        "market_growth": "market_cagr",
        "demand_risk": "demand_clarity",
    }
    findings = {
        topic: finding
        for topic, finding in (analysis.get("findings") or {}).items()
        if (record.get("scores", {}).get(metric_by_topic.get(topic, "")) or {}).get(
            "status"
        )
        == "observed"
    }
    missing = [
        topic
        for topic, metric in metric_by_topic.items()
        if (record.get("scores", {}).get(metric) or {}).get("status") != "observed"
    ]
    return {
        "summary": (
            "채점을 통과한 목표 시장 지표만 서술한다."
            if findings
            else "채점을 통과한 목표 시장 정량 지표가 확인되지 않음."
        ),
        "findings": findings,
        "reference_figures": list(analysis.get("reference_figures") or []),
        "insufficient_topics": missing,
    }


class Catalog:
    """근거 카탈로그: evidence_id ↔ 짧은 번호 E1.. (LLM 인용용)."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.items: list[dict[str, Any]] = []
        self.by_evidence_id: dict[str, str] = {}
        for record in records:
            for evidence in record.get("evidence", []):
                key = evidence["evidence_id"]
                if key not in self.by_evidence_id:
                    self.items.append(evidence)
                    self.by_evidence_id[key] = f"E{len(self.items)}"

    def tag(self, evidence_ids: list[str]) -> str:
        tags: list[str] = []
        for evidence_id in evidence_ids:
            tag = self.by_evidence_id.get(evidence_id)
            if tag and tag not in tags:
                tags.append(tag)
        return "".join(f"[{tag}]" for tag in tags)

    def prompt_lines(self) -> str:
        lines = []
        for index, item in enumerate(self.items, start=1):
            where = f"{item.get('doc_id')} p.{item.get('page')}" if item["source_type"] == "rag" else item.get("url")
            lines.append(
                f"E{index} | {item['company']} | {item['metric']} | 값={item.get('value')} {item.get('unit') or ''} | "
                f"범위={item.get('scope')} | {item.get('title')} ({where}, {item.get('date') or '날짜 미확인'}) | "
                f"\"{(item.get('quote') or '')[:140]}\""
            )
        return "\n".join(lines)

    def get(self, tag: str) -> dict[str, Any] | None:
        index = int(tag[1:]) - 1
        return self.items[index] if 0 <= index < len(self.items) else None


# ---------------------------------------------------------------- REFERENCE
def _year(date: str | None) -> str | None:
    match = re.match(r"(\d{4})", date or "")
    return match.group(1) if match else None


_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)
_SNS_TOKENS = ("linkedin.", "facebook.", "instagram.", "twitter.", "x.com", "threads.net")


def _is_sns_item(item: dict[str, Any]) -> bool:
    blob = " ".join(str(item.get(key) or "") for key in ("url", "source", "title")).lower()
    return any(token in blob for token in _SNS_TOKENS)


def _clean_title_and_outlet(item: dict[str, Any]) -> tuple[str, str]:
    raw = str(item.get("title") or "제목 미확인").strip()
    outlet = None
    article = re.search(r"기사본문\s*-\s*([^<]+)", raw)
    if article:
        outlet = re.split(r"[(\[]", article.group(1).strip())[0].strip()
    body = re.split(r"\s+<\s+", raw, maxsplit=1)[0]
    body = re.sub(r"\s*[|/]\s*\d+\s*comments?\b.*$", "", body, flags=re.IGNORECASE).strip()
    body = re.sub(r"\s*[|/]\s*[^|/]{1,40}\s*[|/]\s*\d+\s*comments?\b.*$", "", body, flags=re.IGNORECASE).strip()
    tail = re.search(r"\s+[-|/]\s+([^</|]{2,40})$", body)
    if tail:
        candidate = tail.group(1).strip()
        if outlet is None or candidate.replace(" ", "") in outlet.replace(" ", ""):
            outlet = outlet or candidate
            body = body[: tail.start()].strip()
    source = str(item.get("source") or "").strip()
    if not outlet:
        outlet = source if source and not _DOMAIN_RE.match(source) else (source or "사이트명 미확인")
    return outlet, (body.strip(" -|/") or "제목 미확인")


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(str(url))
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}" if host else ""


def _normalize_title(title: str | None) -> str:
    _, clean = _clean_title_and_outlet({"title": title or "", "source": ""})
    compact = re.sub(r"\s+", "", clean).lower()
    return compact if len(compact) >= 8 else ""


def _identity_keys(item: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if item.get("doc_id"):
        keys.add(("doc", str(item["doc_id"])))
    url = _normalize_url(item.get("url"))
    if url:
        keys.add(("url", url))
    title = _normalize_title(item.get("title"))
    if title:
        keys.add(("title", title))
    if not keys:
        keys.add(("item", str(item.get("evidence_id") or id(item))))
    return keys


def _merge_references(cited: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge the same document by doc_id, normalized URL, or cleaned title."""
    groups: list[dict[str, Any]] = []
    for item in cited:
        keys = _identity_keys(item)
        hits = [group for group in groups if group["keys"] & keys]
        if not hits:
            groups.append({"keys": set(keys), "item": item, "pages": [item["page"]] if item.get("page") else []})
            continue
        primary = hits[0]
        primary["keys"] |= keys
        if item.get("page") and item["page"] not in primary["pages"]:
            primary["pages"].append(item["page"])
        for extra in hits[1:]:
            primary["keys"] |= extra["keys"]
            for page in extra["pages"]:
                if page not in primary["pages"]:
                    primary["pages"].append(page)
            groups.remove(extra)
    return groups


def _format_references(cited: list[dict[str, Any]]) -> list[str]:
    """cited: 인용 순서대로의 Evidence. 같은 문서는 한 항목으로 합친다."""
    return [
        _reference_line(group["item"], sorted(group["pages"]))
        for group in _merge_references(cited)
    ]


def _reference_line(item: dict[str, Any], pages: list[int]) -> str:
    """Print the real URL as text. Page numbers stay in the body, not here."""
    _ = pages
    outlet, title = _clean_title_and_outlet(item)
    url = str(item.get("url") or "").strip()
    if item["source_type"] == "rag":
        source = item.get("source") or "발행기관 미확인"
        year = _year(item.get("date"))
        head = f"{source}({year})" if year else source
        if url:
            return f"{head}. *{title}*. {url}."
        if item.get("doc_id"):
            return f"{head}. *{title}*. 문서 ID {item['doc_id']}."
        return f"{head}. *{title}*."
    date = item.get("date")
    head = f"{outlet}({date})" if date else f"{outlet}(날짜 미확인, 조회일 {item.get('accessed_at') or '미확인'})"
    if url:
        return f"{head}. *{title}*. {outlet}, {url}."
    return f"{head}. *{title}*."


def _finalize_citations(markdown_text: str, catalog: Catalog) -> str:
    """[E#] 표기를 인용 순서의 번호로 바꾸고, 같은 문서는 REFERENCE에서 한 항목으로 합친다."""
    ordered = [
        item
        for tag in re.findall(r"\[(E\d+)\]", markdown_text)
        if (item := catalog.get(tag)) is not None
    ]
    groups = _merge_references(ordered)

    def number_of(item: dict[str, Any]) -> int | None:
        keys = _identity_keys(item)
        for index, group in enumerate(groups, start=1):
            if group["keys"] & keys:
                return index
        return None

    def replace(match: re.Match) -> str:
        tag = match.group(0)[1:-1]
        item = catalog.get(tag)
        if item is None:
            return f"[INVALID:{tag}]"
        number = number_of(item)
        return f"[{number}]" if number else f"[INVALID:{tag}]"

    body = re.sub(r"\[E\d+\]", replace, markdown_text)
    for group in groups:
        item = group["item"]
        if item["source_type"] == "web" and not item.get("date") and item.get("url"):
            item["date"] = fetch_published_date(item["url"])
    body = re.sub(
        r"(?:\[\d+\])+",
        lambda m: "".join(f"[{n}]" for n in sorted({int(x) for x in re.findall(r"\d+", m.group(0))})),
        body,
    )
    lines = [_reference_line(group["item"], sorted(group["pages"])) for group in groups]
    reference = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1)) or "인용한 외부 자료 없음."
    return (
        f"{body}\n\n"
        '<div class="page-break"></div>\n\n'
        f"# REFERENCE\n\n{reference}\n"
    )


# ---------------------------------------------------------------- 서술(LLM)
def _compact(record: dict[str, Any]) -> dict[str, Any]:
    # Discovery fields are hints, not scored facts. Do not let stale round/funding
    # strings override the Evidence-backed ScoreEntry in the narrative model input.
    excluded = {
        "profile_evidence",
        "description",
        "evaluation_product",
        "total_funding",
        "recent_round",
        "latest_round",
    }
    profile = {k: v for k, v in record["company_profile"].items() if k not in excluded}
    verified_funding = _raw(record, "funding_total")
    if verified_funding:
        profile["verified_funding_total"] = verified_funding
    funding_evidence = [
        item
        for item in record.get("evidence", [])
        if item.get("metric") == "funding_total"
    ]
    funding_text = " ".join(
        f"{item.get('title') or ''} {item.get('quote') or ''}"
        for item in funding_evidence
    )
    round_match = re.search(
        r"(?:series|시리즈)\s*[- ]?([a-c])\b|\b(seed|pre[- ]?[a-c])\b|\b(시드)\b",
        funding_text,
        flags=re.IGNORECASE,
    )
    if round_match:
        if round_match.group(1):
            profile["verified_funding_round"] = f"Series {round_match.group(1).upper()}"
        else:
            profile["verified_funding_round"] = round_match.group(2) or round_match.group(3)
    return {
        "company_profile": profile,
        "selection_uncertain": record.get("selection_uncertain"),
        "tech_summary": record.get("tech_summary"),
        "market_analysis": _report_market_analysis(record),
        "competitor_analysis": record.get("competitor_analysis"),
        "scores": {
            m: {"score": s["score"], "status": s["status"], "raw_value": s["raw_value"], "reason": s["reason"]}
            for m, s in record.get("scores", {}).items()
        },
        "total_score": record.get("total_score"),
        "missing_evidence": record.get("missing_evidence"),
        "decision": record.get("decision"),
        "reason": record.get("reason"),
    }


def _fallback_sections(rtype: str, records: list[dict[str, Any]]) -> ReportSections:
    main = records[0]
    unknown = "확인되지 않음"
    return ReportSections(
        summary=f"{_name(main)} 등 {len(records)}개 기업을 평가했다." if rtype == "all_hold" else f"{_name(main)}을(를) 평가했다.",
        overview=str(main["company_profile"].get("evaluation_product", unknown)),
        team_tech=str(main.get("tech_summary", {}).get("summary", unknown)),
        stakeholders="고객·파트너·투자자·정부 과제에 대한 공개 근거가 없으면 공개 근거 미확인.",
        domain_application="적용 워크로드와 현장 적용에 대한 공개 근거가 없으면 공개 근거 미확인.",
        market_competition=f"{_report_market_analysis(main).get('summary', unknown)} "
        f"{main.get('competitor_analysis', {}).get('summary', '')}",
        risk_market=unknown, risk_tech=unknown, risk_regulation=unknown, risk_competition=unknown,
    )


def _write_sections(rtype: str, records: list[dict[str, Any]], catalog: Catalog, scale: float, as_of: str, feedback: str = "") -> ReportSections:
    limits = {k: int(v * scale) for k, v in BASE_LIMITS.items()}
    human = (
        f"보고서 유형: {rtype}\n평가 기준일: {as_of}\n"
        f"글자 수 상한: SUMMARY {limits['summary']}자, 기업·사업 개요 {limits['overview']}자, "
        f"팀·기술·제품 경쟁력 {limits['team_tech']}자, 시장성·경쟁 환경 {limits['market_competition']}자, "
        f"리스크 4종 각 {limits['risk']}자\n\n"
        f"평가 결과(JSON):\n{json.dumps([_compact(r) for r in records], ensure_ascii=False)}\n\n"
        f"근거 카탈로그:\n{catalog.prompt_lines()}"
    )
    if feedback:
        human += (
            "\n\n품질 평가에서 아래 섹션만 미달이었다. 표시된 섹션의 문장만 고치고, "
            "다른 절의 사실과 점수·판정은 유지하라. 규제 절에는 투자·실적을 넣지 마라.\n"
            f"{feedback}\n"
        )
    try:
        llm = create_llm(temperature=0).with_structured_output(ReportSections)
        result = llm.invoke([("system", load_prompt("report")), ("human", human)])
        if result is not None:
            return result
    except Exception as error:  # noqa: BLE001 - 보고서는 서술 실패 시에도 생성되어야 한다
        print(f"[report] LLM narrative failed, using fallback: {error}")
    return _fallback_sections(rtype, records)


def _cite_numeric_sentences(
    text: str,
    catalog: Catalog,
    categories: tuple[str, ...],
) -> str:
    """Attach only Evidence whose value/quote/scope contains the stated number."""
    parts = re.split(r"(?<=[.!?。])(\s+)", text)
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        if not re.search(r"\d", sentence) or re.search(r"\[E\d+\]", sentence):
            continue
        numbers = {
            token.replace(",", "")
            for token in re.findall(r"\d[\d,]*(?:\.\d+)?", sentence)
        }
        evidence_ids: list[str] = []
        for item in catalog.items:
            if item.get("category") not in categories:
                continue
            evidence_text = " ".join(
                str(item.get(key) or "")
                for key in ("value", "unit", "quote", "scope")
            ).replace(",", "")
            if any(number in evidence_text for number in numbers):
                evidence_ids.append(str(item.get("evidence_id") or ""))
        tags = catalog.tag([item for item in evidence_ids if item])
        if tags:
            parts[index] = sentence.rstrip() + tags
    return "".join(parts)


def _clip(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    cut = compact[:limit]
    pivot = max(cut.rfind(" "), cut.rfind("."), cut.rfind(","))
    if pivot >= limit // 2:
        cut = cut[:pivot]
    return cut.rstrip(" ,./")


def _clean_narrative(text: str) -> str:
    """Remove leftover braces and bare domain labels from narrative fields."""
    text = re.sub(r"\s*\}(?=\s*$)", "", text.strip())
    return re.sub(r"\s*\((?:https?://)?(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}[^)]*\)", "", text, flags=re.IGNORECASE)


# ---------------------------------------------------------------- 표(코드)
def _scorecard(record: dict[str, Any], catalog: Catalog, reason_limit: int) -> str:
    rows = []
    for metric, (group, label) in METRIC_INFO.items():
        entry = record["scores"].get(metric)
        if entry is None:
            rows.append([group, label, "-", "결측", "점수 없음"])
            continue
        missing = entry["status"] == "missing"
        score = f"{entry['score']:g} ({NEUTRAL})" if missing else f"{entry['score']:g}"
        reason = entry["reason"]
        if reason.startswith("설계된 정량·범주 구간을 코드로 적용"):
            reason = "정량·범주 구간 규칙 적용"
        _ = reason_limit
        reason = _clip(reason, 60)
        rows.append([group, label, _clip(_fmt_raw(entry["raw_value"]), 30), score, f"{reason} {catalog.tag(entry['evidence_ids'])}".strip()])
    return _table(["항목", "지표", "원값", "점수", "판단 근거"], rows)


def _overview_table(records: list[dict[str, Any]]) -> str:
    rows = []
    for r in records:
        p = r["company_profile"]
        rows.append([
            _name(r), p.get("country"), TYPE_LABEL.get(p.get("company_type"), "미확정"),
            _cell(p.get("evaluation_product"), 40),
            p.get("latest_round") or p.get("recent_round"),
            _raw(r, "funding_total") or NEUTRAL, _raw(r, "development_stage") or NEUTRAL,
            _raw(r, "technical_headcount") or NEUTRAL,
        ])
    return _table(["기업", "국가", "유형", "평가 제품", "투자 단계", "누적 투자액", "개발 단계", "기술 인력"], rows)


def _market_table(records: list[dict[str, Any]], catalog: "Catalog") -> str:
    rows = []
    for r in records:
        row: list[Any] = [_name(r)]
        for metric in ("market_tam", "market_cagr"):
            entry = r.get("scores", {}).get(metric) or {}
            if entry.get("status") == "observed":
                by_id = {e["evidence_id"]: e for e in r.get("evidence", [])}
                first = next((by_id[i] for i in entry.get("evidence_ids", []) if i in by_id), {})
                row += [f"{_fmt_raw(entry.get('raw_value'))} {catalog.tag(entry.get('evidence_ids', []))}".strip(),
                        _cell(first.get("scope"), 45)]
            else:
                row += [f"{NEUTRAL} (3점)", "-"]
        rows.append(row)
    return _table(["기업", "목표 시장 규모", "규모 범위·기준", "목표 시장 CAGR", "CAGR 범위·기간"], rows)


def _comparison_table(records: list[dict[str, Any]]) -> str:
    rows = [
        [
            _name(r), r["company_profile"].get("country"), TYPE_LABEL.get(r["company_profile"].get("company_type"), "미확정"),
            _raw(r, "development_stage") or NEUTRAL, _raw(r, "funding_total") or NEUTRAL,
            f"{r['total_score']:.2f}" if r.get("total_score") is not None else "미산정",
            len(r.get("missing_evidence", [])), r["decision"],
        ]
        for r in records
    ]
    return _table(["기업", "국가", "유형", "개발 단계", "투자액(원통화·환산 기준)", "총점", "결측", "판정"], rows)


def _missing_text(record: dict[str, Any]) -> str:
    names = [METRIC_INFO.get(m, ("", m))[1] for m in record.get("missing_evidence", [])]
    text = f"결측 {len(names)}개({', '.join(names)})는 3점이 **{NEUTRAL}**으로 계산되었으며 근거 있는 3점과 구분한다." if names else "결측 지표 없음."
    if record.get("selection_uncertain"):
        criteria = record["company_profile"].get("uncertain_criteria") or ["미확인"]
        text += f" 선정 기준 REVIEW가 해소되지 않아(불확실 기준: {', '.join(criteria)}) 최종 판정은 보류다."
    return text


def _limitations(state: dict[str, Any], rtype: str) -> str:
    total = len(state.get("candidates", []))
    done = len(state.get("evaluated", []))
    excluded = sum(1 for c in state.get("candidates", []) if c.get("selection_status") == "FAIL")
    progress = (
        f"- 후보 {total}개를 수집해 선정 기준(A-3)에서 {excluded}개를 제외하고 {done}개를 평가했다. "
    )
    if rtype == "invest":
        progress += (
            "조사 순서상 최초 투자 판정에서 종료했으므로 나머지 후보는 평가되지 않았으며, "
            "검색 순서·후보 상한에 따라 일부 기업이 평가되지 않았을 수 있다.\n"
        )
    else:
        progress += (
            "투자 판정 없이 후보가 소진되거나 평가 상한에 도달했으며, "
            "검색 순서·후보 상한에 따라 일부 기업이 평가되지 않았을 수 있다.\n"
        )
    text = (
        progress
        + "- 정량 구간은 공식 VC 표준이 아닌 프로젝트 초기 설계값이며, 누적 투자액은 조달 이력의 대리 지표(현재 자금 여력 아님)다.\n"
        "- Valuation·지분율이 비공개이므로 ROI는 산출하지 않았고, 기대 성장 여력은 시장·사업화 단계로 정성 평가했다.\n"
        "- 서로 다른 기업 유형의 원수치는 직접 비교하지 않는다."
    )
    conflicts = _conflict_lines(state.get("evaluated") or [])
    if conflicts:
        text += "\n- 출처 충돌: " + " ".join(conflicts)
    return text


def _canonical_amount(value: Any, unit: Any) -> tuple[float, str] | None:
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", "" if value is None else str(value))
    if not match:
        return None
    number = float(match.group(0).replace(",", ""))
    unit_text = str(unit or "").replace(" ", "")
    if "억" in unit_text and "원" in unit_text:
        return number * 100_000_000, "KRW"
    if unit_text.upper() == "KRW" or unit_text == "원":
        return number, "KRW"
    if "%" in unit_text or unit_text == "％":
        return number, "%"
    if "달러" in unit_text or unit_text.upper() == "USD":
        return number, "USD"
    if unit_text:
        return number, unit_text.upper()
    return None


def _quantitative_conflicts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare amounts only after unit, scope, and base date match. Sentences are not conflicts."""
    found: list[dict[str, Any]] = []
    for record in records:
        buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for item in record.get("evidence") or []:
            parsed = _canonical_amount(item.get("value"), item.get("unit"))
            if parsed is None:
                continue
            number, unit = parsed
            key = (
                str(item.get("metric") or ""),
                unit,
                str(item.get("scope") or "").strip(),
                str(item.get("date") or "")[:10],
            )
            buckets.setdefault(key, []).append({**item, "_number": number})
        for key, items in buckets.items():
            numbers = {item["_number"] for item in items}
            if len(numbers) < 2:
                continue
            metric = key[0]
            reason = str(((record.get("scores") or {}).get(metric) or {}).get("reason") or "")
            if "채택" not in reason and "미확인" not in reason:
                reason = "기준이 같아 값이 충돌하며, 어느 값을 채택할지 공개 근거가 없어 미확인"
            found.append({"record": record, "metric": metric, "items": items, "reason": reason})
    return found


def _conflict_lines(records: list[dict[str, Any]]) -> list[str]:
    lines = []
    for conflict in _quantitative_conflicts(records):
        sides = []
        for item in conflict["items"]:
            amount = f"{item.get('value')}{item.get('unit') or ''}"
            source = item.get("source") or item.get("title") or "출처 미확인"
            sides.append(f"{amount} ({source}, {item.get('date') or '기준일 미확인'})")
        label = METRIC_INFO.get(conflict["metric"], ("", conflict["metric"]))[1]
        lines.append(
            f"{_name(conflict['record'])} {label}: {' vs '.join(sides)}. {conflict['reason']}"
        )
    return lines


_REGULATORY = re.compile(r"수출\s*통제|수출통제|수출\s*규제|공급망|정부\s*규제|규제\s*위험|이니셔티브|정책\s*방향")
_FUNDING_TEXT = re.compile(r"투자\s*유치|투자유치|시리즈|매출|실적")


def _regulatory_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for record in records:
        for item in record.get("evidence") or []:
            if item.get("metric") not in {None, "", "demand_risk", "risk_mitigation"}:
                continue
            blob = " ".join(str(item.get(key) or "") for key in ("quote", "scope", "title"))
            if not _REGULATORY.search(blob) or _FUNDING_TEXT.search(blob):
                continue
            matches.append(item)
    return matches


_COMPANY_RISK = re.compile(r"수출\s*통제|수출통제|수출\s*규제|공급망|정부\s*규제|규제\s*위험")
_INDUSTRY_POLICY = re.compile(r"이니셔티브|정책\s*방향|지원\s*정책|지원정책")


def _company_regulation_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    company_items = []
    for item in _regulatory_items(records):
        blob = " ".join(str(item.get(key) or "") for key in ("quote", "scope", "title"))
        if _INDUSTRY_POLICY.search(blob) and not _COMPANY_RISK.search(blob):
            continue
        if _COMPANY_RISK.search(blob):
            company_items.append(item)
    return company_items


def _complete_regulation_summary(quotes: list[str]) -> str:
    """One or two finished sentences. A clipped quote is not a regulation line."""
    response = create_llm(temperature=0).invoke(
        [
            (
                "system",
                "기업의 규제 위험만 한국어 1~2문장으로 요약한다. 문장을 중간에 자르지 말고, "
                "조사로 끝내지 않는다. 산업 지원정책·이니셔티브는 기업 규제 위험이 아니다. "
                "인용 원문을 그대로 붙이지 않는다.",
            ),
            ("human", "\n".join(quotes)),
        ]
    )
    text = str(getattr(response, "content", response) or "").strip()
    if not text or "…" in text or text.endswith("...") or re.search(r"(?:을|를|은|는|이|가|와|과|의)$", text):
        return ""
    return text


def _regulation_text(records: list[dict[str, Any]], narrative: str, catalog: "Catalog") -> str:
    del narrative
    matches = _company_regulation_items(records)
    if not matches:
        return "공개 근거 미확인"
    quotes = [re.sub(r"\s+", " ", str(item.get("quote") or "")).strip() for item in matches[:3]]
    quotes = [quote for quote in quotes if quote]
    if not quotes:
        return "공개 근거 미확인"
    try:
        summary = _complete_regulation_summary(quotes)
    except Exception:  # noqa: BLE001 - 요약 실패 시 잘린 원문 대신 미확인을 적는다
        summary = ""
    if not summary:
        return "공개 근거 미확인"
    tags = catalog.tag([str(item.get("evidence_id") or "") for item in matches[:2]])
    return f"{summary} {tags}".strip()


def _reference_figure_lines(records: list[dict[str, Any]], catalog: "Catalog") -> str:
    lines = []
    for record in records:
        for figure in (record.get("market_analysis") or {}).get("reference_figures") or []:
            label = "시장 규모" if figure.get("metric") == "market_size" else "성장률"
            scope = figure.get("market_scope") or "범위 미기재"
            region = figure.get("region") or "지역 미기재"
            period = figure.get("period") or ""
            tag = catalog.tag(figure.get("evidence_ids") or [])
            where = ""
            if figure.get("doc_id"):
                where = f", 문서 {figure['doc_id']}"
                if figure.get("page"):
                    where += f" p.{figure['page']}"
            lines.append(
                f"- {label}: {figure.get('value')} {figure.get('unit') or ''}".rstrip()
                + f" (시장 범위 {scope}, 지역 {region}{', 기준 ' + period if period else ''}{where}) {tag}".rstrip()
                + f" — {figure.get('note') or '참고(범위 상이)'}"
            )
    return "\n".join(lines)


def _citation_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    cited_ids = {
        evidence_id
        for entry in record.get("scores", {}).values()
        if entry.get("status") == "observed"
        for evidence_id in entry.get("evidence_ids", [])
    }
    for figure in (record.get("market_analysis") or {}).get("reference_figures") or []:
        cited_ids.update(figure.get("evidence_ids") or [])
    items = []
    for item in record.get("evidence") or []:
        evidence_id = item.get("evidence_id")
        blob = " ".join(str(item.get(key) or "") for key in ("quote", "scope", "title"))
        if evidence_id in cited_ids or item.get("metric") == "demand_risk" or _REGULATORY.search(blob):
            items.append(item)
    press_metrics = {item.get("metric") for item in items if not _is_sns_item(item)}
    return [
        item
        for item in items
        if not (_is_sns_item(item) and item.get("metric") in press_metrics)
    ]


def _hold_summary(records: list[dict[str, Any]], reason_limit: int) -> str:
    rows = [[_name(r), f"{r['total_score']:.2f}" if r.get("total_score") is not None else "미산정",
             len(r.get("missing_evidence", [])), _cell(r.get("reason"), reason_limit)] for r in records]
    return _table(["보류 기업", "총점", "결측", "사유"], rows)


# ---------------------------------------------------------------- 조립
def _assemble(state: dict[str, Any], rtype: str, level: int, chart_dir: Path) -> str:
    as_of = state.get("as_of_date", "미설정")
    scale = LENGTH_SCALES[level]
    reason_limit = (60, 40, 25)[level]
    evaluated = state["evaluated"]
    if rtype == "invest":
        main = next(r for r in evaluated if r["decision"] == "투자")
        focus, holds = [main], [r for r in evaluated if r is not main]
    else:
        focus, holds = evaluated, []
    citation_records = [{"evidence": _citation_items(record)} for record in focus]
    catalog = Catalog(citation_records)
    sec = _write_sections(rtype, focus, catalog, scale, as_of, str(state.get("quality_feedback") or ""))
    for field_name in (
        "summary",
        "overview",
        "team_tech",
        "stakeholders",
        "domain_application",
        "market_competition",
        "risk_market",
        "risk_tech",
        "risk_regulation",
        "risk_competition",
    ):
        setattr(sec, field_name, _clean_narrative(getattr(sec, field_name)))
    sec.summary = _cite_numeric_sentences(
        sec.summary, catalog, ("profile", "tech", "market")
    )
    sec.overview = _cite_numeric_sentences(sec.overview, catalog, ("profile",))
    sec.team_tech = _cite_numeric_sentences(
        sec.team_tech, catalog, ("profile", "tech")
    )
    sec.stakeholders = _cite_numeric_sentences(
        sec.stakeholders, catalog, ("profile", "market", "competitor")
    )
    sec.domain_application = _cite_numeric_sentences(
        sec.domain_application, catalog, ("tech", "market")
    )
    sec.market_competition = _cite_numeric_sentences(
        sec.market_competition,
        catalog,
        ("market", "competitor"),
    )

    if rtype == "invest":
        p = main["company_profile"]
        head = (f"**{_name(main)}**({p.get('country')}, {TYPE_LABEL.get(p.get('company_type'), '유형 미확정')}) · "
                f"총점 **{main['total_score']:.2f}/100** · 판정 **투자** · 평가 기준일 {as_of}")
    else:
        best = max((r["total_score"] for r in evaluated if r.get("total_score") is not None), default=None)
        head = (f"평가 후보 {len(evaluated)}개 **전원 보류** · 최고 총점 {best:.2f}/100 · 평가 기준일 {as_of}"
                if best is not None else f"평가 후보 {len(evaluated)}개 전원 보류 · 평가 기준일 {as_of}")

    status_line = _shared_status_line(focus)
    if status_line:
        sec.team_tech = f"{sec.team_tech} {status_line}".strip()
        sec.risk_tech = f"{sec.risk_tech} {status_line}".strip()

    parts = [
        f"# SUMMARY\n\n{head}\n\n{sec.summary}\n\n## 1. 기업 및 사업 개요\n\n{sec.overview}",
        (
            "## 2. 팀·기술 및 제품 경쟁력\n\n"
            f"### 기술 성숙도\n\n{sec.team_tech}\n\n"
            f"### 이해관계자\n\n{sec.stakeholders}\n\n"
            f"### 도메인 적용\n\n{sec.domain_application}"
        ),
    ]
    market = f"## 3. 시장성 및 경쟁 환경\n\n{sec.market_competition}\n\n**시장 지표**\n\n{_market_table(focus, catalog)}"
    reference_lines = _reference_figure_lines(focus, catalog)
    if reference_lines:
        market += "\n\n" + reference_lines
    aliases: set[str] = set()
    for record in focus:
        profile = record.get("company_profile") or {}
        for raw in (profile.get("name"), profile.get("english_name"), profile.get("alias"), _name(record)):
            aliases |= _name_keys(raw)
    comps = [
        c for record in focus for c in record.get("competitor_analysis", {}).get("competitors", [])
        if not (_name_keys(c.get("name")) & aliases)
    ]
    if comps:
        market += "\n\n**경쟁사 비교**\n\n" + _table(["경쟁사", "국가", "비교"], [[c.get("name"), c.get("country"), _cell(c.get("comparison"), 60)] for c in comps[:5]])
    parts.append(market)

    if rtype == "invest":
        group_score_chart(main, chart_dir / "group_scores.png")
        body4 = (f"### Scorecard\n\n![항목별 점수 구성](group_scores.png)\n\n{_scorecard(main, catalog, reason_limit)}\n\n**판단 사유**: {main.get('reason')}\n\n"
                 f"**결측·선정 불확실성**: {_missing_text(main)}")
    else:
        total_score_chart(evaluated, chart_dir / "total_scores.png")
        body4 = (f"### 후보 비교\n\n![후보별 총점 비교](total_scores.png)\n\n{_comparison_table(evaluated)}\n\n" +
                 "\n".join(f"- **{_name(r)}**: {_missing_text(r)}" for r in evaluated))
    if holds:
        body4 += f"\n\n### 앞서 평가한 보류 후보\n\n{_hold_summary(holds, reason_limit)}"
    parts.append(f"## 4. 투자 판단 및 주요 리스크\n\n{body4}")
    risks = (
        f"### 주요 리스크\n\n- **시장**: {sec.risk_market}\n"
        f"- **기술**: {sec.risk_tech}\n"
        f"- **규제**: {_regulation_text(focus, sec.risk_regulation, catalog)}\n"
        f"- **경쟁**: {sec.risk_competition}\n\n"
        f"### 평가 범위와 한계\n\n{_limitations(state, rtype)}"
    )
    parts.append(risks)
    return _finalize_citations("\n\n".join(parts), catalog)


def _assemble_none(state: dict[str, Any]) -> str:
    as_of = state.get("as_of_date", "미설정")
    candidates = state.get("candidates", [])
    sources: list[dict[str, Any]] = []
    rows = []
    for c in candidates:
        tags = ""
        for s in c.get("sources", [])[:1]:
            sources.append({"evidence_id": f"none:{len(sources)}", "company": c.get("name"), "source_type": "web",
                            "title": s.get("title"), "source": re.sub(r"https?://(www\.)?([^/]+).*", r"\2", s.get("url") or ""),
                            "date": s.get("date"), "accessed_at": as_of, "url": s.get("url"), "doc_id": None, "page": None})
            tags = f"[E{len(sources)}]"
        reason = c.get("exclusion_reason") or c.get("reason") or "사유 미기록"
        rows.append([c.get("name"), c.get("country"), c.get("selection_status"), c.get("failed_criterion"), f"{reason} {tags}".strip()])
    catalog = Catalog([{"evidence": sources}])
    if candidates:
        summary = f"수집한 후보 {len(candidates)}개 모두 A-3 선정 기준에서 제외되어 평가 대상이 없다. 점수표와 투자 판정은 생성하지 않았다."
        table = _table(["후보", "국가", "선정 상태", "제외 기준", "사유"], rows)
    else:
        summary = "국내외 탐색에서 평가할 후보를 찾지 못했다. 점수표와 투자 판정은 생성하지 않았다."
        table = "수집된 후보가 없다."
    text = (f"# SUMMARY\n\n평가 기준일 {as_of} · **평가 대상 없음**\n\n{summary}\n\n"
            f"## 1. 탐색 결과\n\n국내외(한국어·영어) 질의로 후보를 탐색했으며 수집 후보 수는 {len(candidates)}개다.\n\n"
            f"## 2. 선정 제외 사유\n\n{table}\n\n"
            "## 3. 한계 및 후속 조치\n\n- 검색 순서·후보 상한 때문에 조건에 맞는 다른 기업이 탐색되지 않았을 수 있다.\n"
            "- 탐색 질의·후보 수집 범위를 넓히거나 평가 기준일을 조정해 재실행할 수 있다.")
    # 카탈로그 태그는 표 안의 [E#]만 사용
    catalog.items = sources
    return _finalize_citations(text, catalog)


def generate_report(state: dict[str, Any], output_dir: Path = OUTPUT_DIR, basename: str = REPORT_BASENAME) -> dict[str, Any]:
    """보고서 md·PDF를 만들고, 10페이지 초과 시 서술을 줄여 최대 2회 재생성한다."""
    rtype = report_type(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path = output_dir / f"{basename}.md", output_dir / f"{basename}.pdf"
    attempts = 1 if rtype == "none" else 1 + MAX_REGENERATIONS
    markdown_text, pages = "", 0
    for level in range(attempts):
        chart_dir = output_dir / "charts"
        markdown_text = _assemble_none(state) if rtype == "none" else _assemble(state, rtype, level, chart_dir)
        pages = render_markdown_pdf(markdown_text, pdf_path, asset_dir=chart_dir if chart_dir.is_dir() else None)
        if pages <= MAX_PAGES:
            break
        print(f"[report] {pages} pages > {MAX_PAGES}; regenerating shorter (attempt {level + 1}/{MAX_REGENERATIONS})")
    md_path.write_text(markdown_text, encoding="utf-8")
    if pages > MAX_PAGES:
        print(f"[report] WARNING: still {pages} pages after {MAX_REGENERATIONS} regenerations")
    return {"type": rtype, "markdown": markdown_text, "md_path": str(md_path), "pdf_path": str(pdf_path), "pages": pages}


def report_agent(state: dict[str, Any], config: RunnableConfig) -> dict[str, str]:
    """그래프 노드 인터페이스: final_report(markdown)만 State에 쓰고 파일은 outputs/에 저장한다."""
    basename = (config or {}).get("configurable", {}).get("report_basename", REPORT_BASENAME)
    return {"final_report": generate_report(state, basename=basename)["markdown"]}
