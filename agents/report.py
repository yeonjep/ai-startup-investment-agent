# Report Agent: docs/DESIGN.md E장(목차·유형·REFERENCE), B-3 export_report_pdf.
# 표·점수·판정·REFERENCE는 코드가 만들고, 서술만 LLM이 쓴다. 근거는 [E번호]로만 인용한다.

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agents.common import create_llm, load_prompt
from tools.charts import group_score_chart, total_score_chart
from tools.pdf_export import render_markdown_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_BASENAME = "RAG-Output_울산캠퍼스-4반_박연제+이정인+정예지"
MAX_PAGES = 5
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
    market_competition: str
    risk_market: str
    risk_tech: str
    risk_regulation: str
    risk_competition: str


# ---------------------------------------------------------------- helpers
def report_type(state: dict[str, Any]) -> str:
    evaluated = state.get("evaluated", [])
    if any(record.get("decision") == "투자" for record in evaluated):
        return "invest"
    return "all_hold" if evaluated else "none"


def _cell(value: Any, limit: int | None = None) -> str:
    text = "-" if value in (None, "") else str(value).replace("|", "/").replace("\n", " ")
    return text[: limit - 1] + "…" if limit and len(text) > limit else text


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _name(record: dict[str, Any]) -> str:
    return record["company_profile"].get("name", "기업")


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
        for evidence_id in evidence_ids:
            if evidence_id in self.by_evidence_id:
                return f"[{self.by_evidence_id[evidence_id]}]"
        return ""

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


def _format_references(cited: list[dict[str, Any]]) -> list[str]:
    """cited: 인용 순서대로의 Evidence. 같은 문서(RAG)·같은 URL(웹)은 한 항목으로 합친다."""
    groups: dict[tuple, dict[str, Any]] = {}
    for item in cited:
        key = ("rag", item["doc_id"]) if item["source_type"] == "rag" else ("web", item.get("url") or item.get("title"))
        group = groups.setdefault(key, {"item": item, "pages": []})
        if item.get("page") and item["page"] not in group["pages"]:
            group["pages"].append(item["page"])
    return [_reference_line(group["item"], sorted(group["pages"])) for group in groups.values()]


def _reference_line(item: dict[str, Any], pages: list[int]) -> str:
    source, title, url = item.get("source"), item.get("title"), item.get("url")
    if item["source_type"] == "rag":
        year = _year(item.get("date"))
        head = f"{source or '발행기관 미확인'}({year})" if year else f"{source or '발행기관 미확인'}"
        tail = url if url else f"문서 ID {item['doc_id']}, 조회일 {item.get('accessed_at') or '미확인'}"
        page_text = f", 인용 PDF 페이지 {', '.join(str(p) for p in pages)}" if pages else ""
        return f"{head}. *{title}*. {tail}{page_text}."
    date = item.get("date")
    head = f"{source or '작성자 미확인'}({date})" if date else f"{source or '작성자 미확인'}(날짜 미확인, 조회일 {item.get('accessed_at') or '미확인'})"
    return f"{head}. *{title}*. {source or '사이트명 미확인'}, {url}." if url else f"{head}. *{title}*."


def _finalize_citations(markdown_text: str, catalog: Catalog) -> str:
    """[E#] 표기를 인용 순서의 번호로 바꾸고, 실제 인용한 출처만 REFERENCE로 만든다."""
    cited: list[dict[str, Any]] = []
    numbers: dict[str, int] = {}
    ref_keys: dict[tuple, int] = {}

    def key_of(item: dict[str, Any]) -> tuple:
        return ("rag", item["doc_id"]) if item["source_type"] == "rag" else ("web", item.get("url") or item.get("title"))

    def replace(match: re.Match) -> str:
        tag = match.group(0)[1:-1]
        item = catalog.get(tag)
        if item is None:
            return ""
        cited.append(item)
        ref_keys.setdefault(key_of(item), len(ref_keys) + 1)
        return f"[{ref_keys[key_of(item)]}]"

    body = re.sub(r"\[E\d+\]", replace, markdown_text)
    # 연속된 인용은 중복을 없애고 번호순으로 정렬한다: [3][1][3] -> [1][3]
    body = re.sub(
        r"(?:\[\d+\])+",
        lambda m: "".join(f"[{n}]" for n in sorted({int(x) for x in re.findall(r"\d+", m.group(0))})),
        body,
    )
    lines = _format_references(cited)
    reference = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1)) or "인용한 외부 자료 없음."
    return f"{body}\n\n# REFERENCE\n\n{reference}\n"


# ---------------------------------------------------------------- 서술(LLM)
def _compact(record: dict[str, Any]) -> dict[str, Any]:
    profile = {k: v for k, v in record["company_profile"].items() if k != "profile_evidence"}
    return {
        "company_profile": profile,
        "selection_uncertain": record.get("selection_uncertain"),
        "tech_summary": record.get("tech_summary"),
        "market_analysis": record.get("market_analysis"),
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
        market_competition=f"{main.get('market_analysis', {}).get('summary', unknown)} "
        f"{main.get('competitor_analysis', {}).get('summary', '')}",
        risk_market=unknown, risk_tech=unknown, risk_regulation=unknown, risk_competition=unknown,
    )


def _write_sections(rtype: str, records: list[dict[str, Any]], catalog: Catalog, scale: float, as_of: str) -> ReportSections:
    limits = {k: int(v * scale) for k, v in BASE_LIMITS.items()}
    human = (
        f"보고서 유형: {rtype}\n평가 기준일: {as_of}\n"
        f"글자 수 상한: SUMMARY {limits['summary']}자, 기업·사업 개요 {limits['overview']}자, "
        f"팀·기술·제품 경쟁력 {limits['team_tech']}자, 시장성·경쟁 환경 {limits['market_competition']}자, "
        f"리스크 4종 각 {limits['risk']}자\n\n"
        f"평가 결과(JSON):\n{json.dumps([_compact(r) for r in records], ensure_ascii=False)}\n\n"
        f"근거 카탈로그:\n{catalog.prompt_lines()}"
    )
    try:
        llm = create_llm(temperature=0).with_structured_output(ReportSections)
        result = llm.invoke([("system", load_prompt("report")), ("human", human)])
        if result is not None:
            return result
    except Exception as error:  # noqa: BLE001 - 보고서는 서술 실패 시에도 생성되어야 한다
        print(f"[report] LLM narrative failed, using fallback: {error}")
    return _fallback_sections(rtype, records)


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
        reason = _cell(reason, reason_limit)
        rows.append([group, label, _cell(_fmt_raw(entry["raw_value"]), 28), score, f"{reason} {catalog.tag(entry['evidence_ids'])}".strip()])
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
                row += [NEUTRAL, "-"]
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
    text = f"결측 {len(names)}개({', '.join(names)})는 3점이 **{NEUTRAL}**로 계산되었으며 근거 있는 3점과 구분한다." if names else "결측 지표 없음."
    if record.get("selection_uncertain"):
        criteria = record["company_profile"].get("uncertain_criteria") or ["미확인"]
        text += f" 선정 기준 REVIEW가 해소되지 않아(불확실 기준: {', '.join(criteria)}) 최종 판정은 보류다."
    return text


def _limitations(state: dict[str, Any]) -> str:
    total = len(state.get("candidates", []))
    done = len(state.get("evaluated", []))
    excluded = sum(1 for c in state.get("candidates", []) if c.get("selection_status") == "FAIL")
    return (
        f"- 후보 {total}개를 수집해 선정 기준(A-3)에서 {excluded}개를 제외하고 {done}개를 평가했다. "
        "조사 순서상 최초 투자 판정에서 종료했으므로 나머지 후보는 평가되지 않았으며, 검색 순서·후보 상한에 따라 일부 기업이 평가되지 않았을 수 있다.\n"
        "- 정량 구간은 공식 VC 표준이 아닌 프로젝트 초기 설계값이며, 누적 투자액은 조달 이력의 대리 지표(현재 자금 여력 아님)다.\n"
        "- Valuation·지분율이 비공개이므로 ROI는 산출하지 않았고, 기대 성장 여력은 시장·사업화 단계로 정성 평가했다.\n"
        "- 서로 다른 기업 유형의 원수치는 직접 비교하지 않는다."
    )


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
    catalog = Catalog(focus)
    sec = _write_sections(rtype, focus, catalog, scale, as_of)

    if rtype == "invest":
        p = main["company_profile"]
        head = (f"**{_name(main)}**({p.get('country')}, {TYPE_LABEL.get(p.get('company_type'), '유형 미확정')}) · "
                f"총점 **{main['total_score']:.2f}/100** · 판정 **투자** · 평가 기준일 {as_of}")
    else:
        best = max((r["total_score"] for r in evaluated if r.get("total_score") is not None), default=None)
        head = (f"평가 후보 {len(evaluated)}개 **전원 보류** · 최고 총점 {best:.2f}/100 · 평가 기준일 {as_of}"
                if best is not None else f"평가 후보 {len(evaluated)}개 전원 보류 · 평가 기준일 {as_of}")

    parts = [f"# SUMMARY\n\n{head}\n\n{sec.summary}", f"## 1. 기업 및 사업 개요\n\n{sec.overview}",
             f"## 2. 팀·기술 및 제품 경쟁력\n\n{sec.team_tech}"]
    market = f"## 3. 시장성 및 경쟁 환경\n\n{sec.market_competition}\n\n**시장 지표**\n\n{_market_table(focus, catalog)}"
    comps = [
        c for r in focus for c in r.get("competitor_analysis", {}).get("competitors", [])
        if str(c.get("name", "")).strip() != _name(r)
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
    body4 += (f"\n\n### 주요 리스크\n\n- **시장**: {sec.risk_market}\n- **기술**: {sec.risk_tech}\n"
              f"- **규제**: {sec.risk_regulation}\n- **경쟁**: {sec.risk_competition}")
    if holds:
        body4 += f"\n\n### 앞서 평가한 보류 후보\n\n{_hold_summary(holds, reason_limit)}"
    body4 += f"\n\n### 한계\n\n{_limitations(state)}"
    parts.append(f"## 4. 투자 판단 및 주요 리스크\n\n{body4}")
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
    """보고서 md·PDF를 만들고, 5장 초과 시 서술을 줄여 최대 2회 재생성한다."""
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
    return {"final_report": generate_report(state)["markdown"]}
