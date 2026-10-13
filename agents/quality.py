# Quality gate: four guide checks. Citation, source mix, and headings are code; neutrality and contradictions are an LLM judge.

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.config import LLM_JUDGE_MODEL
from agents.state import InvestmentState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
QUANTITY = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|억|조|만|달러|원|건|명|배|TOPS|nm)",
    re.IGNORECASE,
)
PERSPECTIVES = ("기술 성숙도", "시장성", "이해관계자", "도메인 적용")
CHECK_NAMES = ("groundedness", "neutrality", "bias_control", "perspective_coverage")


class QualityJudgement(BaseModel):
    neutrality_passed: bool = Field(description="근거 없는 우열·과장이 없으면 true")
    neutrality_reasons: list[str] = Field(default_factory=list)
    neutrality_section: str = Field(default="", description="고칠 절 제목. 통과면 빈 문자열")
    contradiction_passed: bool = Field(description="같은 보고서 안에서 서로 부정하는 사실이 없으면 true")
    contradiction_reasons: list[str] = Field(default_factory=list)
    contradiction_section: str = Field(default="", description="모순이 있는 절 제목")


def check_groundedness(report: str, state: dict[str, Any]) -> tuple[bool, str]:
    item = _groundedness_code(report, state)
    return item["passed"], item["reason"]


def check_bias(report: str, state: dict[str, Any] | None = None) -> tuple[bool, str]:
    item = _bias_code(report, state)
    return item["passed"], item["reason"]


def check_perspectives(report: str) -> tuple[bool, str]:
    item = _perspective_code(report)
    return item["passed"], item["reason"]


def quality_agent(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    report = str(state.get("final_report") or "")
    grounded = _groundedness_code(report, state)
    judged = _judge(report)
    if not judged["contradiction_passed"]:
        grounded = _merge_groundedness(grounded, judged)
    checks = {
        "groundedness": grounded,
        "neutrality": _check(
            "neutrality",
            judged["neutrality_passed"],
            judged["neutrality_reasons"] or ["중립성 판정 결과 없음"],
            judged["neutrality_section"] or "본문",
        ),
        "bias_control": _bias_code(report, state),
        "perspective_coverage": _perspective_code(report),
    }
    if judged["neutrality_passed"]:
        checks["neutrality"] = _check("neutrality", True, ["근거 없는 우열·과장이 없음"], "")
    result = {
        "passed": all(checks[name]["passed"] for name in CHECK_NAMES),
        "checks": checks,
    }
    _save_quality(result, state, config)
    return {"quality_result": result, "last_error": None}


def _groundedness_code(report: str, state: dict[str, Any]) -> dict[str, Any]:
    evaluated = list(state.get("evaluated") or [])
    if not evaluated:
        if "평가 대상 없음" in report or "선정 기준에서 제외" in report:
            return _check("groundedness", True, ["평가 기록이 없는 보고서는 선정 결과만 적는다"], "")
        return _check("groundedness", False, ["평가 기록 없이 투자 평가 문장을 적음"], "SUMMARY")
    if "[INVALID:E" in report:
        return _check("groundedness", False, ["근거 카탈로그에 없는 인용 태그가 포함됨"], _section_at(report, "[INVALID:"))

    body, _, references = report.partition("# REFERENCE")
    reference_numbers = set(_reference_identities(references))
    reasons: list[str] = []
    section = "본문"
    for paragraph, paragraph_section in _prose_paragraphs(body):
        cited = set(re.findall(r"\[(\d+)\]", paragraph))
        missing = sorted(cited - reference_numbers)
        if missing:
            reasons.append("REFERENCE에 없는 인용 번호: " + ", ".join(missing))
            section = paragraph_section
        paragraph_cited = bool(re.search(r"\[\d+\]\s*[.!?。]?\s*$", paragraph.strip()))
        for sentence in re.split(r"(?<=[.!?。])\s+", paragraph):
            if not QUANTITY.search(sentence) or "출처 충돌" in sentence or "상충 근거" in sentence:
                continue
            sentence_cited = bool(re.search(r"\[\d+\]", sentence)) or paragraph_cited
            if not sentence_cited:
                reasons.append(sentence.strip()[:120])
                section = paragraph_section
                break
            mismatch = _citation_mismatch(sentence, evaluated, references)
            if mismatch:
                reasons.append(mismatch)
                section = paragraph_section
                break
    for row in _table_rows(body):
        if not QUANTITY.search(row):
            continue
        mismatch = _citation_mismatch(row, evaluated, references)
        if mismatch:
            reasons.append(mismatch)
            section = "Scorecard"
    if reasons:
        return _check("groundedness", False, reasons[:3], section)
    return _check("groundedness", True, ["수치·사실 문장이 REFERENCE 인용과 연결됨"], "")


def _bias_code(report: str, state: dict[str, Any] | None) -> dict[str, Any]:
    body, _, references = report.partition("# REFERENCE")
    reasons: list[str] = []
    marks = re.findall(r"\[(\d+)\]", body)
    if len(marks) >= 3:
        identities = _reference_identities(references)
        counts = Counter(identities.get(mark, f"unknown:{mark}") for mark in marks)
        total = sum(counts.values())
        source, count = counts.most_common(1)[0]
        if total and count / total > 0.7:
            reasons.append(f"단일 출처 비중 {count}/{total}이 70%를 넘음 ({source})")
    if state and _has_unmarked_conflict(report, state):
        reasons.append("수치 충돌이 보고서에 표기되지 않음")
    if reasons:
        return _check("bias_control", False, reasons, "본문")
    return _check("bias_control", True, ["단일 출처 편중이 70% 이하이고 수치 충돌을 표기함"], "")


def _perspective_code(report: str) -> dict[str, Any]:
    if "평가 대상 없음" in report:
        return _check("perspective_coverage", True, ["평가 대상이 없어 관점 소제목을 요구하지 않음"], "")
    body = report.split("# REFERENCE")[0]
    missing = [name for name in PERSPECTIVES if name not in body]
    if missing:
        section = "팀·기술 및 제품 경쟁력" if any(name != "시장성" for name in missing) else "시장성 및 경쟁 환경"
        return _check("perspective_coverage", False, ["관점 소제목 누락: " + ", ".join(missing)], section)
    return _check("perspective_coverage", True, ["기술 성숙도·시장성·이해관계자·도메인 적용 소제목이 있음"], "")


def _judge(report: str) -> dict[str, Any]:
    body = report.split("# REFERENCE")[0][:6000]
    empty = {
        "neutrality_passed": False,
        "neutrality_reasons": ["중립성 판정 결과 없음"],
        "neutrality_section": "본문",
        "contradiction_passed": True,
        "contradiction_reasons": [],
        "contradiction_section": "",
    }
    try:
        llm = create_llm(LLM_JUDGE_MODEL, temperature=0).with_structured_output(QualityJudgement)
        result = llm.invoke(
            [
                (
                    "system",
                    "투자 보고서 품질을 두 가지만 판정한다. "
                    "neutrality: '견고하다', '획기적', '압도적', '차별화된다'처럼 근거 없이 우열이나 과장을 단정하면 실패다. "
                    "투자/보류 판정 문장과 점수표 규칙은 실패가 아니다. "
                    "contradiction: 같은 보고서가 한 사실을 긍정하고 다른 문장에서 그 사실을 부정하면 groundedness 위반이다. "
                    "예: SDK를 보유한다고 하면서 SW 스택이 미확인이라고 하는 경우. "
                    "실패 시 reasons에 짧은 한국어 사유를 적고, section에는 고칠 절 제목만 적는다.",
                ),
                ("human", body),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - 판정 실패는 품질 실패로 남기고 루프 상한으로 끝낸다
        empty["neutrality_reasons"] = [f"중립성 판정 오류: {type(exc).__name__}: {exc}"]
        return empty
    if result is None:
        return empty
    return {
        "neutrality_passed": bool(result.neutrality_passed),
        "neutrality_reasons": list(result.neutrality_reasons or []),
        "neutrality_section": result.neutrality_section or "본문",
        "contradiction_passed": bool(result.contradiction_passed),
        "contradiction_reasons": list(result.contradiction_reasons or []),
        "contradiction_section": result.contradiction_section or "본문",
    }


def _merge_groundedness(code_item: dict[str, Any], judged: dict[str, Any]) -> dict[str, Any]:
    reasons = [] if code_item["passed"] else list(code_item.get("reasons") or [])
    reasons.extend(judged["contradiction_reasons"] or ["보고서 안에 서로 모순되는 사실이 있음"])
    section = judged["contradiction_section"] or code_item.get("target_section") or "본문"
    return _check("groundedness", False, reasons[:4], section)


def _check(item: str, passed: bool, reasons: list[str], section: str) -> dict[str, Any]:
    cleaned = [reason for reason in reasons if reason]
    return {
        "item": item,
        "passed": passed,
        "reasons": cleaned,
        "target_section": "" if passed else section,
        "reason": "; ".join(cleaned),
    }


def _prose_paragraphs(body: str) -> list[tuple[str, str]]:
    section = "SUMMARY"
    paragraphs: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", body):
        lines: list[str] = []
        block_section = section
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                section = stripped.lstrip("#").strip()
                block_section = section
                continue
            if not stripped or stripped.startswith("|") or stripped.startswith("![") or stripped.startswith("<"):
                continue
            lines.append(stripped)
        if lines:
            paragraphs.append((" ".join(lines), block_section))
    return paragraphs


def _section_at(report: str, snippet: str) -> str:
    current = "SUMMARY"
    for line in report.splitlines():
        if snippet[:40] in line:
            return current
        if line.startswith("#"):
            current = line.lstrip("#").strip()
    return current


def _has_unmarked_conflict(report: str, state: dict[str, Any]) -> bool:
    from agents.report import _quantitative_conflicts

    conflicts = _quantitative_conflicts(list(state.get("evaluated") or []))
    if not conflicts:
        return False
    if "출처 충돌" not in report and "상충" not in report:
        return True
    for conflict in conflicts:
        for item in conflict["items"]:
            if str(item.get("value")) not in report:
                return True
    return False


_MEASURE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(%|％|억원|억\s*원|조원|조\s*원|달러|원|건|명|배)",
    re.IGNORECASE,
)
_MARKET_SCOPE = re.compile(r"[가-힣A-Za-z0-9]{2,24}시장")


def _scaled_measure(number: float, unit: str) -> tuple[float, str]:
    text = str(unit or "").replace(" ", "")
    if "억" in text and "원" in text:
        return number * 100_000_000, "KRW"
    if text.upper() == "KRW" or text == "원":
        return number, "KRW"
    if text in {"%", "％"}:
        return number, "%"
    if "달러" in text or text.upper() == "USD":
        return number, "USD"
    if "명" in text:
        return number, "명"
    return number, text


def _measured(text: str) -> list[tuple[float, str]]:
    found = []
    for match in _MEASURE.finditer(text or ""):
        number = float(match.group(1).replace(",", ""))
        found.append(_scaled_measure(number, match.group(2)))
    return found


def _evidence_measures(evidence: list[dict[str, Any]]) -> list[tuple[float, str, str]]:
    from agents.report import _canonical_amount

    measures = []
    for item in evidence:
        blob = f"{item.get('quote') or ''} {item.get('scope') or ''}"
        parsed = _canonical_amount(item.get("value"), item.get("unit"))
        if parsed is not None:
            measures.append((parsed[0], parsed[1], blob))
        for number, unit in _measured(blob):
            measures.append((number, unit, blob))
        for number, unit in _measured(f"{item.get('value') or ''} {item.get('unit') or ''}"):
            measures.append((number, unit, blob))
    return measures


def _citation_mismatch(text: str, evaluated: list[dict[str, Any]], references: str) -> str:
    evidence = [item for record in evaluated for item in (record.get("evidence") or [])]
    if not evidence:
        return ""
    measures = _evidence_measures(evidence)
    for number, unit in _measured(text):
        same = [item for item in measures if abs(item[0] - number) <= 1e-6 and item[1] == unit]
        if not same:
            return f"인용 값 {number:g}{unit}이 근거 값·단위와 다름"
    scopes = _MARKET_SCOPE.findall(text)
    if scopes:
        blob = " ".join(f"{item.get('quote') or ''} {item.get('scope') or ''}" for item in evidence)
        missing = [scope for scope in scopes if scope not in blob]
        if missing and blob.strip():
            return f"시장 범위 {missing[0]}이 인용 근거와 다름"
    _ = references
    return ""


def _table_rows(body: str) -> list[str]:
    rows = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and not stripped.startswith("|---") and "원값" not in stripped and "지표" not in stripped:
            rows.append(stripped)
    return rows


def _reference_identities(references: str) -> dict[str, str]:
    identities: dict[str, str] = {}
    for line in references.splitlines():
        match = re.match(r"(\d+)\.\s+(.*)", line.strip())
        if not match:
            continue
        number, text = match.group(1), match.group(2)
        document = re.search(r"문서 ID ([A-Za-z0-9_\-]+)", text)
        url = re.search(r"https?://[^\s)]+", text)
        if document:
            identities[number] = f"doc:{document.group(1)}"
        elif url:
            host = urlparse(url.group(0)).netloc.lower().removeprefix("www.")
            identities[number] = f"host:{host}"
        else:
            identities[number] = f"line:{number}"
    return identities


def _save_quality(result: dict[str, Any], state: dict[str, Any], config: RunnableConfig | None) -> None:
    payload = {
        "passed": result["passed"],
        "loop_count": int(state.get("quality_retry") or 0),
        "checks": [result["checks"][name] for name in CHECK_NAMES],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "quality_eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[quality] passed={payload['passed']} loop={payload['loop_count']}")
    for check in payload["checks"]:
        if not check["passed"]:
            print(f"[quality] {check['item']} section={check['target_section']} {check['reason']}")
    _trace_quality(payload)
    _ = config


def _trace_quality(payload: dict[str, Any]) -> None:
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        return
    run = get_current_run_tree()
    if run is None:
        return
    try:
        run.add_metadata({"quality_eval": payload})
    except Exception:
        metadata = dict(getattr(run, "metadata", None) or {})
        metadata["quality_eval"] = payload
        try:
            run.metadata = metadata
        except Exception:
            return
