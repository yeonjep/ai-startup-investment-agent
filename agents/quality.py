# Report quality gate: Groundedness and source concentration in code, neutrality by LLM.
# Perspective coverage is intentionally absent. See docs/DESIGN.md E-1.

import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.common import create_llm
from agents.config import LLM_JUDGE_MODEL
from agents.report import REPORT_BASENAME
from agents.state import InvestmentState
from tools.pdf_export import render_markdown_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
QUANTITY = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|％|억|조|만|달러|원|건|명|배|TOPS|nm)",
    re.IGNORECASE,
)
SUPERIORITY = re.compile(r"세계\s*최고|압도적|독보적|확실히\s*우수|경쟁사를\s*완전히")


class NeutralityJudgement(BaseModel):
    passed: bool = Field(description="근거 없는 기술 우열 단정이 없으면 true")
    reason: str


def check_groundedness(report: str, state: dict[str, Any]) -> tuple[bool, str]:
    evaluated = list(state.get("evaluated") or [])
    if not evaluated:
        if "평가 대상 없음" in report or "선정 기준에서 제외" in report:
            return True, "평가 기록이 없는 보고서는 선정 결과만 적는다"
        return False, "평가 기록 없이 투자 평가 문장을 적음"

    body = report.split("# REFERENCE")[0]
    evidence_text = "\n".join(
        f"{item.get('value') or ''} {item.get('quote') or ''} {item.get('scope') or ''}"
        for record in evaluated
        for item in record.get("evidence") or []
    )
    allowed = _allowed_numbers(state)
    failures: list[str] = []
    for sentence in re.split(r"\n+|(?<=[.!?。])\s+", body):
        if re.search(r"\[\d+\]", sentence):
            continue
        for match in QUANTITY.finditer(sentence):
            token = match.group(0)
            number = re.match(r"\d+(?:\.\d+)?", token)
            number_text = number.group(0) if number else token
            if number_text in allowed or token in evidence_text or number_text in evidence_text:
                continue
            failures.append(sentence.strip()[:120])
            break
    if failures:
        return False, "출처와 연결되지 않은 수치: " + " / ".join(failures[:3])
    return True, "수치 문장이 근거 또는 채점 결과와 연결됨"


def check_bias(report: str) -> tuple[bool, str]:
    body, _, references = report.partition("# REFERENCE")
    marks = re.findall(r"\[(\d+)\]", body)
    if len(marks) < 3:
        return True, "인용이 3개 미만"
    identities = _reference_identities(references)
    counts = Counter(identities.get(mark, f"unknown:{mark}") for mark in marks)
    total = sum(counts.values())
    source, count = counts.most_common(1)[0]
    if total and count / total > 0.7:
        return False, f"{source} 비중이 {count}/{total}로 70%를 넘음"
    return True, "단일 출처 편중이 70% 이하"


def judge_neutrality(report: str) -> tuple[bool, str]:
    body = report.split("# REFERENCE")[0]
    obvious = SUPERIORITY.search(body)
    if obvious:
        return False, f"근거 없는 우열 표현: {obvious.group(0)}"
    try:
        llm = create_llm(LLM_JUDGE_MODEL, temperature=0).with_structured_output(NeutralityJudgement)
        result = llm.invoke(
            [
                (
                    "system",
                    "투자/보류 판정 문장은 허용한다. Evidence에 없는 기술 우열 단정만 실패로 본다. "
                    "점수와 결측 서술은 우열 단정이 아니다.",
                ),
                ("human", body[:6000]),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - 판정 실패는 품질 실패로 남기고 루프 상한으로 끝낸다
        return False, f"중립성 판정 오류: {type(exc).__name__}: {exc}"
    if result is None:
        return False, "중립성 판정 결과 없음"
    return bool(result.passed), result.reason


def quality_agent(state: InvestmentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    report = str(state.get("final_report") or "")
    grounded_ok, grounded_reason = check_groundedness(report, state)
    neutral_ok, neutral_reason = judge_neutrality(report)
    bias_ok, bias_reason = check_bias(report)
    result = {
        "passed": grounded_ok and neutral_ok and bias_ok,
        "checks": {
            "groundedness": {"passed": grounded_ok, "reason": grounded_reason},
            "neutrality": {"passed": neutral_ok, "reason": neutral_reason},
            "bias": {"passed": bias_ok, "reason": bias_reason},
        },
    }
    if result["passed"] or int(state.get("quality_retry") or 0) < 1:
        return {"quality_result": result, "last_error": None if neutral_ok or "오류" not in neutral_reason else neutral_reason}

    note = "\n".join(
        f"- {name}: {check['reason']}"
        for name, check in result["checks"].items()
        if not check["passed"]
    )
    text = report.rstrip() + "\n\n## 품질 평가\n\n재작성 후에도 아래 항목이 미달이다.\n\n" + note + "\n"
    basename = (config or {}).get("configurable", {}).get("report_basename", REPORT_BASENAME)
    _write_report_files(text, basename)
    return {"quality_result": result, "final_report": text}


def _allowed_numbers(state: dict[str, Any]) -> set[str]:
    allowed = {str(state.get("as_of_date") or ""), str(len(state.get("candidates") or [])), str(len(state.get("evaluated") or []))}
    for record in state.get("evaluated") or []:
        total = record.get("total_score")
        if isinstance(total, (int, float)):
            allowed.update({f"{total:.2f}", f"{total:.1f}", f"{total:g}"})
        allowed.add(str(len(record.get("missing_evidence") or [])))
        for entry in (record.get("scores") or {}).values():
            score = entry.get("score")
            if isinstance(score, (int, float)):
                allowed.add(f"{score:g}")
            raw = entry.get("raw_value")
            if raw not in (None, ""):
                allowed.add(str(raw))
    return {item for item in allowed if item}


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


def _write_report_files(markdown_text: str, basename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{basename}.md").write_text(markdown_text, encoding="utf-8")
    render_markdown_pdf(markdown_text, OUTPUT_DIR / f"{basename}.pdf")
