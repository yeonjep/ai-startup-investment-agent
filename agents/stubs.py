# 설계 확정 후 교체할 부분: 아래 여섯 노드는 흐름 확인용 더미 구현입니다.

from typing import Any

from agents.state import InvestmentState


def startup_search(state: InvestmentState) -> dict[str, Any]:
    candidates = state.get("candidates", [])
    current_idx = state.get("current_idx", 0)
    current_startup = candidates[current_idx] if current_idx < len(candidates) else None
    return {
        "current_startup": current_startup,
        "tech_summary": "",
        "market_analysis": "",
        "competitor_analysis": "",
        "scores": {},
        "total_score": None,
        "missing_evidence": [],
        "decision": "",
        "sources": [],
    }


def technology_summary(state: InvestmentState) -> dict[str, str]:
    startup = state.get("current_startup") or {}
    return {"tech_summary": f"[stub] {startup.get('name', '후보')} 기술 요약 자리"}


def market_evaluation(state: InvestmentState) -> dict[str, str]:
    startup = state.get("current_startup") or {}
    return {"market_analysis": f"[stub] {startup.get('name', '후보')} 시장성 평가 자리"}


def competitor_comparison(state: InvestmentState) -> dict[str, str]:
    startup = state.get("current_startup") or {}
    return {
        "competitor_analysis": f"[stub] {startup.get('name', '후보')} 경쟁사 비교 자리"
    }


def investment_decision(state: InvestmentState) -> dict[str, Any]:
    startup = state.get("current_startup") or {}
    scenario = state.get("scenario", "all_hold")
    decision = "투자" if scenario == "invest" and state.get("current_idx", 0) == 0 else "보류"
    evaluated = [
        *state.get("evaluated", []),
        {
            "startup": startup.get("name", "후보"),
            "decision": decision,
            "reason": "[stub] 시나리오 분기 확인용 결과; 평가표 및 판단 기준 확정 후 교체",
        },
    ]
    return {
        "decision": decision,
        "current_idx": state.get("current_idx", 0) + 1,
        "evaluated": evaluated,
    }


def report_generation(state: InvestmentState) -> dict[str, str]:
    evaluated = state.get("evaluated", [])
    candidate_lines = [
        f"- {item.get('startup', '후보')}: {item.get('decision', '미정')}"
        for item in evaluated
    ] or ["- 평가 대상이 없습니다."]
    report = "\n".join(
        [
            "# SUMMARY",
            "",
            "이 문서는 그래프 실행 확인용 더미 보고서입니다. 투자 판단과 분석 내용은 설계 확정 후 교체합니다.",
            "",
            "## 후보 처리 결과",
            *candidate_lines,
            "",
            "# REFERENCE",
            "",
            "실제 활용 자료 없음 (stub).",
        ]
    )
    return {"final_report": report}