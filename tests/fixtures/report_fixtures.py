# Report Agent 개발용 가짜 State. 기업·수치·URL은 모두 가상이며 D-1/D-2 스키마 형식을 따른다.
# 실행: uv run python -m tests.fixtures.report_fixtures  (같은 폴더에 JSON 덤프)

import json
from pathlib import Path

from agents.config import SCORE_WEIGHTS

AS_OF = "2026-09-30"
GROUPS = {
    "team": ["team_experience", "technical_headcount"],
    "market": ["market_tam", "market_cagr", "demand_clarity"],
    "product_technology": ["development_stage", "technical_originality"],
    "competitive_advantage": ["competitive_difference", "entry_barrier"],
    "traction": ["customer_traction", "revenue_stage"],
    "funding_risk": ["funding_total", "risk_mitigation"],
}

RAG_DOCS = {
    "kisdi_2024_perspectives": ("새로운 기회의 창으로 AI반도체 시장 현황과 전망", "정보통신정책연구원", "2024-07"),
    "gov_2025_strategy": ("AI반도체 산업 도약 전략(안)(요약본)", "관계부처 합동", "2025-12"),
    "deloitte_2026_outlook": ("2026 Semiconductor Industry Outlook", "Deloitte", "2026"),
}


def rag_ev(company, metric, doc_id, page, quote, value=None, unit=None, scope=None):
    title, source, date = RAG_DOCS[doc_id]
    return {
        "evidence_id": f"{company}:market:{metric}:{doc_id}-p{page}",
        "company": company, "category": "market", "metric": metric,
        "value": value, "unit": unit, "source_type": "rag",
        "title": title, "source": source, "date": date, "accessed_at": AS_OF,
        "url": None, "doc_id": doc_id, "page": page, "chunk_id": f"{doc_id}-p{page}-c0",
        "quote": quote, "scope": scope,
    }


def web_ev(company, category, metric, key, url, date, title, source, quote, value=None, unit=None, scope=None):
    return {
        "evidence_id": f"{company}:{category}:{metric}:{key}",
        "company": company, "category": category, "metric": metric,
        "value": value, "unit": unit, "source_type": "web",
        "title": title, "source": source, "date": date, "accessed_at": AS_OF,
        "url": url, "doc_id": None, "page": None, "chunk_id": None,
        "quote": quote, "scope": scope,
    }


def build_scores(spec):
    """spec: metric -> (score, status, raw_value, reason, evidence_ids)."""
    scores = {
        metric: {"metric": metric, "raw_value": raw, "score": float(score),
                 "status": status, "reason": reason, "evidence_ids": ids}
        for metric, (score, status, raw, reason, ids) in spec.items()
    }
    total = sum(
        sum(scores[m]["score"] for m in metrics) / len(metrics) / 5 * SCORE_WEIGHTS[group]
        for group, metrics in GROUPS.items()
    )
    missing = [m for m, s in scores.items() if s["status"] == "missing"]
    return scores, total, missing


def _record(profile, uncertain, tech, market, comp, evidence, spec, decision, reason):
    scores, total, missing = build_scores(spec)
    return {
        "company_profile": profile, "selection_uncertain": uncertain,
        "tech_summary": tech, "market_analysis": market, "competitor_analysis": comp,
        "evidence": evidence, "scores": scores, "total_score": total,
        "missing_evidence": missing, "decision": decision, "reason": reason,
    }


def _company(name, country, ctype, product, domain, stage_note):
    return {"name": name, "country": country, "company_type": ctype,
            "evaluation_product": product, "official_domain": domain,
            "latest_round": stage_note, "profile_evidence": []}


def _ev_set(name, domain, tag):
    web = lambda cat, metric, key, path, date, title, quote, **kw: web_ev(  # noqa: E731
        name, cat, metric, key, f"https://{domain}/{path}", date, title, domain, quote, **kw)
    return {
        "team": web("profile", "team_experience", "team", "about", "2026-03-12", f"{tag} 창업팀 소개",
                    "CEO는 글로벌 반도체 기업에서 NPU 개발을 15년간 리드했다."),
        "head": web("profile", "technical_headcount", "head", "careers", "2026-05-02", f"{tag} 채용 페이지",
                    "R&D 인력 85명", value=85, unit="명"),
        "fund": web("profile", "funding_total", "fund", "news/series-b", "2026-06-20", f"{tag} 시리즈 B 유치",
                    "누적 투자 유치액 1,200억 원", value=120000000000, unit="KRW", scope="원값 1,200억 원(KRW)"),
        "cust": web("profile", "customer_traction", "cust", "news/poc", "2026-07-08", f"{tag} 고객 PoC 발표",
                    "국내 3개 고객사와 PoC 진행", value=3, unit="건"),
        "tech": web("tech", "technical_originality", "tech", "tech", "2026-04-15", f"{tag} 기술 백서",
                    "독자 데이터플로 구조로 동일 워크로드 대비 전성비 개선을 보고했다.",
                    scope="INT8, batch 1, 동일 전력 조건 (자체 측정)"),
        "stage": web("tech", "development_stage", "stage", "news/tapeout", "2026-02-10", f"{tag} 테이프아웃 완료",
                     "2세대 칩 테이프아웃 완료", value=4, unit="단계"),
        "comp": web("competitor", "competitive_difference", "comp", "blog/benchmark", "2026-08-01", f"{tag} 벤치마크",
                    "동일 조건에서 경쟁 GPU 대비 처리량 우위를 주장했다."),
    }


def invest_case():
    a, b, c = "[가상]넥스트칩AI", "[가상]오로라실리콘", "[가상]팹옵스AI"
    ea = _ev_set(a, "nextchip-ai.test", "넥스트칩AI")
    ea["mkt"] = rag_ev(a, "market_cagr", "kisdi_2024_perspectives", 7,
                       "글로벌 AI 반도체 시장은 연평균 성장률 약 30%로 전망된다.", value=30, unit="%",
                       scope="AI 반도체 글로벌, 2023~2030 예측")
    ea["tam"] = rag_ev(a, "market_tam", "deloitte_2026_outlook", 4,
                       "Generative AI chips are expected to exceed US$150 billion in 2026 sales.",
                       value=150e9, unit="USD", scope="생성형 AI 칩, 글로벌, 2026")
    ea["pol"] = rag_ev(a, "demand_clarity", "gov_2025_strategy", 3,
                       "정부는 AI 반도체 국산화와 데이터센터 수요 연계를 핵심 과제로 제시했다.", scope="국내 정책, 2025.12")
    spec_a = {
        "team_experience": (5, "observed", "NPU 개발 리드 15년", "해당 유형 개발 리드 이력", [ea["team"]["evidence_id"]]),
        "technical_headcount": (4, "observed", "85명", "R&D 인력 50~99명", [ea["head"]["evidence_id"]]),
        "market_tam": (5, "observed", "USD 150B", "1,000억 달러 이상", [ea["tam"]["evidence_id"]]),
        "market_cagr": (5, "observed", "30%", "20% 이상", [ea["mkt"]["evidence_id"]]),
        "demand_clarity": (3, "observed", "정책·DC 수요", "타깃만 제시", [ea["pol"]["evidence_id"]]),
        "development_stage": (4, "observed", "테이프아웃", "CHIP 4점", [ea["stage"]["evidence_id"]]),
        "technical_originality": (3, "observed", "자체 측정 주장", "주장만 있음(제3자 검증 없음)", [ea["tech"]["evidence_id"]]),
        "competitive_difference": (3, "observed", "벤치마크 주장", "주장만 있음", [ea["comp"]["evidence_id"]]),
        "entry_barrier": (3, "observed", "특허 1종 확인", "1~2종 확인", [ea["tech"]["evidence_id"]]),
        "customer_traction": (4, "observed", "3건", "PoC 3건", [ea["cust"]["evidence_id"]]),
        "revenue_stage": (3, "observed", "초기 매출", "초기·일회성 매출", [ea["cust"]["evidence_id"]]),
        "funding_total": (4, "observed", "1,200억 원 (KRW)", "1,000억~3,000억", [ea["fund"]["evidence_id"]]),
        "risk_mitigation": (3, "observed", "일부 완화", "공급망 이원화 일부", [ea["tech"]["evidence_id"]]),
    }
    rec_a = _record(
        _company(a, "KR", "CHIP", "NX-2 AI 추론 가속기", "nextchip-ai.test", "Series B (2026-06)"), False,
        {"summary": "독자 데이터플로 NPU. 2세대 칩 테이프아웃 완료, 시험 칩 기준 전성비 개선 주장.",
         "development_stage": "테이프아웃(4점)", "strengths": ["팀 상용화 경험", "테이프아웃 완료"],
         "weaknesses": ["제3자 검증 벤치마크 없음", "SW 생태계 미성숙"]},
        {"summary": "AI 반도체 시장 연평균 약 30% 성장 전망(글로벌, 2023~2030).",
         "market_size": "생성형 AI 칩 2026년 1,500억 달러 초과(Deloitte)", "growth": "CAGR 약 30%",
         "demand": "데이터센터·국내 국산화 정책", "regulation": "수출 통제·국내 지원 정책 (국내 정책은 해외에 그대로 적용 불가)"},
        {"summary": "동일 워크로드 GPU 대비 전성비 우위 주장, 실측 검증은 제한적.",
         "competitors": [{"name": "[가상]GPU 기존 방식", "country": "US", "comparison": "SW 생태계 우위, 전력효율 열세"},
                         {"name": "[가상]타사 NPU", "country": "KR", "comparison": "양산 단계 앞섬"}]},
        list(ea.values()), spec_a, "투자",
        "총점 70 이상, 결측 5개 미만, 선정 불확실성 없음. 기대 성장 여력은 시장 규모·성장률과 테이프아웃 단계에서 확인되나 검증 근거는 제한적.")

    eb = _ev_set(b, "aurora-silicon.test", "오로라실리콘")
    spec_b = {
        "team_experience": (3, "observed", "관련 경력", "관련 경력 있음", [eb["team"]["evidence_id"]]),
        "technical_headcount": (2, "observed", "12명", "10~19명", [eb["head"]["evidence_id"]]),
        "market_tam": (3, "missing", None, "EDA 시장 근거 미확보 — 중립 대입", []),
        "market_cagr": (3, "missing", None, "근거 미확보 — 중립 대입", []),
        "demand_clarity": (3, "observed", "고객사 타깃", "타깃만 제시", [eb["cust"]["evidence_id"]]),
        "development_stage": (3, "observed", "프로토타입", "DESIGN_AI 3점", [eb["stage"]["evidence_id"]]),
        "technical_originality": (3, "observed", "주장", "주장만 있음", [eb["tech"]["evidence_id"]]),
        "competitive_difference": (3, "observed", "주장", "주장만 있음", [eb["comp"]["evidence_id"]]),
        "entry_barrier": (3, "missing", None, "자료 없음 — 중립 대입", []),
        "customer_traction": (2, "observed", "1건", "PoC 1건", [eb["cust"]["evidence_id"]]),
        "revenue_stage": (1, "observed", "매출 없음 확인", "매출 없음 확인", [eb["cust"]["evidence_id"]]),
        "funding_total": (2, "observed", "USD 12M → 약 163억 원", "100억~300억", [eb["fund"]["evidence_id"]]),
        "risk_mitigation": (3, "observed", "일부 완화", "일부", [eb["tech"]["evidence_id"]]),
    }
    rec_b = _record(
        _company(b, "US", "DESIGN_AI", "Aurora Place-AI", "aurora-silicon.test", "Series A (2026-01)"), False,
        {"summary": "AI 기반 배치 최적화 EDA. 작동 프로토타입 검증 단계.", "development_stage": "프로토타입(3점)"},
        {"summary": "EDA 시장 근거를 확보하지 못해 시장 지표 결측."},
        {"summary": "기존 EDA 도구 대비 비교 근거 부족.", "competitors": []},
        list(eb.values()), spec_b, "보류", "총점 70 미만.")
    ec = _ev_set(c, "fabops-ai.test", "팹옵스AI")
    spec_c = {m: (3, "missing", None, "근거 미확보 — 중립 대입", []) for m in sum(GROUPS.values(), [])}
    spec_c["team_experience"] = (3, "observed", "관련 경력", "관련 경력 있음", [ec["team"]["evidence_id"]])
    spec_c["technical_headcount"] = (1, "observed", "8명", "10명 미만", [ec["head"]["evidence_id"]])
    spec_c["funding_total"] = (1, "observed", "약 40억 원", "100억 미만", [ec["fund"]["evidence_id"]])
    rec_c = _record(
        {**_company(c, "KR", "PROCESS_AI", "수율 예측 AI", "fabops-ai.test", "Seed (2025-11)"),
         "uncertain_criteria": ["exit_not_completed"]}, True,
        {"summary": "공정 수율 예측 모델. 고객 현장 검증 근거 미확인."},
        {"summary": "시장 근거 미확보."}, {"summary": "경쟁사 비교 근거 미확보.", "competitors": []},
        list(ec.values()), spec_c, "보류", "결측 5개 이상, 선정 불확실성 존재.")
    candidates = [{"name": n, "selection_status": s} for n, s in
                  [(a, "PASS"), (b, "PASS"), (c, "REVIEW")]]
    return {"as_of_date": AS_OF, "max_candidates": 5, "candidates": candidates,
            "evaluated": [rec_b, rec_c, rec_a]}


def all_hold_case():
    inv = invest_case()
    b, c = inv["evaluated"][0], inv["evaluated"][1]
    d_name = "[가상]리소메트릭스"
    ed = _ev_set(d_name, "lithometrix.test", "리소메트릭스")
    spec_d = {m: (3, "missing", None, "근거 미확보 — 중립 대입", []) for m in sum(GROUPS.values(), [])}
    spec_d["team_experience"] = (5, "observed", "장비사 출신", "공정 현장 적용 리드", [ed["team"]["evidence_id"]])
    spec_d["technical_headcount"] = (3, "observed", "25명", "20~49명", [ed["head"]["evidence_id"]])
    spec_d["development_stage"] = (4, "observed", "고객 현장 PoC 완료", "PROCESS_AI 4점", [ed["stage"]["evidence_id"]])
    spec_d["funding_total"] = (2, "observed", "USD 9M → 약 122억 원", "100억~300억", [ed["fund"]["evidence_id"]])
    rec_d = _record(
        _company(d_name, "KR", "PROCESS_AI", "웨이퍼 결함 검사 AI", "lithometrix.test", "Series A (2026-04)"), False,
        {"summary": "AI 결함 검사. 고객 PoC 완료, 수율 효과의 대조 조건 불명확.", "development_stage": "현장 PoC(4점)"},
        {"summary": "공정 AI 세부 시장 근거 미확보로 시장 지표 결측."},
        {"summary": "기존 검사 장비 대비 비교 근거 부족.", "competitors": [
            {"name": "[가상]기존 광학 검사", "country": "JP", "comparison": "비AI 기준, 오탐률 비교 자료 없음"}]},
        list(ed.values()), spec_d, "보류", "결측 5개 이상.")
    b["decision"], c["decision"] = "보류", "보류"
    candidates = [{"name": r["company_profile"]["name"], "selection_status": "PASS"} for r in (b, c, rec_d)]
    return {"as_of_date": AS_OF, "max_candidates": 5, "candidates": candidates,
            "evaluated": [b, c, rec_d]}


def no_target_case():
    reasons = [
        ("[가상]퓨처로직", "FAIL", "unlisted", "코스닥 상장 완료로 확인"),
        ("[가상]클라우드칩스", "FAIL", "funding_stage", "최근 라운드가 Series D로 확인"),
        ("[가상]범용메모리랩", "FAIL", "ai_core", "AI가 부수 기능·마케팅 문구에 그침"),
    ]
    candidates = [
        {"name": n, "selection_status": s, "country": "KR", "failed_criterion": crit, "exclusion_reason": why,
         "sources": [{"title": f"{n} 기업 소개", "url": f"https://{i}-fake.test/about", "date": "2026-05-01"}]}
        for i, (n, s, crit, why) in enumerate(reasons)
    ]
    return {"as_of_date": AS_OF, "max_candidates": 5, "candidates": candidates, "evaluated": []}


FIXTURES = {"invest": invest_case, "all_hold": all_hold_case, "no_target": no_target_case}


def load_fixture(name: str) -> dict:
    return FIXTURES[name]()


if __name__ == "__main__":
    out = Path(__file__).parent
    for key, builder in FIXTURES.items():
        (out / f"{key}.json").write_text(
            json.dumps(builder(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out / f"{key}.json")
