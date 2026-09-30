"""영역별 추출 계약과 검증. source_id와 정확한 원문 없이 채점하지 않는다."""
import hashlib
import re
import unicodedata
from agents.criteria import CATEGORY_METRICS, UNITS, STAGES, RUBRICS

ROLE_TASKS = {
    'startup': '팀: 창업자 실명·경력, 기술 인력 수. 실적: 실제 고객·PoC·계약, 실현 매출. 투자: 누적 투자액과 기준일. 회사 소개 반복 금지.',
    'tech': '핵심 제품명·칩 아키텍처·공정·워크로드·공개 성능/전력효율·개발 단계·자체 SDK. 투자 유치/기업 연혁을 기술 분석으로 대신하지 말 것.',
    'market': '기업 타깃 시장별 규모(TAM), 기준연도/전망기간, CAGR, 실제 수요처, 기술·규제·공급망 리스크. 기업 소개/투자액 반복 금지. 전체 반도체 시장을 해당 기업 TAM으로 대입 금지.',
    'competitor': '동일 워크로드의 국내외 3~5개 경쟁사 이름, 비교 제품·성능 측정 조건·장단점, 특허·자체 SW·파트너십. 기업 소개 반복 금지. 근거 없는 우위 단정 금지.',
}
METRIC_DEFINITIONS = {
    'team_experience': '실명과 반도체 설계/양산 관련 경력. 정성 지표: value에 근거 내용을 짧은 문자열로 기재.',
    'technical_headcount': '명시된 기술/R&D 인력 수만. 전체 직원 수는 사용 불가. scope=technical_team.',
    'customer_count': '중복을 제거한 실제 고객/PoC/계약 수. 목표/예정 제외. 명시된 최소 건수는 scope=confirmed_lower_bound.',
    'revenue_stage': '실제로 발생한 매출만: 없음/초기 매출/양산 매출. 목표나 예상 매출로 매출 발생 판단 금지.',
    'funding_total': '누적 투자 유치액만. 누적/총 투자 문구 필요. value_kind=cumulative. 단일 Series 라운드는 value_kind=round로 구분하여 채점에서 제외. 과거 누적치는 관측일 명시.',
    'development_stage': '실제 도달한 단계만: 구상/설계/시제품/테이프아웃/양산. 양산 예정은 양산이 아님.',
    'technology_originality': '독자 구조, 효율 수치와 검증 수준을 문자열로 기록. 객관 수치 없으면 주장 수준임을 명시.',
    'market_tam': '직접 타깃인 시장의 규모, USD billion. scope=시장명; period=기준연도. 전망은 value_kind=forecast. USD 아닌 수치는 환율 근거 없으면 null.',
    'market_cagr': '기업 타깃 시장 CAGR %, scope=시장명, period=전망기간. 전년비 성장률과 구분.',
    'market_demand': '구체적인 타깃 수요처와 수요 근거를 문자열로 기록.',
    'risk': '기술·공급망·규제 위험 및 실제 완화 근거를 문자열로 기록. 일반 시장 위험을 회사가 해소했다고 단정 금지.',
    'competitive_edge': '비교 대상 제품/경쟁사와 동일 조건 벤치마크 유무를 문자열로 기록.',
    'entry_barriers': '특허/자체 SW 스택/전략 파트너십의 실제 확인 항목을 문자열로 기록.',
}


def contract(category):
    return {m: {'definition': METRIC_DEFINITIONS[m], 'unit': UNITS.get(m, ''),
                'allowed_values': list(STAGES[m]) if m in STAGES else None, 'rubric': RUBRICS.get(m)} for m in CATEGORY_METRICS[category]}


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text or ''))


CONVERSIONS = {
    'market_tam': {'USD billion': 1, 'USD million': .001, 'USD trillion': 1000, 'USD': 1e-9, '억 달러': .1},
    'market_cagr': {'%': 1}, 'technical_headcount': {'명': 1}, 'customer_count': {'건': 1},
    'funding_total': {'억 원': 1, '억원': 1, '원': 1e-8, 'KRW': 1e-8, '조 원': 10000},
}


def numeric_error(metric):
    import math
    if metric.metric not in CONVERSIONS:
        return None
    raw = metric.raw_value
    if raw is None or not math.isfinite(raw):
        return 'raw_number_missing'
    literal = format(raw, '.12g')
    quote = compact(metric.quote).replace(',', '')
    if not re.search(r'(?<![\d.])' + re.escape(literal) + r'(?![\d.])', quote):
        return 'raw_number_not_in_quote'
    factor = CONVERSIONS[metric.metric].get(metric.raw_unit)
    if factor is None or metric.unit != UNITS[metric.metric]:
        return 'unsupported_unit_conversion'
    if isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)) or not math.isclose(metric.value, raw * factor, rel_tol=1e-8):
        return 'numeric_conversion_mismatch'
    return None


def validate_metric(metric, source, company=''):
    if source is None:
        return 'unknown_source_id'
    if not metric.quote.strip() or compact(metric.quote) not in compact(source.get('content', '')):
        return 'quote_not_in_source'
    if metric.value is None:
        return 'value_missing'
    invalid_number = numeric_error(metric)
    if invalid_number:
        return invalid_number
    if metric.metric == 'funding_total':
        if metric.value_kind != 'cumulative' or not re.search(r'누적|총.{0,12}투자|total.{0,30}(fund|rais)|cumulative', metric.quote, re.I):
            return 'not_cumulative_funding'
    if metric.metric == 'technical_headcount' and (metric.scope != 'technical_team' or not re.search(r'기술|연구|개발|엔지니어|engineer|R&D|technical', metric.quote, re.I)):
        return 'not_technical_headcount'
    if metric.metric not in ('market_tam', 'market_cagr') and metric.value_kind in ('forecast', 'target', 'round', 'unknown'):
        return 'not_observed_fact'
    if metric.metric in ('market_tam', 'market_cagr') and (not metric.scope or not metric.period):
        return 'market_scope_or_period_missing'
    if metric.metric in STAGES and metric.value not in STAGES[metric.metric]:
        return 'invalid_stage'
    if metric.metric == 'development_stage' and metric.value == '양산':
        # "내년 양산", "양산에 들어갈 예정"은 현재 양산 실적이 아니다.
        production = r'양산|mass production'
        future = r'예정|목표|계획|전망|향후|내년|앞으로|추후|준비|will|plan|expected|aim'
        if re.search(rf'(?:(?:{future}).{{0,35}}(?:{production})|(?:{production}).{{0,35}}(?:{future}))', metric.quote, re.I | re.S):
            return 'stage_not_confirmed'
    if metric.metric == 'risk' and company and compact(company) not in compact(metric.quote):
        return 'company_risk_not_supported'
    if metric.metric in STAGES and re.search((r'양산[^.\n]{0,20}(예정|목표|계획)' if metric.metric == 'development_stage' and metric.value == '양산' else r'매출[^.\n]{0,25}(예정|목표|계획|전망|예상)'), metric.quote, re.I):
        return 'stage_not_confirmed'
    return None


def bind_analysis(analysis, sources, category, company):
    lookup = {s['source_id']: s for s in sources}
    evidence, rejected, seen = [], [], set()
    for metric in analysis.metrics:
        source = lookup.get(metric.source_id)
        reason = ('unexpected_metric' if metric.metric not in CATEGORY_METRICS[category]
                  else validate_metric(metric, source, company))
        if reason:
            rejected.append({'metric': metric.metric, 'reason': reason, 'value': metric.value,
                             'source_id': metric.source_id, 'quote': metric.quote})
            continue
        key = f'{company}|{category}|{metric.metric}|{source.get("chunk_id", source.get("url"))}|{metric.quote}'
        eid = 'EV-' + hashlib.sha256(key.encode()).hexdigest()[:10]
        if eid in seen:
            continue
        seen.add(eid)
        evidence.append({**metric.model_dump(), 'evidence_id': eid, 'company': company, 'category': category,
            'source_type': 'rag' if source.get('doc_id') else 'web',
            **{k: source.get(k) or '' for k in ('title','source','date','url','doc_id','page','chunk_id')},
            'location': f"p.{source['page']} / {source.get('chunk_id')}" if source.get('doc_id') else ''})
    # 추가 사실은 원문 인용을 검증한 Finding만 사용한다. 요약의 인용 번호만으로는 근거화하지 않는다.
    for finding in analysis.findings:
        source = lookup.get(finding.source_id)
        if not source or not finding.quote.strip() or compact(finding.quote) not in compact(source.get('content', '')):
            rejected.append({'metric': 'context', 'reason': 'unverified_finding', 'source_id': finding.source_id})
            continue
        eid = 'EV-' + hashlib.sha256(f'{company}|{category}|{finding.quote}|{source}'.encode()).hexdigest()[:10]
        if eid in seen:
            continue
        seen.add(eid)
        evidence.append({**finding.model_dump(), 'metric': 'context', 'value': finding.statement,
            'evidence_id': eid, 'company': company, 'category': category, 'unit': '',
            'source_type': 'rag' if source.get('doc_id') else 'web',
            **{k: source.get(k) or '' for k in ('title','source','date','url','doc_id','page','chunk_id')},
            'location': f"p.{source['page']} / {source.get('chunk_id')}" if source.get('doc_id') else ''})
    lines = []
    for e in evidence:
        detail = e.get('statement') or f"{e['metric']}: {e['value']} {e.get('unit', '')} ({e.get('scope','')}; {e.get('period','')})"
        lines.append(f"- {detail} [{e['evidence_id']}]")
    present = {e['metric'] for e in evidence}
    missing = [m for m in CATEGORY_METRICS[category] if m not in present]
    diagnostics = {'category': category, 'company': company, 'source_count': len(sources),
        'extracted_metric_count': len(analysis.metrics), 'accepted_metric_count': sum(e['metric'] != 'context' for e in evidence),
        'missing_metrics': missing, 'rejected': rejected,
        'extraction_missing': [m.model_dump() for m in analysis.missing]}
    summary = '\n'.join(lines) or '검증 가능한 근거를 확보하지 못했습니다.'
    if missing:
        summary += '\n미확인 지표: ' + ', '.join(missing)
    return summary, evidence, diagnostics
