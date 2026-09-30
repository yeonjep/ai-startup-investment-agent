"""설계 C-1: 단위는 인력=명, TAM=USD billion, CAGR=%, 투자액=억 원."""
GROUPS = {
    '창업자/팀': (25, ['team_experience', 'technical_headcount']),
    '시장성': (20, ['market_tam', 'market_cagr', 'market_demand']),
    '제품/기술력': (20, ['development_stage', 'technology_originality']),
    '경쟁 우위': (15, ['competitive_edge', 'entry_barriers']),
    '실적': (10, ['customer_count', 'revenue_stage']),
    '투자조건/리스크': (10, ['funding_total', 'risk']),
}
CATEGORY_METRICS = {
    'startup': ['team_experience', 'technical_headcount', 'customer_count', 'revenue_stage', 'funding_total'],
    'tech': ['development_stage', 'technology_originality'],
    'market': ['market_tam', 'market_cagr', 'market_demand', 'risk'],
    'competitor': ['competitive_edge', 'entry_barriers'],
}
RUBRICS = {
    'team_experience': '칩 설계·양산 리드 이력 5 / 경력 있음 3 / 경력 없음이 확인됨 1',
    'market_demand': '타깃+수요 근거 5 / 타깃만 3 / 불명확 1',
    'technology_originality': '독자 구조+수치 공개 5 / 주장만 3 / 차별 없음 1',
    'competitive_edge': '동일 조건 벤치마크 우위 입증 5 / 주장만 3 / 차별 없음 1',
    'entry_barriers': '특허·자체 SW 스택·전략 파트너십 3개 5 / 1~2개 3 / 없음 1',
    'risk': '기술·규제·공급망 리스크 완화 근거 확인 5 / 일부 완화 3 / 치명적 미해결 1',
}
UNITS = {'technical_headcount': '명', 'market_tam': 'USD billion',
         'market_cagr': '%', 'customer_count': '건', 'funding_total': '억 원'}
THRESHOLDS = {'technical_headcount': [10, 20, 50, 100],
              'market_tam': [1, 10, 50, 100], 'market_cagr': [5, 10, 15, 20],
              'customer_count': [1, 2, 3, 5], 'funding_total': [100, 300, 1000, 3000]}
STAGES = {'development_stage': {'구상': 1, '설계': 2, '시제품': 3, '테이프아웃': 4, '양산': 5},
          'revenue_stage': {'없음': 1, '초기 매출': 3, '양산 매출': 5}}
ALL_METRICS = [m for _, metrics in GROUPS.values() for m in metrics]


def quantitative_score(metric: str, value, unit: str) -> int | None:
    import math
    if metric in STAGES:
        return STAGES[metric].get(value) if isinstance(value, str) else None
    if metric not in THRESHOLDS or unit != UNITS[metric]:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if value < 0 and metric != 'market_cagr':
        return None
    if metric in {'technical_headcount', 'customer_count'} and value != int(value):
        return None
    return 1 + sum(value >= bound for bound in THRESHOLDS[metric])
