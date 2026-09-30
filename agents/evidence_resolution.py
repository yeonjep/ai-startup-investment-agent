"""같은 기준의 최신 관측만 선택한다. 시장/기간/단위를 혼합하지 않는다."""
import re


def resolve_rows(metric, rows):
    if not rows:
        return [], 'accepted_evidence_missing'
    if metric == 'funding_total':
        rows = [e for e in rows if e.get('value_kind') == 'cumulative']
        if not rows:
            return [], 'cumulative_funding_missing'
    if metric in ('market_tam', 'market_cagr'):
        scopes = {(e.get('scope', '').strip().casefold(), e.get('period', '').strip()) for e in rows}
        if len(scopes) > 1:
            return [], 'market_scope_or_period_conflict'
    values = {(str(e['value']), e.get('unit')) for e in rows}
    if len(values) == 1:
        return rows, ''
    # 발행일이 아닌 관측일을 사용한다. 날짜 불명 값을 임의로 과거 값으로 취급하지 않는다.
    if metric not in ('market_tam', 'market_cagr') and all(re.fullmatch(r'\d{4}-\d{2}-\d{2}', e.get('observation_date', '')) for e in rows):
        latest = max(e['observation_date'] for e in rows)
        newest = [e for e in rows if e['observation_date'] == latest]
        if len({(str(e['value']), e.get('unit')) for e in newest}) == 1:
            return newest, ''
    return [], 'conflicting_observations'
