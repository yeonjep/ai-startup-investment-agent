from copy import deepcopy


def run(state):
    total = state['total_score']
    missing = state['missing_evidence']
    decision = '투자' if total >= 70 and len(missing) < 5 else '보류'
    reason = f"총점 {total:.2f}/100, 결측 {len(missing)}/13. "
    reason += ('70점 이상 및 결측 5개 미만 충족.' if decision == '투자' else
               '결측 5개 이상으로 보류.' if len(missing) >= 5 else '70점 미만으로 보류.')
    record = {key: deepcopy(state.get(key)) for key in ('current_startup', 'scores', 'total_score',
              'missing_evidence', 'uncertain', 'tech_summary', 'market_analysis', 'competitor_analysis', 'evidence')}
    record['diagnostics'] = [deepcopy(d) for d in state.get('diagnostics', []) if d.get('company') == state['current_startup']['name']]
    record.update(startup=state['current_startup']['name'], decision=decision, reason=reason)
    return {'decision': decision, 'evaluated': [record]}
