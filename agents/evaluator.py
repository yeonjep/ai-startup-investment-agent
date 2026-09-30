from fractions import Fraction
import re
from agents.criteria import ALL_METRICS, GROUPS, RUBRICS, CATEGORY_METRICS, quantitative_score
from agents.schemas import Judgments
from agents.evidence_resolution import resolve_rows


def run(state, services):
    evidence = state.get('evidence', [])
    qualitative = [e for e in evidence if e['metric'] in RUBRICS and e.get('value') is not None]
    judgments = services.ask(Judgments, 'score: 근거 있는 정성 지표만 루브릭으로 채점. 여러 근거를 종합해 metric별 정확히 한 개의 판단만 반환. metric, score, reason, evidence_ids 반환. 제조사 주장만 있으면 검증된 벤치마크 우위로 채점 금지.',
                             {'rubrics': RUBRICS, 'evidence': qualitative}).judgments if qualitative else []
    scores, missing = {}, []
    for metric in ALL_METRICS:
        rows = [e for e in evidence if e['metric'] == metric and e.get('value') is not None]
        valid_ids = {e['evidence_id'] for e in rows}
        score, reason, ids = None, '검증된 지표 근거 없음: 중립 3점', []
        missing_reason = 'accepted_evidence_missing'
        if metric in RUBRICS:
            votes = [j for j in judgments if j.metric == metric and j.evidence_ids
                     and set(j.evidence_ids).issubset(valid_ids)]
            if votes and len({vote.score for vote in votes}) == 1:
                score = votes[0].score
                reason = votes[0].reason
                ids = sorted({eid for vote in votes for eid in vote.evidence_ids})
                if metric == 'entry_barriers':
                    supported = ' '.join(str(e.get('value','')) + ' ' + e.get('quote','') for e in rows)
                    kinds = sum(bool(re.search(pattern, supported, re.I)) for pattern in
                        (r'특허|patent', r'SDK|자체.?SW|SW.?스택|컴파일러|소프트웨어|software stack', r'파트너|협력|제휴|partnership'))
                    ceiling = 5 if kinds == 3 else 3 if kinds else 1
                    if score > ceiling:
                        score = ceiling
                        reason += f' (검증된 진입장벽 유형 {kinds}개: 상한 {ceiling}점)'
                if metric == 'risk':
                    mitigation = any(re.search(r'완화|해소|확보|다변화|관리|mitigat|secured',
                                     str(e.get('value','')) + ' ' + e.get('quote',''), re.I) for e in rows)
                    if not mitigation:
                        score = None
                        reason = '회사별 리스크 완화 근거 없음'
                        ids = []
                        missing_reason = 'company_mitigation_missing'
        else:
            rows, missing_reason = resolve_rows(metric, rows)
            values = [(quantitative_score(metric, e['value'], e.get('unit', '')), e) for e in rows]
            if values and all(v is not None for v, e in values):
                score = values[0][0]
                reason = f"{values[0][1]['value']} {values[0][1].get('unit', '')}: 설계 C-1 구간 적용"
                ids = [e['evidence_id'] for _, e in values]
            elif rows:
                missing_reason = 'invalid_unit_or_value'
        if score is None:
            missing.append(metric)
            reason = missing_reason + ': 중립 3점'
            score = 3
        scores[metric] = {'score': score, 'reason': reason, 'evidence_ids': ids, 'missing': metric in missing, 'missing_reason': missing_reason if metric in missing else ''}
    groups = {name: {'score': sum(scores[m]['score'] for m in metrics) / len(metrics), 'weight': weight}
              for name, (weight, metrics) in GROUPS.items()}
    total = float(sum(Fraction(sum(scores[m]['score'] for m in metrics), len(metrics) * 5) * weight
                      for weight, metrics in GROUPS.values()))
    target = ''
    if state.get('evidence_retry', 0) == 0:
        counts = {c: sum(m in missing for m in CATEGORY_METRICS[c]) for c in ('tech', 'market', 'competitor')}
        category = max(counts, key=counts.get)
        if counts[category]:
            target = {'tech': 'technology_agent', 'market': 'market_agent', 'competitor': 'competitor_agent'}[category]
    return {'scores': {'metrics': scores, 'groups': groups}, 'total_score': total,
            'missing_evidence': missing, 'retry_target': target,
            'evidence_retry': state.get('evidence_retry', 0) + bool(target)}
