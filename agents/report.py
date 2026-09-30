import json
import re
from pathlib import Path
from agents.config import REPORT_NAME
from agents.schemas import ReportText
from tools.report_pdf import export_report_pdf


METRIC_LABELS = {
    'team_experience': '핵심 인력 경력', 'technical_headcount': '기술 인력 규모',
    'market_tam': '목표 시장 규모', 'market_cagr': '시장 성장률', 'market_demand': '수요처 명확성',
    'development_stage': '개발 단계', 'technology_originality': '기술 독창성',
    'competitive_edge': '경쟁 차별성', 'entry_barriers': '진입장벽',
    'customer_count': '고객·PoC·계약 수', 'revenue_stage': '매출 발생',
    'funding_total': '누적 투자액', 'risk': '기술·규제·공급망 리스크',
}
CRITERION_LABELS = {'domain': '도메인 해당', 'ai_core': 'AI 핵심성', 'unlisted': '비상장',
                    'funding_stage': '투자 단계', 'no_exit': 'Exit 미완료', 'information': '정보 충분성'}


MISSING_LABELS = {
    'accepted_evidence_missing': '검증된 근거 없음',
    'cumulative_funding_missing': '누적 투자액 근거 없음',
    'market_scope_or_period_conflict': '시장 범위/기간 충돌',
    'conflicting_observations': '시점 또는 수치 충돌',
    'invalid_unit_or_value': '값 또는 단위 검증 실패',
    'company_mitigation_missing': '회사별 리스크 완화 근거 없음',
}


def cell(value):
    return str(value).replace('|', '/').replace('\n', ' ')


def run(state, services):
    evaluated = state.get('evaluated', [])
    target = next((r for r in evaluated if r['decision'] == '투자'), None)
    records = [target] if target else evaluated
    evidence = {e['evidence_id']: e for r in evaluated for e in r.get('evidence', [])}
    evidence.update({e['evidence_id']: e for e in state.get('evidence', [])})
    accepted_ids = {eid for r in records for metric in r['scores']['metrics'].values()
                    if not metric['missing'] for eid in metric['evidence_ids']}
    report_evidence = {eid: evidence[eid] for eid in accepted_ids if eid in evidence}
    if records:
        text = services.ask(ReportText, 'report: 설계 보고서 작성. 투자 판정이 있으면 그 기업 중심, 전원 보류이면 후보 비교 중심. '
            '모든 사실 주장에 제공된 [EV-...]를 인용. 제공된 코드 판정과 점수를 변경하지 말 것. '
            'overview=기업/제품/고객/사업 단계; technology=팀/기술/제품 성숙도/차별성. 분야별 해당 category의 근거만 사용. '
            'market=TAM/CAGR/수요/경쟁; risks=시장/기술/규제/경쟁 리스크 및 성장 여력. '
            'ROI 수치 추정 금지, 기업가치·지분 정보 부족 명시. summary 450자 이하. 근거 없는 절은 미확인이라고 명시. '
            '기업 홍보문을 반복하지 말고 근거와 한계를 간결하게 작성. 시장 전망은 기간·범위, 실제 실적과 목표를 구별. '
            '제공된 근거는 채점에 채택된 지표용이다. 각 근거에서 그 지표와 직접 관련된 내용만 사용. '
            '결측 지표에 관한 수치나 사실을 단정하지 말 것. 특히 누적 투자액이 결측이면 투자 라운드 금액도 누적액으로 서술하지 말 것. '
            '자료의 "내년", "올해" 등 상대시점은 현재 시점으로 바꿔 쓰지 말 것. 원문에 없는 최고/최초/입증 표현 금지. ',
            {'records': [{'startup': r['startup'], 'decision': r['decision'], 'total_score': r['total_score'],
                          'missing_evidence': r['missing_evidence']} for r in records],
             'evidence': [{k: e.get(k) for k in ('evidence_id','company','category','metric','value','unit','scope','period','quote','value_kind')} for e in report_evidence.values()]})
    else:
        text = ReportText(summary='선정 기준을 통과한 분석 대상이 없어 투자 판단을 보류합니다.',
                          overview='선정 결과와 제외 사유는 아래에 기록했습니다.', technology='분석 대상 없음.',
                          market='분석 대상 없음.', risks='후보 선정 조건과 최신 공개 자료를 추가 확인해야 합니다.')
    code_summary = ' '.join(f"{r['startup']}: {r['decision']} ({r['total_score']:.2f}점, 근거 확보 {13-len(r['missing_evidence'])}/13)." for r in records)
    if any(len(r['missing_evidence']) >= 5 for r in records):
        code_summary += ' 결측 지표의 3점은 중립 대체값이므로 총점을 기업의 검증된 경쟁력으로 해석할 수 없습니다.'
    if records:
        highlighted = [(metric, item) for metric, item in records[0]['scores']['metrics'].items()
                       if not item['missing'] and item['evidence_ids']]
        highlighted.sort(key=lambda pair: (-pair[1]['score'], pair[0]))
        if highlighted:
            code_summary += '\n핵심 확인 근거: ' + '; '.join(
                f"{METRIC_LABELS[m]} {item['score']}점 [" + item['evidence_ids'][0] + ']'
                for m, item in highlighted[:3]) + '.'
    sections = ['# SUMMARY', ('**합성 데이터 데모 — 실제 투자 평가가 아닙니다.**\n' if state.get('demo') else '') + (code_summary or text.summary),
                '## 1. 기업 및 사업 개요', text.overview,
                '## 2. 팀·기술 및 제품 경쟁력', text.technology,
                '## 3. 시장성 및 경쟁 환경', text.market,
                '## 4. 투자 판단 및 주요 리스크', text.risks,
                '### 4.1 종합 Scorecard']
    sections += ['|기업|항목|점수(5점)|가중치|', '|---|---|---:|---:|'] if records else ['평가 대상 없음.']
    for r in records:
        for name, group in r['scores']['groups'].items():
            sections.append(f"|{cell(r['startup'])}|{name}|{group['score']:.2f}|{group['weight']}%|")
    sections += ['### 4.2 후보별 판단과 한계']
    sections += ['|기업|총점|판정 및 사유|', '|---|---:|---|'] if evaluated else ['분석·점수 산출 대상 없음.']
    for r in evaluated:
        sections.append(f"|{cell(r['startup'])}|{r['total_score']:.2f}|{cell(r['decision'] + ': ' + r['reason'])}|")
    for r in evaluated:
        ids = sorted({eid for s in r['scores']['metrics'].values() for eid in s['evidence_ids']})
        sections.append('')
        sections.append(f"**{cell(r['startup'])} 한계:** 결측 {', '.join(METRIC_LABELS.get(m, m) for m in r['missing_evidence']) or '없음'}; "
                        f"선정 정보 {'불확실' if r.get('uncertain') else '확인'}. 점수 근거: " + ' '.join(f'[{eid}]' for eid in ids))
    sections += ['### 4.3 선정 제외 및 정보 불확실 후보']
    selection_refs = []
    selection_count = 0
    for c in state['candidates']:
        checks = [v for v in c.get('criteria', []) if v['status'] != 'PASS']
        if checks:
            for v in checks:
                selection_count += 1
                labels = []
                for s in v.get('sources', []):
                    label = f'SEL-{len(selection_refs)+1}'
                    selection_refs.append((label, s))
                    labels.append(f'[{label}]')
                sections.append(f"- {cell(c['name'])}: {CRITERION_LABELS[v['criterion']]} {v['status']} — {cell(v['reason'])} " + ' '.join(labels))
    if not selection_count:
        sections.append('해당 없음.')
    for r in evaluated:
        limitations = [item for d in r.get('diagnostics', []) for item in d.get('limitations', [])]
        for item in dict.fromkeys(limitations):
            sections.append(f"- {cell(r['startup'])} 분석 한계: {cell(item)}")
    if state.get('warnings'):
        sections += ['### 실행 한계', *['- ' + cell(w) for w in dict.fromkeys(state['warnings'])]]
    sections.append('ROI는 기업가치·지분율 자료 부족으로 산출하지 않았으며, 시장·개발 단계의 성장 여력으로 대리 평가했습니다.')
    # 기계 판정과 13개 세부 점수의 전체 감사 자료는 별도 JSON으로 보존한다.
    if records and len(records) == 1:
        sections += ['### 4.4 세부 지표와 근거', '|지표|점수|근거/결측 사유|', '|---|---:|---|']
        for metric, item in records[0]['scores']['metrics'].items():
            reason = MISSING_LABELS.get(item.get('missing_reason'), '근거 미확인') if item['missing'] else ' '.join('[' + e + ']' for e in item['evidence_ids'])
            sections.append(f"|{METRIC_LABELS[metric]}|{item['score']}|{cell(reason)}|")
    body = '\n\n'.join(sections)
    # 표의 연속 행에는 빈 줄을 넣지 않는다.
    body = re.sub(r'\|\n\n(?=\|)', '|\n', body)
    valid = set(report_evidence)
    body = re.sub(r'\[(EV-[^\]]+)\]', lambda m: m[0] if m[1] in valid else '[출처 미확인]', body)
    used = set(re.findall(r'\[(EV-[^\]]+)\]', body))
    refs = [(eid, report_evidence[eid]) for eid in sorted(used)] + selection_refs
    grouped = {}
    for label, e in refs:
        key = (e.get('doc_id'), e.get('url'), e.get('title'))
        if key not in grouped:
            grouped[key] = {'source': e, 'labels': [], 'locations': set()}
        grouped[key]['labels'].append(label)
        if e.get('location'):
            grouped[key]['locations'].add(e['location'])
    references = []
    for ref_number, group in enumerate(grouped.values(), 1):
        for label in group['labels']:
            body = body.replace(f'[{label}]', f'[R{ref_number}]')
        e = group['source']
        date = e.get('date') or '발행일 미상'
        if e.get('doc_id'):
            date = date[:4]
        url = e.get('url', '')
        link = f'[원문]({url})' if url.startswith(('https://', 'http://')) else 'URL 미확인'
        references.append(f"- [R{ref_number}] {e.get('source') or '기관/작성자 미상'}({date}). "
                          f"*{e.get('title') or '제목 미상'}*. {link} {'; '.join(sorted(group['locations']))}")
    body = re.sub(r'\[R\d+\](?:[ \t]+\[R\d+\])+',
                  lambda m: ' '.join(dict.fromkeys(re.findall(r'\[R\d+\]', m[0]))), body)
    report = body + '\n\n# REFERENCE\n\n' + '\n'.join(references or ['실제로 인용한 외부 자료 없음.']) + '\n'
    folder = Path(state['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    basename = 'DEMO_' + REPORT_NAME if state.get('demo') else REPORT_NAME
    (folder / (basename + '.md')).write_text(report, encoding='utf-8')
    (folder / 'evaluated.json').write_text(json.dumps(evaluated, ensure_ascii=False, indent=2), encoding='utf-8')
    path = export_report_pdf(report, folder / (basename + '.pdf'))
    import pymupdf
    page_count = None
    if Path(path).is_file():
        with pymupdf.open(path) as pdf:
            page_count = len(pdf)
    quality = {'pdf_pages': page_count, 'max_pages_pass': page_count is not None and page_count <= 5,
               'evidence_coverage': {r['startup']: {'available': 13-len(r['missing_evidence']), 'total': 13} for r in evaluated},
               'all_metrics_missing': any(len(r['missing_evidence']) == 13 for r in evaluated),
               'unverified_citation': '[출처 미확인]' in report,
               'rag_evidence_count': sum(e.get('source_type') == 'rag' for e in evidence.values()),
               'reference_dates_missing': sum(not str(group['source'].get('date') or '').startswith(('19','20')) for group in grouped.values()),
               'peer_group_counts': {r['startup']: max((len(d.get('peers', [])) for d in r.get('diagnostics', []) if d['category'] == 'competitor'), default=0) for r in evaluated},
               'market_rag_metric_count': sum(e.get('category') == 'market' and e.get('source_type') == 'rag' and e.get('metric') not in ('context', None) for e in evidence.values()),
               'note': '자동 검사는 원문과 서술의 의미 일치 여부를 완전히 보장하지 않습니다.'}
    (folder / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'final_report': report, 'report_path': path, 'quality': quality}
