import json
import re
from pathlib import Path
from rapidfuzz.fuzz import ratio
from agents.schemas import Candidates, Selection
from agents.services import safe_web, number_sources, analyze_sources, replace_evidence

CRITERIA = ('domain', 'ai_core', 'unlisted', 'funding_stage', 'no_exit', 'information')
SELECTION_RULES = '''6개 기준 모두 판정: domain=국내 AI반도체 주력 PASS/무관 FAIL/일부 사업 REVIEW;
ai_core=AI 핵심 PASS/마케팅만 FAIL/불명확 REVIEW;
unlisted=미상장 PASS/상장 FAIL/예비심사 또는 주관사 공식 발표 REVIEW;
funding_stage=Seed~Series C PASS/Series D 이상 또는 Pre-IPO FAIL/비공개 REVIEW;
no_exit=Exit 미완료 PASS/인수로 독립법인 소멸 FAIL/합병·지분매각 진행 REVIEW;
information=홈페이지+독립 관련 기사 2건 이상 PASS/공개정보 거의 없음 FAIL/1건 수준 REVIEW.
상장 사실/인수 사실을 검색에서 못 찾았다는 이유만으로 PASS하지 않는다. 단순히 검색 실패나 자료가 없으면 사실을 추측하지 말고 REVIEW. IPO 준비와 공식 주관사 선임/예비심사를 구별한다. 각 판정 사유와 source_ids 필수.'''


def normalize_name(name):
    return re.sub(r'[\W_]+', '', re.sub(r'주식회사|\(주\)|inc\.?|co\.?\s*ltd\.?', '', name, flags=re.I)).casefold()


def deduplicate(candidates):
    result = []
    for candidate in candidates:
        name = normalize_name(candidate['name'])
        if not name:
            continue
        existing = next((c for c in result if ratio(name, normalize_name(c['name'])) >= 95), None)
        if existing:
            existing['urls'] = sorted(set(existing.get('urls', []) + candidate.get('urls', [])))
        else:
            result.append(dict(candidate))
    return result[:10]


def selection_result(selection, sources):
    lookup = {s['source_id']: s for s in sources}
    records = []
    for name in CRITERIA:
        items = [c for c in selection.criteria if c.criterion == name]
        item = items[0] if len(items) == 1 else None
        cited = [lookup[sid] for sid in item.source_ids if sid in lookup] if item else []
        records.append({'criterion': name,
                        'status': item.status if item and cited and all(sid in lookup for sid in item.source_ids) else 'REVIEW',
                        'reason': item.reason if item and cited else '판정 근거 미확인',
                        'sources': cited})
    statuses = [c['status'] for c in records]
    return ('FAIL' if 'FAIL' in statuses else 'REVIEW' if 'REVIEW' in statuses else 'PASS'), records


def run(state, services):
    candidates = [dict(c) for c in state.get('candidates', [])]
    warnings = []
    if not candidates and state['current_idx'] == 0:
        sources, warnings = safe_web(services, f"{state['domain']} 국내 비상장 스타트업 Seed Series A B C 투자 기업")
        if sources:
            discovered = services.ask(Candidates, 'discover: 국내 AI 반도체 기업 후보 최대 10개 추출. 최신 투자 라운드와 출처.',
                                      {'sources': number_sources(sources)})
            candidates = deduplicate([c.model_dump() for c in discovered.candidates])
        if not candidates:
            seed_path = Path(__file__).resolve().parents[1] / 'data' / 'seed_startups.json'
            candidates = deduplicate(json.loads(seed_path.read_text()))
            warnings.append('후보 탐색 실패: 시드 기업명을 사용하며 현재 자격은 다시 확인합니다.')
    idx = state['current_idx']
    if idx >= len(candidates):
        return {'candidates': candidates, 'current_startup': None, 'selection_status': 'FAIL', 'warnings': warnings}
    company = candidates[idx]
    retry = int(state.get('selection_status') == 'REVIEW')
    query = f"{company['name']} 국내 AI 반도체 홈페이지 최신 투자 라운드 상장 인수 합병 창업자"
    if retry:
        query += ' ' + ' '.join(c['criterion'] for c in company.get('criteria', []) if c['status'] == 'REVIEW')
    sources, errors = safe_web(services, query)
    warnings += errors
    if retry:
        sources = company.get('selection_sources', []) + sources
    sources = number_sources(sources)
    if sources:
        selection = services.ask(Selection, 'selection: ' + SELECTION_RULES, {'company': company['name'], 'sources': sources})
    else:
        selection = Selection(criteria=[])
    status, records = selection_result(selection, sources)
    company.update(criteria=records, selection_status=status, selection_sources=sources,
                   uncertain=status == 'REVIEW' and retry == 1)
    update = {'candidates': candidates, 'selection_status': status, 'selection_retry': retry,
              'uncertain': company['uncertain'], 'warnings': warnings, 'current_startup': company}
    if status == 'PASS' or (status == 'REVIEW' and retry == 1):
        profile_sources = list(sources)
        for terms in ('창업자 대표 CTO 반도체 설계 경력 연구개발 기술 인력',
                      '공급 고객 PoC 계약 실제 매출 실적 누적 투자 유치 총액'):
            found, errors = safe_web(services, f"{company['name']} {terms}")
            profile_sources += found
            warnings += errors
        summary, evidence, diagnostics = analyze_sources(services, 'startup', company, profile_sources)
        update['diagnostics'] = [diagnostics]
        company['profile'] = summary
        update['evidence'] = replace_evidence(state, 'startup', evidence)
    out = Path(state['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
    return update
