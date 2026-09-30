import json
from datetime import date
from agents.common import create_llm, load_prompt


class LiveServices:
    repair_extraction = True
    grounded_extraction = True

    def __init__(self, output_dir=None):
        from pathlib import Path
        self.output_dir = Path(output_dir) if output_dir else None
        self.counter = 0

    def record(self, event, payload):
        if self.output_dir is None:
            return
        self.counter += 1
        folder = self.output_dir / 'diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{self.counter:03d}_{event}.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def ask(self, schema, task: str, data: dict):
        llm = create_llm(temperature=0, timeout=90, max_retries=2)
        result = llm.with_structured_output(schema, method='json_schema', strict=True).invoke([
            ('system', load_prompt('system') + f'\n평가 기준일: {date.today().isoformat()}'),
            ('human', task + '\n입력 데이터:\n' + json.dumps(data, ensure_ascii=False)),
        ])
        self.record(schema.__name__, {'task': task, 'input': data, 'output': result.model_dump()})
        return result

    def web(self, query: str) -> list[dict]:
        from tools.web_search import web_search
        return web_search.invoke({'query': query, 'max_results': 5})

    def rag(self, query: str) -> list[dict]:
        from rag.retriever import search
        return search(query)


def number_sources(sources: list[dict]) -> list[dict]:
    unique = {}
    for source in sources:
        key = (source.get('doc_id'), source.get('chunk_id'), source.get('url'), source.get('content'))
        unique.setdefault(key, source)
    return [{**s, 'source_id': f'S{i+1}'} for i, s in enumerate(unique.values())]


def safe_web(services, query: str) -> tuple[list[dict], list[str]]:
    try:
        return services.web(query), []
    except Exception as exc:
        # API 오류 내용에 요청·인증 정보가 포함될 수 있어 클래스명만 기록한다.
        return [], [f'웹검색 실패 ({type(exc).__name__}); 관련 정보 미확인: {query}']


def summarize_sources(services, category: str, startup: dict, sources: list[dict]):
    from agents.schemas import Analysis
    from agents.extraction import ROLE_TASKS, contract
    if not sources:
        return Analysis(summary='검색 근거 부족으로 분석 불가.', metrics=[])
    if getattr(services, 'grounded_extraction', False):
        from agents.grounded_extraction import extract
        return extract(services, category, startup, sources)
    # 선정 사유·이전 기업 소개를 사실 근거로 재사용하지 않는다.
    context = {k: startup[k] for k in ('name', 'tech_summary', 'peers') if k in startup}
    return services.ask(Analysis, 'extract_' + category + ': ' + ROLE_TASKS[category] +
        ' 지정된 metrics 각각에 대해 실제 근거를 추출하거나 missing에 사유를 남긴다. '
        'metric 키는 계약에 있는 영문 ID만 사용. 한글 항목명이나 source_id S1,S2 형태 금지. 하나의 레코드는 하나의 출처. '
        '정성 지표의 value는 근거 내용 문자열이며 수치가 없다는 이유로 null 처리하지 않는다. '
        'quote는 오직 sources.content에서 복사한 하나의 연속 원문이다. 번역/의역/다른 구절 연결 금지. '
        '핵심 사실은 findings에 topic/statement/quote/source_id로 기록. summary는 짧은 영역별 결론. '
        '목표·예정·전망과 실제 실적을 구분. 같은 지표의 다른 시점 값은 observation_date에 ISO 날짜를 적는다. '
        '숫자는 원문의 raw_value/raw_unit과 정규화한 value/unit을 모두 제공한다. 수치를 추측하거나 서로 다른 통화 환산 금지. '
        '날짜가 불명확하면 빈 문자열. 해당 source_id에서 직접 뒷받침되지 않는 기업 정보는 사용 금지.',
        {'category': category, 'company': context, 'metrics': contract(category), 'sources': sources})


def bind_evidence(analysis, sources, category, company):
    from agents.extraction import bind_analysis
    summary, evidence, _ = bind_analysis(analysis, sources, category, company)
    return summary, evidence


def analyze_sources(services, category, company, sources):
    from agents.extraction import bind_analysis
    from agents.schemas import Analysis
    sources = number_sources(sources)
    analysis = summarize_sources(services, category, company, sources)
    summary, evidence, diagnostics = bind_analysis(analysis, sources, category, company['name'])
    # 탈락한 원문 인용만 한 번 교정한다. 새로운 검색/무한 반복/검증 완화 없음.
    invalid = [r for r in diagnostics['rejected'] if r['reason'] in ('quote_not_in_source', 'unknown_source_id')]
    if invalid and getattr(services, 'repair_extraction', False):
        from agents.extraction import contract
        repair = services.ask(Analysis, 'repair_extraction: 아래 거절된 지표만 원문 sources.content의 정확한 '
            '연속 인용으로 다시 추출. 뒷받침할 수 없으면 missing. 다른 지표·요약 재작성 금지.',
            {'category': category, 'company': company['name'], 'rejected': invalid,
             'metrics': contract(category), 'sources': sources})
        analysis.metrics += repair.metrics
        summary, evidence, repaired = bind_analysis(analysis, sources, category, company['name'])
        repaired['initial_rejected'] = diagnostics['rejected']
        diagnostics = repaired
    if hasattr(services, 'record'):
        services.record('extraction_' + category, {'sources': sources, 'raw': analysis.model_dump(), 'diagnostics': diagnostics})
    return summary, evidence, diagnostics


def replace_evidence(state, category, additions):
    return [e for e in state.get('evidence', []) if e['category'] != category] + additions
