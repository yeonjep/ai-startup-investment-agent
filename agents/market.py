from agents.schemas import Queries, Relevance, Rewrite
from agents.services import number_sources, safe_web, analyze_sources, replace_evidence


def grade_relevance(services, query, chunks):
    if not chunks:
        return []
    result = services.ask(Relevance, 'grade_relevance: 질의에 직접 답할 수 있는 근거 source_id만 선택. 기업 타깃 시장 범위가 맞는지 확인.',
                          {'query': query, 'chunks': chunks})
    return [c for c in chunks if c['source_id'] in result.relevant_source_ids]


def run(state, services):
    company = state['current_startup']
    queries = services.ask(Queries, 'market_queries: 타깃 시장에 대한 질의 3개: TAM, CAGR, 수요처·규제.',
                           {'company': company, 'technology': state['tech_summary']}).queries
    all_sources, warnings = [], []
    max_retry = 0
    for original_query in queries:
        query, accepted = original_query, []
        for attempt in range(3):
            max_retry = max(max_retry, attempt)
            try:
                chunks = number_sources(services.rag(query))
            except Exception as exc:
                warnings.append(f'RAG 검색 실패 ({type(exc).__name__}); 웹 보완 수행')
                break
            for chunk in grade_relevance(services, query, chunks):
                if not any(c.get('chunk_id') == chunk.get('chunk_id') for c in accepted):
                    accepted.append(chunk)
            if len(accepted) >= 2:
                break
            if attempt < 2:
                query = services.ask(Rewrite, 'rewrite: 같은 목표 시장과 지표를 유지하며 한국어/영어 동의어로 검색 질의 개선.',
                                     {'original_query': original_query, 'query': query}).query
        if len(accepted) < 2:
            web, errors = safe_web(services, original_query)
            warnings += errors
            accepted += grade_relevance(services, original_query, number_sources(web))
        all_sources += accepted[:5]
    unique = {}
    for s in all_sources:
        unique[(s.get('doc_id'), s.get('chunk_id'), s.get('url'))] = s
    sources = number_sources(list(unique.values()))
    summary, evidence, diagnostics = analyze_sources(services, 'market', {'name': company['name'], 'tech_summary': state['tech_summary']}, sources)
    diagnostics['queries'] = queries
    diagnostics['rag_retry'] = max_retry
    diagnostics['rag_source_count'] = sum(bool(s.get('doc_id')) for s in sources)
    return {'market_analysis': summary, 'evidence': replace_evidence(state, 'market', evidence),
            'rag_retry': max_retry, 'warnings': warnings, 'diagnostics': [diagnostics]}
