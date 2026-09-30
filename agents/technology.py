from agents.services import safe_web, analyze_sources, replace_evidence


def run(state, services):
    company = state['current_startup']
    sources, warnings = [], []
    for terms in ('공식 제품 NPU 아키텍처 공정 전력효율 TOPS SDK', '반도체 양산 테이프아웃 실증 벤치마크 개발 단계'):
        found, errors = safe_web(services, f"{company['name']} {terms}")
        sources += found
        warnings += errors
    summary, evidence, diagnostics = analyze_sources(services, 'tech', company, sources)
    return {'tech_summary': summary, 'evidence': replace_evidence(state, 'tech', evidence),
            'diagnostics': [diagnostics], 'warnings': warnings}
