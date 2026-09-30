from typing import Literal
from pydantic import Field, create_model
from agents.grounded_extraction import prepare_passages
from agents.services import safe_web, number_sources, analyze_sources, replace_evidence


def select_peers(services, company, technical_summary, sources):
    sources = number_sources(sources)
    passages, inputs = prepare_passages(sources)
    if not passages:
        return [], []
    item = create_model('PeerCandidate', name=(str, ...), product=(str, ...),
                        workload=(str, ...), passage_id=(Literal[tuple(passages)], ...))
    schema = create_model('SourcedPeerGroup', peers=(list[item], Field(max_length=5)))
    result = services.ask(schema, 'peer_group: 같은 엣지/온디바이스 또는 데이터센터 워크로드에서 경쟁하는 국내외 회사를 3~5개. '
        '회사 이름이 실제 선택한 passage 텍스트에 나타나야 한다. 이 회사 자신은 제외. '
        '자료에서 찾을 수 없는 회사를 만들지 않는다. 각 경쟁사에는 passage_id 하나.',
        {'company': company, 'technical_context': technical_summary[:1500], 'sources': inputs})
    peers, rejected = [], []
    for peer in result.peers:
        passage = passages[peer.passage_id]
        if peer.name.casefold() == company.casefold() or peer.name.casefold() not in passage['text'].casefold():
            rejected.append({'name': peer.name, 'reason': 'peer_name_not_in_source_passage'})
            continue
        if any(p['name'].casefold() == peer.name.casefold() for p in peers):
            continue
        peers.append({**peer.model_dump(exclude={'passage_id'}), 'source_id': passage['source_id'],
                      'quote': passage['text']})
    return peers, rejected


def run(state, services):
    company = state['current_startup']
    sources, warnings = [], []
    for query in (
        f"{company['name']} NPU 경쟁사 국내 딥엑스 리벨리온 퓨리오사 기술 비교",
        f"edge AI NPU accelerator competitors NVIDIA Jetson Hailo Mobilint benchmark",
    ):
        found, errors = safe_web(services, query)
        sources += found
        warnings += errors
    if getattr(services, 'grounded_extraction', False):
        peers, peer_rejected = select_peers(services, company['name'], state['tech_summary'], sources)
    else:
        # 데모/테스트 서비스는 고정 PeerGroup 스키마를 사용한다.
        from agents.schemas import PeerGroup
        numbered = number_sources(sources)
        result = services.ask(PeerGroup, 'peer_group: 동일 워크로드 경쟁사',
                              {'company': company['name'], 'technology': state['tech_summary'], 'sources': numbered}) if numbered else PeerGroup(peers=[])
        peers = [p.model_dump() for p in result.peers]
        peer_rejected = []
    for peer in peers:
        found, errors = safe_web(services, f"{peer['name']} {peer['product']} {company['name']} {peer['workload']} 성능 전력효율 SDK 비교")
        sources += found
        warnings += errors
    unique = {(s.get('url'), s.get('content')): s for s in sources}
    summary, evidence, diagnostics = analyze_sources(services, 'competitor',
        {'name': company['name'], 'tech_summary': state['tech_summary'], 'peers': peers}, list(unique.values()))
    diagnostics['peers'] = peers
    diagnostics['peer_rejected'] = peer_rejected
    if len(peers) < 3:
        diagnostics['limitations'] = ['동일 워크로드 경쟁사 3개 이상을 출처로 확인하지 못함']
        summary += '\n경쟁사 비교 한계: 출처로 검증된 경쟁사 3개 미만.'
    return {'competitor_analysis': summary, 'evidence': replace_evidence(state, 'competitor', evidence),
            'diagnostics': [diagnostics], 'warnings': warnings}
