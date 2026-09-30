"""외부 API/모델 다운로드 없이 전체 경로를 검증하는 명시적 합성 데이터."""
from agents.criteria import CATEGORY_METRICS, UNITS
from agents.schemas import Candidates, Selection, Criterion, Analysis, Metric, Queries, Relevance, Rewrite, Judgments, Judgment, ReportText, PeerGroup, Finding


class DemoServices:
    def __init__(self, scenario='invest'):
        self.scenario = scenario
        self.values = dict(team_experience='칩 양산 리드 이력', technical_headcount=100,
                           market_tam=100, market_cagr=20, market_demand='타깃 및 수요 확인',
                           development_stage='양산', technology_originality='독자 구조와 측정치',
                           competitive_edge='동일 조건 벤치마크', entry_barriers='특허 자체SW 파트너십',
                           customer_count=5, revenue_stage='양산 매출', funding_total=3000,
                           risk='완화 근거 확인')
        self.risk_quote = '가상반도체 A와 가상반도체 B는 공급망 다변화로 위험을 완화했다.'
        self.content = '합성 테스트 근거. 누적 투자 유치 3000억 원. 가상 경쟁사. ' + self.risk_quote + ' ' + ' '.join(f'{k}={v}' for k, v in self.values.items())

    def web(self, query):
        return [{'title': '합성 테스트 자료', 'url': 'https://example.com/synthetic',
                 'date': '2026-09-29', 'source': '테스트 fixture', 'content': self.content}]

    def rag(self, query):
        return [{**self.web(query)[0], 'doc_id': 'synthetic', 'chunk_id': f'synthetic:{i}', 'page': i+1} for i in range(2)]

    def ask(self, schema, task, data):
        if schema is Candidates:
            return Candidates(candidates=[{'name': '가상반도체 A'}, {'name': '가상반도체 B'}])
        if schema is Selection:
            status = 'FAIL' if self.scenario == 'excluded' else 'REVIEW' if self.scenario == 'uncertain' else 'PASS'
            return Selection(criteria=[Criterion(criterion=c, status=status, reason='합성 선정 테스트', source_ids=['S1'])
                for c in ('domain', 'ai_core', 'unlisted', 'funding_stage', 'no_exit', 'information')])
        if schema is Analysis:
            metrics = [] if self.scenario == 'missing' else [Metric(metric=m, value=self.values[m], unit=UNITS.get(m, ''),
                quote='누적 투자 유치 3000억 원.' if m == 'funding_total' else self.risk_quote if m == 'risk' else f'{m}={self.values[m]}', source_id='S1',
                raw_value=self.values[m] if isinstance(self.values[m], (float, int)) else None, raw_unit=UNITS.get(m, ''),
                value_kind='cumulative' if m == 'funding_total' else 'actual',
                scope='technical_team' if m == 'technical_headcount' else '가상 NPU 시장', period='2025-2030') for m in CATEGORY_METRICS[data['category']]]
            return Analysis(summary='합성 기업의 제품·팀·수요 및 위험 완화 근거를 검증하는 테스트입니다. [S1]', metrics=metrics)
        if schema is PeerGroup:
            return PeerGroup(peers=[{'name': f'가상 경쟁사 {i}', 'product': 'NPU', 'workload': 'edge', 'source_id': 'S1', 'quote': '가상 경쟁사'} for i in range(3)])
        if schema is Queries:
            return Queries(queries=['가상 시장 TAM', '가상 시장 CAGR', '가상 시장 수요 규제'])
        if schema is Relevance:
            return Relevance(relevant_source_ids=[s['source_id'] for s in data['chunks']])
        if schema is Rewrite:
            return Rewrite(query=data['query'] + ' semiconductor')
        if schema is Judgments:
            return Judgments(judgments=[Judgment(metric=e['metric'], score=1 if self.scenario == 'all_hold' else 5,
                reason='합성 루브릭 테스트', evidence_ids=[e['evidence_id']]) for e in data['evidence']])
        if schema is ReportText:
            citations = ' '.join(f"[{e['evidence_id']}]" for e in data['evidence'][:2])
            return ReportText(summary='합성 데이터로 파이프라인과 코드 판정 결과를 검증한 보고서입니다.',
                overview='가상 기업의 AI 반도체 사업 평가 시나리오입니다. ' + citations,
                technology='설계·양산 및 팀 역량은 합성 테스트 값입니다. ' + citations,
                market='시장 규모·성장률·경쟁 정보는 합성값이며 실제 시장 분석에 사용할 수 없습니다.',
                risks='시장·기술·규제·경쟁 리스크의 실제 조사는 수행하지 않았습니다.')
        raise ValueError(f'Unsupported demo schema: {schema}')
