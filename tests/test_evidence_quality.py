import tempfile
import unittest
from pathlib import Path
from agents.schemas import Analysis, Metric, Finding
from agents.extraction import bind_analysis
from agents.evidence_resolution import resolve_rows
from agents.services import summarize_sources
from app import create_run_directory


def extract(metric, content=None, category='market'):
    content = metric.quote if content is None else content
    return bind_analysis(Analysis(summary='임의 요약', metrics=[metric]),
                         [{'source_id':'S1', 'content':content, 'url':'https://example.com/source'}], category, '예시기업')


class EvidenceQualityTests(unittest.TestCase):
    def test_number_must_match_quote(self):
        m = Metric(metric='market_cagr', value=25, raw_value=25, raw_unit='%', unit='%',
                   scope='엣지 NPU', period='2025-2030', quote='시장 CAGR 10%', source_id='S1')
        _, ev, diag = extract(m)
        self.assertEqual(ev, [])
        self.assertEqual(diag['rejected'][0]['reason'], 'raw_number_not_in_quote')

    def test_currency_conversion_is_explicit(self):
        m = Metric(metric='market_tam', value=2.5, raw_value=2500, raw_unit='USD million', unit='USD billion',
                   scope='엣지 NPU', period='2030', value_kind='forecast', quote='Edge NPU market USD 2,500 million in 2030', source_id='S1')
        self.assertEqual(len(extract(m)[1]), 1)
        m.value = 2500
        self.assertEqual(extract(m)[2]['rejected'][0]['reason'], 'numeric_conversion_mismatch')

    def test_round_cannot_be_cumulative(self):
        m = Metric(metric='funding_total', value=700, raw_value=700, raw_unit='억 원', unit='억 원',
                   value_kind='round', quote='시리즈C 700억원 투자 유치', source_id='S1')
        self.assertEqual(extract(m, category='startup')[2]['rejected'][0]['reason'], 'not_cumulative_funding')
        m.value_kind = 'cumulative'
        self.assertEqual(extract(m, category='startup')[1], [])

    def test_forecast_does_not_prove_revenue_or_production(self):
        for key, value, quote in [('development_stage','양산','2027년 양산 예정'),
                                 ('development_stage','양산','고객 평가 후 내년 상반기부터 실제 양산에 들어간다.'),
                                 ('revenue_stage','양산 매출','양산 매출 100억원 목표')]:
            m = Metric(metric=key, value=value, quote=quote, source_id='S1')
            category = 'tech' if key == 'development_stage' else 'startup'
            self.assertEqual(extract(m, category=category)[2]['rejected'][0]['reason'], 'stage_not_confirmed')

    def test_generic_industry_risk_does_not_prove_company_risk(self):
        m = Metric(metric='risk', value='예시기업 공급망 위험',
                   quote='AI 반도체 산업 전반은 공급망 다변화가 필요하다.', source_id='S1')
        self.assertEqual(extract(m)[2]['rejected'][0]['reason'], 'company_risk_not_supported')

    def test_total_staff_is_not_technical_staff(self):
        m = Metric(metric='technical_headcount', value=115, raw_value=115, raw_unit='명', unit='명',
                   scope='technical_team', quote='전체 직원 115명', source_id='S1')
        self.assertEqual(extract(m, category='startup')[1], [])
        m.quote = '연구개발 기술 인력 50명'
        m.value = m.raw_value = 50.0
        self.assertEqual(len(extract(m, category='startup')[1]), 1)

    def test_qualitative_fact_without_number_is_accepted(self):
        m = Metric(metric='team_experience', value='설립자가 반도체 설계 경력 보유',
                   quote='창업자는 반도체 설계팀에서 근무했다.', source_id='S1')
        self.assertEqual(len(extract(m, category='startup')[1]), 1)

    def test_rejected_quotes_leave_diagnostics(self):
        m = Metric(metric='technology_originality', value='독자 구조', quote='원문에 없는 주장', source_id='S1')
        _, ev, diag = extract(m, content='실제 원문', category='tech')
        self.assertFalse(ev)
        self.assertEqual(diag['rejected'][0]['reason'], 'quote_not_in_source')
        self.assertIn('technology_originality', diag['missing_metrics'])

    def test_unverified_summary_does_not_become_evidence(self):
        analysis = Analysis(summary='양산 달성 주장 [S1]', metrics=[])
        _, ev, _ = bind_analysis(analysis, [{'source_id':'S1','content':'개발 중'}], 'tech','예시')
        self.assertEqual(ev, [])

    def test_findings_require_quote(self):
        a = Analysis(summary='',metrics=[],findings=[Finding(topic='제품',statement='독자 칩',quote='없는 내용',source_id='S1')])
        self.assertEqual(bind_analysis(a,[{'source_id':'S1','content':'제품 소개'}],'tech','예시')[1], [])

    def test_prior_profile_not_sent_as_factual_source(self):
        class Service:
            def ask(self, schema, task, data):
                self.data = data
                return Analysis(summary='', metrics=[])
        svc=Service()
        summarize_sources(svc,'market',{'name':'예시','profile':'오염된 소개','criteria':['이전 주장']},[{'content':'시장 보고서'}])
        self.assertEqual(svc.data['company'], {'name':'예시'})
        self.assertIn('market_tam', svc.data['metrics'])

    def test_newest_cumulative_observation_is_selected(self):
        rows=[{'value':300,'unit':'억 원','value_kind':'cumulative','observation_date':'2023-01-01'},
              {'value':1000,'unit':'억 원','value_kind':'cumulative','observation_date':'2026-01-01'}]
        self.assertEqual(resolve_rows('funding_total',rows)[0][0]['value'],1000)
        rows[0]['observation_date']=''
        self.assertEqual(resolve_rows('funding_total',rows)[1], 'conflicting_observations')

    def test_different_market_periods_not_mixed(self):
        rows=[{'value':10,'unit':'USD billion','scope':'NPU','period':'2025'},
              {'value':20,'unit':'USD billion','scope':'NPU','period':'2030'}]
        self.assertEqual(resolve_rows('market_tam',rows)[1], 'market_scope_or_period_conflict')

    def test_new_runs_never_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = create_run_directory(Path(tmp)), create_run_directory(Path(tmp))
            self.assertNotEqual(first,second)
            self.assertTrue(first.is_dir() and second.is_dir())

class EvaluationGuardTests(unittest.TestCase):
    def test_barrier_requires_three_kinds_for_top_score(self):
        from agents.evaluator import run
        from agents.schemas import Judgments, Judgment
        class Judge:
            def ask(self, schema, task, data):
                return Judgments(judgments=[Judgment(metric='entry_barriers',score=5,
                    reason='과장된 판단',evidence_ids=['EV-1'])])
        row={'evidence_id':'EV-1','metric':'entry_barriers','value':'자체 SDK와 파트너십',
             'quote':'자체 SDK와 전략 파트너십 체결'}
        result=run({'evidence':[row], 'evidence_retry':1},Judge())
        self.assertEqual(result['scores']['metrics']['entry_barriers']['score'],3)

    def test_sector_risk_without_company_mitigation_is_missing(self):
        from agents.evaluator import run
        from agents.schemas import Judgments, Judgment
        class Judge:
            def ask(self, schema, task, data):
                return Judgments(judgments=[Judgment(metric='risk',score=5,
                    reason='근거 없는 완화 판단',evidence_ids=['EV-1'])])
        row={'evidence_id':'EV-1','metric':'risk','value':'국내 산업 인력 부족',
             'quote':'국내 반도체 산업은 인력이 부족하다'}
        result=run({'evidence':[row], 'evidence_retry':1},Judge())
        self.assertTrue(result['scores']['metrics']['risk']['missing'])
        self.assertEqual(result['scores']['metrics']['risk']['score'],3)
