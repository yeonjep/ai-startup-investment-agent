import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agents.criteria import quantitative_score
from agents.decision import run as decide
from agents.demo import DemoServices
from agents.graph import build_graph
from agents.services import bind_evidence
from agents.schemas import Analysis, Metric, Selection, Criterion
from agents.startup import selection_result, deduplicate


class RuleTests(unittest.TestCase):
    def test_numeric_boundaries_and_units(self):
        for n, expected in [(-1, 1), (4.9, 1), (5, 2), (10, 3), (15, 4), (20, 5)]:
            self.assertEqual(quantitative_score('market_cagr', n, '%'), expected)
        self.assertEqual(quantitative_score('funding_total', 3000, '억 원'), 5)
        self.assertIsNone(quantitative_score('funding_total', 3000, 'USD'))
        self.assertIsNone(quantitative_score('technical_headcount', 1.5, '명'))
        self.assertIsNone(quantitative_score('market_tam', float('nan'), 'USD billion'))
        self.assertEqual(quantitative_score('customer_count', 0, '건'), 1)
        self.assertIsNone(quantitative_score('development_stage', '미확인', ''))

    def test_decision_threshold_and_missing_override(self):
        state = dict(current_startup={'name': 'test'}, total_score=70, missing_evidence=[])
        self.assertEqual(decide(state)['decision'], '투자')
        self.assertEqual(decide({**state, 'total_score': 69.999})['decision'], '보류')
        self.assertEqual(decide({**state, 'total_score': 100, 'missing_evidence': list('abcde')})['decision'], '보류')

    def test_missing_selection_source_cannot_pass(self):
        s = Selection(criteria=[Criterion(criterion='domain', status='PASS', reason='test', source_ids=['fake'])])
        status, criteria = selection_result(s, [])
        self.assertEqual(status, 'REVIEW')
        self.assertEqual(len(criteria), 6)
        self.assertTrue(all(c['status'] == 'REVIEW' for c in criteria))

    def test_fabricated_quote_dropped(self):
        analysis = Analysis(summary='예시', metrics=[Metric(metric='market_cagr', value=25, unit='%',
                                  quote='CAGR 25%', source_id='S1')])
        _, evidence = bind_evidence(analysis, [{'source_id': 'S1', 'content': 'CAGR 10%'}], 'market', 'test')
        self.assertEqual(evidence, [])

    def test_duplicate_companies(self):
        self.assertEqual(len(deduplicate([{'name':'주식회사 테스트'}, {'name':'테스트 Inc.'}])), 1)


class GraphTests(unittest.TestCase):
    def run_graph(self, scenario, limit=5, service=None):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('agents.report.export_report_pdf', return_value='test.pdf'):
                graph = build_graph(service or DemoServices(scenario))
                updates = list(graph.stream(dict(domain='AI 반도체', max_candidates=limit,
                         candidates=[], evaluated=[], warnings=[], output_dir=tmp, demo=True),
                         config={'recursion_limit': 200}, stream_mode='values'))
        return updates[-1], updates

    def test_invest_stops_after_first(self):
        state, _ = self.run_graph('invest')
        self.assertEqual(len(state['evaluated']), 1)
        self.assertEqual(state['decision'], '투자')
        self.assertAlmostEqual(state['total_score'], 100)
        self.assertTrue(state['final_report'].startswith('# SUMMARY'))
        self.assertIn('# REFERENCE', state['final_report'])
        for title in ['## 2. 팀', '## 3. 시장', '### 4.3 선정']:
            self.assertIn('\n\n' + title, state['final_report'])

    def test_all_hold_accumulates_and_resets(self):
        state, _ = self.run_graph('all_hold')
        self.assertEqual(len(state['evaluated']), 2)
        self.assertEqual(state['current_idx'], 2)
        self.assertEqual(state['evidence'], [])
        self.assertEqual({r['decision'] for r in state['evaluated']}, {'보류'})

    def test_missing_retries_once_per_candidate(self):
        state, history = self.run_graph('missing')
        self.assertEqual(len(state['evaluated']), 2)
        self.assertEqual([len(r['missing_evidence']) for r in state['evaluated']], [13, 13])
        self.assertTrue(all(s['evidence_retry'] <= 1 for s in history if 'evidence_retry' in s))
        self.assertTrue(all(r['total_score'] == 60 for r in state['evaluated']))

    def test_excluded_never_evaluates(self):
        state, _ = self.run_graph('excluded')
        self.assertEqual(state['evaluated'], [])
        self.assertEqual(len(state['candidates']), 2)

    def test_review_research_once(self):
        state, _ = self.run_graph('uncertain')
        self.assertEqual(state['selection_retry'], 1)
        self.assertTrue(state['uncertain'])
        self.assertTrue(state['evaluated'][0]['uncertain'])

    def test_limit_counts_only_evaluated(self):
        state, _ = self.run_graph('all_hold', 1)
        self.assertEqual(len(state['evaluated']), 1)

    def test_rag_retries_are_bounded_and_fallback(self):
        class EmptyRag(DemoServices):
            calls = 0
            def rag(self, query):
                self.calls += 1
                return []
        service = EmptyRag('invest')
        state, _ = self.run_graph('invest', service=service)
        self.assertEqual(service.calls, 9)
        self.assertEqual(state['rag_retry'], 2)
        self.assertEqual(state['decision'], '투자')


if __name__ == '__main__':
    unittest.main()
