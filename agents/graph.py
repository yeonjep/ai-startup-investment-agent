from functools import partial
from langgraph.graph import END, START, StateGraph
from agents.state import InvestmentState, initialize_state, next_candidate
from agents import startup, technology, market, competitor, evaluator, decision, report
from agents.services import LiveServices


def route_startup(state):
    if not state.get('current_startup'):
        return 'report_agent'
    if state['selection_status'] == 'FAIL':
        return 'next_candidate'
    if state['selection_status'] == 'REVIEW' and state['selection_retry'] == 0:
        return 'startup_agent'
    return 'technology_agent'


def route_next(state):
    return ('startup_agent' if state['current_idx'] < len(state['candidates'])
            and len(state.get('evaluated', [])) < state['max_candidates'] else 'report_agent')


def build_graph(services=None):
    services = services or LiveServices()
    builder = StateGraph(InvestmentState)
    builder.add_node('initialize_state', initialize_state)
    builder.add_node('next_candidate', next_candidate)
    for name, module in [('startup', startup), ('technology', technology), ('market', market),
                         ('competitor', competitor), ('evaluator', evaluator), ('report', report)]:
        builder.add_node(name + '_agent', partial(module.run, services=services))
    builder.add_node('decision_agent', decision.run)
    builder.add_edge(START, 'initialize_state')
    builder.add_edge('initialize_state', 'startup_agent')
    builder.add_conditional_edges('startup_agent', route_startup,
                                  ['report_agent', 'next_candidate', 'startup_agent', 'technology_agent'])
    for a, b in [('technology', 'market'), ('market', 'competitor'), ('competitor', 'evaluator')]:
        builder.add_edge(a + '_agent', b + '_agent')
    builder.add_conditional_edges('evaluator_agent', lambda s: s.get('retry_target') or 'decision_agent',
                                  ['technology_agent', 'market_agent', 'competitor_agent', 'decision_agent'])
    builder.add_conditional_edges('decision_agent', lambda s: 'report_agent' if s['decision'] == '투자' else 'next_candidate',
                                  ['report_agent', 'next_candidate'])
    builder.add_conditional_edges('next_candidate', route_next, ['startup_agent', 'report_agent'])
    builder.add_edge('report_agent', END)
    return builder.compile()
