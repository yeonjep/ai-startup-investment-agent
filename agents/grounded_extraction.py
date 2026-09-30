"""LLM은 제공된 원문 구간 ID를 선택한다. 인용문/출처는 코드가 원문으로 복원한다."""
from typing import Literal
from pydantic import Field, create_model
from agents.criteria import CATEGORY_METRICS, STAGES, RUBRICS
from agents.schemas import Analysis, Finding, Metric, MissingMetric
from agents.extraction import ROLE_TASKS, contract


def prepare_passages(sources):
    passages, inputs = {}, []
    for source in sources:
        text = source.get('content') or ''
        items = []
        # 구간을 원문 그대로 보존하고 경계에 걸친 사실은 overlap으로 보완한다.
        for start in range(0, len(text), 500):
            excerpt = text[start:start+650]
            if not excerpt.strip():
                continue
            pid = f'P{len(passages)+1}'
            passages[pid] = {'source_id': source['source_id'], 'text': excerpt}
            items.append({'passage_id': pid, 'text': excerpt})
        inputs.append({k: source.get(k) for k in ('source_id', 'title', 'date', 'doc_id', 'page')})
        inputs[-1]['passages'] = items
    return passages, inputs


def extract(services, category, company, sources):
    passages, inputs = prepare_passages(sources)
    if not passages:
        return Analysis(summary='', metrics=[], findings=[])
    PassageID = Literal[tuple(passages)]
    fields = {}
    for metric in CATEGORY_METRICS[category]:
        value_type = Literal[tuple(STAGES[metric])] if metric in STAGES else str if metric in RUBRICS else float
        observation = create_model(category.title() + metric.title().replace('_','') + 'Observation',
            value=(value_type, ...), passage_id=(PassageID, ...),
            raw_value=(float | None, None), raw_unit=(str, ''),
            value_kind=(Literal['actual','cumulative','round','forecast','target','unknown'], ...),
            scope=(str, ''), period=(str, ''), observation_date=(str, ''))
        fields[metric] = (list[observation], Field(max_length=1 if metric in STAGES else 3))
    finding = create_model(category.title() + 'GroundedFinding',
                           topic=(str,...), statement=(str,Field(max_length=250)), passage_id=(PassageID,...))
    missing_item = create_model(category.title() + 'MissingMetric',
                                metric=(Literal[tuple(CATEGORY_METRICS[category])], ...), reason=(str, ...))
    schema = create_model(category.title() + 'Extraction',
                          **fields, findings=(list[finding], Field(max_length=5)),
                          missing=(list[missing_item], Field(max_length=len(CATEGORY_METRICS[category]))))
    result = services.ask(schema, 'grounded_extract_' + category + ': ' + ROLE_TASKS[category] +
        ' 각 지표 필드에 해당 사실만 넣고 없으면 빈 배열 및 missing 사유. '
        '각 관측은 단 하나의 passage_id의 원문만 사용한다. 인용문/출처를 새로 쓰지 않는다. '
        '숫자는 원문의 raw_value/raw_unit과 지정 단위로 변환한 value를 함께 제공. 범위/추정/목표는 실제 값으로 대체하지 않는다. '
        '정성 value는 150자 이내 사실. 공개 주장과 입증을 구분. 개발 단계/매출은 점수(숫자)가 아닌 enum 명칭. '
        'development_stage는 평가 대상의 대표 제품 하나의 실제 도달 단계만 추출하고 다른 제품 단계는 findings에 설명. 목표 양산/미래 매출은 실제 단계 근거가 아님. '
        'observation_date는 값의 관측일. 원문에 없으면 빈 문자열. 문서 발행일을 임의 관측일로 대입하지 않는다. '
        'findings는 원문 구간이 직접 뒷받침하는 영역별 핵심 사실 최대 5개. 원문에 없는 우위·해결·완료를 덧붙이지 않는다.',
        {'category': category, 'company': {k: company[k] for k in ('name','tech_summary','peers') if k in company},
         'metrics': contract(category), 'sources': inputs})
    metrics = []
    for name in CATEGORY_METRICS[category]:
        for observation in getattr(result, name):
            raw = observation.model_dump()
            passage = passages[raw.pop('passage_id')]
            metrics.append(Metric(metric=name, quote=passage['text'], source_id=passage['source_id'], **raw))
    findings = [Finding(topic=f.topic, statement=f.statement, quote=passages[f.passage_id]['text'],
                        source_id=passages[f.passage_id]['source_id']) for f in result.findings]
    return Analysis(summary='', metrics=metrics, findings=findings, missing=[MissingMetric(**m.model_dump()) for m in result.missing])
