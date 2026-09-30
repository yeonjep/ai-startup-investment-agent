# Market · Evaluator · Decision 구현 현황과 설계 미달 사항

기준 문서: `docs/DESIGN.md`

작성 범위는 담당 영역인 Market Agent, Evaluator Agent, Decision Agent로 한정한다. 공유 파일인
`agents/state.py`, `agents/graph.py`, `agents/config.py`는 수정하지 않았다.

## 이번 구현에 포함된 내용

- Market
  - `market_size`, `market_growth`, `demand_risk` topic별 Hybrid RAG 검색
  - 청크 관련성 평가, 부족 topic 재검색, 재시도 상한 후 웹검색 보완
  - Evidence ID 병합 및 RAG 문서의 기관·발행일·공식 URL 보강
  - 시장 규모·CAGR·수요/위험을 구조화하고 근거 ID와 연결
  - 범위가 다르거나 충돌한 수치는 임의로 결합하지 않고 결측 처리
- Evaluator
  - 설계의 13개 지표 전체 생성
  - 정량·범주 지표의 코드 채점
  - 정성 지표의 1/3/5점 구조화 채점과 Evidence ID 검증
  - 결측 지표 3점 중립 대입 및 `status="missing"` 구분
  - 6개 항목 내부 평균과 25·20·20·15·10·10 가중 총점 계산
- Decision
  - 총점 70점 이상, 결측 5개 미만, 선정 불확실성 없음의 세 조건을 코드로 판정
  - 후보별 분석·근거·점수·판정 스냅샷 저장
  - 충족/미충족 조건을 판정 사유에 기록

## 현재 설계 미달 또는 통합 전 제약

1. Startup, Technology, Competitor가 아직 stub이므로 실제 실행에서는 팀·기술·경쟁·실적·투자
   Evidence가 충분히 공급되지 않는다. Evaluator는 이 경우 설계대로 결측 3점을 적용하며, 임의의
   관측값을 만들지 않는다.
2. `app.py`의 `invest`, `all_hold`, `evidence_retry` 등은 기존 고정점수 stub을 검증하던 시나리오다.
   Evaluator가 실제 근거 기반 계산으로 변경되어 시나리오 이름만으로 투자/보류 결과가 강제되지 않는다.
   실제 Agent fixture가 준비될 때 통합 실행 시나리오를 다시 구성해야 한다.
3. 설계에 기재된 독립 `summarize_sources` 도구는 없다. 현재 Market Agent 내부의 구조화 LLM 호출이
   해당 역할을 수행하지만, 설계에 적힌 별도 도구 단위로 분리된 상태는 아니다.
4. 웹검색 결과는 발행일이 제공되지 않을 수 있다. 질의에 `as_of_date`를 포함하지만 검색 API 단계에서
   기준일 이후 자료를 완전히 차단할 수 없으므로, 날짜가 없는 웹 근거의 기준일 준수 여부는 미확정이다.
5. 금액 정규화는 설계와 현재 config에 있는 USD/KRW 고정 환율만 지원한다. 다른 통화이거나 단위가
   불명확하면 결측으로 처리한다.
6. 정성 채점은 동일 프롬프트와 `temperature=0`을 사용하지만 LLM 출력의 완전한 결정성을 보장하지
   않는다. 설계가 요구하는 실제 프롬프트·모델 응답의 영속 보관은 현재 LangSmith 설정에 의존한다.
7. 실제 OpenAI·Tavily 호출을 포함한 end-to-end 검증과 최종 보고서/PDF 인용 검증은 아직 수행하지
   않았다. 이 문서에 포함된 자동 테스트는 외부 호출을 대역으로 교체한 담당 로직 단위 테스트다.
8. 로컬에 Qwen 임베딩 모델이 캐시되어 있어도 Hugging Face가 시작 시 원격 메타데이터를 확인할 수
   있다. 네트워크가 제한된 실행 환경에서는 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`을 설정해야
   저장된 모델과 인덱스만으로 즉시 검색된다.

## 검증 방법

```bash
uv run python -m unittest discover -s tests -v
```

담당 테스트 파일: `tests/test_market_evaluator_decision.py`
