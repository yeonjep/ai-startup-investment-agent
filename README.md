# AI Startup Investment Evaluation Agent

본 프로젝트는 AI 반도체(Semiconductor) 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다. AI 연산용 칩을 설계·개발하거나, AI로 반도체 설계·제조 공정을 개선하는 국내외 비상장 스타트업(Seed~Series C)을 대상으로 합니다.

> 설계 기준 문서: [docs/DESIGN.md](docs/DESIGN.md) · 울산 캠퍼스 4반

## Overview

- Objective : AI 반도체 스타트업의 **팀, 시장성, 제품/기술력, 경쟁 우위, 실적, 자금조달·리스크**를 기준으로 투자 적합성을 분석하고 투자 평가 보고서(PDF)를 자동 생성
- Method : Supervisor 패턴의 LangGraph Multi-Agent, Agentic RAG(Hybrid 검색 + 관련성 평가 + 재검색 + 웹 보완), 근거(Evidence) 기반 13개 지표 채점, 코드 규칙에 의한 투자/보류 판정, 보고서 품질 평가

### Pattern 및 선정 이유

- **Pattern: Supervisor (hub-and-spoke)**
- 워커의 역할은 고정돼 있지만, 다음에 누구를 부를지는 후보 선정·근거 충분성·결측·판정·품질 결과에 따라 달라진다. 워커가 다음 워커를 직접 부르면 그 분기가 그래프 엣지에 고정된다. Supervisor가 State만 보고 다음 워커를 고르면 재작업과 종료를 한곳에서 판단할 수 있다.
- 투자 점수와 판정은 코드 규칙을 유지한다. Supervisor는 점수를 계산하지 않고, 근거가 충분한지와 재시도 상한만 본다.

### 동적 처리 — 고정 순서와의 차이

고정 순서는 `Startup → Technology → Market → Competitor → Evaluator → Decision → Report`를 항상 같은 횟수로 실행한다. 이 그래프에는 워커에서 워커로 가는 엣지가 없고, Supervisor가 매 반환마다 다음 대상을 고른다.

- 근거가 부족한 영역만 해당 워커에게 다시 맡긴다. 세 분석을 항상 처음부터 다시 하지 않는다.
- 근거가 충분하거나 재분석 상한에 닿은 뒤에만 Decision을 거쳐 Report로 보낸다.
- 품질 평가가 미달이면 Supervisor가 Report 재작성을 한 번 배정한 뒤 Quality를 다시 실행한다.
- `step_count`가 상한(기본 80)에 도달하면 Supervisor가 Report 작성 후 종료한다. 평소 종료는 이 상한이 아니라 후보 수와 재시도 예산이다.

## Problem & Approach

- **문제**: AI 반도체 스타트업 투자 검토는 기술 검증·시장성·경쟁력 정보가 홈페이지·기사·산업 보고서에 흩어져 있어 수작업 조사가 오래 걸리고, 평가자마다 점수 기준이 달라집니다.
- **접근**: 후보 탐색과 기술·시장·경쟁 분석을 Agent별로 분담하고, 공통 **Evidence(출처 포함)** 와 평가표로 통합합니다. LLM은 정보 추출·요약·정성 채점만 맡고, **점수 계산·투자/보류 판정·분기는 코드**가 처리합니다.

## 차별점 (Differentiators)

1. **Corrective Agentic RAG — 부족한 topic만 다시 검색**

   시장 분석 질의를 규모·성장률·수요/리스크 3개 topic으로 나눠 Hybrid 검색(Dense + BM25, Weighted RRF)을 하고, topic별로 관련성을 평가합니다. 기준에 못 미친 topic만 질의를 재작성해 최대 2회 재검색하고, 그래도 부족하면 웹 검색으로 보완합니다. 임베딩은 한·영 혼합 과제 문서로 3종을 직접 비교해 Qwen3-Embedding-0.6B를 골랐습니다.

2. **LLM과 코드의 역할 분리**

   LLM은 지표별 근거(URL·원문 발췌)를 추출하고 보고서 문장을 씁니다. 가중 합산 점수와 투자/보류 분기(`총점 ≥ 70` · `미확인 < 5` · `선정 불확실성 없음`)는 코드가 계산하므로 입력이 같으면 결론도 같습니다. 시장 수치는 인용 원문에 그 숫자가 실제로 있는지 코드로 대조해 LLM이 만든 수치를 걸러냅니다.

3. **3-상태 채점: 확인 / 부재 확인 / 미확인**

   "없음이 확인됨"(1점)과 "정보를 못 찾음"(미확인)을 구분합니다. 미확인 지표는 중립 3점을 넣고 보고서에 표시하며, 미확인 지표만 골라 근거 보완 검색을 1회 더 실행합니다. 그래도 5개 이상 남으면 판정을 보류합니다. API 오류나 빈 검색 결과도 미확인으로 처리합니다.

4. **기업 유형별 해석 + 공통 13개 지표**

   후보를 칩(CHIP)·AI 설계/EDA(DESIGN_AI)·AI 공정(PROCESS_AI)으로 분류하고, 유형마다 개발 단계·기술 검증 조건·진입장벽·리스크를 다르게 해석합니다. 13개 지표와 가중치는 고정이라 유형 분류가 점수를 유리하게 바꾸지 못합니다.

## Features

- **국내외 후보 탐색**: 한국어·영어 웹검색으로 후보를 최대 10개 수집하고, 6개 선정 기준(도메인·AI 핵심성·비상장·투자 단계·Exit 미완료·정보 충분성)으로 PASS/FAIL/REVIEW 판정
- **PDF 자료 기반 시장 분석(Agentic RAG)**: 정부·연구기관·컨설팅 보고서 4종(100페이지)을 Hybrid 검색(Dense + BM25, Weighted RRF)으로 조회하고, topic별 관련성 평가 → 질의 재작성 → 웹 보완
- **근거 기반 채점**: 모든 지표는 출처가 연결된 Evidence로만 채점하고, 근거가 없으면 중립 3점 + `missing`으로 구분. 정량 점수·총점·투자/보류 분기는 LLM이 아닌 코드가 계산
- **확증 편향 방지**: 한 출처가 인용의 70%를 넘으면 품질 평가가 실패한다. 금액이 다른 근거는 보고서에 충돌로 적고, 실측·제3자 근거가 없는 기술·경쟁·리스크 완화 주장은 점수 상한을 3점으로 둔다
- **보고서 품질 평가**: Groundedness, 중립성, 편향 통제, 관점 커버리지 4항목. 인용·출처 다양성·수치 충돌·관점 제목은 코드가 검사하고, 중립성과 본문 모순은 LLM(gpt-4.1-mini, temperature 0)이 검사한다. 미달이면 Supervisor가 해당 섹션만 1회 다시 쓰게 하고, 결과는 보고서 본문이 아니라 `outputs/quality_eval.json`에 남긴다
- **분기·루프**: Supervisor가 근거 충분성에 따라 필요한 워커만 다시 호출한다. 종료는 선정 재검색 1회, 시장 재검색 topic별 2회, 근거 재분석 후보당 1라운드, 후보 수, 품질 재작성 1회, `step_count` 상한으로 보장한다
- **투자 보고서 자동 생성**: 완료된 평가 스냅샷만 받아 SUMMARY~REFERENCE 구조의 10페이지 이내 PDF를 생성한다. 주장별 실제 Evidence만 인용한다

## Tech Stack

- Framework : LangGraph, LangChain (Python 3.11, uv)
- LLM/Generator : gpt-4.1-mini (분석·보고서 작성)
- LLM/Judge : gpt-4.1-mini (정성 채점·관련성 평가, temperature=0)
- Retrieval : FAISS(Dense) + BM25(Sparse), Weighted RRF (0.5/0.5, c=60), Top-5 — 사전 벤치마크 Hit Rate@1/3/5 = 0.85 / 1.00 / 1.00, MRR = 0.925 (Qwen3, 125청크 파일럿·합성 질문 20개 기준, [outputs/embedding_eval.csv](outputs/embedding_eval.csv))
- Embedding : Qwen/Qwen3-Embedding-0.6B (오픈소스, 로컬 실행)
- Web Search : Tavily
- Report : PyMuPDF(PDF 변환), matplotlib(차트), 나눔고딕(한글 폰트, OFL)
- Tracing : LangSmith

### Embedding 선정

BAAI/bge-m3, intfloat/multilingual-e5-large, Qwen/Qwen3-Embedding-0.6B 3종을 동일한 청크·질문·정답으로 비교했습니다. 리더보드 순위가 아니라 **본 과제 문서(한·영 혼합 4종)에서의 실측 검색 성능**으로 선택했습니다.

| 모델                           |    Hit@1 |    Hit@3 |    Hit@5 |       MRR | 인덱싱(초) |
| ------------------------------ | -------: | -------: | -------: | --------: | ---------: |
| BAAI/bge-m3                    |     0.75 |     0.95 |     0.95 |     0.842 |       48.4 |
| intfloat/multilingual-e5-large |     0.70 |     0.95 |     0.95 |     0.817 |       43.7 |
| **Qwen/Qwen3-Embedding-0.6B**  | **0.85** | **1.00** | **1.00** | **0.925** |      110.4 |

인덱싱 시간은 최초 1회 비용이며 인덱스를 재사용합니다. 현재 RAG 적재본은 4개 문서, PDF 100페이지, 117청크(chunk_size=1000, overlap=100)입니다. 평가는 125청크 파일럿·합성 질문 20개 기준의 상대 비교이며 Hybrid 적용 효과는 별도로 검증하지 않았습니다.

## Agents

- **Supervisor** : 모든 워커가 작업 후 돌아오는 허브. State의 관점·근거 충분성·결측·판정·품질 결과·`step_count`를 보고 다음 워커를 고른다. 라우팅 판단은 코드 규칙이고, 각 결정은 `outputs/logs/{trace_id}.jsonl`에 남긴다
- **Startup Agent** : 국내외 후보 수집·정규화·중복 제거, A-3 선정 기준 판정과 기업 유형(CHIP / DESIGN_AI / PROCESS_AI) 분류, 팀·인력·투자·고객·매출 근거 수집
- **Technology Agent** : 기업 유형별 기술 지표·개발 단계·검증 조건 수집, 기업 주장과 검증된 효과 구분, 기술 영역 충분성 투표
- **Market Agent (RAG)** : `market_size` / `market_growth` / `demand_risk` topic별 Hybrid RAG → 관련성 평가 → 재검색 → 웹 보완, 범위·기준연도가 다른 수치는 결합하지 않음, 시장 영역 충분성 투표
- **Competitor Agent** : 같은 유형·고객 문제의 국내외 경쟁 3~5개(기존 비AI 방식 포함) 비교, 진입장벽 근거 수집, 경쟁 영역 충분성 투표
- **Evaluator Agent** : 13개 지표 채점(정량·범주는 코드, 정성은 1/3/5 루브릭), 결측 처리, 가중 총점 계산
- **Decision Agent** : `total_score ≥ 70` · `결측 < 5` · `uncertain == False` 세 조건을 코드로 판정하고 후보별 스냅샷 저장. 충분성 투표는 이 판정을 바꾸지 않는다
- **Report Agent** : 투자 기업 중심 / 전원 보류 비교 / 평가 대상 없음 3유형을 생성한다. PASS·REVIEW 후보의 평가가 미완료이면 생성을 거부하고, 품질 수정 시 기존 판정·점수를 유지한다
- **Quality Agent** : Groundedness(코드), 중립성(LLM), 편향 통제(코드), 관점 커버리지(코드). 미달이면 Supervisor가 보고서를 1회 재작성하게 한다

## State Schema

가이드의 예시 State는 관점 합성용 참고라 그대로 쓰지 않았다. 이 그래프의 State는 투자 평가 본문과, Gate·Supervisor가 분기와 재작업을 결정하는 제어 정보로 나뉜다.

작업 결과는 `candidates`, `current_startup`, 기술·시장·경쟁 분석, Evidence, `scores`, `decision`, `evaluated`, `final_report`다. 제어 정보는 `trace_id`, `step_count`, `node_status`, `last_error`, `next_routes`, `route_reason`, 재시도 카운터, `current_idx`, `sufficiency_votes`, `refresh_areas`, `evidence_gate_status`, `quality_retry`다.

| 항목 | 이 그래프에서의 답 |
| --- | --- |
| 제어 vs 페이로드 분리 | 라우팅은 제어 필드만 본다. 기업 분석, 근거, 점수, 보고서 문장은 작업 결과로 두고 라우팅 로그는 State에 넣지 않는다 |
| 관측성 위치 | `{trace_id, node, decision, reason, ts}`는 `outputs/logs/{trace_id}.jsonl`에 쌓는다. State에는 최신 `next_routes`, `route_reason`, `node_status`, `last_error`만 남긴다 |
| 지속성 비용 | 검색 원문과 RAG 청크는 State에 넣지 않고 Evidence 인용문은 500자로 자른다. 후보가 바뀌면 그 후보의 Evidence를 비우고, 스냅샷은 `evaluated`에 후보당 1건만 남긴다 |
| 상관 | 실행 시작 시 `trace_id`를 만들어 LangGraph `run_name`·`metadata`·`tags`에 넣는다. 같은 값이 체크포인터 `thread_id`와 로그 파일명이다 |
| 재개/복구 | `MemorySaver`에 `thread_id=trace_id`로 컴파일한다. `node_status`는 pending→running→done/failed로 바뀌고, 실패 시 `last_error`를 남긴다 |
| 동시 처리 | 분석 3종은 서로 다른 결과·Evidence 필드에 쓴다. `sufficiency_votes`와 `node_status`는 dict merge reducer로 워커별 key를 합치고, `evaluated`는 `operator.add`로 잇는다 |
| 종료 보장 | Supervisor가 `step_count`를 올리고, `MAX_STEP_COUNT`(기본 80, 환경 변수로 변경)에 도달하면 Report 후 END로 강제 종료한다. 평소에는 후보 수·선정 1회·근거 재분석 1라운드·품질 재작성 1회에서 멈춘다 |

## Architecture

![Graph](outputs/graph.png)

`initialize_state` 다음 모든 분기는 Supervisor다. Startup, Technology, Market, Competitor, Evaluator, Decision, Report, Quality는 끝나면 Supervisor로만 돌아간다. (상세: [docs/DESIGN.md](docs/DESIGN.md) D장)

- 근거가 부족한 분석만 다시 보내고, 충분하거나 재분석 상한 뒤에 Decision과 Report로 보낸다
- 품질 미달이면 Supervisor가 Report를 한 번 다시 쓰게 한 뒤 Quality로 되돌린다
- `step_count` 상한에 도달하면 Report 작성 후 종료한다. 투자/보류 규칙은 코드가 유지한다

## 실행 결과 (2026-10-13 검증)

실제 API와 RAG로 Supervisor 허브 그래프를 한 번 실행했다. trace `b3ec877a7887492ab3c04405d335c6a6`.

```text
initialize_state → supervisor
→ startup(탐색) → supervisor → startup(FAIL, 다음 후보) → supervisor → startup(FAIL, 다음 후보)
→ supervisor → startup(REVIEW 보완) → supervisor
→ [technology ∥ market ∥ competitor] → supervisor → evaluator
→ supervisor → [technology ∥ market] 재분석 → supervisor → evaluator
→ supervisor → decision(보류) → supervisor → startup(다음 후보) → supervisor
→ report → supervisor → quality → supervisor → report 재작성 → supervisor → quality
→ supervisor → END
```

| 단계 | 결과 |
| --- | --- |
| 라우팅 | 워커는 모두 supervisor로 복귀. 근거 재분석 1회(기술·시장), 품질 재작성 1회 후 상한으로 종료 |
| 평가 | **아이에이치더블유(iHW)** — 총점 **72.67**, 보류. 이어서 평가할 후보가 없어 보고서 작성 |
| 품질 | Groundedness·중립성·관점 커버리지 통과. 편향 통제는 본문 수치 충돌 미표기로 실패. 재작성 1회 후에도 실패해 보고서는 유지하고 `outputs/quality_eval.json`에만 기록 |
| 보고서 | **5페이지**. SUMMARY~REFERENCE |

### 투자 보고서 핵심 포인트

- **판정**: 아이에이치더블유(iHW, 한국, ACiM AI 반도체) — 총점 72.67/100, 보류.
- 목표 시장 규모·CAGR은 타깃과 범위가 달라 채점에서 제외하고, 참고 수치로만 적었다.
- 품질 재작성 상한(1회) 이후에도 편향 통제가 미달이다. 미달 내용은 보고서 본문에 넣지 않았다.
- 보고서 구성: SUMMARY → 1. 기업 및 사업 개요 → 2. 팀·기술 및 제품 경쟁력 → 3. 시장성 및 경쟁 환경 → 4. 투자 판단 및 주요 리스크 → REFERENCE (5페이지)

> 웹검색과 LLM 응답은 실행마다 달라져 후보·점수가 매번 같지는 않습니다. 위 내용은 제출본 PDF와 같은 실행 기준입니다.

미리보기 이미지는 이전 레이아웃 확인본이다. 이번 실행 PDF는 5페이지다.

| 1페이지 | 2페이지 | 4페이지 |
| --- | --- | --- |
| ![p1](docs/report_preview/page-1.png) | ![p2](docs/report_preview/page-2.png) | ![p4](docs/report_preview/page-4.png) |

## Directory Structure

```text
├── agents/             # Agent 모듈(startup, technology, competitor, report), State, Graph, config
│   ├── supervisor.py   # 다음 워커 선택
│   ├── route_log.py    # 라우팅 결정 JSONL
│   └── stubs.py        # Market·Evaluator·Decision, initialize_state, 후보 초기화
├── data/
│   ├── raw/            # 원본 PDF
│   ├── docs/           # RAG 적재용 PDF (4종, 100페이지)
│   └── vectorstores/   # FAISS·BM25 인덱스
├── prompts/            # 프롬프트 템플릿
├── rag/                # 문서 적재·청킹, Hybrid 검색, 임베딩 평가
├── tools/              # web_search, 차트, PDF 변환
├── tests/              # 단위 테스트, 보고서 fixture
├── assets/fonts/       # 보고서용 한글 폰트
├── outputs/            # 보고서, 후보 목록, graph.png, logs/{trace_id}.jsonl, quality_eval.json
├── docs/DESIGN.md      # 설계 문서(구현 기준)
├── app.py              # 실행 스크립트
└── README.md
```

## Usage

```bash
# 1. 의존성 설치
uv sync

# 2. 환경 변수 설정 (.env는 커밋하지 않습니다)
cp .env.example .env
# OPENAI_API_KEY, TAVILY_API_KEY 등을 입력

# 3. (선택) RAG 인덱스 재생성 — 저장소에 인덱스가 포함되어 있어 보통 필요 없습니다
uv run python -m rag.ingest

# 4. 실행: 후보 탐색 → 분석 → 채점 → 판정 → 보고서 생성
uv run python app.py
```

- 결과: `outputs/RAG-Output_울산캠퍼스-4반_박연제+이정인+정예지+박태준.pdf` (+ 같은 이름의 `.md`)
- 평가 수 상한은 환경 변수 `MAX_CANDIDATES`(1~10, 기본 5), 전역 스텝 상한은 `MAX_STEP_COUNT`(기본 80), 기준일은 `AS_OF_DATE`(YYYY-MM-DD)로 조정합니다.
- 실행이 끝나면 LangSmith run URL과 `outputs/logs/{trace_id}.jsonl` 경로를 출력합니다. 그래프는 `thread_id=trace_id`인 MemorySaver로 컴파일합니다.
- 최초 실행 시 임베딩 모델(Qwen3-Embedding-0.6B)이 다운로드됩니다.
- 웹 검색과 LLM 응답은 실행마다 달라질 수 있어 후보·점수가 매번 같지는 않습니다.
- `--scenario invest|all_hold|zero_pass|evidence_retry|no_candidates|uncertain` 옵션은 그래프 분기 검증용 stub 후보이며, 결과는 `stub-*` 파일로 분리 저장됩니다.
- 보고서 유형별 렌더링 확인: `uv run python -m tests.run_report_fixtures` (가상 데이터, `outputs/fixtures/`)
- 단위 테스트: `uv run --with pytest python -m pytest -q`

## Lessons Learned

- **State 형식 불일치**: Agent 통합 시 `tech_summary`가 dict가 아닌 문자열로 전달되어 Decision 노드에서 실행 중단. State·Evidence 규격에 맞춰 수정. 개발 초기 공통 fixture와 스키마 테스트의 선행 구축 필요.
- **LLM 실패 시 기본값**: 초기 코드는 LLM 호출 실패 시 기본 후보·기술 정보를 채워 넣는 구조. 실제 근거로 오인될 위험이 있어 전부 제거, 실패 시 결측으로 기록하도록 변경.
- **단위 환산 버그**: `"1,228억"`과 `"달러"`가 1,228달러로 환산되어 시장 규모가 $0M으로 채점. 환산 로직 수정, 인용 원문–수치 대조 검증 및 회귀 테스트 추가.
- **웹 검색 발행일**: 웹 검색 결과 다수에서 발행일 누락. URL·본문·페이지 메타데이터 기반 날짜 추출, 미확인 시 조회일로 표기.
- **판정 로직 분리**: 점수 계산·판정·분기의 코드 기반 처리로, 실행마다 검색 결과가 달라도 결과 도출 과정 추적 가능.
- **정성 지표 점수**: 하이퍼엑셀의 일부 정성 지표는 기업·투자사 발표 근거 비중이 높아, 외부 자료 교차 검증을 강화할 필요가 있습니다.

## Limitations

- 정량 구간·투자 임계값(70점, 결측 5개)은 검증된 VC 표준이 아닌 프로젝트 정책값이며, 공통 점수는 선별 보조값입니다.
- Valuation·지분율이 비공개이므로 ROI는 계산하지 않고, 누적 투자액은 조달 이력의 대리 지표로만 사용합니다.
- 조사 순서상 최초 투자 판정에서 종료하므로 투자 기업이 전체 후보 중 최우수 기업이라는 뜻은 아닙니다.
- RAG 문서 4종은 AI 반도체 산업 배경 중심이라 AI 기반 EDA·공정 개선의 세부 시장은 웹 보완으로 확인하며, 확인하지 못한 지표는 결측으로 처리합니다.
- 글로벌 **HBM 시장** CAGR 36.3%(2022~2027)는 하이퍼엑셀의 LPU 타깃 시장과 범위가 달라 제출본의 목표 시장 성장률 점수에서 제외했습니다.
- 정성 지표는 LLM 채점이라 실행마다 다소 달라질 수 있고, 기업 발표 위주의 근거는 우호적으로 평가될 수 있습니다.
- 웹 자료의 발행일은 검색 결과의 공개일 또는 URL·본문에 명시된 날짜로만 표기하며, 확인되지 않으면 조회일을 기재합니다.

## Contributors

• 박연제 : RAG 인프라·그래프 통합·보고서 생성
   ◦ RAG 문서 준비·청킹·메타데이터 스키마, Qwen3 로컬 임베딩.
   ◦ FAISS + BM25 Hybrid 검색, 인덱스 재사용, 임베딩 3종 비교 평가.
   ◦ Supervisor 허브 조립. 워커는 작업 후 supervisor로만 복귀한다.
   ◦ 병렬 결과 병합 Reducer와 재작업 State 스키마 통합.
   ◦ 미완료 평가의 보고서 생성 차단, 투자·전원 보류·대상 없음 유형별 작성 조건 정리.
   ◦ 검증된 시장 지표만 보고서 입력에 반영, REFERENCE 생성·PDF 편집·보고서 계약 테스트.
   ◦ 설계서·README·그래프 이미지 갱신 및 통합 실행 점검.

• 이정인 : 후보 탐색·기술 분석·경쟁사 분석
   ◦ 웹검색·출처 요약 도구, 국내외 후보 탐색·정규화.
   ◦ Startup Agent의 PASS/FAIL/REVIEW 선정, 기업 유형 분류, 프로필 근거 수집.
   ◦ Technology·Competitor Agent의 기술 검증 조건·차별성·진입장벽 분석.
   ◦ 기술·경쟁 분석 결과의 충분성 투표와 근거 반환 형식 정리.
   ◦ Supervisor가 지정한 재조사 영역에 맞춰 추가 검색 수행.
   ◦ 재조사 시 기존 근거 보존·병합과 분석 Worker 입출력 점검.

• 정예지 : 시장 근거 검증·채점·투자 판정
   ◦ Market Agent의 주제별 Corrective RAG, 최대 2회 재검색과 웹 보완.
   ◦ Evaluator Agent의 13개 지표 채점, 단위·통화 환산, 가중 총점 계산.
   ◦ Decision Agent의 투자/보류 규칙과 후보별 평가 스냅샷 저장.
   ◦ 투자유치 금액과 시장 규모 구분, 타깃 시장과 다른 수치의 검증 실패 처리.
   ◦ 검증 실패한 시장 수치를 결측으로 유지하도록 채점 로직 보완.
   ◦ 결측 목록을 Supervisor에 전달하고, 근거 보완 후 재채점하는 흐름 연동.
   ◦ 시장 수치 검증·결측 처리·채점·판정 경계값 회귀 테스트.

• 박태준 : Supervisor 근거 Gate·품질 평가·실행 추적
   ◦ 근거 충분성·결측·재시도 예산으로 재분석 또는 Decision 진행을 결정.
   ◦ 재시도 소진 시 잔여 결측을 유지하고 판정으로 진행.
   ◦ 품질 미달 시 기존 점수·판정을 유지하며 보고서 수정을 1회 요청. 상한 후에는 보고서를 유지.
   ◦ Groundedness·중립성·편향 통제·관점 커버리지 평가 점검.
   ◦ Supervisor 분기·재작업·step 상한·종료 테스트, LangSmith Tracing 캡처와 제출 결과 점검.
