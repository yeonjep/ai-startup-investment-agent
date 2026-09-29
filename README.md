# AI Startup Investment Evaluation Agent

본 프로젝트는 [TODO: 확정된 도메인 및 평가 범위] 스타트업의 투자 가능성을 분석하는 에이전트를 설계하고 구현하는 실습 프로젝트입니다.

## Overview

- Objective: [TODO]
- Method: [TODO]

## Features

- [TODO]

## Tech Stack

- Framework: [TODO]
- LLM/Generator: [TODO]
- LLM/Judge: [TODO]
- Retrieval: [TODO]
- Embedding: [TODO]

## Agents

- [TODO: 에이전트별 역할]

## Architecture

[TODO: 확정된 그래프 구조 및 이미지]

## Directory Structure

```text
├── agents/             # 에이전트와 그래프 공통 모듈
├── data/
│   ├── docs/           # RAG 입력 PDF
│   ├── raw/            # 원본 문서
│   └── seed_startups.json  # [TODO: 확정 후 추가]
├── outputs/            # 생성된 결과물
├── prompts/            # 프롬프트 템플릿
├── rag/                # 문서 적재, 검색, 평가 모듈
├── tools/              # 외부 도구
├── app.py              # 실행 스크립트
└── README.md
```

## Usage

```bash
# 의존성 설치
uv sync

# 환경 변수 파일 생성 후 필요한 API 키 설정
cp .env.example .env
# .env의 OPENAI_API_KEY, TAVILY_API_KEY 등을 설정

# RAG 문서 로딩, 청킹, 임베딩 및 FAISS 인덱스 생성
uv run python -m rag.ingest

# 애플리케이션 실행
uv run python app.py
```

## Contributors

- [TODO: 이름 및 개인별 수행 역할]
