# 설계 기준은 docs/DESIGN.md의 B-3/B-4입니다.

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstores"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
MIN_MEANINGLESS_FRAGMENT_CHARS = 20
QWEN_QUERY_INSTRUCTION = "Given a question, retrieve relevant passages that answer the question"
DENSE_RETRIEVAL_WEIGHT = 0.5
BM25_RETRIEVAL_WEIGHT = 0.5
ENSEMBLE_RRF_C = 60


def vectorstore_path_for_model(model_name: str) -> Path:
	return VECTORSTORE_DIR / model_name.replace("/", "--")