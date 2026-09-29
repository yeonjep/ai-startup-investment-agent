# 설계 확정 후 교체할 부분: 임베딩 모델과 검색 설정은 조 최종 결정에 맞춰 변경합니다.

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstores"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def vectorstore_path_for_model(model_name: str) -> Path:
	return VECTORSTORE_DIR / model_name.replace("/", "--")