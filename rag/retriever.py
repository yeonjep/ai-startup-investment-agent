# 설계 확정 후 교체할 부분: 검색 필터와 반환 스키마는 RAG 설계 확정 후 조정합니다.

from functools import lru_cache
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

from agents.common import configure_runtime
from rag.config import EMBEDDING_MODEL, vectorstore_path_for_model


@lru_cache(maxsize=2)
def _load_vectorstore(embedding_model: str) -> FAISS:
    configure_runtime()
    vectorstore_path = vectorstore_path_for_model(embedding_model)
    if not (vectorstore_path / "index.faiss").is_file():
        raise FileNotFoundError(
            f"FAISS index for EMBEDDING_MODEL='{embedding_model}' was not found at "
            f"{vectorstore_path}. Build it with: uv run python -m rag.ingest"
        )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search(
    query: str,
    k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    vectorstore = _load_vectorstore(EMBEDDING_MODEL)
    documents = vectorstore.similarity_search(query, k=k, filter=filters)
    return [
        {
            "content": document.page_content,
            "doc_id": document.metadata.get("doc_id"),
            "title": document.metadata.get("title"),
            "page": document.metadata.get("page"),
            "source": document.metadata.get("source"),
            "url": document.metadata.get("url"),
            "chunk_id": document.metadata.get("chunk_id"),
        }
        for document in documents
    ]


@tool("search_rag_documents")
def search_rag_documents(
    query: str,
    k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search the stored RAG index and return matching source metadata."""
    return search(query=query, k=k, filters=filters)