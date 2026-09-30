# 설계 기준: docs/DESIGN.md B-1 2장, B-3, B-4.

from functools import lru_cache
import pickle
from typing import Any, Literal

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from agents.common import configure_runtime, create_llm
from agents.config import LLM_JUDGE_MODEL
from rag.config import (
    BM25_RETRIEVAL_WEIGHT,
    BM25_INDEX_FILENAME,
    BM25_TOP_K,
    DENSE_RETRIEVAL_WEIGHT,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    ENSEMBLE_RRF_C,
    HYBRID_TOP_K,
    QWEN_QUERY_INSTRUCTION,
    vectorstore_path_for_model,
)
from rag.ingest import load_and_split_documents


class QueryInstructionEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.model_name == "Qwen/Qwen3-Embedding-0.6B":
            text = f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery: {text}"
        return self.base_embeddings.embed_query(text)


@lru_cache(maxsize=2)
def _load_vectorstore(embedding_model: str) -> FAISS:
    configure_runtime()
    vectorstore_path = vectorstore_path_for_model(embedding_model)
    if not (vectorstore_path / "index.faiss").is_file():
        raise FileNotFoundError(
            f"FAISS index for EMBEDDING_MODEL='{embedding_model}' was not found at "
            f"{vectorstore_path}. Build it with: uv run python -m rag.ingest"
        )

    embeddings = QueryInstructionEmbeddings(embedding_model)
    return FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[Document, ...]:
    chunks, _ = load_and_split_documents()
    return tuple(chunks)


@lru_cache(maxsize=2)
def _load_bm25_retriever(embedding_model: str) -> BM25Retriever:
    index_path = vectorstore_path_for_model(embedding_model) / BM25_INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"BM25 index was not found at {index_path}. "
            "Build or update indexes with: uv run python -m rag.ingest"
        )
    with index_path.open("rb") as index_file:
        return pickle.load(index_file)


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


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
    corpus = [chunk for chunk in _load_chunks() if _matches_filters(chunk.metadata, filters)]
    if not corpus:
        return []

    dense_search_kwargs: dict[str, Any] = {"k": DENSE_TOP_K, "fetch_k": len(corpus)}
    if filters:
        dense_search_kwargs["filter"] = filters
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=dense_search_kwargs,
    )
    if filters:
        bm25_retriever = BM25Retriever.from_documents(corpus, k=BM25_TOP_K)
    else:
        bm25_retriever = _load_bm25_retriever(EMBEDDING_MODEL)
        bm25_retriever.k = BM25_TOP_K
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[DENSE_RETRIEVAL_WEIGHT, BM25_RETRIEVAL_WEIGHT],
        c=ENSEMBLE_RRF_C,
        id_key="chunk_id",
    )
    component_results = [dense_retriever.invoke(query), bm25_retriever.invoke(query)]
    documents = hybrid_retriever.weighted_reciprocal_rank(component_results)[
        : min(k, HYBRID_TOP_K)
    ]
    weights = (DENSE_RETRIEVAL_WEIGHT, BM25_RETRIEVAL_WEIGHT)
    scores_by_chunk_id: dict[str, float] = {}
    for weight, component_documents in zip(weights, component_results, strict=True):
        for rank, document in enumerate(component_documents, start=1):
            chunk_id = str(document.metadata.get("chunk_id"))
            scores_by_chunk_id[chunk_id] = scores_by_chunk_id.get(chunk_id, 0.0) + (
                weight / (rank + ENSEMBLE_RRF_C)
            )
    return [
        {
            "rank": rank,
            "content": document.page_content,
            "doc_id": document.metadata.get("doc_id"),
            "title": document.metadata.get("title"),
            "page": document.metadata.get("page"),
            "chunk_id": document.metadata.get("chunk_id"),
            "score": scores_by_chunk_id.get(str(document.metadata.get("chunk_id")), 0.0),
        }
        for rank, document in enumerate(documents, start=1)
    ]


@tool("rag_search")
def rag_search(
    query: str,
    k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search RAG documents with Dense FAISS and BM25 hybrid retrieval."""
    return search(query=query, k=k, filters=filters)


class ChunkGrade(BaseModel):
    chunk_id: str = Field(description="ID of the chunk being evaluated")
    relevance: Literal["yes", "no"] = Field(
        description="Whether the chunk contains evidence relevant to the query"
    )
    reason: str = Field(description="Brief reason for the relevance decision")
    answerable_metrics: list[str] = Field(
        description="Metric IDs or market topics that this chunk can answer"
    )


class RelevanceResponse(BaseModel):
    results: list[ChunkGrade]


@tool("grade_relevance")
def grade_relevance(query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Grade each supplied RAG chunk as yes/no for answering the query."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not chunks:
        return []

    configure_runtime()
    structured_llm = create_llm(LLM_JUDGE_MODEL, temperature=0).with_structured_output(
        RelevanceResponse
    )
    chunk_inputs = [
        {
            "chunk_id": str(chunk.get("chunk_id", index)),
            "content": str(chunk.get("content", "")),
        }
        for index, chunk in enumerate(chunks)
    ]
    response = structured_llm.invoke(
        [
            (
                "system",
                "Assess each chunk independently. Answer yes only if it contains useful "
                "evidence for answering the query. For every chunk return yes/no, a brief "
                "reason, and the metric IDs or topics it can answer. Return one result for "
                "every chunk, preserving its chunk_id exactly.",
            ),
            ("human", f"Query: {query}\nChunks: {chunk_inputs}"),
        ]
    )
    grades_by_id = {grade.chunk_id: grade for grade in response.results}
    expected_ids = [item["chunk_id"] for item in chunk_inputs]
    if set(grades_by_id) != set(expected_ids):
        raise ValueError("Structured relevance output did not grade every supplied chunk")

    return [
        {
            "chunk_id": chunk.get("chunk_id", index),
            "relevance": grades_by_id[item["chunk_id"]].relevance,
            "reason": grades_by_id[item["chunk_id"]].reason,
            "answerable_metrics": grades_by_id[item["chunk_id"]].answerable_metrics,
        }
        for index, (chunk, item) in enumerate(zip(chunks, chunk_inputs, strict=True))
    ]