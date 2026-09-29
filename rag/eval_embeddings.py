import csv
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from rag.config import PROJECT_ROOT, vectorstore_path_for_model
from rag.ingest import load_and_split_documents


MODEL_NAMES = (
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "Qwen/Qwen3-Embedding-0.6B",
)
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.json"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "embedding_eval.csv"
EVAL_SAMPLE_COUNT = 20
MIN_CHUNK_LENGTH = 100
RANDOM_SEED = 42
MAX_K = 5
QWEN_QUERY_INSTRUCTION = "Given a question, retrieve relevant passages that answer the question"


class RetrievalEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.dimension = len(self.embed_query("embedding dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model_name == "intfloat/multilingual-e5-large":
            texts = [f"passage: {text}" for text in texts]
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.model_name == "intfloat/multilingual-e5-large":
            text = f"query: {text}"
        elif self.model_name == "Qwen/Qwen3-Embedding-0.6B":
            text = f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery: {text}"
        return self.model.embed_query(text)


def generate_evaluation_set(chunks: list) -> list[dict[str, str | int]]:
    eligible_chunks = [
        chunk for chunk in chunks if len(chunk.page_content.strip()) >= MIN_CHUNK_LENGTH
    ]
    if len(eligible_chunks) < EVAL_SAMPLE_COUNT:
        raise ValueError(
            f"Need at least {EVAL_SAMPLE_COUNT} eligible chunks; found {len(eligible_chunks)}"
        )

    sampled_chunks = random.Random(RANDOM_SEED).sample(
        eligible_chunks,
        EVAL_SAMPLE_COUNT,
    )
    load_dotenv(PROJECT_ROOT / ".env")
    prompt = PromptTemplate.from_template(
        """
다음 문서 청크의 내용으로 답할 수 있는 구체적인 한국어 질문을 하나 만드세요.

<content>
{content}
</content>

질문만 JSON 형식으로 응답하세요:
{{"question": "질문 내용"}}
"""
    )
    chain = prompt | ChatOpenAI(model="gpt-4.1-nano", temperature=0) | JsonOutputParser()
    evaluation_set = []

    for index, chunk in enumerate(sampled_chunks, start=1):
        result = chain.invoke({"content": chunk.page_content})
        question = result.get("question") if isinstance(result, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question generation returned invalid output for sample {index}")
        evaluation_set.append(
            {
                "question": question.strip(),
                "chunk_id": int(chunk.metadata["chunk_id"]),
            }
        )
        print(f"Generated evaluation question {index}/{EVAL_SAMPLE_COUNT}")

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SET_PATH.write_text(
        json.dumps(evaluation_set, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evaluation_set


def load_or_create_evaluation_set(chunks: list) -> list[dict[str, str | int]]:
    valid_chunk_ids = {int(chunk.metadata["chunk_id"]) for chunk in chunks}
    if EVAL_SET_PATH.is_file():
        evaluation_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
        if not isinstance(evaluation_set, list) or len(evaluation_set) != EVAL_SAMPLE_COUNT:
            raise ValueError(f"{EVAL_SET_PATH} must contain exactly {EVAL_SAMPLE_COUNT} items")
        for item in evaluation_set:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("question"), str)
                or not isinstance(item.get("chunk_id"), int)
                or item["chunk_id"] not in valid_chunk_ids
            ):
                raise ValueError(f"Invalid question or chunk_id in {EVAL_SET_PATH}")
        print(f"Reusing evaluation set: {EVAL_SET_PATH}")
        return evaluation_set

    print(f"Generating {EVAL_SAMPLE_COUNT} questions with gpt-4.1-nano")
    return generate_evaluation_set(chunks)


def evaluate_retrieval(
    vectorstore: FAISS,
    evaluation_set: list[dict[str, str | int]],
) -> dict[str, float]:
    hits = {k: 0 for k in (1, 3, 5)}
    reciprocal_rank_sum = 0.0

    for item in evaluation_set:
        results = vectorstore.similarity_search(item["question"], k=MAX_K)
        retrieved_ids = [document.metadata.get("chunk_id") for document in results]
        gold_chunk_id = item["chunk_id"]
        for k in hits:
            if gold_chunk_id in retrieved_ids[:k]:
                hits[k] += 1
        rank = next(
            (position for position, chunk_id in enumerate(retrieved_ids, start=1)
             if chunk_id == gold_chunk_id),
            None,
        )
        if rank is not None:
            reciprocal_rank_sum += 1 / rank

    count = len(evaluation_set)
    return {
        "hit_rate_at_1": hits[1] / count,
        "hit_rate_at_3": hits[3] / count,
        "hit_rate_at_5": hits[5] / count,
        "mrr": reciprocal_rank_sum / count,
    }


def print_results_table(results: list[dict[str, str | float | int]]) -> None:
    headers = (
        "model",
        "hit_rate@1",
        "hit_rate@3",
        "hit_rate@5",
        "mrr",
        "indexing_seconds",
        "embedding_dimension",
    )
    rows = []
    for result in results:
        rows.append(
            (
                str(result["model"]),
                f"{result['hit_rate_at_1']:.4f}",
                f"{result['hit_rate_at_3']:.4f}",
                f"{result['hit_rate_at_5']:.4f}",
                f"{result['mrr']:.4f}",
                f"{result['indexing_seconds']:.2f}",
                str(result["embedding_dimension"]),
            )
        )
    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def run_evaluation() -> list[dict[str, str | float | int]]:
    chunks, _ = load_and_split_documents()
    evaluation_set = load_or_create_evaluation_set(chunks)
    results = []

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name}")
        embeddings = RetrievalEmbeddings(model_name)
        vectorstore_path = vectorstore_path_for_model(model_name)
        start = time.perf_counter()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vectorstore_path))
        indexing_seconds = time.perf_counter() - start

        metrics = evaluate_retrieval(vectorstore, evaluation_set)
        results.append(
            {
                "model": model_name,
                **metrics,
                "indexing_seconds": indexing_seconds,
                "embedding_dimension": embeddings.dimension,
            }
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "model",
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "mrr",
        "indexing_seconds",
        "embedding_dimension",
    )
    with RESULTS_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved: {RESULTS_PATH}")
    print_results_table(results)
    return results


if __name__ == "__main__":
    run_evaluation()