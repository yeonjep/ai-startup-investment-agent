import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLEANING_VERSION,
    BM25_INDEX_FILENAME,
    BM25_TOP_K,
    BM25_RETRIEVAL_WEIGHT,
    DOCS_DIR,
    DENSE_RETRIEVAL_WEIGHT,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    ENSEMBLE_RRF_C,
    HYBRID_TOP_K,
    MIN_MEANINGLESS_FRAGMENT_CHARS,
    QWEN_QUERY_INSTRUCTION,
    vectorstore_path_for_model,
)
from rag.doc_meta import DOC_META, PDF_PAGE_STARTS


def clean_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)

    cleaned_lines = []
    for line in text.split("\n"):
        fragment = line.strip()
        if (
            fragment
            and len(fragment) < MIN_MEANINGLESS_FRAGMENT_CHARS
            and not any(character.isalpha() for character in fragment)
        ):
            continue
        cleaned_lines.append(fragment)

    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_index_manifest(embedding_model: str) -> dict:
    documents = []
    for filename, metadata in DOC_META.items():
        pdf_path = DOCS_DIR / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"RAG PDF not found: {pdf_path}")
        documents.append(
            {
                "filename": filename,
                "sha256": _sha256(pdf_path.read_bytes()),
                "metadata": metadata,
                "original_page_start": PDF_PAGE_STARTS.get(filename, 1),
            }
        )
    return {
        "manifest_version": 2,
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "cleaning_version": CLEANING_VERSION,
        "min_meaningless_fragment_chars": MIN_MEANINGLESS_FRAGMENT_CHARS,
        "query_instruction": (
            QWEN_QUERY_INSTRUCTION
            if embedding_model == "Qwen/Qwen3-Embedding-0.6B"
            else None
        ),
        "bm25_top_k": BM25_TOP_K,
        "dense_top_k": DENSE_TOP_K,
        "hybrid_top_k": HYBRID_TOP_K,
        "dense_weight": DENSE_RETRIEVAL_WEIGHT,
        "bm25_weight": BM25_RETRIEVAL_WEIGHT,
        "rrf_constant": ENSEMBLE_RRF_C,
        "documents": documents,
    }


def load_and_split_documents() -> tuple[list, list[tuple[str, int]]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    chunk_counts = []

    for filename, doc_meta in DOC_META.items():
        pdf_path = DOCS_DIR / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"RAG PDF not found: {pdf_path}")

        page_start = PDF_PAGE_STARTS.get(filename, 1)
        pages = PyMuPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.page_content = clean_page_text(page.page_content)
            page_index = page.metadata.get("page")
            if not isinstance(page_index, int):
                raise ValueError(f"PDF page metadata is missing for {pdf_path}")
            page.metadata["page"] = page_index + page_start
            page.metadata.update(
                {
                    **doc_meta,
                    "mentioned_companies": list(doc_meta["mentioned_companies"]),
                }
            )

        for page in pages:
            page_chunks = splitter.split_documents([page])
            stable_page = page.metadata["page"]
            for page_chunk_index, chunk in enumerate(page_chunks):
                content = chunk.page_content.strip()
                if not content or (
                    len(content) < MIN_MEANINGLESS_FRAGMENT_CHARS
                    and not any(character.isalpha() for character in content)
                ):
                    continue
                stable_payload = (
                    f"{doc_meta['doc_id']}\0{stable_page}\0{page_chunk_index}\0{content}"
                ).encode("utf-8")
                chunk.metadata["chunk_id"] = hashlib.sha256(stable_payload).hexdigest()
                chunks.append(chunk)
        chunk_counts.append((doc_meta["doc_id"], len(pages)))

    return chunks, chunk_counts


def ingest(embedding_model: str = EMBEDDING_MODEL, rebuild: bool = False) -> FAISS:
    expected_manifest = build_index_manifest(embedding_model)
    vectorstore_path = vectorstore_path_for_model(embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    index_file = vectorstore_path / "index.faiss"
    metadata_file = vectorstore_path / "index.pkl"
    manifest_file = vectorstore_path / "index_manifest.json"
    bm25_file = vectorstore_path / BM25_INDEX_FILENAME
    manifest_matches = False
    if manifest_file.is_file():
        existing_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest_matches = existing_manifest == expected_manifest

    can_reuse = (
        not rebuild
        and manifest_matches
        and index_file.is_file()
        and metadata_file.is_file()
    )
    if can_reuse:
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        chunks, page_counts = load_and_split_documents()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vectorstore_path))

    if not can_reuse or not bm25_file.is_file():
        if can_reuse:
            chunks, page_counts = load_and_split_documents()
        bm25_retriever = BM25Retriever.from_documents(chunks, k=BM25_TOP_K)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        with bm25_file.open("wb") as bm25_file_handle:
            pickle.dump(bm25_retriever, bm25_file_handle)
        manifest_file.write_text(
            json.dumps(expected_manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        for doc_id, page_count in page_counts:
            print(f"{doc_id}: {page_count} pages")
        print(f"Total chunks: {len(chunks)}")

    return vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Load RAG PDFs and build a FAISS index.")
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"Hugging Face model ID (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recreate the selected model's FAISS index from cleaned documents.",
    )
    args = parser.parse_args()
    ingest(args.embedding_model, rebuild=args.rebuild)


if __name__ == "__main__":
    main()