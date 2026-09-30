import argparse
import re
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    MIN_MEANINGLESS_FRAGMENT_CHARS,
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

        document_chunks = splitter.split_documents(pages)
        document_chunks = [
            chunk
            for chunk in document_chunks
            if chunk.page_content.strip()
            and not (
                len(chunk.page_content.strip()) < MIN_MEANINGLESS_FRAGMENT_CHARS
                and not any(character.isalpha() for character in chunk.page_content)
            )
        ]
        for chunk in document_chunks:
            chunk.metadata["chunk_id"] = f"chunk_{len(chunks):06d}"
            chunks.append(chunk)
        chunk_counts.append((doc_meta["doc_id"], len(document_chunks)))

    return chunks, chunk_counts


def ingest(embedding_model: str = EMBEDDING_MODEL, rebuild: bool = False) -> FAISS:
    chunks, chunk_counts = load_and_split_documents()
    vectorstore_path = vectorstore_path_for_model(embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    index_file = vectorstore_path / "index.faiss"
    metadata_file = vectorstore_path / "index.pkl"
    if not rebuild and index_file.is_file() and metadata_file.is_file():
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        action = "Reused"
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vectorstore_path))
        action = "Created"

    print(f"Vector store {action}: {vectorstore_path}")
    for doc_id, count in chunk_counts:
        print(f"{doc_id}: {count} chunks")
    print(f"Total: {len(chunks)} chunks")
    if chunks:
        print(f"Sample chunk metadata: {chunks[0].metadata}")

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