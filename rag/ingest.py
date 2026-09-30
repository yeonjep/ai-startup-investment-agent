"""안전한 JSON 메타데이터+FAISS 인덱스. 기존 평가용 pickle 인덱스는 보존한다."""
import argparse
import hashlib
import json
import re
import faiss
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_DIR, EMBEDDING_MODEL, vectorstore_path_for_model
from rag.doc_meta import DOC_META, PDF_PAGE_STARTS
from rag.embeddings import RetrievalEmbeddings


INDEX_VERSION = 2


def load_and_split_documents():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks, counts, total_pages = [], [], 0
    for filename, metadata in DOC_META.items():
        path = DOCS_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f'{path}: run python -m rag.prepare_docs first')
        pages = []
        with pymupdf.open(path) as pdf:
            total_pages += len(pdf)
            for i, page in enumerate(pdf):
                text = re.sub(r'[ \t]+', ' ', page.get_text())
                text = re.sub(r'\n{3,}', '\n\n', text).strip()
                if len(text) < 20:
                    continue
                pages.append(Document(page_content=text, metadata={**metadata,
                    'page': i + PDF_PAGE_STARTS.get(filename, 1)}))
        doc_chunks = splitter.split_documents(pages)
        for local_id, chunk in enumerate(doc_chunks):
            chunk.metadata['chunk_id'] = f"{metadata['doc_id']}:{chunk.metadata['page']}:{local_id}"
            chunks.append(chunk)
        counts.append((metadata['doc_id'], len(doc_chunks)))
    if total_pages > 200:
        raise ValueError(f'RAG page limit exceeded: {total_pages} > 200')
    if not chunks:
        raise ValueError('No extractable text found; OCR is required')
    return chunks, counts


def fingerprint(chunks, model):
    payload = {'version': INDEX_VERSION, 'model': model, 'size': CHUNK_SIZE, 'overlap': CHUNK_OVERLAP,
               'chunks': [{'text': c.page_content, 'metadata': c.metadata} for c in chunks]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ingest(embedding_model=None, force=False, embeddings=None):
    from agents.common import configure_runtime
    import os
    configure_runtime()
    model = embedding_model or os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL)
    chunks, counts = load_and_split_documents()
    folder = vectorstore_path_for_model(model) / 'v2'
    manifest_path = folder / 'manifest.json'
    digest = fingerprint(chunks, model)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    embeddings = embeddings or RetrievalEmbeddings(model)
    if not force and manifest.get('fingerprint') == digest and (folder / 'index.faiss').is_file():
        raw = (folder / 'index.faiss').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest.get('index_sha256'):
            raise ValueError('Index checksum mismatch; rebuild with --force')
        index = faiss.read_index(str(folder / 'index.faiss'))
        ids = [str(i) for i in range(len(chunks))]
        store = FAISS(embeddings, index, InMemoryDocstore(dict(zip(ids, chunks))), dict(enumerate(ids)))
        action = 'Reused'
    else:
        store = FAISS.from_documents(chunks, embeddings)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(store.index, str(folder / 'index.faiss'))
        manifest_path.write_text(json.dumps({'fingerprint': digest, 'model': model, 'version': INDEX_VERSION,
            'index_sha256': hashlib.sha256((folder / 'index.faiss').read_bytes()).hexdigest(),
            'chunks': [{'text': c.page_content, 'metadata': c.metadata} for c in chunks]}, ensure_ascii=False, indent=2))
        action = 'Created'
    print(f'{action} {folder}: {len(chunks)} chunks')
    return store


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-model')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    ingest(args.embedding_model, args.force)
