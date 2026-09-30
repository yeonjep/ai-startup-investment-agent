"""Dense + BM25 weighted reciprocal-rank fusion (EnsembleRetriever와 같은 RRF 방식)."""
import os
import re
from functools import lru_cache
from rank_bm25 import BM25Okapi
from langchain_core.tools import tool
from agents.common import configure_runtime
from rag.config import EMBEDDING_MODEL


def tokenize(text):
    tokens = re.findall(r'[a-z0-9]+|[가-힣]+', text.lower())
    # 한국어 조사/합성어의 완전 일치 의존성을 줄이는 한글 2-gram.
    return tokens + [word[i:i+2] for word in tokens if re.fullmatch('[가-힣]{3,}', word) for i in range(len(word)-1)]


def matches(metadata, filters):
    for key, value in (filters or {}).items():
        actual = metadata.get(key)
        if isinstance(actual, list):
            if value not in actual:
                return False
        elif actual != value:
            return False
    return True


def hybrid_search(store, query, k=5, filters=None):
    if not query.strip() or k < 1:
        raise ValueError('Nonempty query and k >= 1 required')
    docs = [store.docstore.search(key) for key in store.index_to_docstore_id.values()]
    docs = [d for d in docs if matches(d.metadata, filters)]
    if not docs:
        return []
    n = min(max(k * 4, 20), len(docs))
    dense = store.similarity_search(query, k=n, filter=lambda m: matches(m, filters), fetch_k=store.index.ntotal)
    corpus = [tokenize(d.page_content) or ['__empty__'] for d in docs]
    bm25 = BM25Okapi(corpus)
    sparse_scores = bm25.get_scores(tokenize(query))
    sparse = [docs[i] for i in sorted(range(len(docs)), key=lambda i: (-sparse_scores[i], str(docs[i].metadata['chunk_id'])))[:n]
              if sparse_scores[i] > 0]
    fused, lookup = {}, {}
    for ranking in (dense, sparse):
        for rank, doc in enumerate(ranking, 1):
            key = doc.metadata['chunk_id']
            fused[key] = fused.get(key, 0) + 0.5 / (60 + rank)
            lookup[key] = doc
    return [{'content': lookup[key].page_content, **lookup[key].metadata, 'score': fused[key]}
            for key in sorted(fused, key=lambda key: (-fused[key], str(key)))[:k]]


@lru_cache(maxsize=2)
def _load_vectorstore(model):
    from rag.ingest import ingest
    return ingest(model)


def search(query, k=5, filters=None):
    configure_runtime()
    return hybrid_search(_load_vectorstore(os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL)), query, k, filters)


@tool('rag_search')
def rag_search(query: str, k: int = 5, filters: dict | None = None) -> list[dict]:
    """Search the local industry corpus using dense/BM25 fusion and metadata filters."""
    return search(query, k, filters)


search_rag_documents = rag_search
