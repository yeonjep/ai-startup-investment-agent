import unittest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from rag.retriever import hybrid_search
from rag.ingest import load_and_split_documents, fingerprint
from rag.eval_embeddings import evaluate_retrieval


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]
    def embed_query(self, text):
        return [float('NPU' in text), float('GPU' in text), 1.0]


class RetrievalTests(unittest.TestCase):
    def test_filter_applies_to_both_rankings(self):
        docs = [Document(page_content='NPU 성장률', metadata={'chunk_id': 'a', 'language': 'ko'}),
                Document(page_content='NPU GPU growth', metadata={'chunk_id': 'b', 'language': 'en'}),
                Document(page_content='GPU 시장', metadata={'chunk_id': 'c', 'language': 'ko'})]
        store = FAISS.from_documents(docs, TinyEmbeddings())
        results = hybrid_search(store, 'NPU', filters={'language': 'ko'})
        self.assertTrue(results)
        self.assertNotIn('b', [r['chunk_id'] for r in results])
        self.assertEqual(results[0]['chunk_id'], 'a')
        self.assertEqual(hybrid_search(store, 'NPU', filters={'language': 'missing'}), [])
        with self.assertRaises(ValueError):
            hybrid_search(store, '')

    def test_pdf_chunks_and_original_page_numbers(self):
        chunks, counts = load_and_split_documents()
        self.assertEqual(len(counts), 4)
        self.assertEqual(len({c.metadata['chunk_id'] for c in chunks}), len(chunks))
        research = [c for c in chunks if c.metadata['doc_id'] == 'kisdi_2024_research']
        self.assertGreaterEqual(min(c.metadata['page'] for c in research), 27)
        self.assertLessEqual(max(c.metadata['page'] for c in research), 81)
        self.assertNotEqual(fingerprint(chunks, 'model-a'), fingerprint(chunks, 'model-b'))

    def test_known_retrieval_metrics(self):
        class Store:
            def similarity_search(self, query, k):
                return [Document(page_content='', metadata={'chunk_id': c}) for c in ['a', 'b', 'c']]
        result = evaluate_retrieval(Store(), [{'question': 'q', 'chunk_id': 'b'}, {'question': 'q', 'chunk_id': 'missing'}])
        self.assertEqual(result['hit_rate_at_1'], 0)
        self.assertEqual(result['hit_rate_at_3'], 0.5)
        self.assertEqual(result['mrr'], 0.25)

class PersistenceTests(unittest.TestCase):
    def test_index_reuse_and_checksum(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from rag.ingest import ingest
        docs = [Document(page_content='NPU', metadata={'chunk_id': 'a'}),
                Document(page_content='GPU', metadata={'chunk_id': 'b'})]
        class CountingEmbeddings(TinyEmbeddings):
            document_calls = 0
            def embed_documents(self, texts):
                self.document_calls += 1
                return super().embed_documents(texts)
        embeddings = CountingEmbeddings()
        with tempfile.TemporaryDirectory() as tmp, \
             patch('rag.ingest.load_and_split_documents', return_value=(docs, [])), \
             patch('rag.ingest.vectorstore_path_for_model', return_value=Path(tmp)):
            ingest('test', embeddings=embeddings)
            store = ingest('test', embeddings=embeddings)
            self.assertEqual(embeddings.document_calls, 1)
            self.assertEqual(store.similarity_search('NPU', k=1)[0].metadata['chunk_id'], 'a')
            path = Path(tmp) / 'v2/index.faiss'
            path.write_bytes(path.read_bytes() + b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                ingest('test', embeddings=embeddings)
