from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

QWEN_QUERY_INSTRUCTION = 'Given a question, retrieve relevant passages that answer the question'


class RetrievalEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(model_name=model_name,
                                          encode_kwargs={'normalize_embeddings': True})

    @property
    def dimension(self):
        return len(self.embed_query('embedding dimension probe'))

    def embed_documents(self, texts):
        if self.model_name == 'intfloat/multilingual-e5-large':
            texts = [f'passage: {text}' for text in texts]
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        if self.model_name == 'intfloat/multilingual-e5-large':
            text = f'query: {text}'
        elif self.model_name == 'Qwen/Qwen3-Embedding-0.6B':
            text = f'Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery: {text}'
        return self.model.embed_query(text)
