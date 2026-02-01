from pinecone_text.sparse import BM25Encoder


class SparseEmbeddingManager:
    _instance = None
    _encoder = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparseEmbeddingManager, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        print("[Sparse] Initializing BM25 Encoder...")
        self._encoder = BM25Encoder().default()

    def get_sparse_vector(self, text: str):
        return self._encoder.encode_documents(text)

    def get_sparse_query(self, text: str):
        return self._encoder.encode_queries(text)
