import os

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
        json_path = "tools/BM25/bm25_params.json"

        if os.path.exists(json_path):
            print(f"[Sparse] Loading BM25 from local file ({json_path})...")
            self._encoder = BM25Encoder().load(json_path)

        else:
            print("[Sparse] Local params not found. Downloading default...")
            self._encoder = BM25Encoder().default()

            self._encoder.dump(json_path)
            print(f"[Sparse] Saved params to {json_path} for future speedups.")

    def get_sparse_vector(self, text: str):
        return self._encoder.encode_documents(text)

    def get_sparse_query(self, text: str):
        return self._encoder.encode_queries(text)
