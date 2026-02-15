import os

from pinecone_text.sparse import BM25Encoder


from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

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
            logger.info(f"Loading BM25 encoder parameters from local file: {json_path}")
            self._encoder = BM25Encoder().load(json_path)

        else:
            logger.info("Local BM25 parameters not found. Downloading default encoder parameters.")
            self._encoder = BM25Encoder().default()

            self._encoder.dump(json_path)
            logger.info(f"Saved default BM25 parameters to {json_path} for future initialization speed")

    def get_sparse_vector(self, text: str):
        return self._encoder.encode_documents(text)

    def get_sparse_query(self, text: str):
        return self._encoder.encode_queries(text)
