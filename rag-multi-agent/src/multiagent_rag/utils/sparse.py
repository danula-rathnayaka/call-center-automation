import os
from typing import List

from pinecone_text.sparse import BM25Encoder

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "tools", "BM25", "bm25_params.json"
)


class SparseEmbeddingManager:
    _instance = None
    _encoder = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparseEmbeddingManager, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        abs_path = os.path.abspath(_DEFAULT_JSON_PATH)
        if os.path.exists(abs_path):
            logger.info(f"Loading BM25 parameters from: {abs_path}")
            self._encoder = BM25Encoder().load(abs_path)
        else:
            logger.warning("BM25 params not found — using default MS MARCO weights. Run refit_bm25.py after ingestion.")
            self._encoder = BM25Encoder().default()
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            self._encoder.dump(abs_path)
            logger.info(f"Saved default BM25 params to {abs_path}")

    def fit_on_corpus(self, texts: List[str], save_path: str = None):
        if not texts:
            logger.warning("fit_on_corpus called with empty text list — skipping")
            return

        logger.info(f"Fitting BM25 on {len(texts)} documents...")
        self._encoder = BM25Encoder()
        self._encoder.fit(texts)

        path = save_path or os.path.abspath(_DEFAULT_JSON_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._encoder.dump(path)
        logger.info(f"BM25 refitted and saved to {path}")

    def reload(self, path: str = None):
        load_path = path or os.path.abspath(_DEFAULT_JSON_PATH)
        if not os.path.exists(load_path):
            logger.error(f"Cannot reload — file not found: {load_path}")
            return
        self._encoder = BM25Encoder().load(load_path)
        logger.info(f"BM25 encoder reloaded from {load_path}")

    def get_sparse_vector(self, text: str):
        return self._encoder.encode_documents(text)

    def get_sparse_query(self, text: str):
        return self._encoder.encode_queries(text)
