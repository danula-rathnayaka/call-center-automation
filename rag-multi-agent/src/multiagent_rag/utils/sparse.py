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
        current_file_path = os.path.abspath(__file__)

        utils_dir = os.path.dirname(current_file_path)
        multiagent_rag_dir = os.path.dirname(utils_dir)
        src_dir = os.path.dirname(multiagent_rag_dir)
        project_root = os.path.dirname(src_dir)

        abs_json_path = os.path.join(project_root, "tools", "BM25", "bm25_params.json")

        if os.path.exists(abs_json_path):
            logger.info(f"Loading BM25 parameters from local file: {abs_json_path}")
            self._encoder = BM25Encoder().load(abs_json_path)
        else:
            logger.info("Local BM25 parameters not found. Downloading default...")
            self._encoder = BM25Encoder().default()

            directory = os.path.dirname(abs_json_path)
            os.makedirs(directory, exist_ok=True)

            self._encoder.dump(abs_json_path)
            logger.info(f"Saved default BM25 parameters to {abs_json_path}")

    def get_sparse_vector(self, text: str):
        return self._encoder.encode_documents(text)

    def get_sparse_query(self, text: str):
        return self._encoder.encode_queries(text)
