from datetime import datetime, timezone

from langchain_experimental.text_splitter import SemanticChunker
from multiagent_rag.utils.embeddings import EmbeddingManager
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class Chunker:
    _instance = None
    _splitter = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Chunker, cls).__new__(cls)
            cls._instance._initialize_splitter()
        return cls._instance

    def _initialize_splitter(self):
        logger.info("Initializing SemanticChunker with All-MiniLM embeddings")
        embedding_manager = EmbeddingManager()
        self._splitter = SemanticChunker(
            embedding_manager.get_model(),
            breakpoint_threshold_type="percentile"
        )

    def split_text(self, text: str, metadata: dict) -> list:
        if not text:
            return []

        logger.info("Splitting text using semantic boundaries")
        docs = self._splitter.create_documents([text])

        clean_chunks = []
        total_chunks = len(docs)
        ingested_at = datetime.now(timezone.utc).isoformat()

        for i, doc in enumerate(docs):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_index"] = i
            doc_metadata["total_chunks"] = total_chunks
            doc_metadata["ingested_at"] = ingested_at
            clean_chunks.append({
                "content": doc.page_content,
                "metadata": doc_metadata
            })

        logger.info(f"Generated {len(clean_chunks)} semantic chunks")
        return clean_chunks
