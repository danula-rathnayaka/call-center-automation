import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from multiagent_rag.utils.embeddings import EmbeddingManager
from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.sparse import SparseEmbeddingManager

load_dotenv()
logger = get_logger(__name__)

_ALPHA_BY_INTENT = {
    "technical": 0.3,
    "customer_service": 0.4,
    "casual": 0.8,
    "escalation": 0.7,
    "blocked": 0.5,
    "unknown": 0.5,
}
_DEFAULT_ALPHA = 0.5
_MIN_RELEVANCE_SCORE = 0.15


class PineconeClient:
    _instance = None
    _pc_client = None
    _index_name = "call-center-automation"
    _index = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self._pc_client = Pinecone(api_key=api_key)
        existing_indexes = [i.name for i in self._pc_client.list_indexes()]

        if self._index_name not in existing_indexes:
            logger.info(f"Creating hybrid index '{self._index_name}'...")
            self._pc_client.create_index(
                name=self._index_name,
                dimension=384,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self._pc_client.describe_index(self._index_name).status["ready"]:
                time.sleep(1)

        logger.info("Connection to Pinecone successful")
        self._index = self._pc_client.Index(self._index_name)
        self._dense_manager = EmbeddingManager()
        self._sparse_manager = SparseEmbeddingManager()

    def insert_documents(self, documents: list) -> bool:
        if not documents:
            return False
        try:
            texts = [doc.page_content for doc in documents]
            dense_vectors = self._dense_manager.get_embeddings(texts)
            vectors_to_upsert = []
            for i, doc in enumerate(documents):
                sparse_vector = self._sparse_manager.get_sparse_vector(doc.page_content)
                unique_id = f"doc_{uuid.uuid4()}"
                vectors_to_upsert.append({
                    "id": unique_id,
                    "values": dense_vectors[i],
                    "sparse_values": sparse_vector,
                    "metadata": doc.metadata | {"text": doc.page_content},
                })
            self._index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Upserted {len(vectors_to_upsert)} hybrid vectors to Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False

    def search(self, query: str, k: int = 5, intent: str = "unknown") -> list:
        try:
            alpha = _ALPHA_BY_INTENT.get(intent, _DEFAULT_ALPHA)
            dense_query = self._dense_manager.get_embedding(query)
            sparse_query = self._sparse_manager.get_sparse_query(query)

            scaled_dense = [v * alpha for v in dense_query]
            scaled_sparse = {
                "indices": sparse_query["indices"],
                "values": [v * (1 - alpha) for v in sparse_query["values"]],
            }

            logger.info(f"Hybrid search | intent={intent} alpha={alpha} | query: {query[:60]}")

            results = self._index.query(
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=k,
                include_metadata=True,
            )

            docs = []
            for match in results["matches"]:
                score = match.get("score", 0.0)
                if score < _MIN_RELEVANCE_SCORE:
                    logger.info(f"Filtered low-relevance chunk (score={score:.3f}): {match.get('id', '')}")
                    continue
                text_content = match["metadata"].pop("text", "")
                doc = Document(page_content=text_content, metadata=match["metadata"])
                doc.metadata["_score"] = round(score, 4)
                docs.append(doc)

            logger.info(f"Returned {len(docs)} chunks above score threshold {_MIN_RELEVANCE_SCORE}")
            return docs

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def fetch_all_texts(self, batch_size: int = 100) -> list:
        texts = []
        try:
            stats = self._index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            dummy_vec = [0.0] * 384

            if total == 0:
                logger.warning("Index is empty — no texts to fetch for BM25 fitting")
                return []

            results = self._index.query(
                vector=dummy_vec,
                top_k=min(total, 10000),
                include_metadata=True,
            )

            for match in results.get("matches", []):
                text = match.get("metadata", {}).get("text", "")
                if text.strip():
                    texts.append(text)

            logger.info(f"Fetched {len(texts)} texts from Pinecone for BM25 fitting")
            return texts
        except Exception as e:
            logger.error(f"Failed to fetch texts from Pinecone: {e}")
            return []

    def check_duplicate(self, document_hash: str) -> bool:
        try:
            dummy_vector = [0.0] * 384
            results = self._index.query(
                vector=dummy_vector,
                top_k=1,
                include_metadata=True,
                filter={"document_hash": {"$eq": document_hash}},
            )
            return bool(results and results.get("matches"))
        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return False

    def list_by_type(self, doc_type: str) -> list:
        seen_sources: set = set()
        entries: list = []
        try:
            stats = self._index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            if total == 0:
                return []

            dummy_vec = [0.0] * 384
            results = self._index.query(
                vector=dummy_vec,
                top_k=min(total, 10000),
                include_metadata=True,
                filter={"type": {"$eq": doc_type}},
            )

            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                source = meta.get("source", "")
                if not source or source in seen_sources:
                    continue
                seen_sources.add(source)
                entries.append({
                    "source": source,
                    "title": meta.get("title", ""),
                    "type": meta.get("type", doc_type),
                    "document_hash": meta.get("document_hash", ""),
                })

            logger.info(f"Listed {len(entries)} unique sources of type='{doc_type}'")
            return entries
        except Exception as e:
            logger.error(f"list_by_type failed for type='{doc_type}': {e}")
            return []

    def delete_by_source(self, source: str) -> int:
        try:
            stats = self._index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            if total == 0:
                return 0

            dummy_vec = [0.0] * 384
            results = self._index.query(
                vector=dummy_vec,
                top_k=min(total, 10000),
                include_metadata=False,
                filter={"source": {"$eq": source}},
            )
            ids_to_delete = [m["id"] for m in results.get("matches", [])]
            if not ids_to_delete:
                return 0

            batch_size = 1000
            for i in range(0, len(ids_to_delete), batch_size):
                self._index.delete(ids=ids_to_delete[i: i + batch_size])

            logger.info(f"Deleted {len(ids_to_delete)} vectors for source='{source}'")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"delete_by_source failed for source='{source}': {e}")
            raise

    def delete_all(self):
        try:
            self._index.delete(delete_all=True)
            logger.info("Wiped all data from Pinecone index")
        except Exception as e:
            logger.error(f"Failed to wipe Pinecone index: {e}")
