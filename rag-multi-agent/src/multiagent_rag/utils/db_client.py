import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from multiagent_rag.utils.embeddings import EmbeddingManager
from multiagent_rag.utils.sparse import SparseEmbeddingManager

from multiagent_rag.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

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
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self._pc_client.describe_index(self._index_name).status['ready']:
                time.sleep(1)

        logger.info("Connection to Pinecone successful")
        self._index = self._pc_client.Index(self._index_name)

        self._dense_manager = EmbeddingManager()
        self._sparse_manager = SparseEmbeddingManager()

    def insert_documents(self, documents: list):
        if not documents:
            return False

        try:
            logger.info(f"Processing {len(documents)} documents for hybrid search insertion")

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
                    "metadata": doc.metadata | {"text": doc.page_content}
                })

            self._index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} hybrid vectors to Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to insert documents into Pinecone: {str(e)}")
            return False

    def search(self, query: str, k: int = 5):
        try:
            dense_query = self._dense_manager.get_embedding(query)
            sparse_query = self._sparse_manager.get_sparse_query(query)

            results = self._index.query(
                vector=dense_query,
                sparse_vector=sparse_query,
                top_k=k,
                include_metadata=True
            )

            docs = []
            for match in results['matches']:
                text_content = match['metadata'].pop('text', '')
                docs.append(Document(page_content=text_content, metadata=match['metadata']))

            return docs

        except Exception as e:
            logger.error(f"Hybrid search operation failed: {str(e)}")
            return []

    def delete_all(self):
        try:
            self._index.delete(delete_all=True)
            logger.info("Successfully wiped all data from the Pinecone index")
        except Exception as e:
            logger.error(f"Failed to wipe Pinecone index data: {str(e)}")


if __name__ == '__main__':
    client = PineconeClient()

    sample_text = "Error 503: The Huawei B310 router service is unavailable."
    sample_meta = {"source": "hybrid_test"}

    doc = Document(page_content=sample_text, metadata=sample_meta)

    print("\n--- Testing Hybrid Insertion ---")
    client.insert_documents([doc])
    time.sleep(2)

    print("\n--- Testing Hybrid Search ---")
    query = "Error 503"
    results = client.search(query, k=1)

    if results:
        print(f"Search Successful!")
        print(f"Content: {results[0].page_content}")
    else:
        print("Search returned no results.")
