import os
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from multiagent_rag.utils.embeddings import EmbeddingManager

load_dotenv()


class PineconeClient:
    _instance = None
    _vector_store = None
    _pc_client = None
    _index_name = "call-center-automation"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self._pc_client = Pinecone()

        if not self._pc_client.has_index(self._index_name):
            self._pc_client.create_index(
                name=self._index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self._pc_client.describe_index(self._index_name).status['ready']:
                time.sleep(1)

        print("Connected to PINECONE")

        embedding_model = EmbeddingManager().get_model()
        self._vector_store = PineconeVectorStore(
            index_name=self._index_name,
            embedding=embedding_model,
            pinecone_api_key=api_key
        )

    def insert_documents(self, documents: list):
        if not documents:
            return False

        try:
            self._vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"[DB] Error inserting: {e}")
            return False

    def search(self, query: str, k: int = 5, query_filter: dict = None):
        try:
            results = self._vector_store.similarity_search(
                query=query,
                k=k,
                filter=query_filter
            )
            return results
        except Exception as e:
            print(f"[DB] Search failed: {e}")
            return []

    def delete_all(self):
        self._vector_store.delete(delete_all=True)


if __name__ == '__main__':
    client = PineconeClient()

    sample_text = "The Huawei B310 router has a red light error when the SIM card is missing."
    sample_meta = {"source": "test_script", "topic": "troubleshooting"}

    doc = Document(page_content=sample_text, metadata=sample_meta)

    print("\n--- Testing Insertion ---")
    if client.insert_documents([doc]):
        print("Insert command sent.")
        print("Waiting 5s for indexing...")
        time.sleep(5)

    print("\n--- Testing Search ---")
    query = "red light router"
    results = client.search(query, k=1)

    if results:
        print(f"Search Successful! Found {len(results)} match.")
        print(f"   Context: {results[0].page_content}")
        print(f"   Metadata: {results[0].metadata}")
    else:
        print("Search returned no results.")

    print("\n--- Testing Cleanup ---")
    client.delete_all()
    print("Database wiped.")
