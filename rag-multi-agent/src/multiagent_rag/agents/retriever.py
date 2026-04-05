from langfuse import observe
from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self):
        self.db = PineconeClient()

    @observe()
    def retrieve(self, query: str, k: int = 5, intent: str = "unknown") -> list:
        results = self.db.search(query, k=k, intent=intent)

        if not results:
            logger.warning(f"No relevant documents found for query: {query}")
            return []

        clean_docs = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]

        logger.info(f"Retrieved {len(clean_docs)} chunks (intent={intent})")
        return clean_docs

    def format_docs(self, docs: list) -> str:
        formatted_text = ""
        for doc in docs:
            source = doc["metadata"].get("source", "Unknown")
            formatted_text += f"\n--- Source: {source} ---\n{doc['content']}\n"
        return formatted_text
