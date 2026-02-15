from multiagent_rag.utils.db_client import PineconeClient


from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.db = PineconeClient()

    def retrieve(self, query: str, k: int = 5):
        results = self.db.search(query, k=k)

        if not results:
            logger.warning(f"No relevant documents found for query: {query}")
            return []

        clean_docs = []
        for doc in results:
            clean_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        logger.info(f"Retrieved {len(clean_docs)} relevant document chunks.")
        return clean_docs

    def format_docs(self, docs: list) -> str:
        formatted_text = ""
        for i, doc in enumerate(docs):
            source = doc['metadata'].get('source', 'Unknown')
            formatted_text += f"\n--- Source: {source} ---\n{doc['content']}\n"

        return formatted_text
