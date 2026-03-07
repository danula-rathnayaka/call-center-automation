from sentence_transformers import CrossEncoder

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        logger.info("Initializing Cross-Encoder reranker model")
        self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: list, top_k: int = 3) -> list:
        if not docs:
            return []

        if len(docs) <= top_k:
            return docs

        pairs = [(query, doc["content"]) for doc in docs]
        scores = self._model.predict(pairs)

        scored_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        reranked = [doc for doc, score in scored_docs[:top_k]]
        logger.info(
            f"Reranked {len(docs)} documents to top {top_k}. "
            f"Score range: {scored_docs[-1][1]:.4f} - {scored_docs[0][1]:.4f}"
        )
        return reranked
