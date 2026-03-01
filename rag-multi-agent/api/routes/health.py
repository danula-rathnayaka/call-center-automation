from fastapi import APIRouter

from api.schemas import HealthResponse, ComponentStatus
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    components = []
    overall_healthy = True

    try:
        from multiagent_rag.utils.db_client import PineconeClient
        client = PineconeClient()
        stats = client._index.describe_index_stats()
        components.append(ComponentStatus(
            name="Pinecone Vector DB",
            status="healthy",
            details=f"Connected. Vectors: {stats.get('total_vector_count', 0)}"
        ))
    except Exception as e:
        components.append(ComponentStatus(name="Pinecone Vector DB", status="unhealthy", details=str(e)))
        overall_healthy = False

    try:
        from multiagent_rag.utils.embeddings import EmbeddingManager
        em = EmbeddingManager()
        test_emb = em.get_embedding("test")
        components.append(ComponentStatus(
            name="Embedding Model",
            status="healthy",
            details=f"all-MiniLM-L6-v2 loaded. Dim: {len(test_emb)}"
        ))
    except Exception as e:
        components.append(ComponentStatus(name="Embedding Model", status="unhealthy", details=str(e)))
        overall_healthy = False

    try:
        from multiagent_rag.utils.sparse import SparseEmbeddingManager
        _ = SparseEmbeddingManager()
        components.append(ComponentStatus(name="BM25 Sparse Encoder", status="healthy", details="BM25 encoder loaded"))
    except Exception as e:
        components.append(ComponentStatus(name="BM25 Sparse Encoder", status="unhealthy", details=str(e)))
        overall_healthy = False

    try:
        from multiagent_rag.agents.emotion_agent import EmotionAgent
        agent = EmotionAgent()
        result = agent.detect_from_text("test")
        pipeline_status = "real model" if agent._pipeline_available else "fallback"
        components.append(ComponentStatus(
            name="Emotion Model",
            status="healthy",
            details=f"Using {pipeline_status}. Test emotion: {result['emotion']}"
        ))
    except Exception as e:
        components.append(ComponentStatus(name="Emotion Model", status="unhealthy", details=str(e)))

    try:
        from multiagent_rag.agents.confidence_agent import ConfidenceAgent
        agent = ConfidenceAgent()
        pipeline_status = "real model" if agent._pipeline_available else "fallback"
        components.append(ComponentStatus(name="Confidence Model", status="healthy", details=f"Using {pipeline_status}"))
    except Exception as e:
        components.append(ComponentStatus(name="Confidence Model", status="unhealthy", details=str(e)))

    try:
        from multiagent_rag.agents.finetuned_llm_agent import FinetunedLLMAgent
        agent = FinetunedLLMAgent()
        pipeline_status = "real model" if agent._pipeline_available else "Groq fallback"
        components.append(ComponentStatus(name="Fine-tuned LLM", status="healthy", details=f"Using {pipeline_status}"))
    except Exception as e:
        components.append(ComponentStatus(name="Fine-tuned LLM", status="unhealthy", details=str(e)))

    try:
        from multiagent_rag.graph.rag_workflow import rag_app
        components.append(ComponentStatus(name="RAG Workflow", status="healthy", details="LangGraph workflow compiled"))
    except Exception as e:
        components.append(ComponentStatus(name="RAG Workflow", status="unhealthy", details=str(e)))
        overall_healthy = False

    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        components=components,
    )
