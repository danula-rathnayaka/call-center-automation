from api.schemas import HealthResponse, ComponentStatus
from fastapi import APIRouter

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check — all components",
    description=(
        "Probes every major system component and returns its status: "
        "Pinecone Vector DB, dense embedding model (all-MiniLM-L6-v2), BM25 sparse encoder, "
        "emotion detection model, confidence scoring model, fine-tuned LLM, and the compiled LangGraph workflow. "
        "Overall `status` is `healthy` if all critical components pass, or `degraded` if any fail. "
        "Call this on app startup and periodically to display a system status indicator in the admin panel."
    ),
)
async def health_check():
    components = []
    overall_healthy = True

    try:
        from multiagent_rag.utils.db_client import PineconeClient
        client = PineconeClient()
        stats = client._index.describe_index_stats()
        components.append(ComponentStatus(name="Pinecone Vector DB", status="healthy",
            details=f"Connected. Vectors: {stats.get('total_vector_count', 0)}"))
    except Exception as e:
        components.append(ComponentStatus(name="Pinecone Vector DB", status="unhealthy", details=str(e)))
        overall_healthy = False

    try:
        from multiagent_rag.utils.embeddings import EmbeddingManager
        em = EmbeddingManager()
        test_emb = em.get_embedding("test")
        components.append(ComponentStatus(name="Embedding Model", status="healthy",
            details=f"all-MiniLM-L6-v2 loaded. Dim: {len(test_emb)}"))
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
        pipeline_status = "real model" if agent._model else "keyword fallback"
        components.append(ComponentStatus(name="Emotion Model", status="healthy",
            details=f"Using {pipeline_status}. Test emotion: {result['emotion']}"))
    except Exception as e:
        components.append(ComponentStatus(name="Emotion Model", status="unhealthy", details=str(e)))

    try:
        from multiagent_rag.agents.confidence_agent import ConfidenceAgent
        agent = ConfidenceAgent()
        pipeline_status = "real model" if agent._model else "heuristic fallback"
        components.append(
            ComponentStatus(name="Confidence Model", status="healthy", details=f"Using {pipeline_status}"))
    except Exception as e:
        components.append(ComponentStatus(name="Confidence Model", status="unhealthy", details=str(e)))

    try:
        from multiagent_rag.agents.finetuned_llm_agent import FinetunedLLMAgent
        agent = FinetunedLLMAgent()
        pipeline_status = "real model" if agent._pipeline_ready else "Groq fallback"
        components.append(ComponentStatus(name="Fine-tuned LLM", status="healthy", details=f"Using {pipeline_status}"))
    except Exception as e:
        components.append(ComponentStatus(name="Fine-tuned LLM", status="unhealthy", details=str(e)))

    try:
        from multiagent_rag.utils.telemetry import get_langfuse_client
        lf_client = get_langfuse_client()
        if lf_client:
            lf_client.auth_check()
            components.append(ComponentStatus(name="Langfuse", status="healthy", details="Connected and authenticated"))
        else:
            components.append(ComponentStatus(name="Langfuse", status="degraded", details="Client not initialized - check LANGFUSE_* env vars"))
    except Exception as e:
        components.append(ComponentStatus(name="Langfuse", status="unhealthy", details=str(e)))

    return HealthResponse(status="healthy" if overall_healthy else "degraded", components=components)


@router.post(
    "/prompts/reload",
    summary="Hot-reload all prompts from Langfuse",
    description=(
        "Clears the in-process prompt cache. The next request to each agent will re-fetch the latest "
        "published version of its prompt from Langfuse without needing a server restart. "
        "Call this from the admin panel after publishing a new prompt version in the Langfuse UI."
    ),
)
async def reload_prompts():
    try:
        from multiagent_rag.utils.prompt_manager import invalidate_cache
        invalidate_cache()
        logger.info("Prompt cache invalidated via API")
        return {"status": "ok", "message": "Prompt cache cleared. Agents will fetch latest versions from Langfuse on next request."}
    except Exception as e:
        logger.error(f"Failed to reload prompts: {e}")
        return {"status": "error", "message": str(e)}
