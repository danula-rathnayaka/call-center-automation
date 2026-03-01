from fastapi import APIRouter, HTTPException

from api.schemas import KnowledgeBaseStatus, KnowledgeResetResponse
from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])


@router.get("/status", response_model=KnowledgeBaseStatus)
async def get_knowledge_status():
    try:
        client = PineconeClient()
        index = client._index
        index_name = client._index_name

        stats = index.describe_index_stats()

        return KnowledgeBaseStatus(
            index_name=index_name,
            total_vectors=stats.get("total_vector_count", 0),
            status="ready",
        )

    except Exception as e:
        logger.error(f"Knowledge status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge base status: {str(e)}")


@router.delete("/reset", response_model=KnowledgeResetResponse)
async def reset_knowledge_base():
    try:
        client = PineconeClient()
        client.delete_all()

        return KnowledgeResetResponse(
            status="completed",
            message="Knowledge base has been successfully wiped. All vectors deleted."
        )

    except Exception as e:
        logger.error(f"Knowledge base reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset knowledge base: {str(e)}")
