from fastapi import APIRouter, HTTPException

from multiagent_rag.utils.human_handoff_store import (get_pending_handoffs, get_handoff_detail, mark_handled,
                                                      get_handoff_stats, )
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/handoff", tags=["Human Handoff"])


@router.get("/queue")
async def get_handoff_queue():
    try:
        return {"items": get_pending_handoffs(), "stats": get_handoff_stats()}
    except Exception as e:
        logger.error(f"Failed to fetch handoff queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/{handoff_id}")
async def get_handoff(handoff_id: int):
    try:
        item = get_handoff_detail(handoff_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Handoff {handoff_id} not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch handoff {handoff_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/queue/{handoff_id}/resolve")
async def resolve_handoff(handoff_id: int):
    try:
        item = get_handoff_detail(handoff_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Handoff {handoff_id} not found")
        if item["status"] != "pending":
            raise HTTPException(status_code=400, detail=f"Handoff {handoff_id} is already {item['status']}")
        mark_handled(handoff_id)
        return {"status": "resolved", "handoff_id": handoff_id, "session_id": item["session_id"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve handoff {handoff_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    try:
        return get_handoff_stats()
    except Exception as e:
        logger.error(f"Failed to fetch handoff stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
