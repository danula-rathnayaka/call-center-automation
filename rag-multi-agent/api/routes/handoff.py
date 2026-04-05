from fastapi import APIRouter, HTTPException

from api.schemas import (
    HandoffQueueResponse,
    HandoffHistoryResponse,
    HandoffDashboardResponse,
    HandoffActionResponse,
    HandoffQueueItem,
)
from multiagent_rag.utils.human_handoff_store import (
    get_active_handoffs,
    get_ended_handoffs,
    get_handoff_detail,
    mark_answered,
    mark_ended,
    get_handoff_stats,
)
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/handoff", tags=["Human Handoff"])


@router.get(
    "/queue",
    response_model=HandoffQueueResponse,
    summary="Get active handoff queue (ringing + answered)",
    description=(
        "Returns all calls currently requiring human agent attention, sorted oldest-first. "
        "Only ringing and answered calls appear here - ended calls are excluded. "
        "The response includes per-status counts so the frontend can render queue depth badges. "
        "Poll this endpoint every 5-10 seconds on the agent dashboard to keep the queue live."
    ),
)
async def get_handoff_queue():
    try:
        items = get_active_handoffs()
        ringing_count = sum(1 for i in items if i["status"] == "ringing")
        answered_count = sum(1 for i in items if i["status"] == "answered")
        return HandoffQueueResponse(
            total=len(items),
            ringing=ringing_count,
            answered=answered_count,
            items=[HandoffQueueItem(**i) for i in items],
        )
    except Exception as e:
        logger.error(f"Failed to fetch handoff queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    response_model=HandoffHistoryResponse,
    summary="Get all ended (completed) handoff calls",
    description=(
        "Returns every call that has been ended via POST /{id}/end, sorted most-recently-ended first. "
        "These calls no longer appear in /queue. "
        "Use this to build a call log tab on the agent dashboard. "
        "Each item includes actioned_at (when the call was ended) and answered_at (when it was picked up) "
        "so you can compute handle time."
    ),
)
async def get_handoff_history():
    try:
        items = get_ended_handoffs()
        return HandoffHistoryResponse(
            total=len(items),
            items=[HandoffQueueItem(**i) for i in items],
        )
    except Exception as e:
        logger.error(f"Failed to fetch handoff history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/dashboard",
    response_model=HandoffDashboardResponse,
    summary="Live status board - counts and full active call list",
    description=(
        "Single endpoint for a call-center status board. Returns: "
        "ringing - number of unanswered calls waiting; "
        "answered - calls currently being handled by an agent; "
        "ended - total historical call count; "
        "active_calls - the full active queue list for rendering individual call cards. "
        "Use this as the primary testing_dataset source for a supervisor dashboard that needs "
        "both aggregate metrics and call detail in a single request."
    ),
)
async def get_handoff_dashboard():
    try:
        stats = get_handoff_stats()
        active = get_active_handoffs()
        return HandoffDashboardResponse(
            ringing=stats.get("ringing", 0),
            answered=stats.get("answered", 0),
            ended=stats.get("ended", 0),
            active_calls=[HandoffQueueItem(**i) for i in active],
        )
    except Exception as e:
        logger.error(f"Failed to fetch handoff dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{handoff_id}/answer",
    response_model=HandoffActionResponse,
    summary="Mark a call as answered (agent picked up)",
    description=(
        "Transitions the call from ringing to answered. "
        "Call this when an agent clicks Answer on a queue item. "
        "The call remains visible in /queue but moves to the answered section. "
        "Records answered_at timestamp for handle-time reporting. "
        "Returns 409 Conflict if the call is not in ringing state."
    ),
)
async def answer_handoff(handoff_id: str):
    try:
        item = get_handoff_detail(handoff_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Handoff {handoff_id} not found")
        if item["status"] != "ringing":
            raise HTTPException(
                status_code=409,
                detail=f"Cannot answer handoff {handoff_id}: current status is '{item['status']}'. Must be 'ringing'."
            )
        mark_answered(handoff_id)
        logger.info(f"Handoff {handoff_id} answered (session={item['session_id']})")
        return HandoffActionResponse(
            status="answered",
            handoff_id=handoff_id,
            session_id=item["session_id"],
            message=f"Call {handoff_id} is now answered. Customer is on the line.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to answer handoff {handoff_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{handoff_id}/end",
    response_model=HandoffActionResponse,
    summary="End a call - removes it from the active queue",
    description=(
        "Transitions the call from ringing or answered to ended. "
        "Call this when the agent completes the interaction and hangs up. "
        "The call is immediately removed from /queue and will appear in /history. "
        "Records actioned_at timestamp. "
        "Returns 409 Conflict if the call is already ended."
    ),
)
async def end_handoff(handoff_id: str):
    try:
        item = get_handoff_detail(handoff_id)
        if not item:
            raise HTTPException(status_code=404, detail=f"Handoff {handoff_id} not found")
        if item["status"] == "ended":
            raise HTTPException(
                status_code=409,
                detail=f"Handoff {handoff_id} is already ended."
            )
        mark_ended(handoff_id)
        logger.info(f"Handoff {handoff_id} ended (session={item['session_id']})")
        return HandoffActionResponse(
            status="ended",
            handoff_id=handoff_id,
            session_id=item["session_id"],
            message=f"Call {handoff_id} has been ended and moved to history.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end handoff {handoff_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{handoff_id}",
    summary="Get full detail for a single handoff record",
    description=(
        "Returns the complete record for any handoff by its UUID - regardless of status. "
        "Includes the full chat_history, conversation_summary, latency_ms breakdown, "
        "emotion scores, confidence score, escalation reason, and customer phone number. "
        "Use this to populate a side-panel when an agent clicks on a queue item."
    ),
)
async def get_handoff_detail_endpoint(handoff_id: str):
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
