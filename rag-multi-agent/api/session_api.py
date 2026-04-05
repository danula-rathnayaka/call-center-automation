import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.session_store import (load_history, load_summary, delete_session, )

logger = get_logger(__name__)
router = APIRouter(prefix="/api/session", tags=["Session"])


class SessionStartResponse(BaseModel):
    session_id: str
    phone_number: Optional[str] = None
    message: str


class SessionEndResponse(BaseModel):
    session_id: str
    turns_in_history: int
    summary_saved: bool
    message: str


@router.post("/start", response_model=SessionStartResponse)
async def start_session(session_id: Optional[str] = None, phone_number: Optional[str] = None):
    sid = session_id or str(uuid.uuid4())
    
    from multiagent_rag.utils.session_store import create_session
    create_session(sid, phone_number)
    
    logger.info(f"Session started: {sid} (phone: {phone_number})")
    return SessionStartResponse(
        session_id=sid,
        phone_number=phone_number,
        message="Session initialized. Phone number linked to this session.",
    )


@router.post("/end/{session_id}", response_model=SessionEndResponse)
async def end_session(session_id: str):
    try:
        history = load_history(session_id)
        summary = load_summary(session_id)

        turns = len([m for m in history if getattr(m, "type", "") == "human"])
        summary_saved = summary is not None

        delete_session(session_id)
        logger.info(f"Session ended and cleaned up: {session_id} ({turns} turns)")

        return SessionEndResponse(session_id=session_id, turns_in_history=turns, summary_saved=summary_saved,
            message=f"Session ended. {turns} turns processed. Data cleared from store.", )
    except Exception as e:
        logger.error(f"Failed to end session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    try:
        summary = load_summary(session_id)
        history = load_history(session_id)
        return {"session_id": session_id, "summary": summary or "", "recent_turns": len(history),
            "has_summary": summary is not None, }
    except Exception as e:
        logger.error(f"Failed to get session summary {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
