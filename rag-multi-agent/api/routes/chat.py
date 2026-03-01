import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, SessionHistoryResponse, SessionMessage
from multiagent_rag.graph.rag_workflow import rag_app
from multiagent_rag.utils.interaction_logger import InteractionLogger
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

_interaction_logger = InteractionLogger()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a query to the multi-agent RAG system and get a response.

    The system will:
    1. Detect the emotion of the query
    2. Route to the appropriate agent (technical, customer service, casual, escalation)
    3. Generate a response using RAG or tools
    4. Evaluate response confidence
    5. Log the interaction

    If no session_id is provided, one will be auto-generated.
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        config = {"configurable": {"thread_id": session_id}}

        result = rag_app.invoke(
            {
                "query": request.query,
                "session_id": session_id,
            },
            config=config,
        )

        return ChatResponse(
            response=result.get("final_answer", ""),
            session_id=session_id,
            emotion=result.get("emotion", "neutral"),
            emotion_confidence=result.get("emotion_confidence", 0.0),
            confidence_score=result.get("response_confidence", 0.0),
            should_escalate=result.get("should_escalate", False),
            intent=result.get("intent", "unknown"),
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """
    Retrieve the conversation history for a specific session.
    """
    try:
        history = _interaction_logger.get_session_history(session_id)

        messages = []
        for entry in history:
            messages.append(SessionMessage(
                timestamp=entry.get("timestamp", ""),
                query=entry.get("query", ""),
                response=entry.get("response", ""),
                emotion=entry.get("emotion", "neutral"),
                confidence_score=entry.get("response_confidence", 0.0),
            ))

        return SessionHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
        )

    except Exception as e:
        logger.error(f"Session history retrieval error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session history: {str(e)}"
        )
