import asyncio
import json
import os
import uuid

from api.schemas import (ChatRequest, ChatResponse, RetrievedChunk, EmotionResult, ConfidenceResult,
                         SessionHistoryResponse, SessionMessage, )
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from multiagent_rag.graph.rag_workflow import rag_app
from multiagent_rag.utils.interaction_logger import InteractionLogger
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

_interaction_logger = InteractionLogger()

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_voice_upload_dir = os.path.join(_project_root, "data", "voice_uploads")
os.makedirs(_voice_upload_dir, exist_ok=True)


def _build_response(result: dict, session_id: str, transcribed_text: str = "") -> ChatResponse:
    retrieved_chunks = []
    for doc in result.get("retrieved_docs", []):
        retrieved_chunks.append(
            RetrievedChunk(content=doc.get("content", ""), source=doc.get("metadata", {}).get("source", ""),
                chunk_type=doc.get("metadata", {}).get("type", ""), ))

    raw_latency = result.get("latency_ms", {})
    total_ms = round(sum(raw_latency.values())) if raw_latency else 0
    latency_out = {"total": total_ms, **raw_latency}

    return ChatResponse(response=result.get("final_answer", ""), session_id=session_id,
        transcribed_text=transcribed_text, emotion=EmotionResult(emotion=result.get("emotion", "neutral"),
            confidence=result.get("emotion_confidence", 0.0), ), intent=result.get("intent", "unknown"),
        retrieved_chunks=retrieved_chunks, confidence=ConfidenceResult(score=result.get("response_confidence", 0.0),
            should_escalate=result.get("should_escalate", False), ), latency_ms=latency_out, )


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a text message to the AI agent",
    description=(
        "Submit a plain-text customer query to the full multi-agent RAG pipeline. "
        "The system runs through: session loading → emotion detection → safety guardrail → "
        "intent routing → (RAG retrieval + reranking + generation) OR (tool calls) OR (casual response) → "
        "confidence evaluation → history summarization. "
        "**Always pass a `session_id`** so conversation history is preserved across turns. "
        "If omitted, a new session UUID is generated per call and history will not persist. "
        "Check `confidence.should_escalate` in the response — if `true`, the backend has already "
        "enqueued a human handoff and the frontend should display a handoff notification to the customer."
    ),
)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    try:
        config = {"configurable": {"thread_id": session_id}}

        result = rag_app.invoke({"query": request.query, "audio_path": "", "session_id": session_id, }, config=config, )

        return _build_response(result, session_id, transcribed_text="")

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@router.post(
    "/voice",
    response_model=ChatResponse,
    summary="Send a voice/audio message to the AI agent",
    description=(
        "Upload an audio file (WAV, MP3, etc.) containing the customer's spoken query. "
        "The pipeline first transcribes the audio using the STT engine, then runs the "
        "transcribed text through the identical multi-agent RAG pipeline as the text endpoint. "
        "Emotion is additionally detected from the audio signal itself when available. "
        "The response includes `transcribed_text` so the UI can show the customer what was heard. "
        "Pass `session_id` as a query parameter to link this voice turn to an ongoing conversation."
    ),
)
async def chat_voice(audio: UploadFile = File(...), session_id: str = None):
    session_id = session_id or str(uuid.uuid4())

    try:
        filename = audio.filename or f"voice_{uuid.uuid4()}.wav"
        audio_path = os.path.join(_voice_upload_dir, f"{uuid.uuid4()}_{filename}")

        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        logger.info(f"Voice file saved: {audio_path} ({len(content)} bytes)")

        config = {"configurable": {"thread_id": session_id}}

        result = rag_app.invoke({"query": "", "audio_path": audio_path, "session_id": session_id, }, config=config, )

        transcribed_text = result.get("query", "")
        response = _build_response(result, session_id, transcribed_text=transcribed_text)

        try:
            os.remove(audio_path)
        except Exception:
            pass

        return response

    except Exception as e:
        logger.error(f"Voice chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process voice: {str(e)}")


@router.post(
    "/stream",
    summary="Stream AI response tokens in real time (Server-Sent Events)",
    description=(
        "Identical to `POST /api/chat` but streams the response token-by-token using "
        "Server-Sent Events (SSE). Connect with `EventSource` or `fetch` with a readable stream. "
        "Each SSE event has a `type` field: "
        "`token` — a partial response chunk to append to the UI; "
        "`done` — full metadata (emotion, intent, confidence, latency_ms, should_escalate) sent once generation finishes; "
        "`error` — emitted if the pipeline fails mid-stream. "
        "If `should_escalate` is `true` in the `done` event, show the escalation notice and poll `GET /api/handoff/queue`."
    ),
)
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            config = {"configurable": {"thread_id": session_id}}
            input_data = {"query": request.query, "audio_path": "", "session_id": session_id, }

            async for event in rag_app.astream_events(input_data, config=config, version="v2"):
                event_name = event.get("event", "")

                if event_name == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        payload = json.dumps({"type": "token", "content": chunk.content})
                        yield f"data: {payload}\n\n"

                elif event_name == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    final_payload = json.dumps(
                        {"type": "done", "session_id": session_id, "emotion": output.get("emotion", "neutral"),
                            "intent": output.get("intent", "unknown"),
                            "confidence": output.get("response_confidence", 0.0),
                            "should_escalate": output.get("should_escalate", False),
                            "latency_ms": output.get("latency_ms", {}), })
                    yield f"data: {final_payload}\n\n"

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", }, )


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Retrieve full conversation history for a session",
    description=(
        "Returns all logged interaction turns for the given `session_id`, ordered chronologically. "
        "Each message includes the customer's query, the AI's response, detected emotion, "
        "response confidence score, and UTC timestamp. "
        "Use this to power a chat history panel or to pre-load context when a user reopens an existing conversation."
    ),
)
async def get_session_history(session_id: str):
    try:
        history = _interaction_logger.get_session_history(session_id)

        messages = []
        for entry in history:
            messages.append(SessionMessage(timestamp=entry.get("timestamp", ""), query=entry.get("query", ""),
                response=entry.get("response", ""), emotion=entry.get("emotion", "neutral"),
                confidence_score=entry.get("response_confidence", 0.0), ))

        return SessionHistoryResponse(session_id=session_id, messages=messages, total_messages=len(messages), )

    except Exception as e:
        logger.error(f"Session history retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session history: {str(e)}")
