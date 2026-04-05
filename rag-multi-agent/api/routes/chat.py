import asyncio
import json
import os
import uuid

from api.schemas import (ChatRequest, ChatResponse, RetrievedChunk, EmotionResult, ConfidenceResult,
                         SessionHistoryResponse, SessionMessage)
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langfuse import observe, propagate_attributes, get_client

from multiagent_rag.graph.rag_workflow import rag_app
from multiagent_rag.utils.interaction_logger import InteractionLogger
from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.telemetry import get_langchain_handler, flush

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])

_ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".webm", ".flac", ".aiff", ".m4a", ".aac"}
_interaction_logger = InteractionLogger()

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_voice_upload_dir = os.path.join(_project_root, "data", "voice_uploads")
os.makedirs(_voice_upload_dir, exist_ok=True)


def _build_response(result: dict, session_id: str, transcribed_text: str = "") -> ChatResponse:
    retrieved_chunks = []
    for doc in result.get("retrieved_docs", []):
        retrieved_chunks.append(RetrievedChunk(
            content=doc.get("content", ""),
            source=doc.get("metadata", {}).get("source", ""),
            chunk_type=doc.get("metadata", {}).get("type", ""),
        ))
    raw_latency = result.get("latency_ms", {})
    total_ms = round(sum(raw_latency.values())) if raw_latency else 0
    latency_out = {"total": total_ms, **raw_latency}
    return ChatResponse(
        response=result.get("final_answer", ""),
        session_id=session_id,
        transcribed_text=transcribed_text,
        emotion=EmotionResult(
            emotion=result.get("emotion", "neutral"),
            confidence=result.get("emotion_confidence", 0.0),
        ),
        intent=result.get("intent", "unknown"),
        retrieved_chunks=retrieved_chunks,
        confidence=ConfidenceResult(
            score=result.get("response_confidence", 0.0),
            should_escalate=result.get("should_escalate", False),
        ),
        latency_ms=latency_out,
        handoff_uuid=result.get("handoff_uuid"),
    )


def _write_scores(result: dict):
    try:
        lf = get_client()
        if not lf:
            return
        scores = [
            ("response_confidence", round(result.get("response_confidence", 0.5), 4),
             f"intent={result.get('intent', 'unknown')}"),
            ("emotion_confidence", round(result.get("emotion_confidence", 0.0), 4),
             f"emotion={result.get('emotion', 'neutral')}"),
            ("was_escalated", 1.0 if result.get("should_escalate") else 0.0,
             result.get("escalation_reason", "") or "no escalation"),
            ("retrieved_docs_count", float(len(result.get("retrieved_docs", []))),
             "number of chunks retrieved from Pinecone"),
        ]
        for name, value, comment in scores:
            lf.score_current_trace(name=name, value=value, comment=comment)
        logger.info(f"Langfuse scores written: confidence={scores[0][1]}, escalated={scores[2][1]}")
    except Exception as e:
        logger.warning(f"Langfuse score write failed: {e}")


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a text message to the AI agent",
    description=(
        "Submit a plain-text customer query to the full multi-agent RAG pipeline. "
        "The system runs through: session loading -> emotion detection -> safety guardrail -> "
        "intent routing -> (RAG retrieval + reranking + generation) OR (tool calls) OR (casual response) -> "
        "confidence evaluation -> history summarization. "
        "Always pass a session_id so conversation history is preserved across turns. "
        "Check confidence.should_escalate - if true, the backend has already enqueued a human handoff. "
        "handoff_uuid in the response is the tracking ID for the agent dashboard."
    ),
)
@observe(name="chat_text")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    with propagate_attributes(
        session_id=session_id,
        user_id=request.phone_number or session_id,
        tags=["rag", "channel:text"],
        metadata={"session_id": session_id, "phone_number": request.phone_number or ""},
    ):
        lf_handler = get_langchain_handler()
        callbacks = [lf_handler] if lf_handler else []
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "run_name": f"rag_pipeline:{session_id[:8]}",
        }

        try:
            result = rag_app.invoke(
                {"query": request.query, "audio_path": "", "session_id": session_id,
                 "phone_number": request.phone_number},
                config=config,
            )
            _write_scores(result)
            flush()
            return _build_response(result, session_id)

        except Exception as e:
            logger.error(f"Chat endpoint error: {str(e)}")
            flush()
            raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@router.post(
    "/voice",
    response_model=ChatResponse,
    summary="Send a voice/audio message to the AI agent",
    description=(
        "Upload an audio file containing the customer's spoken query. "
        "Supported formats: WAV, MP3, OGG, WebM, FLAC, AIFF, M4A, AAC. "
        "The pipeline runs STT transcription -> audio-based emotion detection -> "
        "the identical multi-agent RAG pipeline as the text endpoint. "
        "The response includes transcribed_text so the UI can show the customer what was heard. "
        "Pass session_id as a query parameter to link this voice turn to an ongoing conversation."
    ),
)
@observe(name="chat_voice")
async def chat_voice(audio: UploadFile = File(...), session_id: str = None, phone_number: str = None):
    session_id = session_id or str(uuid.uuid4())

    filename = audio.filename or f"voice_{uuid.uuid4()}.wav"
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in _ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXTENSIONS))}"
        )

    with propagate_attributes(
        session_id=session_id,
        user_id=phone_number or session_id,
        tags=["rag", "channel:voice"],
        metadata={"session_id": session_id, "phone_number": phone_number or "",
                  "audio_format": ext or "unknown"},
    ):
        lf_handler = get_langchain_handler()
        callbacks = [lf_handler] if lf_handler else []
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "run_name": f"rag_pipeline_voice:{session_id[:8]}",
        }

        try:
            audio_path = os.path.join(_voice_upload_dir, f"{uuid.uuid4()}_{filename}")
            with open(audio_path, "wb") as f:
                content = await audio.read()
                f.write(content)
            logger.info(f"Voice file saved: {audio_path} ({len(content)} bytes)")

            result = rag_app.invoke(
                {"query": "", "audio_path": audio_path, "session_id": session_id,
                 "phone_number": phone_number},
                config=config,
            )
            transcribed_text = result.get("query", "")
            _write_scores(result)

            try:
                os.remove(audio_path)
            except Exception:
                pass

            flush()
            return _build_response(result, session_id, transcribed_text=transcribed_text)

        except HTTPException:
            flush()
            raise
        except Exception as e:
            logger.error(f"Voice chat endpoint error: {str(e)}")
            flush()
            raise HTTPException(status_code=500, detail=f"Failed to process voice: {str(e)}")


@router.post(
    "/stream",
    summary="Stream AI response tokens in real time (Server-Sent Events)",
    description=(
        "Identical to POST /api/chat but streams the response token-by-token using "
        "Server-Sent Events (SSE). Each SSE event has a type field: "
        "token - a partial response chunk; "
        "done - full metadata (emotion, intent, confidence, latency_ms, should_escalate, handoff_uuid); "
        "error - emitted if the pipeline fails mid-stream."
    ),
)
@observe(name="chat_stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        with propagate_attributes(
            session_id=session_id,
            user_id=request.phone_number or session_id,
            tags=["rag", "channel:stream"],
        ):
            lf_handler = get_langchain_handler()
            callbacks = [lf_handler] if lf_handler else []
            config = {
                "configurable": {"thread_id": session_id},
                "callbacks": callbacks,
                "run_name": f"rag_pipeline_stream:{session_id[:8]}",
            }
            input_data = {"query": request.query, "audio_path": "", "session_id": session_id,
                          "phone_number": request.phone_number}
            try:
                async for event in rag_app.astream_events(input_data, config=config, version="v2"):
                    event_name = event.get("event", "")
                    if event_name == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            payload = json.dumps({"type": "token", "content": chunk.content})
                            yield f"data: {payload}\n\n"
                    elif event_name == "on_chain_end" and event.get("name") == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        _write_scores(output)
                        final_payload = json.dumps({
                            "type": "done",
                            "session_id": session_id,
                            "emotion": output.get("emotion", "neutral"),
                            "intent": output.get("intent", "unknown"),
                            "confidence": output.get("response_confidence", 0.0),
                            "should_escalate": output.get("should_escalate", False),
                            "handoff_uuid": output.get("handoff_uuid"),
                            "latency_ms": output.get("latency_ms", {}),
                        })
                        yield f"data: {final_payload}\n\n"
                        flush()
            except asyncio.CancelledError:
                logger.info(f"Stream cancelled for session {session_id}")
                flush()
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                flush()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Retrieve full conversation history for a session",
    description=(
        "Returns all logged interaction turns for the given session_id, ordered chronologically. "
        "Each message includes the customer query, AI response, detected emotion, "
        "response confidence score, and UTC timestamp."
    ),
)
async def get_session_history(session_id: str):
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
        return SessionHistoryResponse(session_id=session_id, messages=messages, total_messages=len(messages))
    except Exception as e:
        logger.error(f"Session history retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session history: {str(e)}")
