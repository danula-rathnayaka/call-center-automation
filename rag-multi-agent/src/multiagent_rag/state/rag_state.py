from typing import TypedDict, List, Annotated, Dict, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    merged = a.copy() if a else {}
    if b:
        merged.update(b)
    return merged


class RAGState(TypedDict):
    # --- Core input ---
    query: str
    audio_path: str
    session_id: str

    # --- Emotion ---
    emotion: str
    emotion_confidence: float

    # --- Query processing ---
    reformulated_query: str
    sub_queries: List[str]

    # --- Retrieval ---
    retrieved_docs: List[Dict]

    # --- Routing ---
    intent: str
    guardrail_passed: bool

    # --- Output ---
    final_answer: str

    # --- Memory ---
    chat_history: Annotated[List[BaseMessage], add_messages]

    conversation_summary: Optional[str]

    # --- Confidence / Escalation ---
    response_confidence: float
    should_escalate: bool

    # --- Observability ---
    latency_ms: Annotated[Dict[str, float], merge_dicts]
    status: str
