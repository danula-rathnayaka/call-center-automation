from typing import TypedDict, List, Annotated, Dict, Optional

from langchain_core.messages import BaseMessage


def replace_messages(current: List[BaseMessage], update: List[BaseMessage]) -> List[BaseMessage]:
    if update is None:
        return current or []
    return update


def merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    merged = a.copy() if a else {}
    if b:
        merged.update(b)
    return merged


class RAGState(TypedDict):
    query: str
    audio_path: str
    session_id: str
    phone_number: Optional[str]
    emotion: str
    emotion_confidence: float
    reformulated_query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict]
    intent: str
    guardrail_passed: bool
    final_answer: str
    chat_history: Annotated[List[BaseMessage], replace_messages]
    conversation_summary: Optional[str]
    response_confidence: float
    should_escalate: bool
    escalation_reason: Optional[str]
    latency_ms: Annotated[Dict[str, float], merge_dicts]
    status: str
    handoff_uuid: Optional[str]
