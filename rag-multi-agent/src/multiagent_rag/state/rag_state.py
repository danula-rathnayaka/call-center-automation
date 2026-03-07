from typing import TypedDict, List, Annotated, Dict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    merged = a.copy() if a else {}
    if b:
        merged.update(b)
    return merged


class RAGState(TypedDict):
    query: str
    audio_path: str
    session_id: str
    emotion: str
    emotion_confidence: float
    reformulated_query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict]
    intent: str
    final_answer: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    response_confidence: float
    should_escalate: bool
    guardrail_passed: bool
    latency_ms: Annotated[Dict[str, float], merge_dicts]
    status: str
