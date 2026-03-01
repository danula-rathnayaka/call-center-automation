from typing import TypedDict, List, Annotated, Dict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    query: str
    audio_path: str
    session_id: str
    emotion: str
    emotion_confidence: float
    reformulated_query: str
    retrieved_docs: List[Dict]
    intent: str
    final_answer: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    response_confidence: float
    should_escalate: bool
    status: str
