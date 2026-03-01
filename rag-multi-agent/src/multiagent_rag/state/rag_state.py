from typing import TypedDict, List, Annotated, Dict, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    query: str
    reformulated_query: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Dict]
    final_answer: str
    status: str
    # Session tracking
    session_id: str
    # Emotion detection fields
    emotion: str
    emotion_confidence: float
    # Confidence evaluation fields
    response_confidence: float
    should_escalate: bool
    # Intent routing
    intent: str

