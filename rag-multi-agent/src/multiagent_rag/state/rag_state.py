from typing import TypedDict, List, Annotated, Dict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    query: str
    reformulated_query: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Dict]
    final_answer: str
    status: str
