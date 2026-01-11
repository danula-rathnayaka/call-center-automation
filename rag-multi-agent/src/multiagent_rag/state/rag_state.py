from typing import TypedDict, List, Dict


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Dict]
    final_answer: str
    status: str
