from langgraph.graph import StateGraph, END

from multiagent_rag.agents.generator import Generator
from multiagent_rag.agents.retriever import Retriever
from multiagent_rag.state.rag_state import RAGState


def retrieve_node(state: RAGState):
    query = state["query"]

    retriever = Retriever()
    docs = retriever.retrieve(query)

    status = "found_docs" if docs else "no_docs"

    return {"retrieved_docs": docs, "status": status}


def generate_node(state: RAGState):
    query = state["query"]
    docs = state["retrieved_docs"]

    generator = Generator()

    retriever = Retriever()
    context_text = retriever.format_docs(docs)

    if not context_text:
        answer = "I'm sorry, I couldn't find any relevant information in the database."
    else:
        answer = generator.generate(query, context_text)

    return {"final_answer": answer, "status": "completed"}


workflow = StateGraph(RAGState)

workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.set_entry_point("retriever")

workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

rag_app = workflow.compile()
