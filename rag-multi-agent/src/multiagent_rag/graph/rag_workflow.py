from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from multiagent_rag.agents.contextualizer import Contextualizer
from multiagent_rag.agents.generator import Generator
from multiagent_rag.agents.retriever import Retriever
from multiagent_rag.state.rag_state import RAGState


def contextualize_node(state: RAGState):
    query = state["query"]
    history = state.get("chat_history", [])

    contextualizer = Contextualizer()
    new_query = contextualizer.reformulate(query, history)

    return {"reformulated_query": new_query}


def retrieve_node(state: RAGState):
    query = state["reformulated_query"]
    retriever = Retriever()
    docs = retriever.retrieve(query)
    return {"retrieved_docs": docs}


def generate_node(state: RAGState):
    query = state["query"]
    docs = state["retrieved_docs"]
    history = state.get("chat_history", [])

    generator = Generator()
    retriever = Retriever()
    context_text = retriever.format_docs(docs)

    answer = generator.generate(query, context_text, history)

    return {
        "final_answer": answer,
        "chat_history": [
            HumanMessage(content=query),
            AIMessage(content=answer)
        ]
    }


workflow = StateGraph(RAGState)

workflow.add_node("contextualizer", contextualize_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.set_entry_point("contextualizer")
workflow.add_edge("contextualizer", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

memory = MemorySaver()
rag_app = workflow.compile(checkpointer=memory)
