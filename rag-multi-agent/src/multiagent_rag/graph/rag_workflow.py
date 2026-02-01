from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from multiagent_rag.agents.contextualizer import Contextualizer
from multiagent_rag.agents.generator import Generator
from multiagent_rag.agents.retriever import Retriever
from multiagent_rag.agents.tool_agent import ToolAgent
from multiagent_rag.graph.rag_router import IntentRouter
from multiagent_rag.state.rag_state import RAGState
from multiagent_rag.tools.crm_tools import crm_tools


def check_for_tool_calls(state: RAGState):
    history = state.get("chat_history", [])
    if not history:
        return "__end__"

    last_message = history[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"


def route_query(state: RAGState):
    query = state["query"]
    router = IntentRouter()
    intent = router.route(query)

    if intent == "technical":
        return "contextualizer"
    elif intent == "customer_service":
        return "tool_agent"
    elif intent == "casual":
        return "casual_responder"
    elif intent == "escalation":
        return "escalator"
    else:
        return "contextualizer"


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
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)]
    }


def casual_node(state: RAGState):
    query = state["query"]
    generator = Generator()
    answer = generator.generate(query, "User is normally chatting. Reply politely.", state.get("chat_history", []))

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)]
    }


def escalation_node(state: RAGState):
    query = state["query"]
    answer = ("I understand you'd like to speak with a representative."
              "I am transferring you to a human agent now. Please hold on...")

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)]
    }


def tool_agent_node(state: RAGState):
    query = state["query"]
    history = state.get("chat_history", [])

    agent = ToolAgent()
    response = agent.invoke(query, history)

    # --- FIX START ---
    # We must return 'final_answer' so main.py doesn't crash.
    # response.content is the text the agent wants to say (e.g., "What is your phone number?")
    return {
        "chat_history": [HumanMessage(content=query), response],
        "final_answer": response.content
    }
    # --- FIX END ---


workflow = StateGraph(RAGState)

workflow.add_node("contextualizer", contextualize_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)

workflow.add_node("tool_agent", tool_agent_node)
workflow.add_node("tools", ToolNode(crm_tools, messages_key="chat_history"))

workflow.add_node("casual_responder", casual_node)
workflow.add_node("escalator", escalation_node)

workflow.set_conditional_entry_point(
    route_query,
    {
        "contextualizer": "contextualizer",
        "tool_agent": "tool_agent",
        "casual_responder": "casual_responder",
        "escalator": "escalator"
    }
)

workflow.add_edge("contextualizer", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

workflow.add_conditional_edges(
    "tool_agent",
    check_for_tool_calls,
    {
        "tools": "tools",
        "__end__": END
    }
)
workflow.add_edge("tools", "tool_agent")

workflow.add_edge("casual_responder", END)
workflow.add_edge("escalator", END)

memory = MemorySaver()
rag_app = workflow.compile(checkpointer=memory)
