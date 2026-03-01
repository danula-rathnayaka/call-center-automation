from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from multiagent_rag.agents.contextualizer import Contextualizer
from multiagent_rag.agents.confidence_agent import ConfidenceAgent
from multiagent_rag.agents.emotion_agent import EmotionAgent
from multiagent_rag.agents.finetuned_llm_agent import FinetunedLLMAgent
from multiagent_rag.agents.generator import Generator
from multiagent_rag.agents.retriever import Retriever
from multiagent_rag.agents.tool_agent import ToolAgent
from multiagent_rag.graph.rag_router import IntentRouter
from multiagent_rag.state.rag_state import RAGState
from multiagent_rag.tools.crm_tools import crm_tools
from multiagent_rag.utils.interaction_logger import InteractionLogger
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Shared Singletons ─────────────────────────────────────────────────────────
_emotion_agent = EmotionAgent()
_confidence_agent = ConfidenceAgent()
_finetuned_llm = FinetunedLLMAgent()
_interaction_logger = InteractionLogger()


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def emotion_detector_node(state: RAGState):
    """Detect the emotion of the incoming user query."""
    query = state["query"]
    logger.info("Running emotion detection on user query")

    result = _emotion_agent.detect(query)

    return {
        "emotion": result.get("emotion", "neutral"),
        "emotion_confidence": result.get("confidence", 0.0),
    }


def route_query(state: RAGState):
    """Route the user query to the appropriate processing pipeline."""
    query = state["query"]
    router = IntentRouter()
    intent = router.route(query)
    logger.info(f"Query routed with intent: {intent}")

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
    """Reformulate the query using conversation history for better retrieval."""
    query = state["query"]
    history = state.get("chat_history", [])

    if not history:
        return {"reformulated_query": query, "intent": "technical"}

    logger.info("Starting query contextualization")
    contextualizer = Contextualizer()
    new_query = contextualizer.reformulate(query, history)

    return {"reformulated_query": new_query, "intent": "technical"}


def retrieve_node(state: RAGState):
    """Retrieve relevant documents from the vector database."""
    query = state["reformulated_query"]
    logger.info(f"Starting document retrieval for query: {query}")
    retriever = Retriever()
    docs = retriever.retrieve(query)
    return {"retrieved_docs": docs}


def generate_node(state: RAGState):
    """Generate a response using the fine-tuned LLM (or fallback to Groq)."""
    query = state["query"]
    docs = state["retrieved_docs"]
    history = state.get("chat_history", [])

    logger.info("Starting response generation")
    retriever = Retriever()
    context_text = retriever.format_docs(docs)

    # Use the fine-tuned LLM agent (falls back to Groq if not available)
    answer = _finetuned_llm.generate(query, context_text, history)

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)]
    }


def casual_node(state: RAGState):
    """Handle casual/greeting queries."""
    query = state["query"]
    generator = Generator()
    answer = generator.generate(
        query,
        "User is normally chatting. Reply politely and warmly.",
        state.get("chat_history", [])
    )

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
        "intent": "casual",
    }


def escalation_node(state: RAGState):
    """Handle requests to speak with a human agent."""
    query = state["query"]
    answer = (
        "I understand you'd like to speak with a representative. "
        "I am transferring you to a human agent now. Please hold on..."
    )

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
        "intent": "escalation",
        "should_escalate": True,
    }


def tool_agent_node(state: RAGState):
    """Invoke the CRM tool agent for customer service actions."""
    query = state["query"]
    history = state.get("chat_history", [])

    logger.info("Invoking tool agent")
    agent = ToolAgent()
    response = agent.invoke(query, history)

    return {
        "chat_history": [HumanMessage(content=query), response],
        "final_answer": response.content,
        "intent": "customer_service",
    }


def check_for_tool_calls(state: RAGState):
    """Check if the tool agent wants to call any tools."""
    history = state.get("chat_history", [])
    if not history:
        return "__end__"

    last_message = history[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"


def confidence_evaluator_node(state: RAGState):
    """Evaluate the confidence of the generated response."""
    query = state.get("query", "")
    response = state.get("final_answer", "")
    docs = state.get("retrieved_docs", [])

    # Build context text from retrieved docs
    context_text = ""
    for doc in docs:
        context_text += doc.get("content", "") + "\n"

    logger.info("Running confidence evaluation on generated response")
    result = _confidence_agent.evaluate(query, response, context_text)

    return {
        "response_confidence": result.get("confidence_score", 0.5),
        "should_escalate": result.get("should_escalate", False),
    }


def interaction_logger_node(state: RAGState):
    """Log the complete interaction for transparency and analysis."""
    session_id = state.get("session_id", "unknown")

    _interaction_logger.log_interaction(
        session_id=session_id,
        query=state.get("query", ""),
        response=state.get("final_answer", ""),
        emotion=state.get("emotion", "neutral"),
        emotion_confidence=state.get("emotion_confidence", 0.0),
        response_confidence=state.get("response_confidence", 0.0),
        should_escalate=state.get("should_escalate", False),
        intent=state.get("intent", "unknown"),
        retrieved_docs_count=len(state.get("retrieved_docs", [])),
    )

    return {"status": "completed"}


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

workflow = StateGraph(RAGState)

# ── Register all nodes ──────────────────────────────────────────────────────────
workflow.add_node("emotion_detector", emotion_detector_node)
workflow.add_node("contextualizer", contextualize_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)
workflow.add_node("tool_agent", tool_agent_node)
workflow.add_node("tools", ToolNode(crm_tools, messages_key="chat_history"))
workflow.add_node("casual_responder", casual_node)
workflow.add_node("escalator", escalation_node)
workflow.add_node("confidence_evaluator", confidence_evaluator_node)
workflow.add_node("interaction_logger", interaction_logger_node)

# ── Entry point: always start with emotion detection ────────────────────────────
workflow.set_entry_point("emotion_detector")

# ── After emotion detection → route by intent ──────────────────────────────────
workflow.add_conditional_edges(
    "emotion_detector",
    route_query,
    {
        "contextualizer": "contextualizer",
        "tool_agent": "tool_agent",
        "casual_responder": "casual_responder",
        "escalator": "escalator",
    }
)

# ── Technical path: contextualize → retrieve → generate ────────────────────────
workflow.add_edge("contextualizer", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "confidence_evaluator")

# ── Tool agent path: tool_agent → (tools loop or end) ─────────────────────────
workflow.add_conditional_edges(
    "tool_agent",
    check_for_tool_calls,
    {
        "tools": "tools",
        "__end__": "confidence_evaluator",
    }
)
workflow.add_edge("tools", "tool_agent")

# ── Casual & escalation → confidence evaluator ────────────────────────────────
workflow.add_edge("casual_responder", "confidence_evaluator")
workflow.add_edge("escalator", "confidence_evaluator")

# ── Confidence evaluator → interaction logger → END ───────────────────────────
workflow.add_edge("confidence_evaluator", "interaction_logger")
workflow.add_edge("interaction_logger", END)

# ── Compile with memory checkpointer ─────────────────────────────────────────
memory = MemorySaver()
rag_app = workflow.compile(checkpointer=memory)
