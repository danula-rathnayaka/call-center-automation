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
from multiagent_rag.utils.stt import STTEngine

logger = get_logger(__name__)

_emotion_agent = EmotionAgent()
_confidence_agent = ConfidenceAgent()
_finetuned_llm = FinetunedLLMAgent()
_stt_engine = STTEngine()
_interaction_logger = InteractionLogger()


def voice_processing_node(state: RAGState):
    audio_path = state.get("audio_path", "")
    query = state.get("query", "")

    if audio_path:
        logger.info("Processing voice input - STT + Emotion Detection")

        transcribed_text = _stt_engine.transcribe(audio_path)
        if not transcribed_text:
            transcribed_text = query or "Unable to transcribe audio"
            logger.warning("STT returned empty, using fallback text")

        emotion_result = _emotion_agent.detect_from_audio(audio_path)

        return {
            "query": transcribed_text,
            "emotion": emotion_result.get("emotion", "neutral"),
            "emotion_confidence": emotion_result.get("confidence", 0.0),
        }
    else:
        logger.info("Processing text input - keyword emotion detection")
        emotion_result = _emotion_agent.detect_from_text(query)

        return {
            "emotion": emotion_result.get("emotion", "neutral"),
            "emotion_confidence": emotion_result.get("confidence", 0.0),
        }


def route_query(state: RAGState):
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
    query = state["query"]
    history = state.get("chat_history", [])

    if not history:
        return {"reformulated_query": query, "intent": "technical"}

    logger.info("Starting query contextualization")
    contextualizer = Contextualizer()
    new_query = contextualizer.reformulate(query, history)
    return {"reformulated_query": new_query, "intent": "technical"}


def retrieve_node(state: RAGState):
    query = state["reformulated_query"]
    logger.info(f"Starting document retrieval for query: {query}")
    retriever = Retriever()
    docs = retriever.retrieve(query)
    return {"retrieved_docs": docs}


def generate_node(state: RAGState):
    query = state["query"]
    docs = state["retrieved_docs"]
    history = state.get("chat_history", [])
    emotion = state.get("emotion", "neutral")

    logger.info(f"Starting emotion-aware response generation (emotion: {emotion})")

    retriever = Retriever()
    context_text = retriever.format_docs(docs)

    answer = _finetuned_llm.generate(query, context_text, emotion, history)

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
    }


def casual_node(state: RAGState):
    query = state["query"]
    emotion = state.get("emotion", "neutral")
    generator = Generator()

    emotion_hint = f"The customer's detected emotion is: {emotion}. "
    context = emotion_hint + "User is normally chatting. Reply politely and warmly."
    answer = generator.generate(query, context, state.get("chat_history", []))

    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
        "intent": "casual",
    }


def escalation_node(state: RAGState):
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
    history = state.get("chat_history", [])
    if not history:
        return "__end__"

    last_message = history[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def confidence_evaluator_node(state: RAGState):
    query = state.get("query", "")
    response = state.get("final_answer", "")
    retrieved_docs = state.get("retrieved_docs", [])
    emotion = state.get("emotion", "neutral")

    logger.info("Running confidence evaluation")
    result = _confidence_agent.evaluate(query, response, retrieved_docs, emotion)

    return {
        "response_confidence": result.get("confidence_score", 0.5),
        "should_escalate": result.get("should_escalate", False),
    }


def interaction_logger_node(state: RAGState):
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


workflow = StateGraph(RAGState)

workflow.add_node("voice_processor", voice_processing_node)
workflow.add_node("contextualizer", contextualize_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)
workflow.add_node("tool_agent", tool_agent_node)
workflow.add_node("tools", ToolNode(crm_tools, messages_key="chat_history"))
workflow.add_node("casual_responder", casual_node)
workflow.add_node("escalator", escalation_node)
workflow.add_node("confidence_evaluator", confidence_evaluator_node)
workflow.add_node("interaction_logger", interaction_logger_node)

workflow.set_entry_point("voice_processor")

workflow.add_conditional_edges(
    "voice_processor",
    route_query,
    {
        "contextualizer": "contextualizer",
        "tool_agent": "tool_agent",
        "casual_responder": "casual_responder",
        "escalator": "escalator",
    }
)

workflow.add_edge("contextualizer", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "confidence_evaluator")

workflow.add_conditional_edges(
    "tool_agent",
    check_for_tool_calls,
    {
        "tools": "tools",
        "__end__": "confidence_evaluator",
    }
)
workflow.add_edge("tools", "tool_agent")

workflow.add_edge("casual_responder", "confidence_evaluator")
workflow.add_edge("escalator", "confidence_evaluator")

workflow.add_edge("confidence_evaluator", "interaction_logger")
workflow.add_edge("interaction_logger", END)

memory = MemorySaver()
rag_app = workflow.compile(checkpointer=memory)
