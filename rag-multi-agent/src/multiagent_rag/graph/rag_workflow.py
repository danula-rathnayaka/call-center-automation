import concurrent.futures
import os
import sqlite3
import time

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END

from multiagent_rag.agents.confidence_agent import ConfidenceAgent
from multiagent_rag.agents.contextualizer import Contextualizer
from multiagent_rag.agents.emotion_agent import EmotionAgent
from multiagent_rag.agents.finetuned_llm_agent import FinetunedLLMAgent
from multiagent_rag.agents.generator import Generator
from multiagent_rag.agents.guardrail import Guardrail
from multiagent_rag.agents.query_decomposer import QueryDecomposer
from multiagent_rag.agents.reranker import Reranker
from multiagent_rag.agents.retriever import Retriever
from multiagent_rag.agents.summarizer import ConversationSummarizer
from multiagent_rag.agents.tool_agent import ToolAgent
from multiagent_rag.graph.rag_router import IntentRouter
from multiagent_rag.state.rag_state import RAGState
from multiagent_rag.tools.crm_tools import get_dynamic_tools
from multiagent_rag.utils.interaction_logger import InteractionLogger
from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.session_store import (
    load_history, save_history,
    load_summary, save_summary,
)
from multiagent_rag.utils.stt import STTEngine

logger = get_logger(__name__)

_emotion_agent = EmotionAgent()
_confidence_agent = ConfidenceAgent()
_finetuned_llm = FinetunedLLMAgent()
_stt_engine = STTEngine()
_interaction_logger = InteractionLogger()
_router = IntentRouter()
_contextualizer = Contextualizer()
_retriever = Retriever()
_generator = Generator()
_reranker = Reranker()
_guardrail = Guardrail()
_query_decomposer = QueryDecomposer()
_summarizer = ConversationSummarizer()
_tool_agent = ToolAgent()


def session_manager_node(state: RAGState):
    session_id = state.get("session_id", "")
    if not session_id:
        logger.warning("session_manager: no session_id in state, skipping load")
        return {}

    history = load_history(session_id)
    summary = load_summary(session_id)

    logger.info(
        f"session_manager: loaded {len(history)} messages "
        f"and {'a summary' if summary else 'no summary'} "
        f"for session {session_id}"
    )

    result = {}
    if history:
        result["chat_history"] = history
    if summary:
        result["conversation_summary"] = summary
    return result


def stt_node(state: RAGState):
    start = time.time()
    audio_path = state.get("audio_path", "")
    query = state.get("query", "")

    if audio_path:
        logger.info("STT: Transcribing audio input")
        transcribed = _stt_engine.transcribe(audio_path)
        if not transcribed:
            transcribed = query or "Unable to transcribe audio"
            logger.warning("STT returned empty result, using fallback text")
        elapsed = round((time.time() - start) * 1000)
        return {"query": transcribed, "latency_ms": {"stt": elapsed}}

    elapsed = round((time.time() - start) * 1000)
    return {"latency_ms": {"stt": elapsed}}


def emotion_node(state: RAGState):
    start = time.time()
    audio_path = state.get("audio_path", "")
    query = state.get("query", "")

    if audio_path:
        emotion_result = _emotion_agent.detect_from_audio(audio_path)
    else:
        emotion_result = _emotion_agent.detect_from_text(query)

    elapsed = round((time.time() - start) * 1000)
    return {
        "emotion": emotion_result.get("emotion", "neutral"),
        "emotion_confidence": emotion_result.get("confidence", 0.0),
        "latency_ms": {"emotion_detection": elapsed},
    }


def guardrail_node(state: RAGState):
    start = time.time()
    query = state["query"]
    history = state.get("chat_history", [])

    result = _guardrail.validate(query, history)
    elapsed = round((time.time() - start) * 1000)

    updates = {
        "guardrail_passed": result["safe"],
        "latency_ms": {"guardrail": elapsed},
    }

    if result["safe"]:
        updates["query"] = result["sanitized_query"]
    else:
        updates["final_answer"] = result["reason"]

    return updates


def route_after_guardrail(state: RAGState):
    if not state.get("guardrail_passed", True):
        return "blocked_response"

    query = state["query"]
    history = state.get("chat_history", [])
    intent = _router.route(query, history)
    logger.info(f"Query routed with intent: {intent}")

    state["intent"] = intent

    if intent == "technical":
        return "contextualizer"
    elif intent == "customer_service":
        return "tool_agent"
    elif intent == "casual":
        return "casual_responder"
    elif intent == "escalation":
        return "escalation_responder"
    else:
        return "contextualizer"


def blocked_response_node(state: RAGState):
    query = state["query"]
    answer = state.get("final_answer") or (
        "I am not able to process that request as stated. "
        "Could you please rephrase your question? "
        "I am here to help with anything related to our telecom services."
    )
    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
        "intent": "blocked",
    }


def contextualize_node(state: RAGState):
    start = time.time()
    query = state["query"]
    history = state.get("chat_history", [])
    summary = state.get("conversation_summary")

    new_query = _contextualizer.reformulate(query, history, summary)
    elapsed = round((time.time() - start) * 1000)
    return {
        "reformulated_query": new_query,
        "intent": "technical",
        "latency_ms": {"contextualizer": elapsed},
    }


def query_decomposer_node(state: RAGState):
    start = time.time()
    query = state["reformulated_query"]
    sub_queries = _query_decomposer.decompose(query)
    elapsed = round((time.time() - start) * 1000)
    return {"sub_queries": sub_queries, "latency_ms": {"query_decomposer": elapsed}}


def retrieve_node(state: RAGState):
    start = time.time()
    sub_queries = state.get("sub_queries", [])
    query = state.get("reformulated_query", state["query"])

    if not sub_queries:
        sub_queries = [query]

    def _fetch(sq: str) -> list:
        return _retriever.retrieve(sq)

    all_docs: list = []
    seen_contents: set = set()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_fetch, sq): sq for sq in sub_queries}
        for future in concurrent.futures.as_completed(futures):
            try:
                for doc in future.result():
                    key = doc["content"][:100]
                    if key not in seen_contents:
                        seen_contents.add(key)
                        all_docs.append(doc)
            except Exception as e:
                logger.error(f"Retrieval sub-query failed: {e}")

    elapsed = round((time.time() - start) * 1000)
    return {"retrieved_docs": all_docs, "latency_ms": {"retriever": elapsed}}


def reranker_node(state: RAGState):
    start = time.time()
    query = state.get("reformulated_query", state["query"])
    docs = state["retrieved_docs"]
    reranked = _reranker.rerank(query, docs, top_k=3)
    elapsed = round((time.time() - start) * 1000)
    return {"retrieved_docs": reranked, "latency_ms": {"reranker": elapsed}}


def generate_node(state: RAGState):
    start = time.time()
    query = state["query"]
    docs = state["retrieved_docs"]
    history = state.get("chat_history", [])
    summary = state.get("conversation_summary")
    emotion = state.get("emotion", "neutral")

    context_text = _retriever.format_docs(docs)

    answer = _finetuned_llm.generate(query, context_text, emotion, history, summary)
    answer = _guardrail.sanitize_response(answer)

    elapsed = round((time.time() - start) * 1000)
    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
        "latency_ms": {"generator": elapsed},
    }


def casual_node(state: RAGState):
    start = time.time()
    query = state["query"]
    emotion = state.get("emotion", "neutral")
    intent = state.get("intent", "casual")
    summary = state.get("conversation_summary")

    if intent == "escalation":
        context = (
            "The customer has requested to speak with a human agent. "
            "Acknowledge their request warmly, let them know their concern has been noted, "
            "and inform them that a representative will follow up. "
            f"Customer emotion: {emotion}."
        )
    else:
        context = (
            f"[Customer emotion: {emotion}] "
            "User is making casual conversation. Reply politely and warmly."
        )

    answer = _generator.generate(
        query, context, state.get("chat_history", []), summary
    )

    elapsed = round((time.time() - start) * 1000)
    return {
        "final_answer": answer,
        "chat_history": [HumanMessage(content=query), AIMessage(content=answer)],
        "intent": intent,
        "latency_ms": {"casual_responder": elapsed},
    }


def tool_agent_node(state: RAGState):
    start = time.time()
    query = state["query"]
    history = state.get("chat_history", [])

    is_looping = len(history) > 0 and history[-1].type == "tool"
    messages_to_pass = history if is_looping else history + [HumanMessage(content=query)]
    history_to_return = [] if is_looping else [HumanMessage(content=query)]

    response = _tool_agent.invoke(query, messages_to_pass)
    history_to_return.append(response)

    elapsed = round((time.time() - start) * 1000)
    return {
        "chat_history": history_to_return,
        "final_answer": response.content,
        "intent": "customer_service",
        "latency_ms": {"tool_agent": elapsed},
    }


def check_for_tool_calls(state: RAGState):
    history = state.get("chat_history", [])
    if not history:
        return "__end__"
    last = history[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


def dynamic_tools_node(state: RAGState):
    start = time.time()
    history = state.get("chat_history", [])
    last = history[-1]

    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return {"latency_ms": {"tools": round((time.time() - start) * 1000)}}

    tools_map = {t.name: t for t in get_dynamic_tools()}
    new_messages = []

    for tc in last.tool_calls:
        if tc["name"] in tools_map:
            try:
                result = tools_map[tc["name"]].invoke(tc["args"])
                tool_msg = ToolMessage(content=str(result), name=tc["name"], tool_call_id=tc["id"])
            except Exception as e:
                tool_msg = ToolMessage(content=f"Error: {e}", name=tc["name"], tool_call_id=tc["id"])
        else:
            tool_msg = ToolMessage(
                content=f"Error: Tool {tc['name']} not found.",
                name=tc["name"], tool_call_id=tc["id"]
            )
        new_messages.append(tool_msg)

    elapsed = round((time.time() - start) * 1000)
    return {"chat_history": new_messages, "latency_ms": {"tools": elapsed}}


def confidence_evaluator_node(state: RAGState):
    start = time.time()
    result = _confidence_agent.evaluate(
        state.get("query", ""),
        state.get("final_answer", ""),
        state.get("retrieved_docs", []),
        state.get("emotion", "neutral"),
    )
    elapsed = round((time.time() - start) * 1000)
    return {
        "response_confidence": result.get("confidence_score", 0.5),
        "should_escalate": result.get("should_escalate", False),
        "latency_ms": {"confidence_evaluator": elapsed},
    }


def history_summarizer_node(state: RAGState):
    start = time.time()
    history = state.get("chat_history", [])
    summary = state.get("conversation_summary")

    if len(history) <= 4:
        elapsed = round((time.time() - start) * 1000)
        return {"latency_ms": {"history_summarizer": elapsed}}

    recent_history, new_summary = _summarizer.summarize(
        history, keep_recent=4, existing_summary=summary
    )

    elapsed = round((time.time() - start) * 1000)
    return {
        "chat_history": recent_history,
        "conversation_summary": new_summary,
        "latency_ms": {"history_summarizer": elapsed},
    }


def interaction_logger_node(state: RAGState):
    session_id = state.get("session_id", "unknown")

    history = state.get("chat_history", [])
    summary = state.get("conversation_summary")

    if session_id != "unknown":
        save_history(session_id, history)
        if summary:
            save_summary(session_id, summary)

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
        latency_ms=state.get("latency_ms", {}),
    )
    return {"status": "completed"}


workflow = StateGraph(RAGState)

workflow.add_node("session_manager", session_manager_node)
workflow.add_node("stt_processor", stt_node)
workflow.add_node("emotion_detector", emotion_node)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("blocked_response", blocked_response_node)
workflow.add_node("contextualizer", contextualize_node)
workflow.add_node("query_decomposer", query_decomposer_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("reranker", reranker_node)
workflow.add_node("generator", generate_node)
workflow.add_node("tool_agent", tool_agent_node)
workflow.add_node("tools", dynamic_tools_node)
workflow.add_node("casual_responder", casual_node)
workflow.add_node("confidence_evaluator", confidence_evaluator_node)
workflow.add_node("history_summarizer", history_summarizer_node)
workflow.add_node("interaction_logger", interaction_logger_node)

workflow.set_entry_point("session_manager")

workflow.add_edge("session_manager", "stt_processor")

workflow.add_edge("stt_processor", "emotion_detector")
workflow.add_edge("emotion_detector", "guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_after_guardrail,
    {
        "contextualizer": "contextualizer",
        "tool_agent": "tool_agent",
        "casual_responder": "casual_responder",
        "escalation_responder": "casual_responder",
        "blocked_response": "blocked_response",
    }
)

workflow.add_edge("contextualizer", "query_decomposer")
workflow.add_edge("query_decomposer", "retriever")
workflow.add_edge("retriever", "reranker")
workflow.add_edge("reranker", "generator")
workflow.add_edge("generator", "confidence_evaluator")

workflow.add_conditional_edges(
    "tool_agent",
    check_for_tool_calls,
    {"tools": "tools", "__end__": "confidence_evaluator"}
)
workflow.add_edge("tools", "tool_agent")

workflow.add_edge("casual_responder", "confidence_evaluator")
workflow.add_edge("blocked_response", "interaction_logger")

workflow.add_edge("confidence_evaluator", "history_summarizer")
workflow.add_edge("history_summarizer", "interaction_logger")
workflow.add_edge("interaction_logger", END)

_checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
os.makedirs(_checkpoint_dir, exist_ok=True)
_checkpoint_path = os.path.join(_checkpoint_dir, "checkpoints.db")
_conn = sqlite3.connect(_checkpoint_path, check_same_thread=False)
memory = SqliteSaver(_conn)
rag_app = workflow.compile(checkpointer=memory)
