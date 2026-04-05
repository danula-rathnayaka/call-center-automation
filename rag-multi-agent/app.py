import logging
import time
import uuid

import streamlit as st

logging.getLogger("httpx").setLevel(logging.WARNING)

from multiagent_rag.graph.rag_workflow import rag_app
from multiagent_rag.utils.db_client import PineconeClient
from multiagent_rag.utils.embeddings import EmbeddingManager
from multiagent_rag.utils.sparse import SparseEmbeddingManager
from multiagent_rag.utils.voice import VoiceHandler
from multiagent_rag.utils.tts import TTSEngine
from multiagent_rag.utils.telemetry import get_langfuse_client, get_langchain_handler
from langfuse import propagate_attributes, get_client

st.set_page_config(page_title="Telecom AI Agent", layout="centered")


@st.cache_resource(show_spinner="Booting up AI infrastructure... Please wait.")
def initialize_system():
    _ = PineconeClient()
    _ = EmbeddingManager()
    _ = SparseEmbeddingManager()
    voice = VoiceHandler()
    tts = TTSEngine()
    return voice, tts


voice_handler, tts_engine = initialize_system()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "phone_number" not in st.session_state:
    st.session_state.phone_number = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Telecom Support Agent. Click the mic to speak or type below."}
    ]

if "last_result" not in st.session_state:
    st.session_state.last_result = {}


def _invoke_pipeline(user_query: str, audio_path: str = "") -> dict:
    session_id = st.session_state.thread_id
    phone = st.session_state.phone_number or session_id

    lf_handler = get_langchain_handler()
    callbacks = [lf_handler] if lf_handler else []
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": callbacks,
        "run_name": f"rag_pipeline:{session_id[:8]}",
    }

    with propagate_attributes(
        session_id=session_id,
        user_id=phone,
        tags=["rag", "streamlit", "channel:voice" if audio_path else "channel:text"],
        metadata={"session_id": session_id, "phone_number": phone},
    ):
        result = rag_app.invoke(
            {"query": user_query, "audio_path": audio_path, "session_id": session_id, "phone_number": phone},
            config=config,
        )
        try:
            lf = get_client()
            if lf:
                lf.score_current_trace(
                    name="response_confidence",
                    value=round(result.get("response_confidence", 0.5), 4),
                    comment=f"intent={result.get('intent', 'unknown')}",
                )
                lf.score_current_trace(
                    name="emotion_confidence",
                    value=round(result.get("emotion_confidence", 0.0), 4),
                    comment=f"emotion={result.get('emotion', 'neutral')}",
                )
                lf.score_current_trace(
                    name="was_escalated",
                    value=1.0 if result.get("should_escalate") else 0.0,
                    comment=result.get("escalation_reason", "") or "no escalation",
                )
                lf.score_current_trace(
                    name="retrieved_docs_count",
                    value=float(len(result.get("retrieved_docs", []))),
                    comment="chunks retrieved from Pinecone",
                )
                lf.flush()
        except Exception:
            pass

    return result


st.title("Telecom AI Support")
st.caption(f"Session ID: {st.session_state.thread_id}")

with st.sidebar:
    st.header("Session")
    phone_input = st.text_input(
        "Phone Number",
        value=st.session_state.phone_number or "",
        placeholder="+94 7X XXX XXXX",
    )
    if phone_input and phone_input != st.session_state.phone_number:
        st.session_state.phone_number = phone_input.strip()
        st.success("Phone number saved for this session.")

    st.divider()
    st.header("Last Call Summary")
    result = st.session_state.last_result
    if result:
        st.metric("Confidence", f"{result.get('response_confidence', 0):.0%}")
        st.metric("Emotion", result.get("emotion", "neutral").capitalize())
        st.metric("Intent", result.get("intent", "unknown").replace("_", " ").title())
        latency = result.get("latency_ms", {})
        if latency:
            total = sum(latency.values())
            st.metric("Total Latency", f"{round(total)} ms")
        if result.get("should_escalate"):
            st.warning(f"Escalated: {result.get('escalation_reason', '')}")
            if result.get("handoff_uuid"):
                st.caption(f"Handoff ID: {result['handoff_uuid']}")
    else:
        st.caption("No interaction yet.")

    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am your Telecom Support Agent. Click the mic to speak or type below."}
        ]
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_result = {}
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].caption(f"Emotion: {meta.get('emotion', 'neutral')}")
            cols[1].caption(f"Intent: {meta.get('intent', 'unknown')}")
            cols[2].caption(f"Confidence: {meta.get('response_confidence', 0):.0%}")

_pulsing_circle_html = """
<div style="display:flex;justify-content:center;align-items:center;padding:20px;">
    <div style="width:80px;height:80px;background-color:#4F46E5;border-radius:50%;
                animation:pulse 1.2s infinite;display:flex;justify-content:center;
                align-items:center;color:white;font-size:30px;">mic</div>
</div>
<style>
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(79,70,229,0.7); transform: scale(0.95); }
    70%  { box-shadow: 0 0 0 25px rgba(79,70,229,0); transform: scale(1); }
    100% { box-shadow: 0 0 0 0 rgba(79,70,229,0); transform: scale(0.95); }
}
</style>
<p style="text-align:center;color:#4F46E5;font-weight:bold;">Listening...</p>
"""

col1, col2, col3 = st.columns([1, 1, 1])
animation_placeholder = st.empty()

with col2:
    if st.button("Tap to Speak", use_container_width=True):
        animation_placeholder.markdown(_pulsing_circle_html, unsafe_allow_html=True)
        time.sleep(0.1)
        user_query = voice_handler.listen()
        animation_placeholder.empty()

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    result = _invoke_pipeline(user_query)
                    final_answer = result.get("final_answer", "")
                    st.session_state.last_result = result
                    st.markdown(final_answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "meta": result,
                    })
                    tts_engine.speak(final_answer)
            st.rerun()

text_query = st.chat_input("Or type your message here...")
if text_query:
    st.session_state.messages.append({"role": "user", "content": text_query})
    with st.chat_message("user"):
        st.markdown(text_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = _invoke_pipeline(text_query)
            final_answer = result.get("final_answer", "")
            st.session_state.last_result = result
            st.markdown(final_answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "meta": result,
            })
            tts_engine.speak(final_answer)
    st.rerun()
