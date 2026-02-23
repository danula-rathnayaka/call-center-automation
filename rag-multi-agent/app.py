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

st.set_page_config(page_title="Telecom AI Agent", page_icon="🎧", layout="centered")


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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am your Telecom Support Agent. Click the mic to speak or type below."}
    ]

st.title("🎧 Telecom AI Support")
st.caption(f"Session ID: {st.session_state.thread_id}")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pulsing_circle_html = """
<div style="display: flex; justify-content: center; align-items: center; padding: 20px;">
    <div style="
        width: 80px; 
        height: 80px; 
        background-color: #4CAF50; 
        border-radius: 50%; 
        animation: pulse 1.2s infinite;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 30px;
    ">🎙️</div>
</div>
<style>
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); transform: scale(0.95); }
    70% { box-shadow: 0 0 0 25px rgba(76, 175, 80, 0); transform: scale(1); }
    100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); transform: scale(0.95); }
}
</style>
<p style="text-align: center; color: #4CAF50; font-weight: bold;">Listening...</p>
"""

col1, col2, col3 = st.columns([1, 1, 1])

animation_placeholder = st.empty()

with col2:
    if st.button("🎙️ Tap to Speak", use_container_width=True):
        animation_placeholder.markdown(pulsing_circle_html, unsafe_allow_html=True)

        time.sleep(0.1)

        user_query = voice_handler.listen()

        animation_placeholder.empty()

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = rag_app.invoke({"query": user_query}, config=config)
                    final_answer = result["final_answer"]

                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                    tts_engine.speak(final_answer)

                    st.rerun()

text_query = st.chat_input("Or type your message here...")
if text_query:
    st.session_state.messages.append({"role": "user", "content": text_query})
    with st.chat_message("user"):
        st.markdown(text_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = rag_app.invoke({"query": text_query}, config=config)
            final_answer = result["final_answer"]

            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            tts_engine.speak(final_answer)

            st.rerun()
