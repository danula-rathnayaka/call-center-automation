import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multiagent_rag.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_MAX_RESPONSE_WORDS = 120


def _trim_to_word_limit(text: str, limit: int = _MAX_RESPONSE_WORDS) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    trimmed = " ".join(words[:limit])
    if not trimmed.endswith((".", "!", "?")):
        last_stop = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
        if last_stop > len(trimmed) // 2:
            trimmed = trimmed[:last_stop + 1]
    logger.info(f"Response trimmed from {len(words)} to {len(trimmed.split())} words for voice delivery")
    return trimmed


class Generator:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512, )

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        with open(os.path.join(_prompts_dir, "rag_prompt.txt"), "r", encoding="utf-8") as f:
            template_text = f.read()

        from multiagent_rag.utils.telemetry import get_langfuse_client
        client = get_langfuse_client()
        if client:
            try:
                lf_prompt = client.get_prompt("rag_prompt", type="text", fallback=template_text)
                template_text = lf_prompt.get_langchain_prompt()
            except Exception as e:
                logger.warning(f"Could not load rag_prompt from Langfuse: {e}")

        self.prompt = ChatPromptTemplate.from_messages([("system", template_text), ("placeholder", "{chat_history}"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")])
        self.chain = self.prompt | self.llm | StrOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception), reraise=True, )
    def _invoke_with_retry(self, payload: dict) -> str:
        return self.chain.invoke(payload)

    from langfuse import observe
    @observe(as_type="generation")
    def generate(self, query: str, context: str, history: List[BaseMessage], summary: Optional[str] = None, ) -> str:
        try:
            augmented_history = list(history)
            if summary:
                augmented_history = [SystemMessage(
                    content=f"Summary of earlier conversation:\n{summary}")] + augmented_history

            response = self._invoke_with_retry(
                {"context": context, "question": query, "chat_history": augmented_history, })
            return _trim_to_word_limit(response)
        except Exception as e:
            logger.error(f"Response generation failed after retries: {e}")
            return "I am sorry, I encountered a technical issue. Please try again in a moment."
