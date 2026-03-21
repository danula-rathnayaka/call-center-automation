import os
from typing import List, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class Contextualizer:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        with open(os.path.join(_prompts_dir, "contextualizer_prompt.txt"), "r", encoding="utf-8") as f:
            template_text = f.read()

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", template_text), ("placeholder", "{chat_history}"), ("human", "{input}"), ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception), reraise=False, )
    def _invoke_with_retry(self, payload: dict) -> str:
        return self.chain.invoke(payload)

    def reformulate(self, query: str, history: List[BaseMessage], summary: Optional[str] = None, ) -> str:
        if not history and not summary:
            return query

        augmented_history = list(history)
        if summary:
            augmented_history = [SystemMessage(
                content=f"Summary of earlier conversation:\n{summary}")] + augmented_history

        try:
            new_query = self._invoke_with_retry({"chat_history": augmented_history, "input": query})
            logger.info(f"Query reformulated: '{query}' -> '{new_query}'")
            return new_query
        except Exception as e:
            logger.error(f"Contextualizer failed after retries: {e}")
            return query
