from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langfuse import observe

from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.prompt_manager import get_prompt

logger = get_logger(__name__)


class Guardrail:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=10)
        template_text = get_prompt("guardrail_prompt", "guardrail_prompt.txt")
        self.prompt = PromptTemplate.from_template(template_text)
        self.chain = self.prompt | self.llm | StrOutputParser()

    @observe()
    def validate(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {"safe": False, "reason": "I didn't catch that. Could you please repeat your question?", "block_type": "empty"}

        if len(query) > 2000:
            return {"safe": False, "reason": "Your message is too long. Could you please keep it shorter?", "block_type": "length"}

        try:
            response = self.chain.invoke({"query": query}).strip().upper()
            if "UNSAFE" in response:
                logger.warning(f"Guardrail blocked query: {query[:80]}")
                return {
                    "safe": False,
                    "reason": "I can only assist with company-related inquiries, support requests, or our services. Could you please rephrase how I can help you regarding the company?",
                    "block_type": "off_topic",
                }
        except Exception as e:
            logger.error(f"Guardrail LLM check failed (letting it pass): {e}")

        return {"safe": True, "reason": "", "block_type": None}

    @observe()
    def sanitize_response(self, response: str) -> str:
        return response
