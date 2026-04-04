import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class QueryDecomposer:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        with open(os.path.join(_prompts_dir, "query_decomposer_prompt.txt"), "r", encoding="utf-8") as f:
            template_text = f.read()

        from multiagent_rag.utils.telemetry import get_langfuse_client
        client = get_langfuse_client()
        if client:
            try:
                lf_prompt = client.get_prompt("query_decomposer_prompt", type="text", fallback=template_text)
                template_text = lf_prompt.get_langchain_prompt()
            except Exception as e:
                logger.warning(f"Could not load query_decomposer_prompt from Langfuse: {e}")

        self.prompt = ChatPromptTemplate.from_messages([("system", template_text), ("human", "{query}"), ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception), reraise=False, )
    def _invoke_with_retry(self, query: str) -> str:
        return self.chain.invoke({"query": query})

    from langfuse import observe
    @observe(as_type="generation")
    def decompose(self, query: str) -> list:
        try:
            result = self._invoke_with_retry(query)
            sub_queries = [q.strip().lstrip("0123456789.-) ") for q in result.strip().split("\n") if
                q.strip() and len(q.strip()) > 3]
            if not sub_queries:
                return [query]
            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries
        except Exception as e:
            logger.error(f"Query decomposition failed after retries: {e}")
            return [query]
