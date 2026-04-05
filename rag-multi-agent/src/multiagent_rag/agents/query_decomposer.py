from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse import observe
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.prompt_manager import get_prompt_template

logger = get_logger(__name__)


class QueryDecomposer:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        template_text = get_prompt_template("query_decomposer_prompt", "query_decomposer_prompt.txt")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template_text),
            ("human", "{query}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception), reraise=False)
    def _invoke_with_retry(self, query: str) -> str:
        return self.chain.invoke({"query": query})

    @observe(as_type="generation")
    def decompose(self, query: str) -> list:
        try:
            result = self._invoke_with_retry(query)
            sub_queries = [q.strip().lstrip("0123456789.-) ") for q in result.strip().split("\n")
                           if q.strip() and len(q.strip()) > 3]
            if not sub_queries:
                return [query]
            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries
        except Exception as e:
            logger.error(f"Query decomposition failed after retries: {e}")
            return [query]
