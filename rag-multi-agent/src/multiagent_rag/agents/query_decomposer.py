import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class QueryDecomposer:

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        with open(os.path.join(_prompts_dir, "query_decomposer_prompt.txt"), "r", encoding="utf-8") as f:
            template_text = f.read()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template_text),
            ("human", "{query}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def decompose(self, query: str) -> list:
        try:
            result = self.chain.invoke({"query": query})
            sub_queries = [
                q.strip().lstrip("0123456789.-) ")
                for q in result.strip().split("\n")
                if q.strip() and len(q.strip()) > 3
            ]

            if not sub_queries:
                return [query]

            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries

        except Exception as e:
            logger.error(f"Query decomposition failed: {str(e)}")
            return [query]
