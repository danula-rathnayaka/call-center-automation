import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

class Contextualizer:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        with open(os.path.join(_prompts_dir, "contextualizer_prompt.txt"), "r", encoding="utf-8") as f:
            template_text = f.read()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template_text),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def reformulate(self, query: str, history: list) -> str:
        if not history:
            return query

        new_query = self.chain.invoke({
            "chat_history": history,
            "input": query
        })
        logger.info(f"Query reformulated. Original: '{query}' -> Reformulated: '{new_query}'")
        return new_query
