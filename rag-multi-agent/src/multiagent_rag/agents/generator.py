import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from multiagent_rag.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

class Generator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1024
        )

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        with open(os.path.join(_prompts_dir, "rag_prompt.txt"), "r", encoding="utf-8") as f:
            template_text = f.read()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template_text),
            ("placeholder", "{chat_history}"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, query: str, context: str, history: list) -> str:
        try:
            response = self.chain.invoke({
                "context": context,
                "question": query,
                "chat_history": history
            })
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return "System Error: Could not generate response."
