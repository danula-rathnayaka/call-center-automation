import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class Generator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=1024
        )

        with open(os.path.join("src", "prompts", "rag_prompt.txt"), "r", encoding="utf-8") as f:
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
            print(f"[Generator] Error: {e}")
            return "System Error: Could not generate response."
