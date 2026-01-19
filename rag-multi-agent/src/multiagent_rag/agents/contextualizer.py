import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


class Contextualizer:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        with open(os.path.join("src", "prompts", "contextualizer_prompt.txt"), "r", encoding="utf-8") as f:
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
        print(f"   Original: '{query}' -> New: '{new_query}'")
        return new_query
