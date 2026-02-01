import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from multiagent_rag.tools.crm_tools import crm_tools


class ToolAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

        self.llm_with_tools = self.llm.bind_tools(crm_tools)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Customer Service Action Bot. "
                       "You have access to CRM tools to check balances and verify identities. "
                       "If you need more info (like a phone number) to call a tool, ASK the user. "
                       "Once you get the tool output, summarize it clearly."),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
        ])

    def invoke(self, query: str, history: list):
        chain = self.prompt | self.llm_with_tools

        try:
            return chain.invoke({
                "query": query,
                "chat_history": history
            })
        except Exception as e:
            print(f"[ToolAgent] Error: {e}")
            return "System Error"