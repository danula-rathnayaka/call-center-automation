import os
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from multiagent_rag.tools.crm_tools import crm_tools
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

class ToolAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
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
            logger.error(f"Tool agent invocation failed: {str(e)}")
            return AIMessage(content="I'm sorry, I encountered a system error. Please try again.")