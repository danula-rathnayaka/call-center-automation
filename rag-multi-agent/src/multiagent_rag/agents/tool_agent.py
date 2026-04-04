from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from multiagent_rag.tools.crm_tools import get_dynamic_tools
from multiagent_rag.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class ToolAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a highly capable Customer Service Action Bot. "
                       "You have access to a dynamic set of CRM tools to interact with company systems. "
                       "If you need more info to execute a tool (like a phone number or NIC), ASK the user directly. "
                       "Once the tool executes, summarize the returned data naturally and clearly for the customer."),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
        ])

    from langfuse import observe
    @observe(as_type="generation")
    def invoke(self, query: str, history: list):
        current_tools = get_dynamic_tools()

        if not current_tools:
            logger.warning("No tools are currently registered in the system.")
            return AIMessage(content="I currently don't have any system tools configured to help with that request.")

        llm_with_tools = self.llm.bind_tools(current_tools)
        chain = self.prompt | llm_with_tools

        try:
            return chain.invoke({
                "query": query,
                "chat_history": history
            })
        except Exception as e:
            logger.error(f"Tool agent invocation failed: {str(e)}")
            return AIMessage(content="I'm sorry, I encountered a system error while trying to process your request.")
