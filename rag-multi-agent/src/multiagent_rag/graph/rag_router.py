from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from multiagent_rag.state.router_response_schema import RouteResponse
from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.prompt_manager import get_prompt_template

load_dotenv()
logger = get_logger(__name__)


class IntentRouter:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=RouteResponse)
        template_text = get_prompt_template("rag_router_prompt", "rag_router_prompt.txt")
        self.prompt = ChatPromptTemplate.from_template(template_text)
        self.chain = self.prompt | self.llm | self.parser

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception), reraise=False)
    def _invoke_with_retry(self, payload: dict) -> RouteResponse:
        return self.chain.invoke(payload)

    def route(self, query: str, history: list = None) -> str:
        try:
            history_context = ""
            if history:
                recent = history[-4:]
                history_lines = []
                for msg in recent:
                    role = "Customer" if msg.type == "human" else "Agent"
                    history_lines.append(f"{role}: {msg.content}")
                history_context = "\n".join(history_lines)

            response: RouteResponse = self._invoke_with_retry({
                "query": query,
                "format_instructions": self.parser.get_format_instructions(),
                "history_context": history_context,
            })
            return response.intent
        except Exception as e:
            logger.error(f"Routing failed after retries for query: {query}. Error: {e}")
            return "technical"
