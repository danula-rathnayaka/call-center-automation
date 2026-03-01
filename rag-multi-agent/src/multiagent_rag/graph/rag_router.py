import os

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from multiagent_rag.state.router_response_schema import RouteResponse
from multiagent_rag.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

class IntentRouter:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        self.parser = PydanticOutputParser(pydantic_object=RouteResponse)

        _prompts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
        prompt_path = os.path.join(_prompts_dir, "rag_router_prompt.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found at: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        self.prompt = ChatPromptTemplate.from_template(template_text)

        self.chain = self.prompt | self.llm | self.parser

    def route(self, query: str) -> str:
        try:
            response: RouteResponse = self.chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })

            return response.intent

        except Exception as e:
            logger.error(f"Routing failed for query: {query}. Error: {str(e)}")
            return "technical"


if __name__ == "__main__":
    router = IntentRouter()

    test_queries = [
        "My internet is very slow",
        "Good morning",
        "I want to speak to a manager!",
        "Huawei B310 red light"
    ]

    print("\n--- ROUTER TEST ---")
    for q in test_queries:
        router.route(q)
