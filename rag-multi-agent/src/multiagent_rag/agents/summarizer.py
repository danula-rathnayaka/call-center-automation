from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

SUMMARIZER_TEMPLATE = (
    "You are a conversation summarizer for a telecom customer support system. "
    "Condense the following conversation history into a brief summary that preserves "
    "key context: customer issues, solutions provided, account details mentioned, "
    "and any unresolved matters. Keep it under 150 words."
)


class ConversationSummarizer:

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=256,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZER_TEMPLATE),
            ("human", "Conversation to summarize:\n{conversation}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize(self, history: list, keep_recent: int = 4) -> list:
        if len(history) <= keep_recent:
            return history

        older_messages = history[:-keep_recent]
        recent_messages = history[-keep_recent:]

        try:
            conversation_text = "\n".join(
                f"{'Customer' if msg.type == 'human' else 'Agent'}: {msg.content}"
                for msg in older_messages
                if hasattr(msg, "content") and msg.content
            )

            if not conversation_text.strip():
                return history

            summary = self.chain.invoke({"conversation": conversation_text})

            summarized_history = [
                SystemMessage(content=f"Previous conversation summary: {summary}")
            ] + recent_messages

            logger.info(
                f"Summarized {len(older_messages)} older messages into summary, "
                f"keeping {len(recent_messages)} recent messages"
            )
            return summarized_history

        except Exception as e:
            logger.error(f"Conversation summarization failed: {str(e)}")
            return history
