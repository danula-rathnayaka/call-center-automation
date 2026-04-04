from typing import List, Tuple, Optional

from langchain_core.messages import BaseMessage
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

    from langfuse import observe
    @observe(as_type="generation")
    def summarize(
            self,
            history: List[BaseMessage],
            keep_recent: int = 4,
            existing_summary: Optional[str] = None,
    ) -> Tuple[List[BaseMessage], str]:
        """
        Summarizes turns older than `keep_recent` into a plain string.

        Returns:
            recent_history  — the last `keep_recent` BaseMessage objects
                              (to be stored back in state["chat_history"])
            new_summary     — a plain string summary of ALL older context,
                              folding in `existing_summary` if provided
                              (to be stored in state["conversation_summary"])

        If history is short enough, returns it unchanged with the existing summary.
        """
        if len(history) <= keep_recent:
            return history, existing_summary or ""

        older_messages = history[:-keep_recent]
        recent_messages = history[-keep_recent:]

        try:
            conversation_text = "\n".join(
                f"{'Customer' if msg.type == 'human' else 'Agent'}: {msg.content}"
                for msg in older_messages
                if hasattr(msg, "content") and msg.content
            )

            if not conversation_text.strip():
                return history, existing_summary or ""

            if existing_summary:
                conversation_text = (
                    f"[Previous summary]\n{existing_summary}\n\n"
                    f"[New turns]\n{conversation_text}"
                )

            new_summary = self.chain.invoke({"conversation": conversation_text})

            logger.info(
                f"Summarized {len(older_messages)} older messages, "
                f"keeping {len(recent_messages)} recent messages"
            )
            return recent_messages, new_summary

        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return history, existing_summary or ""
