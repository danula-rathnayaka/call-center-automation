import logging
from typing import Optional

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

try:
    langfuse_client: Optional[Langfuse] = get_client()
    logger.info("Langfuse v4 client initialized")
except Exception as e:
    logger.warning(f"Langfuse could not be initialized: {e}")
    langfuse_client = None


def get_langfuse_client() -> Optional[Langfuse]:
    return langfuse_client


def get_langchain_handler() -> Optional[CallbackHandler]:
    try:
        return CallbackHandler()
    except Exception as e:
        logger.warning(f"Failed to create Langfuse LangChain handler: {e}")
        return None


def flush():
    if langfuse_client:
        try:
            langfuse_client.flush()
        except Exception:
            pass
