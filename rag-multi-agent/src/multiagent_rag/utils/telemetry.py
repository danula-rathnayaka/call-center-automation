import os
import logging
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

logger = logging.getLogger(__name__)

try:
    langfuse_handler = CallbackHandler()
    langfuse_client = Langfuse()
except Exception as e:
    logger.warning(f"Langfuse could not be initialized. Please check your environment variables: {e}")
    langfuse_handler = None
    langfuse_client = None

def get_langfuse_handler():
    return langfuse_handler

def get_langfuse_client():
    return langfuse_client
