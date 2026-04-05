import os
from functools import lru_cache

from multiagent_rag.utils.logger import get_logger
from multiagent_rag.utils.telemetry import get_langfuse_client

logger = get_logger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def _read_local(filename: str) -> str:
    path = os.path.join(_PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Local prompt fallback not found: {path}")
        return ""


@lru_cache(maxsize=None)
def get_prompt(name: str, filename: str) -> str:
    local_text = _read_local(filename)
    client = get_langfuse_client()
    if not client:
        logger.warning(f"Langfuse unavailable - using local file for prompt '{name}'")
        return local_text
    try:
        lf_prompt = client.get_prompt(name, type="text", fallback=local_text)
        text = lf_prompt.compile()
        logger.info(f"Loaded prompt '{name}' from Langfuse (version={getattr(lf_prompt, 'version', '?')})")
        return text
    except Exception as e:
        logger.warning(f"Langfuse prompt '{name}' failed ({e}) - using local fallback")
        return local_text


def get_prompt_template(name: str, filename: str) -> str:
    local_text = _read_local(filename)
    client = get_langfuse_client()
    if not client:
        logger.warning(f"Langfuse unavailable - using local file for prompt '{name}'")
        return local_text
    try:
        lf_prompt = client.get_prompt(name, type="text", fallback=local_text)
        text = lf_prompt.get_langchain_prompt()
        logger.info(f"Loaded langchain prompt '{name}' from Langfuse (version={getattr(lf_prompt, 'version', '?')})")
        return text
    except Exception as e:
        logger.warning(f"Langfuse prompt '{name}' failed ({e}) - using local fallback")
        return local_text


def invalidate_cache():
    get_prompt.cache_clear()
    logger.info("Prompt cache cleared - next calls will re-fetch from Langfuse")
