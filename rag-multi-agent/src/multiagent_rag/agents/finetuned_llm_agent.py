import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class FinetunedLLMAgent:
    """
    Integration agent for the Fine-tuned LLM.

    This agent wraps the fine-tuned LLM's inference pipeline for generating
    customer service responses. When not available, it falls back to the
    existing Groq-based Generator.

    Expected inference pipeline location:
        finetuned-LLM/inference.py -> generate(query, context, history) -> str

    To integrate the real model:
        Update the _call_inference_pipeline() method to import and call
        the actual inference function from the finetuned-LLM component.
    """

    def __init__(self):
        self._pipeline_available = self._check_pipeline()
        self._fallback_generator = None

    def _check_pipeline(self) -> bool:
        """Check if the fine-tuned LLM inference pipeline is available."""
        try:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            finetuned_llm_dir = os.path.join(project_root, "finetuned-LLM")

            if finetuned_llm_dir not in sys.path:
                sys.path.insert(0, finetuned_llm_dir)

            from inference import generate as finetuned_generate
            self._generate_fn = finetuned_generate
            logger.info("Fine-tuned LLM inference pipeline loaded successfully")
            return True
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "Fine-tuned LLM inference pipeline not available. Using Groq fallback. "
                "To enable: create finetuned-LLM/inference.py with "
                "generate(query, context, history) -> str"
            )
            return False

    def _get_fallback_generator(self):
        """Lazy-load the fallback Generator to avoid circular imports."""
        if self._fallback_generator is None:
            from multiagent_rag.agents.generator import Generator
            self._fallback_generator = Generator()
        return self._fallback_generator

    def generate(self, query: str, context: str, history: list) -> str:
        """
        Generate a response using the fine-tuned LLM.

        Args:
            query: The user's question
            context: Retrieved knowledge context
            history: Conversation history (list of BaseMessage)

        Returns:
            str: The generated response text
        """
        try:
            if self._pipeline_available:
                return self._call_inference_pipeline(query, context, history)
            else:
                return self._fallback_generate(query, context, history)
        except Exception as e:
            logger.error(f"Fine-tuned LLM generation failed: {str(e)}")
            return self._fallback_generate(query, context, history)

    def _call_inference_pipeline(self, query: str, context: str, history: list) -> str:
        """
        Call the real fine-tuned LLM inference pipeline.

        When the finetuned-LLM team provides their inference.py,
        this method will use it automatically.
        """
        try:
            # Convert LangChain message history to a simple format for the inference pipeline
            simple_history = []
            for msg in history:
                role = "user" if msg.type == "human" else "assistant"
                simple_history.append({"role": role, "content": msg.content})

            result = self._generate_fn(query, context, simple_history)
            return str(result)
        except Exception as e:
            logger.error(f"Fine-tuned LLM inference pipeline error: {str(e)}")
            return self._fallback_generate(query, context, history)

    def _fallback_generate(self, query: str, context: str, history: list) -> str:
        """
        Fallback: use the existing Groq-based Generator.
        This ensures the system always works even without the fine-tuned model.
        """
        logger.info("Using Groq Generator as fallback for fine-tuned LLM")
        generator = self._get_fallback_generator()
        return generator.generate(query, context, history)
