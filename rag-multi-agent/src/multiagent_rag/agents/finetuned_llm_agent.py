import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class FinetunedLLMAgent:

    def __init__(self):
        self._pipeline_available = self._check_pipeline()
        self._fallback_generator = None

    def _check_pipeline(self) -> bool:
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
                "generate(query, context, emotion, history) -> str"
            )
            return False

    def _get_fallback_generator(self):
        if self._fallback_generator is None:
            from multiagent_rag.agents.generator import Generator
            self._fallback_generator = Generator()
        return self._fallback_generator

    def generate(self, query: str, context: str, emotion: str, history: list) -> str:
        try:
            if self._pipeline_available:
                return self._call_inference_pipeline(query, context, emotion, history)
            else:
                return self._fallback_generate(query, context, emotion, history)
        except Exception as e:
            logger.error(f"Fine-tuned LLM generation failed: {str(e)}")
            return self._fallback_generate(query, context, emotion, history)

    def _call_inference_pipeline(self, query: str, context: str, emotion: str, history: list) -> str:
        try:
            simple_history = []
            for msg in history:
                role = "user" if msg.type == "human" else "assistant"
                simple_history.append({"role": role, "content": msg.content})

            result = self._generate_fn(query, context, emotion, simple_history)
            return str(result)
        except Exception as e:
            logger.error(f"Fine-tuned LLM inference pipeline error: {str(e)}")
            return self._fallback_generate(query, context, emotion, history)

    def _fallback_generate(self, query: str, context: str, emotion: str, history: list) -> str:
        logger.info(f"Using Groq Generator as fallback (emotion: {emotion})")
        generator = self._get_fallback_generator()

        emotion_context = self._build_emotion_context(emotion)
        enriched_context = f"{emotion_context}\n\n{context}"

        return generator.generate(query, enriched_context, history)

    def _build_emotion_context(self, emotion: str) -> str:
        emotion_instructions = {
            "angry": (
                "[CUSTOMER EMOTION: Angry] "
                "The customer is upset. Respond with extra empathy, acknowledge their frustration, "
                "apologize sincerely, and provide a clear solution. Use a calm, reassuring tone."
            ),
            "frustrated": (
                "[CUSTOMER EMOTION: Frustrated] "
                "The customer is frustrated. Show understanding, validate their experience, "
                "and focus on resolving their issue step by step. Be patient and supportive."
            ),
            "happy": (
                "[CUSTOMER EMOTION: Happy] "
                "The customer is in a positive mood. Match their energy, be warm and friendly, "
                "and ensure they leave even more satisfied."
            ),
            "sad": (
                "[CUSTOMER EMOTION: Sad] "
                "The customer seems disappointed. Show genuine empathy, be gentle in your response, "
                "and focus on how you can help improve their situation."
            ),
            "worried": (
                "[CUSTOMER EMOTION: Worried] "
                "The customer is anxious or concerned. Reassure them, provide clear information, "
                "and help them feel confident that their issue will be resolved."
            ),
            "neutral": (
                "[CUSTOMER EMOTION: Neutral] "
                "The customer's tone is neutral. Respond professionally and helpfully."
            ),
        }
        return emotion_instructions.get(emotion, emotion_instructions["neutral"])
