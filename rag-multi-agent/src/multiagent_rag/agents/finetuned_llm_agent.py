import os
import sys
from typing import List, Optional

from langchain_core.messages import BaseMessage

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class FinetunedLLMAgent:

    def __init__(self):
        self._pipeline_ready = self._load_pipeline()
        self._fallback_generator = None

    def _load_pipeline(self) -> bool:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            finetuned_llm_dir = os.path.join(project_root, "finetuned-LLM")
            pipeline_path = os.path.join(finetuned_llm_dir, "inference_pipeline.py")

            if not os.path.exists(pipeline_path):
                logger.warning("finetuned-LLM/inference_pipeline.py not found. Using Groq fallback.")
                return False

            if finetuned_llm_dir not in sys.path:
                sys.path.insert(0, finetuned_llm_dir)

            import inference_pipeline as _pipeline
            _pipeline.initialize()

            self._generate_fn = _pipeline.generate_response
            logger.info("Fine-tuned LLM (HuggingFace Inference API) ready")
            return True

        except Exception as e:
            logger.warning(f"Fine-tuned LLM could not be loaded ({e}). Using Groq fallback.")
            return False

    def _get_fallback_generator(self):
        if self._fallback_generator is None:
            from multiagent_rag.agents.generator import Generator
            self._fallback_generator = Generator()
        return self._fallback_generator

    from langfuse import observe
    @observe(as_type="generation")
    def generate(self, query: str, context: str, emotion: str, history: List[BaseMessage],
            summary: Optional[str] = None, ) -> str:
        if self._pipeline_ready:
            try:
                return self._call_finetuned_llm(query, context, emotion)
            except Exception as e:
                logger.error(f"Fine-tuned LLM generation failed, falling back to Groq: {e}")

        return self._groq_fallback(query, context, emotion, history, summary)

    def _call_finetuned_llm(self, query: str, context: str, emotion: str) -> str:
        response = self._generate_fn(customer_query=query, facts=context, emotion=emotion, max_new_tokens=200, )
        return str(response).strip()

    def _groq_fallback(self, query: str, context: str, emotion: str, history: List[BaseMessage],
            summary: Optional[str] = None, ) -> str:
        logger.info(f"Using Groq Generator as fallback (emotion: {emotion})")
        generator = self._get_fallback_generator()
        emotion_prefix = self._emotion_instruction(emotion)
        enriched_context = f"{emotion_prefix}\n\n{context}"
        return generator.generate(query, enriched_context, history, summary)

    def _emotion_instruction(self, emotion: str) -> str:
        instructions = {"angry": ("[CUSTOMER EMOTION: Angry] "
                                  "The customer is upset. Respond with extra empathy, acknowledge their frustration, "
                                  "apologize sincerely, and provide a clear solution. Use a calm, reassuring tone."),
            "frustrated": ("[CUSTOMER EMOTION: Frustrated] "
                           "The customer is frustrated. Show understanding, validate their experience, "
                           "and focus on resolving their issue step by step. Be patient and supportive."),
            "happy": ("[CUSTOMER EMOTION: Happy] "
                      "The customer is in a positive mood. Match their energy, be warm and friendly, "
                      "and ensure they leave even more satisfied."), "sad": ("[CUSTOMER EMOTION: Sad] "
                                                                             "The customer seems disappointed. Show genuine empathy, be gentle in your response, "
                                                                             "and focus on how you can help improve their situation."),
            "worried": ("[CUSTOMER EMOTION: Worried] "
                        "The customer is anxious or concerned. Reassure them, provide clear information, "
                        "and help them feel confident that their issue will be resolved."),
            "neutral": ("[CUSTOMER EMOTION: Neutral] "
                        "The customer's tone is neutral. Respond professionally and helpfully."), }
        return instructions.get(emotion, instructions["neutral"])
