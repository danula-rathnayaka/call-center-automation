import importlib.util
import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceAgent:

    ESCALATION_THRESHOLD = 0.4

    def __init__(self):
        self._model = self._load_model()

    def _load_model(self):
        try:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            confidence_model_dir = os.path.join(project_root, "confidence-model")

            if confidence_model_dir not in sys.path:
                sys.path.insert(0, confidence_model_dir)

            spec = importlib.util.spec_from_file_location(
                "confidence_model_main",
                os.path.join(confidence_model_dir, "main.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.ConfidentModel()
            logger.info("ConfidentModel loaded from confidence-model/main.py")
            return model
        except (ImportError, ModuleNotFoundError, Exception) as e:
            logger.warning(
                f"ConfidentModel could not be loaded ({str(e)}). Using heuristic fallback."
            )
            return None

    def evaluate(self, query: str, response: str, retrieved_chunks: list, emotion: str) -> dict:
        try:
            if self._model is not None:
                return self._call_model(response)
            return self._fallback_evaluate(query, response, retrieved_chunks, emotion)
        except Exception as e:
            logger.error(f"Confidence evaluation failed: {str(e)}")
            return {
                "confidence_score": 0.5,
                "should_escalate": False,
                "reason": "Evaluation error - defaulting to moderate confidence"
            }

    def _call_model(self, response: str) -> dict:
        try:
            result = self._model.predict_confident_level(response)
            score = float(result.get("confidence_score", 0.5))
            label = result.get("confidence_label", "low")

            return {
                "confidence_score": score,
                "should_escalate": score < self.ESCALATION_THRESHOLD,
                "reason": f"Model prediction: {label} confidence (score={score:.3f})",
            }
        except Exception as e:
            logger.error(f"ConfidentModel.predict_confident_level() failed: {str(e)}")
            raise

    def _fallback_evaluate(self, query: str, response: str, retrieved_chunks: list, emotion: str) -> dict:
        score = 0.5

        if len(response) < 20:
            score -= 0.2
        elif len(response) > 50:
            score += 0.1

        low_confidence_phrases = [
            "i don't have that information",
            "i'm not sure",
            "i cannot find",
            "no relevant information",
            "unable to determine",
            "system error",
            "i don't know"
        ]
        response_lower = response.lower()
        for phrase in low_confidence_phrases:
            if phrase in response_lower:
                score -= 0.3
                break

        if not retrieved_chunks:
            score -= 0.2
        else:
            total_content = sum(len(c.get("content", "")) for c in retrieved_chunks)
            if total_content > 100:
                score += 0.1

        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        if overlap > 0:
            score += min(0.15, overlap * 0.05)

        if emotion in ["angry", "frustrated"]:
            score -= 0.05

        score = max(0.0, min(1.0, score))
        should_escalate = score < self.ESCALATION_THRESHOLD

        return {
            "confidence_score": round(score, 3),
            "should_escalate": should_escalate,
            "reason": "Low confidence - consider human agent handoff" if should_escalate else "Heuristic evaluation"
        }
