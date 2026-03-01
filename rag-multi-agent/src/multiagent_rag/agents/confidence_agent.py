import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceAgent:

    ESCALATION_THRESHOLD = 0.4

    def __init__(self):
        self._pipeline_available = self._check_pipeline()

    def _check_pipeline(self) -> bool:
        try:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            confidence_model_dir = os.path.join(project_root, "confidence-model")

            if confidence_model_dir not in sys.path:
                sys.path.insert(0, confidence_model_dir)

            from inference import predict as confidence_predict
            self._predict_fn = confidence_predict
            logger.info("Confidence model inference pipeline loaded successfully")
            return True
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "Confidence model inference pipeline not available. Using fallback. "
                "To enable: create confidence-model/inference.py with "
                "predict(query, response, retrieved_chunks, emotion) -> dict"
            )
            return False

    def evaluate(self, query: str, response: str, retrieved_chunks: list, emotion: str) -> dict:
        try:
            if self._pipeline_available:
                return self._call_inference_pipeline(query, response, retrieved_chunks, emotion)
            else:
                return self._fallback_evaluate(query, response, retrieved_chunks, emotion)
        except Exception as e:
            logger.error(f"Confidence evaluation failed: {str(e)}")
            return {
                "confidence_score": 0.5,
                "should_escalate": False,
                "reason": "Evaluation error - defaulting to moderate confidence"
            }

    def _call_inference_pipeline(self, query, response, retrieved_chunks, emotion) -> dict:
        try:
            result = self._predict_fn(query, response, retrieved_chunks, emotion)
            if isinstance(result, dict):
                score = float(result.get("confidence_score", 0.5))
            else:
                score = float(result)

            return {
                "confidence_score": score,
                "should_escalate": score < self.ESCALATION_THRESHOLD,
                "reason": result.get("reason", "") if isinstance(result, dict) else ""
            }
        except Exception as e:
            logger.error(f"Confidence inference pipeline error: {str(e)}")
            return self._fallback_evaluate(query, response, retrieved_chunks, emotion)

    def _fallback_evaluate(self, query, response, retrieved_chunks, emotion) -> dict:
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

        if not retrieved_chunks or len(retrieved_chunks) == 0:
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

        reason = "Heuristic evaluation"
        if should_escalate:
            reason = "Low confidence - consider human agent handoff"

        return {
            "confidence_score": round(score, 3),
            "should_escalate": should_escalate,
            "reason": reason
        }
