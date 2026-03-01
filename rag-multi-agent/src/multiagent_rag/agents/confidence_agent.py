import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceAgent:
    """
    Integration agent for the Response Confidence Model.

    This agent evaluates the confidence of a generated response by calling
    the confidence model's inference pipeline. It determines whether the
    response is reliable or should be escalated to a human agent.

    Expected inference pipeline location:
        confidence-model/inference.py -> predict(query, response, context) -> dict

    To integrate the real model:
        Update the _call_inference_pipeline() method to import and call
        the actual inference function from the confidence-model component.
    """

    # Confidence threshold below which the system recommends escalation
    ESCALATION_THRESHOLD = 0.4

    def __init__(self):
        self._pipeline_available = self._check_pipeline()

    def _check_pipeline(self) -> bool:
        """Check if the real confidence model inference pipeline is available."""
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
                "predict(query, response, context) -> dict"
            )
            return False

    def evaluate(self, query: str, response: str, context: str) -> dict:
        """
        Evaluate the confidence of a generated response.

        Args:
            query: The original user query
            response: The generated response text
            context: The retrieved context used for generation

        Returns:
            dict with keys:
                - confidence_score (float): Score between 0.0 and 1.0
                - should_escalate (bool): True if confidence is below threshold
                - reason (str): Brief explanation of the score
        """
        try:
            if self._pipeline_available:
                return self._call_inference_pipeline(query, response, context)
            else:
                return self._fallback_evaluate(query, response, context)
        except Exception as e:
            logger.error(f"Confidence evaluation failed: {str(e)}")
            return {
                "confidence_score": 0.5,
                "should_escalate": False,
                "reason": "Evaluation error - defaulting to moderate confidence"
            }

    def _call_inference_pipeline(self, query: str, response: str, context: str) -> dict:
        """
        Call the real confidence model inference pipeline.

        When the confidence-model team provides their inference.py,
        this method will use it automatically.
        """
        try:
            result = self._predict_fn(query, response, context)
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
            return self._fallback_evaluate(query, response, context)

    def _fallback_evaluate(self, query: str, response: str, context: str) -> dict:
        """
        Heuristic-based fallback confidence evaluation.
        Used when the real model is not yet available.
        """
        score = 0.5  # Base score

        # Factor 1: Response length - very short responses are suspicious
        if len(response) < 20:
            score -= 0.2
        elif len(response) > 50:
            score += 0.1

        # Factor 2: "I don't know" type responses indicate low confidence
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

        # Factor 3: Context availability
        if not context or len(context.strip()) < 10:
            score -= 0.2
        else:
            score += 0.1

        # Factor 4: Query-response keyword overlap (basic relevance check)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        if overlap > 0:
            score += min(0.15, overlap * 0.05)

        # Clamp score to [0, 1]
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
