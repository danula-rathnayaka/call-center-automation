import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionAgent:
    """
    Integration agent for the Emotion Detection Model.

    This agent wraps the emotion model's inference pipeline.
    Currently uses a fallback implementation until the real model is available.

    Expected inference pipeline location:
        emotion-model/inference.py -> predict(text: str) -> dict

    To integrate the real model:
        Update the _call_inference_pipeline() method to import and call
        the actual inference function from the emotion-model component.
    """

    def __init__(self):
        self._pipeline_available = self._check_pipeline()

    def _check_pipeline(self) -> bool:
        """Check if the real emotion model inference pipeline is available."""
        try:
            # Resolve path to emotion-model directory (sibling of rag-multi-agent)
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            emotion_model_dir = os.path.join(project_root, "emotion-model")

            if emotion_model_dir not in sys.path:
                sys.path.insert(0, emotion_model_dir)

            # Try to import the inference module
            from inference import predict as emotion_predict
            self._predict_fn = emotion_predict
            logger.info("Emotion model inference pipeline loaded successfully")
            return True
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "Emotion model inference pipeline not available. Using fallback. "
                "To enable: create emotion-model/inference.py with predict(text) -> dict"
            )
            return False

    def detect(self, text: str) -> dict:
        """
        Detect the emotion in the given text.

        Args:
            text: The input text to analyze

        Returns:
            dict with keys:
                - emotion (str): Detected emotion (e.g., "angry", "happy", "neutral", "frustrated", "sad")
                - confidence (float): Confidence score between 0.0 and 1.0
        """
        try:
            if self._pipeline_available:
                return self._call_inference_pipeline(text)
            else:
                return self._fallback_detect(text)
        except Exception as e:
            logger.error(f"Emotion detection failed: {str(e)}")
            return {"emotion": "neutral", "confidence": 0.0}

    def _call_inference_pipeline(self, text: str) -> dict:
        """
        Call the real emotion model inference pipeline.

        When the emotion-model team provides their inference.py,
        this method will use it automatically.
        """
        try:
            result = self._predict_fn(text)
            # Normalize the result to expected format
            if isinstance(result, dict):
                return {
                    "emotion": result.get("emotion", "neutral"),
                    "confidence": float(result.get("confidence", 0.0))
                }
            else:
                return {"emotion": str(result), "confidence": 1.0}
        except Exception as e:
            logger.error(f"Emotion inference pipeline error: {str(e)}")
            return self._fallback_detect(text)

    def _fallback_detect(self, text: str) -> dict:
        """
        Simple keyword-based fallback emotion detection.
        Used when the real model is not yet available.
        """
        text_lower = text.lower()

        emotion_keywords = {
            "angry": ["angry", "furious", "outraged", "terrible", "worst", "hate", "ridiculous",
                       "unacceptable", "disgusting", "awful"],
            "frustrated": ["frustrated", "annoying", "irritating", "stuck", "can't", "won't work",
                           "not working", "broken", "useless", "waste", "still", "again", "keep"],
            "happy": ["thank", "thanks", "great", "awesome", "excellent", "perfect", "amazing",
                       "love", "wonderful", "appreciate", "helpful", "solved"],
            "sad": ["sad", "disappointed", "unfortunately", "sorry", "lost", "miss", "upset",
                     "unhappy", "regret"],
            "worried": ["worried", "concerned", "afraid", "scared", "anxious", "urgent",
                        "emergency", "help", "please"],
        }

        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:
                return {"emotion": emotion, "confidence": min(0.7, matches * 0.2)}
            elif matches == 1:
                return {"emotion": emotion, "confidence": 0.3}

        return {"emotion": "neutral", "confidence": 0.5}
