import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionAgent:

    def __init__(self):
        self._pipeline_available = self._check_pipeline()

    def _check_pipeline(self) -> bool:
        try:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            emotion_model_dir = os.path.join(project_root, "emotion-model")

            if emotion_model_dir not in sys.path:
                sys.path.insert(0, emotion_model_dir)

            from inference import predict as emotion_predict
            self._predict_fn = emotion_predict
            logger.info("Emotion model inference pipeline loaded successfully")
            return True
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "Emotion model inference pipeline not available. Using text fallback. "
                "To enable: create emotion-model/inference.py with predict(audio_path) -> dict"
            )
            return False

    def detect_from_audio(self, audio_path: str) -> dict:
        try:
            if self._pipeline_available:
                return self._call_inference_pipeline(audio_path)
            else:
                logger.info("Emotion model not available - returning neutral for audio")
                return {"emotion": "neutral", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Emotion detection from audio failed: {str(e)}")
            return {"emotion": "neutral", "confidence": 0.0}

    def detect_from_text(self, text: str) -> dict:
        try:
            if self._pipeline_available:
                try:
                    result = self._predict_fn(text)
                    return self._normalize_result(result)
                except Exception:
                    pass
            return self._keyword_fallback(text)
        except Exception as e:
            logger.error(f"Emotion detection from text failed: {str(e)}")
            return {"emotion": "neutral", "confidence": 0.0}

    def _call_inference_pipeline(self, audio_path: str) -> dict:
        try:
            result = self._predict_fn(audio_path)
            return self._normalize_result(result)
        except Exception as e:
            logger.error(f"Emotion inference pipeline error: {str(e)}")
            return {"emotion": "neutral", "confidence": 0.0}

    def _normalize_result(self, result) -> dict:
        if isinstance(result, dict):
            return {
                "emotion": result.get("emotion", "neutral"),
                "confidence": float(result.get("confidence", 0.0))
            }
        else:
            return {"emotion": str(result), "confidence": 1.0}

    def _keyword_fallback(self, text: str) -> dict:
        text_lower = text.lower()

        emotion_keywords = {
            "angry": ["angry", "furious", "outraged", "terrible", "worst", "hate",
                       "ridiculous", "unacceptable", "disgusting", "awful"],
            "frustrated": ["frustrated", "annoying", "irritating", "stuck", "can't",
                           "won't work", "not working", "broken", "useless", "waste",
                           "still", "again", "keep"],
            "happy": ["thank", "thanks", "great", "awesome", "excellent", "perfect",
                       "amazing", "love", "wonderful", "appreciate", "helpful", "solved"],
            "sad": ["sad", "disappointed", "unfortunately", "sorry", "lost", "miss",
                     "upset", "unhappy", "regret"],
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
