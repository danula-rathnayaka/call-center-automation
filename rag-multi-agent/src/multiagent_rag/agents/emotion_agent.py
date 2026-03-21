import importlib.util
import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionAgent:

    def __init__(self):
        self._model = self._load_model()

    def _load_model(self):
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            emotion_model_dir = os.path.join(project_root, "emotion-model")

            if emotion_model_dir not in sys.path:
                sys.path.insert(0, emotion_model_dir)

            spec = importlib.util.spec_from_file_location("emotion_model_main",
                os.path.join(emotion_model_dir, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._detect_fn = module.detect_emotion
            logger.info("Emotion model loaded from emotion-model/main.py")
            return True
        except (ImportError, ModuleNotFoundError, Exception) as e:
            logger.warning(f"Emotion model not available ({e}). Using keyword fallback.")
            self._detect_fn = None
            return False

    def detect_from_audio(self, audio_path: str) -> dict:
        try:
            if self._model and self._detect_fn is not None:
                result = self._detect_fn(audio_path)
                if result is not None:
                    return self._normalize_result(result)
            logger.info("Emotion model not implemented — returning neutral for audio")
            return {"emotion": "neutral", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Emotion detection from audio failed: {e}")
            return {"emotion": "neutral", "confidence": 0.0}

    def detect_from_text(self, text: str) -> dict:
        try:
            if self._model and self._detect_fn is not None:
                result = self._detect_fn(text)
                if result is not None:
                    return self._normalize_result(result)
            return self._keyword_fallback(text)
        except Exception as e:
            logger.error(f"Emotion detection from text failed: {e}")
            return self._keyword_fallback(text)

    def _normalize_result(self, result) -> dict:
        if isinstance(result, dict):
            return {"emotion": result.get("emotion", "neutral"), "confidence": float(result.get("confidence", 0.0))}
        return {"emotion": str(result), "confidence": 1.0}

    def _keyword_fallback(self, text: str) -> dict:
        text_lower = text.lower()
        emotion_keywords = {
            "angry": ["angry", "furious", "outraged", "terrible", "worst", "hate", "ridiculous", "unacceptable",
                      "disgusting", "awful"],
            "frustrated": ["frustrated", "annoying", "irritating", "stuck", "can't", "won't work", "not working",
                           "broken", "useless", "waste", "still", "again", "keep"],
            "happy": ["thank", "thanks", "great", "awesome", "excellent", "perfect", "amazing", "love", "wonderful",
                      "appreciate", "helpful", "solved"],
            "sad": ["sad", "disappointed", "unfortunately", "sorry", "lost", "miss", "upset", "unhappy", "regret"],
            "worried": ["worried", "concerned", "afraid", "scared", "anxious", "urgent", "emergency", "help",
                        "please"], }
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:
                return {"emotion": emotion, "confidence": min(0.7, matches * 0.2)}
            elif matches == 1:
                return {"emotion": emotion, "confidence": 0.3}
        return {"emotion": "neutral", "confidence": 0.5}
