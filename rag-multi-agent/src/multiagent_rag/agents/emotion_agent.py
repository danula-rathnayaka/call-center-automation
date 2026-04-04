import importlib.util
import os
import sys

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionAgent:

    def __init__(self):
        self._model = False
        self._predict_fn = None
        self._model_path = None
        self._load_model()

    def _load_model(self):
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            emotion_model_dir = os.path.join(project_root, "emotion-model")
            model_path = os.path.join(emotion_model_dir, "models", "emotion_model_final_v2.keras")

            if not os.path.exists(model_path):
                logger.warning(f"Emotion model file not found at {model_path}. Using keyword fallback.")
                return

            if emotion_model_dir not in sys.path:
                sys.path.insert(0, emotion_model_dir)

            spec = importlib.util.spec_from_file_location("emotion_model_main",
                os.path.join(emotion_model_dir, "main.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self._predict_fn = module.predict_emotion
            self._model_path = model_path
            self._model = True
            logger.info("Emotion model loaded successfully")

        except Exception as e:
            logger.warning(f"Emotion model not available ({e}). Using keyword fallback.")
            self._model = False
            self._predict_fn = None

    from langfuse import observe
    @observe(as_type="generation")
    def detect_from_audio(self, audio_path: str) -> dict:
        if self._model and self._predict_fn and audio_path:
            try:
                result = self._run_prediction(audio_path)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Emotion detection from audio failed: {e}")
        return {"emotion": "neutral", "confidence": 0.0}

    from langfuse import observe
    @observe(as_type="generation")
    def detect_from_text(self, text: str) -> dict:
        return self._keyword_fallback(text)

    def _run_prediction(self, audio_path: str) -> dict:
        import io
        from contextlib import redirect_stdout

        captured = io.StringIO()
        with redirect_stdout(captured):
            self._predict_fn(audio_path, self._model_path)

        output = captured.getvalue()

        classes = ["Angry", "Sad", "Neutral", "Happy"]
        probs = {}
        best_class = "Neutral"
        best_prob = 0.0

        for line in output.splitlines():
            for cls in classes:
                if line.strip().startswith(cls):
                    try:
                        prob = float(line.split(":")[1].strip().replace("%", "")) / 100.0
                        probs[cls] = prob
                        if prob > best_prob:
                            best_prob = prob
                            best_class = cls
                    except (IndexError, ValueError):
                        pass

        emotion_map = {"Angry": "angry", "Sad": "sad", "Neutral": "neutral", "Happy": "happy", }

        return {"emotion": emotion_map.get(best_class, "neutral"), "confidence": round(best_prob, 3), }

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
