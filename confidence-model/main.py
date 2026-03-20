from features.confidence_feature_extractor import ConfidenceFeatureExtractor
from sentence_transformers import SentenceTransformer

import numpy as np
from xgboost import XGBClassifier
import re
import pickle

class ConfidentModel:
    def __init__(self):
        self.confidentExtractor = ConfidenceFeatureExtractor()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = XGBClassifier()
        self.model.load_model('model/best_xgb.json')
        self.threshold = 0.7

        with open("model/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

    

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text


    def predict_confident_level(self, text: str):
        """Predict confidence level for a given text."""
        text = self.clean_text(text)

        # Generate embedding
        emb = self.embedder.encode([text])

        ling = self.confidentExtractor.extract_features(text)
        ling_vec = np.array(list(ling.values())).reshape(1, -1)
        ling_vec_scaled = self.scaler.transform(ling_vec)

        # Combine features
        X_input = np.hstack([emb, ling_vec_scaled])

        # Predict
        prob = self.model.predict_proba(X_input)[0][1]
        label = "high" if prob >= self.threshold else "low"

        return {
            "confidence_label": label,
            "confidence_score": float(prob),
            "linguistic_features": ling
        }
 


cm = ConfidentModel()
print(cm.predict_confident_level("Um, maybe we could try that, I guess?"))