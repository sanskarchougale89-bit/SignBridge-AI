"""
ISL Pretrained Recognizer — wraps the Abs6187/isl-models Random Forest model.
Input: A single hand's MediaPipe 21 landmarks (x, y only = 42 features).
Output: Predicted ISL letter/digit + confidence.
"""
import numpy as np
import warnings
import os

class ISLPretrainedRecognizer:
    """
    Real-time ISL alphabet recognizer using a pretrained RandomForest.
    Recognizes: A-Z and 1-9 (35 classes).
    Input per frame: 21 hand landmarks → flattened (x, y) → 42 features.
    """

    LABELS = [
        '1','2','3','4','5','6','7','8','9',
        'A','B','C','D','E','F','G','H','I','J','K',
        'L','M','N','O','P','Q','R','S','T','U','V',
        'W','X','Y','Z'
    ]

    def __init__(self, model_path: str = "models/isl_pretrained/isl_rf_model.pkl"):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[ISLPretrainedRecognizer] Model not found at {self.model_path}")
            return
        try:
            import joblib
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(self.model_path)
            print(f"[ISLPretrainedRecognizer] Loaded RF model — {len(self.LABELS)} classes")
        except Exception as e:
            print(f"[ISLPretrainedRecognizer] Failed to load model: {e}")

    def _extract_features(self, landmarks: list) -> np.ndarray:
        """
        Convert 21 MediaPipe landmarks to 42-feature vector (x, y only).
        Normalizes wrist-relative, same spirit as training.
        """
        if not landmarks or len(landmarks) < 21:
            return None

        coords = np.array([[lm["x"], lm["y"]] for lm in landmarks])  # (21, 2)

        # Wrist-relative normalization
        wrist = coords[0]
        coords = coords - wrist

        # Scale by wrist→middle-finger-mcp distance
        scale = np.linalg.norm(coords[9])
        if scale > 1e-6:
            coords = coords / scale

        return coords.flatten()  # 42 features

    def predict(self, hands: list) -> dict:
        """
        Given a list of detected hands (from LandmarkExtractor output),
        predict the ISL letter/digit being shown.

        Returns dict:
          {
            "detected": bool,
            "sign": str or None,
            "confidence": float,
            "source": "ISL_PRETRAINED"
          }
        """
        if self.model is None:
            return {"detected": False, "sign": None, "confidence": 0.0, "source": "ISL_PRETRAINED"}

        # Use the first detected hand
        if not hands:
            return {"detected": False, "sign": None, "confidence": 0.0, "source": "ISL_PRETRAINED"}

        primary_hand = hands[0]
        landmarks = primary_hand.get("landmarks", [])

        features = self._extract_features(landmarks)
        if features is None:
            return {"detected": False, "sign": None, "confidence": 0.0, "source": "ISL_PRETRAINED"}

        try:
            # Predict with probabilities
            proba = self.model.predict_proba([features])[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            predicted_class = self.model.classes_[best_idx]

            # Apply confidence threshold
            if confidence < 0.50:
                return {
                    "detected": True,
                    "sign": None,
                    "confidence": confidence,
                    "source": "ISL_PRETRAINED",
                    "message": "Please try again"
                }

            return {
                "detected": True,
                "sign": str(predicted_class),
                "confidence": confidence,
                "source": "ISL_PRETRAINED"
            }
        except Exception as e:
            return {"detected": False, "sign": None, "confidence": 0.0, "source": "ISL_PRETRAINED"}
