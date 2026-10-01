import numpy as np
from typing import Dict, Any, Optional

from engine.landmark_extractor import LandmarkExtractor
from engine.feature_pipeline import FeaturePipeline
from engine.prediction_buffer import PredictionBuffer
from engine.tts import TTSEngine

import json
import torch
import torch.nn as nn
import os

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class PyTorchProvider:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        
        # Load label map
        label_map_path = "data/label_map_temporal.json"
        if os.path.exists(label_map_path):
            with open(label_map_path, "r") as f:
                mapping = json.load(f)
                self.classes = {v: k for k, v in mapping.items()}
                num_classes = len(self.classes)
        else:
            self.classes = {0: "HELLO", 1: "IDLE"}
            num_classes = 2
            
        self._classes = [self.classes[i] for i in range(num_classes)]
        
        # Initialize model
        self.model = LSTMModel(input_size=270, hidden_size=64, num_layers=2, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

class MockProvider:
    def predict_proba(self, features: np.ndarray) -> tuple[str, float]:
        return "HELLO", 0.9
        
    @property
    def classes_(self):
        return ["HELLO"]

class SignRecognitionEngine:
    def __init__(self, model_path: Optional[str] = None, hand_landmarker_path: str = "hand_landmarker.task", tts_engine=None, auto_speak: bool = False):
        self.auto_speak = auto_speak
        
        self.extractor = LandmarkExtractor(model_path=hand_landmarker_path)
        self.pipeline = FeaturePipeline()
        self.buffer = PredictionBuffer()
        self.tts = tts_engine if tts_engine else TTSEngine()
        
        self.model = self._load_model(model_path)
        
        # Sequence buffer for LSTM
        self.sequence_length = 30
        self.feature_sequence = []

    def _load_model(self, model_path: Optional[str]):
        if model_path is None:
            return MockProvider()
            
        try:
            if model_path.endswith('.joblib'):
                import joblib
                return joblib.load(model_path)
            elif model_path.endswith('.pt'):
                return PyTorchProvider(model_path)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return MockProvider()
            
        return MockProvider()

    def process_frame(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        extraction = self.extractor.process_frame(frame)
        hands = extraction.get("hands", [])
        
        if not hands:
            # If no hands, clear sequence buffer to avoid corrupted gestures
            self.feature_sequence.clear()
            self.pipeline.previous_features = None
            result = self.buffer.process_prediction(None, 0.0, False)
            result["annotated_frame"] = frame
            return result
            
        features = self.pipeline.process(hands)
        annotated_frame = self.extractor.draw_landmarks(frame, hands)
        
        sign = None
        confidence = 0.0
        
        if self.model is None:
            result = self.buffer.process_prediction(None, 0.0, True)
            result["status"] = "no_model"
            result["annotated_frame"] = annotated_frame
            return result
            
        try:
            if isinstance(self.model, PyTorchProvider):
                # Accumulate temporal sequence
                self.feature_sequence.append(features)
                if len(self.feature_sequence) > self.sequence_length:
                    self.feature_sequence.pop(0)
                    
                if len(self.feature_sequence) == self.sequence_length:
                    seq_tensor = torch.tensor(np.array(self.feature_sequence), dtype=torch.float32).unsqueeze(0).to(self.model.device)
                    with torch.no_grad():
                        outputs = self.model.model(seq_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        conf, predicted_idx = torch.max(probabilities, 0)
                        sign = self.model._classes[predicted_idx.item()]
                        confidence = conf.item()
            elif isinstance(self.model, MockProvider):
                sign, confidence = self.model.predict_proba(features)
            elif hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba([features])[0]
                max_idx = np.argmax(probas)
                sign = self.model.classes_[max_idx]
                confidence = float(probas[max_idx])
            else:
                sign, confidence = "UNKNOWN", 0.0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Prediction error: {e}")
            sign, confidence = "ERROR", 0.0
            
        result = self.buffer.process_prediction(sign, confidence, True)
        result["annotated_frame"] = annotated_frame
        
        if self.auto_speak and result["current_sign"] is not None:
            self.tts.speak(result["current_sign"])
            
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "status": self.buffer.state,
            "sentence": self.buffer.sentence,
            "sequence": list(self.buffer.prediction_window)
        }
        
    def reset(self):
        self.buffer.clear()
        self.feature_sequence.clear()
        
    def close(self):
        self.extractor.close()
        self.tts.shutdown()
