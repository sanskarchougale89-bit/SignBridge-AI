from collections import deque, Counter
from typing import Dict, Any, Optional

# ============================================================
# TUNABLE CONSTANTS — adjust these to trade sensitivity vs accuracy
# ============================================================
MIN_CONFIDENCE_THRESHOLD = 0.40      # Minimum softmax confidence to accept a prediction
WINDOW_SIZE = 10                     # Sliding window of recent predictions
STABILITY_FRAMES = 5                 # Same sign must dominate this many frames in the window
COOLDOWN_FRAMES = 15                 # Frames to wait after a detection before accepting another
SUPPRESSED_CLASSES = {"IDLE"}        # Classes that mean "no sign" — never emit to frontend


class PredictionBuffer:
    def __init__(self,
                 window_size: int = WINDOW_SIZE,
                 confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD,
                 stability_frames: int = STABILITY_FRAMES,
                 cooldown_frames: int = COOLDOWN_FRAMES):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames

        self.prediction_window = deque(maxlen=window_size)
        self.current_cooldown = 0
        self.last_detected_sign = None

        self.sentence = []
        self.state = "IDLE"

        # Track the "live" top prediction for the UI (not just detection events)
        self._live_sign = None
        self._live_confidence = 0.0

    def clear(self):
        self.prediction_window.clear()
        self.current_cooldown = 0
        self.last_detected_sign = None
        self.sentence = []
        self.state = "IDLE"
        self._live_sign = None
        self._live_confidence = 0.0

    def process_prediction(self, sign: Optional[str], confidence: float, landmarks_detected: bool) -> Dict[str, Any]:
        # ── No hand in frame ──
        if not landmarks_detected:
            self.state = "IDLE"
            self.prediction_window.clear()
            self._live_sign = None
            self._live_confidence = 0.0
            if self.current_cooldown > 0:
                self.current_cooldown -= 1
            return self._build_result(False, None, 0.0)

        # ── Cooldown period after a detection ──
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            self.state = "COOLDOWN"
            return self._build_result(True, None, self._live_confidence)

        self.state = "RECOGNIZING"

        # ── Apply confidence threshold ──
        # If confidence is too low, or the sign is a suppressed class,
        # push None into the window so it doesn't count as a vote
        is_suppressed = sign in SUPPRESSED_CLASSES
        if sign is not None and confidence >= self.confidence_threshold and not is_suppressed:
            self.prediction_window.append(sign)
            self._live_sign = sign
            self._live_confidence = confidence
        else:
            self.prediction_window.append(None)
            if is_suppressed or confidence < self.confidence_threshold:
                self._live_sign = None
                self._live_confidence = confidence

        # ── Majority vote in the window ──
        valid_preds = [p for p in self.prediction_window if p is not None]

        detected_sign = None
        if len(valid_preds) >= self.stability_frames:
            counter = Counter(valid_preds)
            most_common_sign, count = counter.most_common(1)[0]

            if count >= self.stability_frames and most_common_sign != self.last_detected_sign:
                detected_sign = most_common_sign
                self.last_detected_sign = detected_sign
                self.sentence.append(detected_sign)
                self.current_cooldown = self.cooldown_frames
                self.prediction_window.clear()
                self.state = "DETECTED"

        return self._build_result(True, detected_sign, confidence)

    def _build_result(self, landmarks_detected: bool, detected_sign: Optional[str], confidence: float) -> Dict[str, Any]:
        return {
            "landmarks_detected": landmarks_detected,
            "current_sign": detected_sign,       # Only set on the frame where a new sign is confirmed
            "live_sign": self._live_sign,         # The current best guess (for continuous UI display)
            "confidence": self._live_confidence,
            "status": self.state,
            "sequence": list(self.prediction_window),
            "sentence": self.sentence.copy()
        }
