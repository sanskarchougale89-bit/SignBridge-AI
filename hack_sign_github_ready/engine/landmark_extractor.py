import time
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class LandmarkExtractor:
    def __init__(self, model_path: str = "hand_landmarker.task", num_hands: int = 2):
        self.model_path = model_path
        self.num_hands = num_hands
        
        # We use VIDEO mode for synchronous per-frame processing
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def process_frame(self, frame: np.ndarray, timestamp_ms: int = None) -> Dict[str, Any]:
        """
        Process a single frame synchronously.
        """
        if self.landmarker is None:
            raise RuntimeError("Landmarker is not initialized.")
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate monotonic timestamp in ms if not provided
        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000
        
        # Process frame
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        parsed_hands = []
        if result and result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[i][0].category_name if result.handedness else "Unknown"
                
                landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
                
                world_landmarks = []
                if result.hand_world_landmarks:
                    wlm = result.hand_world_landmarks[i]
                    world_landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in wlm]
                
                parsed_hands.append({
                    "handedness": handedness,
                    "landmarks": landmarks,
                    "world_landmarks": world_landmarks
                })
                
        self._last_hands = parsed_hands  # Cache for ISL recognizer access
        return {
            "timestamp_ms": timestamp_ms,
            "hands": parsed_hands
        }

    def draw_landmarks(self, frame: np.ndarray, hands: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw landmarks on the frame (green dots + connections)
        """
        annotated_image = np.copy(frame)
        h, w, _ = annotated_image.shape
        
        # Hand connections defined by mediapipe
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for hand in hands:
            landmarks = hand.get("landmarks", [])
            for connection in HAND_CONNECTIONS:
                if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                    p1 = landmarks[connection[0]]
                    p2 = landmarks[connection[1]]
                    x1, y1 = int(p1["x"] * w), int(p1["y"] * h)
                    x2, y2 = int(p2["x"] * w), int(p2["y"] * h)
                    cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            for lm in landmarks:
                x, y = int(lm["x"] * w), int(lm["y"] * h)
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
                
        return annotated_image
