import numpy as np
from typing import List, Dict, Any, Optional

class FeaturePipeline:
    def __init__(self, 
                 use_velocity: bool = True, 
                 use_angles: bool = True):
        self.use_velocity = use_velocity
        self.use_angles = use_angles
        self.previous_features = None

    def get_feature_dim(self) -> int:
        dim = 60 # 20 landmarks * 3 coords
        if self.use_velocity:
            dim += 60
        if self.use_angles:
            dim += 15 # 5 base/joints + 10 finger joints = 15 angles
        return dim * 2 # Multiply by 2 for Left and Right hands

    def _extract_single_hand(self, hand: Dict[str, Any], previous_features: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        landmarks = hand.get("landmarks", [])
        dim = self.get_feature_dim() // 2
        
        if len(landmarks) < 21:
            return np.zeros(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32)
            
        coords = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
        
        # 1. Wrist-relative coords
        wrist = coords[0]
        rel_coords = coords[1:] - wrist # 20x3
        
        # 2. Scale normalization
        mcp_middle = coords[9]
        scale = np.linalg.norm(mcp_middle - wrist)
        if scale > 1e-6:
            rel_coords = rel_coords / scale
            
        base_features = rel_coords.flatten()
        features = [base_features]
        
        # 3. Velocity features
        current_vel_feature = base_features.copy()
        if self.use_velocity:
            if previous_features is not None:
                velocity = base_features - previous_features
            else:
                velocity = np.zeros_like(base_features)
            features.append(velocity)
            
        # 4. Joint angles
        if self.use_angles:
            angles = self._calculate_joint_angles(coords)
            features.append(angles)
            
        return np.concatenate(features).astype(np.float32), current_vel_feature

    def process(self, hands: List[Dict[str, Any]]) -> np.ndarray:
        dim_per_hand = self.get_feature_dim() // 2
        
        if not hands:
            self.previous_features = None
            return np.zeros(self.get_feature_dim(), dtype=np.float32)

        # Sort hands by handedness to ensure deterministic ordering (Left then Right)
        left_hand = None
        right_hand = None
        
        for h in hands:
            # MediaPipe's 'Left' and 'Right' are flipped if you don't mirror, but we just need consistency.
            if h.get("handedness") == "Left" and left_hand is None:
                left_hand = h
            elif h.get("handedness") == "Right" and right_hand is None:
                right_hand = h
            elif left_hand is None: # Fallback if unknown
                left_hand = h
            elif right_hand is None:
                right_hand = h

        # Initialize previous features dict if empty
        if self.previous_features is None:
            self.previous_features = {"Left": None, "Right": None}

        # Extract Left Hand
        if left_hand:
            left_feat, left_base = self._extract_single_hand(left_hand, self.previous_features["Left"])
            self.previous_features["Left"] = left_base
        else:
            left_feat = np.zeros(dim_per_hand, dtype=np.float32)
            self.previous_features["Left"] = None

        # Extract Right Hand
        if right_hand:
            right_feat, right_base = self._extract_single_hand(right_hand, self.previous_features["Right"])
            self.previous_features["Right"] = right_base
        else:
            right_feat = np.zeros(dim_per_hand, dtype=np.float32)
            self.previous_features["Right"] = None

        # Concatenate Left and Right (total 270 features)
        return np.concatenate([left_feat, right_feat]).astype(np.float32)
        
    def _calculate_joint_angles(self, coords: np.ndarray) -> np.ndarray:
        joints = [
            (1, 2, 3), (2, 3, 4), # Thumb
            (5, 6, 7), (6, 7, 8), # Index
            (9, 10, 11), (10, 11, 12), # Middle
            (13, 14, 15), (14, 15, 16), # Ring
            (17, 18, 19), (18, 19, 20) # Pinky
        ]
        
        base_joints = [
            (0, 1, 2), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18)
        ]
        all_joints = base_joints + joints
        
        angles = []
        for (p1_idx, p2_idx, p3_idx) in all_joints:
            p1 = coords[p1_idx]
            p2 = coords[p2_idx]
            p3 = coords[p3_idx]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            
            if n1 > 1e-6 and n2 > 1e-6:
                cosine_angle = np.dot(v1, v2) / (n1 * n2)
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            else:
                angle = 0.0
                
            angles.append(angle)
            
        return np.array(angles, dtype=np.float32)
