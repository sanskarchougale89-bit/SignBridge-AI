import os
import numpy as np
import csv
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.feature_pipeline import FeaturePipeline

def test():
    label = "HELLO"
    signer_id = "test_robot"
    session_id = "sess_001"
    seq_length = 30
    
    base_dir = os.path.join("data", "sequences")
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    metadata_path = os.path.join(base_dir, "metadata.csv")
    
    file_exists = os.path.isfile(metadata_path)
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['sample_id', 'label', 'sequence_length', 'feature_dimension', 'signer_id', 'session_id', 'timestamp', 'filename'])
            
    pipeline = FeaturePipeline()
    feature_dim = pipeline.get_feature_dim()
    
    # Generate mock hands (similar to MediaPipe output)
    sequence = []
    for _ in range(seq_length):
        mock_landmarks = [{"x": np.random.rand(), "y": np.random.rand(), "z": np.random.rand()} for _ in range(21)]
        hands = [{"landmarks": mock_landmarks, "handedness": "Right"}]
        features = pipeline.process(hands)
        sequence.append(features)
        
    seq_array = np.array(sequence)
    print(f"Mock sequence shape: {seq_array.shape}")
    
    seq_num = len([f for f in os.listdir(label_dir) if f.startswith('seq_') and f.endswith('.npy')])
    sample_id = f"{label}_{signer_id}_{session_id}_{seq_num:04d}"
    filename = f"seq_{seq_num:04d}.npy"
    npy_path = os.path.join(label_dir, filename)
    
    np.save(npy_path, seq_array)
    timestamp = datetime.now().isoformat()
    
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sample_id, label, seq_length, feature_dim, signer_id, session_id, timestamp, os.path.join(label, filename)])
        
    print(f"Successfully saved {npy_path}")

if __name__ == '__main__':
    test()
