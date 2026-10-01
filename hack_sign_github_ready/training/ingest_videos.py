import os
import glob
import cv2
import numpy as np
import csv
from datetime import datetime
import sys

# Ensure engine modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.landmark_extractor import LandmarkExtractor
from engine.feature_pipeline import FeaturePipeline

def process_video(video_path, extractor, pipeline, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    frames_features = []
    
    # We must process every frame sequentially to keep velocity features accurate
    pipeline.previous_features = None 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Keep a strictly increasing global timestamp for the shared MediaPipe context
        process_video.global_ts = getattr(process_video, "global_ts", 0) + 33
        timestamp_ms = process_video.global_ts

        # Extract landmarks (pass raw BGR frame, extractor handles RGB conversion)
        result = extractor.process_frame(frame, timestamp_ms=timestamp_ms)
        
        # Convert to features
        features = pipeline.process(result["hands"])
        frames_features.append(features)
        
    cap.release()
    
    if len(frames_features) == 0:
        return None
        
    # Now we need to temporally interpolate/resample to exactly target_frames (30)
    features_array = np.array(frames_features) # Shape: (N, 270)
    N, D = features_array.shape
    
    if N == target_frames:
        return features_array
        
    # Interpolate across the time axis
    # Create an array of target indices
    old_indices = np.linspace(0, N - 1, num=N)
    new_indices = np.linspace(0, N - 1, num=target_frames)
    
    resampled = np.zeros((target_frames, D), dtype=np.float32)
    for i in range(D):
        resampled[:, i] = np.interp(new_indices, old_indices, features_array[:, i])
        
    return resampled

def main():
    base_dir = r"C:\Users\Sanskar\OneDrive\Desktop\hack_sign\data\include_landmarks\ProcessedData_vivit"
    out_dir = r"C:\Users\Sanskar\OneDrive\Desktop\hack_sign\data\sequences"
    metadata_path = os.path.join(out_dir, "metadata.csv")
    
    if not os.path.exists(base_dir):
        print(f"Error: Could not find dataset at {base_dir}")
        return
        
    classes = os.listdir(base_dir)
    print(f"Found {len(classes)} classes to ingest.")
    
    extractor = LandmarkExtractor()
    pipeline = FeaturePipeline()
    feature_dim = pipeline.get_feature_dim()
    
    # Setup metadata CSV
    file_exists = os.path.isfile(metadata_path)
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['sample_id', 'label', 'sequence_length', 'feature_dimension', 'signer_id', 'session_id', 'timestamp', 'filename'])
            
        with extractor: # Start MediaPipe context
            for c in classes:
                class_name = c.strip().upper() # e.g. "AFTERNOON"
                class_out_dir = os.path.join(out_dir, class_name)
                os.makedirs(class_out_dir, exist_ok=True)
                
                # Count existing files to avoid overwriting
                existing = glob.glob(os.path.join(class_out_dir, "*.npy"))
                seq_num = len(existing)
                
                videos = glob.glob(os.path.join(base_dir, c, "*.MOV"))
                print(f"Processing {len(videos)} videos for class {class_name}...")
                
                success_count = 0
                for v_path in videos:
                    seq_array = process_video(v_path, extractor, pipeline, target_frames=30)
                    if seq_array is None:
                        continue
                        
                    # Save
                    sample_id = f"{class_name}_include_import_{seq_num:04d}"
                    filename = f"{sample_id}.npy"
                    filepath = os.path.join(class_out_dir, filename)
                    
                    np.save(filepath, seq_array)
                    
                    # Write metadata
                    timestamp = datetime.now().isoformat()
                    writer.writerow([sample_id, class_name, 30, feature_dim, "include_dataset", "import_session", timestamp, filename])
                    
                    seq_num += 1
                    success_count += 1
                    
                print(f"  -> Successfully ingested {success_count} samples for {class_name}")

if __name__ == "__main__":
    main()
