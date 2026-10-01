import cv2
import numpy as np
import os
import csv
import time
import argparse
from datetime import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.landmark_extractor import LandmarkExtractor
from engine.feature_pipeline import FeaturePipeline

def main():
    parser = argparse.ArgumentParser(description="Collect temporal sequences for Sign Language Recognition.")
    parser.add_argument("--label", type=str, required=True, help="Label for the sign (e.g., HELLO, THANK_YOU)")
    parser.add_argument("--signer-id", type=str, required=True, help="Unique ID for the person signing (e.g., signer_01)")
    parser.add_argument("--session-id", type=str, required=True, help="Unique ID for this session (e.g., sess_01)")
    parser.add_argument("--length", type=int, default=30, help="Sequence length in frames (default: 30)")
    parser.add_argument("--fps", type=int, default=30, help="Target collection FPS (default: 30)")
    
    args = parser.parse_args()

    label = args.label.strip().upper()
    seq_length = args.length
    signer_id = args.signer_id.strip()
    session_id = args.session_id.strip()
    target_fps = args.fps
    frame_time = 1.0 / target_fps

    # Directories
    base_dir = os.path.join("data", "sequences")
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    metadata_path = os.path.join(base_dir, "metadata.csv")

    # Initialize Metadata CSV if not exists
    file_exists = os.path.isfile(metadata_path)
    with open(metadata_path, 'a', newline='') as f:
        writer = cv2.csv_writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['sample_id', 'label', 'sequence_length', 'feature_dimension', 'signer_id', 'session_id', 'timestamp', 'filename'])

    # Initialize Engine Components
    print("Initializing Pipeline...")
    pipeline = FeaturePipeline()
    feature_dim = pipeline.get_feature_dim()
    
    # Try default first, fallback to DSHOW if needed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Default camera index failed. Trying DirectShow...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        print("CRITICAL ERROR: Could not connect to the webcam. Is another app using it?")
        return
        
    with LandmarkExtractor() as extractor:
        print(f"--- COLLECTION READY ---")
        print(f"Label: {label} | Signer: {signer_id} | Session: {session_id} | Seq Length: {seq_length}")
        print("Press 'r' to record a sequence.")
        print("Press 'q' to quit.")

        failed_frames = 0
        while True:
            # Determine next sample ID
            existing_files = [f for f in os.listdir(label_dir) if f.startswith('seq_') and f.endswith('.npy')]
            seq_num = len(existing_files)
            
            ret, frame = cap.read()
            if not ret:
                failed_frames += 1
                if failed_frames > 30:
                    print("ERROR: Camera is connected but sending empty frames! Close any other apps using the camera (Zoom, websites, etc).")
                    failed_frames = 0
                time.sleep(0.1)
                continue
                
            failed_frames = 0

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Live visualization
            extraction = extractor.process_frame(frame)
            display_frame = extractor.draw_landmarks(display_frame, extraction["hands"])
            
            cv2.putText(display_frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Collected: {seq_num}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'r' to record, 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Temporal Sequence Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Countdown
                for i in range(3, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, f"Start signing in {i}...", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow('Temporal Sequence Collection', frame)
                    cv2.waitKey(1000)
                
                sequence = []
                frames_collected = 0
                no_hand_count = 0
                print(f"Recording sequence {seq_num}...")
                
                # We need to reset the pipeline state before each sequence
                pipeline.previous_features = None 
                
                # Collection loop
                while frames_collected < seq_length:
                    t_start = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    frame = cv2.flip(frame, 1)
                    display_frame = frame.copy()
                    
                    # Extract & Pipeline
                    extraction = extractor.process_frame(frame)
                    hands = extraction["hands"]
                    features = pipeline.process(hands)
                    
                    if not hands:
                        no_hand_count += 1
                        
                    sequence.append(features)
                    frames_collected += 1
                    
                    # Draw
                    display_frame = extractor.draw_landmarks(display_frame, hands)
                    cv2.putText(display_frame, f"Recording: {frames_collected}/{seq_length}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Warning if hand is missing
                    if not hands:
                        cv2.putText(display_frame, "NO HAND DETECTED!", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                    cv2.imshow('Temporal Sequence Collection', display_frame)
                    cv2.waitKey(1)
                    
                    # FPS lock
                    elapsed = time.perf_counter() - t_start
                    sleep_time = frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Validation & Saving
                seq_array = np.array(sequence)
                
                if seq_array.shape != (seq_length, feature_dim):
                    print(f"Error: Malformed sequence shape {seq_array.shape}. Expected ({seq_length}, {feature_dim}). Discarding.")
                    continue
                    
                # Removed strict no_hand_count check because two-hand signs like Namaste naturally drop frames due to occlusion.
                
                # Save
                sample_id = f"{label}_{signer_id}_{session_id}_{seq_num:04d}"
                filename = f"seq_{seq_num:04d}.npy"
                npy_path = os.path.join(label_dir, filename)
                
                np.save(npy_path, seq_array)
                timestamp = datetime.now().isoformat()
                
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([sample_id, label, seq_length, feature_dim, signer_id, session_id, timestamp, os.path.join(label, filename)])
                
                print(f"SUCCESS: Saved {npy_path} with shape {seq_array.shape}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
