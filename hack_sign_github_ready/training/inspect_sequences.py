import os
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.join("data", "sequences")
    metadata_path = os.path.join(base_dir, "metadata.csv")
    
    print("=== TEMPORAL DATASET INSPECTION ===\n")
    
    if not os.path.exists(metadata_path):
        print(f"Error: No metadata found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    total_sequences = len(df)
    print(f"Total Sequences in Metadata: {total_sequences}")
    
    if total_sequences == 0:
        return
        
    print("\n-- Missing / Corrupt File Check --")
    missing = 0
    corrupt = 0
    valid = 0
    
    expected_shapes = set()
    
    for idx, row in df.iterrows():
        filepath = os.path.join(base_dir, row['filename'])
        if not os.path.exists(filepath):
            print(f"  Missing: {filepath}")
            missing += 1
            continue
            
        try:
            arr = np.load(filepath)
            expected_shape = (row['sequence_length'], row['feature_dimension'])
            if arr.shape != expected_shape:
                print(f"  Shape mismatch in {filepath}: found {arr.shape}, expected {expected_shape}")
                corrupt += 1
            else:
                expected_shapes.add(arr.shape)
                valid += 1
        except Exception as e:
            print(f"  Corrupt file {filepath}: {e}")
            corrupt += 1
            
    print(f"Valid files: {valid}/{total_sequences}")
    if missing > 0 or corrupt > 0:
        print(f"WARNING: {missing} missing, {corrupt} corrupt.")
    else:
        print("All metadata entries have valid, correctly-shaped .npy files.")
        
    print(f"\nSequence Shapes (T, F): {list(expected_shapes)}")
    
    print("\n-- Distribution by Label --")
    print(df['label'].value_counts().to_string())
    
    print("\n-- Distribution by Signer --")
    print(df['signer_id'].value_counts().to_string())
    
    print("\n-- Distribution by Session --")
    print(df['session_id'].value_counts().to_string())
    
    # Check for duplicates based on exact file content (optional, skipped for perf if large)
    # But as per prompt "detectable duplicates", we can check if any two files are binary identical.
    
    print("\n-- Duplicates Check (Binary Match) --")
    hashes = {}
    dupes = 0
    for idx, row in df.iterrows():
        filepath = os.path.join(base_dir, row['filename'])
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                h = hash(f.read())
                if h in hashes:
                    dupes += 1
                else:
                    hashes[h] = filepath
    print(f"Exact binary duplicate sequences found: {dupes}")
    
    print("\nInspection Complete.")

if __name__ == '__main__':
    main()
