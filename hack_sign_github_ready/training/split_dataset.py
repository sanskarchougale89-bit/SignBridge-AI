import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split temporal dataset by signer")
    parser.add_argument("--test-signer", type=str, help="Signer ID to hold out for test set")
    parser.add_argument("--val-signer", type=str, help="Signer ID to hold out for validation set")
    args = parser.parse_args()
    
    base_dir = os.path.join("data", "sequences")
    metadata_path = os.path.join(base_dir, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return
        
    df = pd.read_csv(metadata_path)
    
    signers = df['signer_id'].unique()
    print(f"Total sequences: {len(df)}")
    print(f"Available signers: {signers}")
    
    if len(signers) < 3:
        print("\nWARNING: Too few signers to do a clean disjoint train/val/test split (need at least 3).")
        print("For a small hackathon team, it's recommended to do a session-based split instead,")
        print("or carefully stratify at the row level while acknowledging the leakage risk.")
        print("Fallback: Creating a stratified random split 80/10/10 (Leakage likely!).")
        
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
        
    else:
        if not args.test_signer or not args.val_signer:
            print("Please specify --test-signer and --val-signer.")
            return
            
        test_df = df[df['signer_id'] == args.test_signer]
        val_df = df[df['signer_id'] == args.val_signer]
        train_df = df[(df['signer_id'] != args.test_signer) & (df['signer_id'] != args.val_signer)]
        
    print("\nSplit Results:")
    print(f"Train: {len(train_df)} sequences")
    print(f"Val:   {len(val_df)} sequences")
    print(f"Test:  {len(test_df)} sequences")
    
    train_df.to_csv(os.path.join(base_dir, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(base_dir, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(base_dir, "test_metadata.csv"), index=False)
    
    print("\nSaved split metadata files to data/sequences/")

if __name__ == '__main__':
    main()
