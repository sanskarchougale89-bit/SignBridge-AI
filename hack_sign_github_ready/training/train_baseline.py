import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from engine.feature_pipeline import FeaturePipeline

def preprocess_features(X):
    # X is (N, 63)
    pipeline = FeaturePipeline()
    features_list = []
    
    for row in X:
        # Convert row (63,) back to the list of dicts format expected by FeaturePipeline
        landmarks = []
        for i in range(21):
            landmarks.append({
                "x": row[i*3],
                "y": row[i*3 + 1],
                "z": row[i*3 + 2]
            })
        
        hands = [{"landmarks": landmarks}]
        features = pipeline.process(hands)
        features_list.append(features)
        
    return np.array(features_list)

def main():
    data_path = 'data/landmarks.csv'
    label_map_path = 'data/label_map.json'
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'baseline_rf.joblib')
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return
        
    print(f"Data shape: {df.shape}")
    
    y = df['label'].to_numpy(dtype=str)
    X_raw = df.drop('label', axis=1).to_numpy(dtype=float)
    
    print("Preprocessing features...")
    X = preprocess_features(X_raw)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)
    
    # Update label map
    unique_labels = sorted(list(set(y)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"Updated label map saved to {label_map_path}")

if __name__ == '__main__':
    main()
