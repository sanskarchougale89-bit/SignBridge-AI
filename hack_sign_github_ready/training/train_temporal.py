import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SignSequenceDataset(Dataset):
    def __init__(self, metadata_path, base_dir="data/sequences"):
        self.base_dir = base_dir
        self.metadata = pd.read_csv(metadata_path)
        
        # Build label mapping
        self.labels = sorted(self.metadata['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.base_dir, row['filename'])
        
        # Load sequence data (T, F)
        sequence = np.load(file_path).copy()
        
        # Simple Data Augmentation (Random Jitter to prevent overfitting)
        # We apply small random noise to the coordinates
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.02, sequence.shape)
            sequence = sequence + noise
            
        # Convert to PyTorch tensors
        x = torch.tensor(sequence, dtype=torch.float32)
        y = torch.tensor(self.label_to_idx[row['label']], dtype=torch.long)
        
        return x, y

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer expects input shape: (batch_size, seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def main():
    base_dir = "data/sequences"
    train_meta = os.path.join(base_dir, "train_metadata.csv")
    val_meta = os.path.join(base_dir, "val_metadata.csv")
    
    if not os.path.exists(train_meta) or not os.path.exists(val_meta):
        print("Error: train_metadata.csv or val_metadata.csv not found.")
        print("Please run `python training/split_dataset.py` first.")
        return
        
    print("Loading datasets...")
    train_dataset = SignSequenceDataset(train_meta)
    val_dataset = SignSequenceDataset(val_meta)
    
    # Save label mapping
    label_map = train_dataset.label_to_idx
    os.makedirs("models", exist_ok=True)
    with open("data/label_map_temporal.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label mapping with {len(label_map)} classes.")
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Check data shape
    sample_x, sample_y = train_dataset[0]
    seq_length, input_size = sample_x.shape
    num_classes = len(label_map)
    print(f"Sequence length: {seq_length}, Feature size: {input_size}, Classes: {num_classes}")
    
    # Model configuration
    hidden_size = 64
    num_layers = 2
    num_epochs = 30
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
              
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/lstm_model.pt")
            print("  [Saved new best model]")
            
    print("Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("Model saved to models/lstm_model.pt")

if __name__ == "__main__":
    main()
