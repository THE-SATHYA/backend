import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold

# Constants
MAX_FRAMES = 100
BATCH_SIZE = 6
LEARNING_RATE = 0.0005  
WEIGHT_DECAY = 1e-5  
DROPOUT_RATE = 0.1
EPOCHS = 85
K_FOLDS = 8
# Custom dataset for loading features
class AudioDataset(Dataset):
    def __init__(self, feature_file):
        with open(feature_file, "rb") as f:
            self.data = pickle.load(f)
        
        self.files = list(self.data.keys())
        
        self.mfccs = [self._pad_or_truncate(self.data[f]["mfccs"]) for f in self.files]
        self.labels = np.array([self._extract_label(f) for f in self.files])
    
    def _pad_or_truncate(self, mfcc):
        if mfcc.shape[1] > MAX_FRAMES:
            return mfcc[:, :MAX_FRAMES]
        else:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    def _extract_label(self, filename):
        numbers = re.findall(r'\d+', filename)  # Extract digits
        return int(numbers[-1]) % 4 if numbers else 0  
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        mfcc = self.mfccs[idx]  # Shape: (13, 100)
        mfcc = np.expand_dims(mfcc, axis=0)  # Ensure shape (1, 13, 100)
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Define Model
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=13, num_classes=4):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout_cnn = nn.Dropout(DROPOUT_RATE)
        self.lstm = nn.LSTM(64 * (MAX_FRAMES // 4), 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# K-Fold Cross Validation

def train_kfold():
    dataset = AudioDataset("test.pkl")
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{K_FOLDS}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = AudioClassifier()
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        for epoch in range(10):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            
            val_loss, val_correct, val_total = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            print(f'Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={correct/total:.2%}, Val Loss={val_loss:.4f}, Val Acc={val_correct/val_total:.2%}')

if __name__ == "__main__":
    train_kfold()