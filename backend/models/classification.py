import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset

# Constants
MAX_FRAMES = 100  # Fixed length for MFCC padding/truncation
BATCH_SIZE = 6
LEARNING_RATE = 0.0005  
WEIGHT_DECAY = 1e-5  
DROPOUT_RATE = 0.1
EPOCHS = 85

# Custom dataset for loading features
class AudioDataset(Dataset):
    def __init__(self, feature_file):
        with open(feature_file, "rb") as f:
            self.data = pickle.load(f)
       
        self.files = list(self.data.keys())
       
        # Process MFCCs: Pad or truncate to MAX_FRAMES
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
        return int(numbers[-1]) % 4 if numbers else 0  # Use last number as label, default to 0
   
    def __len__(self):
        return len(self.files)
   
    def __getitem__(self, idx):
        mfcc = self.mfccs[idx]  # Shape: (13, 100)
        mfcc = np.expand_dims(mfcc, axis=0)  # ✅ Ensure shape (1, 13, 100) for CNN
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Define Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_scores = self.attn_weights(lstm_output).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_output = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return attn_output, attn_weights

# Define Model
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=13, num_classes=4):
        super(AudioClassifier, self).__init__()
       
        # CNN for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout_cnn = nn.Dropout(DROPOUT_RATE)
       
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(64 * (MAX_FRAMES // 4), 128, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(DROPOUT_RATE)
       
        # Attention Mechanism
        self.attention = Attention(256)
       
        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)
        self.dropout_fc = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_cnn(x)
       
        # Ensure correct shape for LSTM
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)  # (Batch, Time, Features)
       
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout_lstm(lstm_out)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.dropout_fc(attn_out)
       
        output = self.fc(attn_out)
        return output

# Training function
def train_model():
    dataset = AudioDataset("test.pkl")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    model = AudioClassifier()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing for stability
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  
   
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
       
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)  # ✅ No unsqueeze needed, dataset already has correct shape
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
       
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={correct/total:.2%}")

if __name__ == "__main__":
    train_model()
