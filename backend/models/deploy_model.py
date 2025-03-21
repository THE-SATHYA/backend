import os
import io
import torch
import torch.nn as nn
import librosa
import numpy as np

# Constants (must match training)
N_MFCC = 13          # Number of MFCC coefficients
PADDING_LENGTH = 100 # Fixed number of frames (MAX_FRAMES)
SAMPLE_RATE = 16000  # Sample rate used during training

# Define the model exactly as in the checkpoint
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # CNN defined as a sequential block (matches keys: cnn.0.weight, etc.)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # cnn.0
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # cnn.3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # LSTM expects input features of size 64 (not 64 * (PADDING_LENGTH//4))
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 4)  # 4 classes: flow, breathy, neutral, pressed

    def forward(self, x):
        # x shape: (batch, 1, N_MFCC, PADDING_LENGTH)
        x = self.cnn(x)  # Output shape should be (batch, 64, H, W)
        # Flatten the spatial dimensions (H, W) into a single time dimension
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        # Now x shape: (batch, time_steps, 64) where time_steps = H*W (likely 75)
        x, _ = self.lstm(x)
        # Use the output of the last time step for classification
        x = self.fc(x[:, -1, :])
        return x

# Preprocessing function that matches training:
def preprocess_audio(file_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC, padding_length=PADDING_LENGTH):
   
    audio_file = io.BytesIO(file_data)
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Loaded audio is empty")
    # Extract 13 MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Normalize each coefficient (mean=0, std=1)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
    # Pad or truncate along the time axis to PADDING_LENGTH (100 frames)
    if mfcc.shape[1] < padding_length:
        pad_width = padding_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :padding_length]
    # Expand dimensions: (13, 100) -> (1, 13, 100)
    mfcc = np.expand_dims(mfcc, axis=0)
    # Add channel dimension: final shape (1, 1, 13, 100)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    return mfcc_tensor

# Function to process a folder of audio files and print predictions
def predict_folder(folder_path, model_path="best_model.pth"):
    device = torch.device("cpu")
    model = AudioClassifier().to(device)
    # Load checkpoint (the keys now match exactly)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Map numeric predictions to string labels
    label_map = {0: "flow", 1: "breathy", 2: "neutral", 3: "pressed"}
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    if not audio_files:
        print("No .wav files found in the specified folder.")
        return
    

    for file_name in audio_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()
            input_tensor = preprocess_audio(file_data)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
                label = label_map.get(pred, "unknown")
            print(f"{file_name}: {label}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # Set the folder path to your audio files
    folder_path = "C:\\Users\\SATHYA\\Documents\\new\\multi\\pressed"
    predict_folder(folder_path)
