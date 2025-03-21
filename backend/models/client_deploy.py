import os
import io
import torch
import torch.nn as nn
import librosa
import numpy as np

# Constants
N_MFCC = 13
PADDING_LENGTH = 100
SAMPLE_RATE = 16000

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 4)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

def preprocess_audio(file_data, sr=SAMPLE_RATE, n_mfcc=N_MFCC, padding_length=PADDING_LENGTH):
    audio_file = io.BytesIO(file_data)
    y, _ = librosa.load(audio_file, sr=sr, mono=True)

    if y.size == 0:
        raise ValueError("Loaded audio is empty")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)

    if mfcc.shape[1] < padding_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, padding_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :padding_length]

    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mfcc_tensor

def predict_folder(folder_path, model_path="best_model.pth"):
    device = torch.device("cpu")

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model = AudioClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    label_map = {0: "flow", 1: "breathy", 2: "neutral", 3: "pressed"}
    reverse_label_map = {v: k for k, v in label_map.items()}

    folder_name = os.path.basename(folder_path).lower()
    if folder_name not in label_map.values():
        print(f"Warning: Folder name '{folder_name}' is not a recognized label.")
        return

    true_label_idx = reverse_label_map[folder_name]
    correct = 0
    total = 0

    print(f"\nEvaluating folder: {folder_path}")
    print(f"Expected class index: {true_label_idx} ({folder_name})\n")

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
                output = model(input_tensor)  # Raw output before softmax
                pred_idx = output.argmax(dim=1).item()
                predicted_label = label_map.get(pred_idx, "unknown")

            print(f"\nFile: {file_name}")
            print(f"Model Raw Output: {output.numpy()}")  # Debugging raw model output
            print(f"Predicted Class Index: {pred_idx} -> {predicted_label}")
            print(f"Expected Class Index: {true_label_idx} -> {folder_name}")

            is_correct = pred_idx == true_label_idx
            correct += int(is_correct)
            total += 1

            print(f"{file_name}: Predicted = {predicted_label}, Actual = {folder_name}, {'✔' if is_correct else '✘'}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == "__main__":
    folder_path = "C:\\Users\\SATHYA\\Documents\\new\\multi\\flow"
    predict_folder(folder_path)
