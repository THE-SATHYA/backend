import librosa
import numpy as np
import torch
import io

def extract_mfcc(audio_data, sr=16000, n_mfcc=13, padding_length=100):
    """
    Extracts MFCC features from raw audio bytes.
    """
    try:
        # Load the audio file
        audio_file = io.BytesIO(audio_data)
        y, _ = librosa.load(audio_file, sr=sr, mono=True)
        if y.size == 0:
            raise ValueError("Loaded audio is empty")
        
        # Compute MFCCs and normalize
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-6)
        
        # Pad or truncate
        if mfcc.shape[1] < padding_length:
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_length - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :padding_length]
        
        # Convert to tensor
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mfcc_tensor
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {e}")
