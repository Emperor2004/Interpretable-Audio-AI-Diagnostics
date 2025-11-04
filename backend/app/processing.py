"""
processing.py - Audio preprocessing and analysis
This module handles audio loading, preprocessing, and prediction.
"""

import torch
import numpy as np
import librosa

# Import from centralized model loader
from . import model_loader

# Re-export constants for backward compatibility
SAMPLE_RATE = model_loader.SAMPLE_RATE
WINDOW_SIZE = model_loader.WINDOW_SIZE
HOP_SIZE = model_loader.HOP_SIZE
MEL_BINS = model_loader.MEL_BINS
FMIN = model_loader.FMIN
FMAX = model_loader.FMAX
device = model_loader.device

def load_panns_model():
    """
    Loads the PANNs model using the centralized model loader.
    This function is kept for backward compatibility.
    
    Returns:
        tuple: (model, labels) or (None, error_dict)
    """
    try:
        model = model_loader.get_model()
        labels = model_loader.get_labels()
        return model, labels
    except Exception as e:
        return None, {"error": f"AI Model or labels failed to load: {e}"}

def preprocess_audio(audio_path: str) -> tuple[np.ndarray, int, torch.Tensor, torch.Tensor]:
    """
    Loads audio, resamples, and prepares both raw waveform and spectrogram tensors.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        tuple containing:
            - audio (np.ndarray): Raw audio waveform as numpy array
            - sr (int): Sample rate
            - waveform_tensor (torch.Tensor): Raw audio tensor for model input (batch, samples)
            - spectrogram_tensor (torch.Tensor): Pre-computed spectrogram for visualization (batch, 1, mels, time)
    """
    # 1. Load and Resample Audio
    (audio, sr) = librosa.core.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * 10)  # Fix to 10 seconds

    # 2. Create waveform tensor for MODEL INPUT
    # PANNs expects raw audio waveform with shape (batch_size, samples)
    waveform_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    print(f"Debug: Waveform tensor shape for model: {waveform_tensor.shape}")

    # 3. Generate Mel Spectrogram for VISUALIZATION ONLY
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=WINDOW_SIZE,
        hop_length=HOP_SIZE,
        n_mels=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # 4. Convert to Log Spectrogram 
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 5. Create spectrogram tensor for visualization
    # Shape: (batch, channel, mels, time) = (1, 1, 64, time_steps)
    spectrogram_tensor = torch.from_numpy(log_mel_spectrogram).float().unsqueeze(0).unsqueeze(0).to(device)
    
    print(f"Debug: Spectrogram tensor shape for visualization: {spectrogram_tensor.shape}")

    return audio, sr, waveform_tensor, spectrogram_tensor

def get_full_analysis(audio_path: str) -> tuple[np.ndarray, int, torch.Tensor, list | dict]:
    """
    Combines preprocessing, prediction, and result formatting.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        tuple containing:
            - audio_np (np.ndarray): Raw audio as numpy array
            - sr (int): Sample rate
            - spectrogram_tensor (torch.Tensor): Spectrogram for visualization
            - predictions (list | dict): List of predictions or error dict
    """
    model, labels = load_panns_model()
    
    if model is None or labels is None:
        return None, None, None, {"error": "AI Model or labels failed to load."}

    try:
        audio_np, sr, waveform_tensor, spectrogram_tensor = preprocess_audio(audio_path)
        
        # Pass raw waveform to the model
        print(f"Debug: Passing waveform tensor to model with shape: {waveform_tensor.shape}")
        
        with torch.no_grad():
            output = model(waveform_tensor)
            clipwise_output = output['clipwise_output'].data.cpu().numpy()[0]
            
        # Get prediction probabilities and indices
        sorted_indices = np.argsort(clipwise_output)[::-1]
        
        # Format predictions (top 5)
        predictions = []
        for i in range(5):
            idx = sorted_indices[i]
            predictions.append({
                "index": int(idx),
                "label": labels[idx],
                "probability": float(clipwise_output[idx])
            })
        
        # Return the pre-computed spectrogram for visualization purposes
        return audio_np, sr, spectrogram_tensor, predictions

    except Exception as e:
        error_msg = f"Error during prediction: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None, {"error": error_msg}