import torch
import numpy as np
import librosa
import os
import requests
from panns_inference.models import Cnn14
from panns_inference.config import labels

# --- Constants for PANNs Cnn14 ---
# Consistent parameters for audio processing matching PANNs Cnn14
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 0    # Frequency minimum for Mel filter bank
FMAX = 16000 # Frequency maximum (Nyquist for 32kHz)
MODEL_FILENAME = 'Cnn14_mAP=0.431.pth'
DOWNLOAD_URL = f'https://huggingface.co/thelou1s/panns-inference/resolve/main/{MODEL_FILENAME}'

# PyTorch device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global storage for the loaded model and labels
PANN_MODEL = None
LABELS = labels # Direct import from panns_inference.config

def download_file_if_missing(file_path, url):
    """Downloads a file from a URL if it does not exist locally."""
    if not os.path.exists(file_path):
        print(f"Model checkpoint not found. Downloading {file_path} from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete: {file_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            raise RuntimeError(f"Failed to download model checkpoint from {url}")

def load_panns_model(checkpoint_path=MODEL_FILENAME):
    global PANN_MODEL
    if PANN_MODEL is None:
        try:
            download_file_if_missing(checkpoint_path, DOWNLOAD_URL)
            
            PANN_MODEL = Cnn14(sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, 
                               hop_size=HOP_SIZE, mel_bins=MEL_BINS, classes_num=len(LABELS), 
                               fmin=FMIN, fmax=FMAX)
            
            PANN_MODEL.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
            PANN_MODEL.to(device)
            PANN_MODEL.eval()
            print("PANNs Cnn14 model loaded successfully.")
            print(f"Loaded {len(LABELS)} AudioSet labels.")
            
        except RuntimeError as e:
            print(f"Error loading PANNs model: {e}")
            PANN_MODEL = None 
            return None, {"error": f"AI Model or labels failed to load: {e}"}
        except Exception as e:
            print(f"Error loading PANNs model: {e}")
            PANN_MODEL = None 
            return None, {"error": f"AI Model or labels failed to load: {e}"}

    return PANN_MODEL, LABELS

def preprocess_audio(audio_path: str) -> tuple[np.ndarray, int, torch.Tensor, torch.Tensor]:
    """
    Loads audio, resamples, and prepares both raw waveform and spectrogram tensors.
    
    Returns:
        audio (np.ndarray): Raw audio waveform as numpy array
        sr (int): Sample rate
        waveform_tensor (torch.Tensor): Raw audio as tensor for model input (batch, samples)
        spectrogram_tensor (torch.Tensor): Pre-computed spectrogram for visualization (batch, 1, mels, time)
    """
    # 1. Load and Resample Audio
    (audio, sr) = librosa.core.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * 10)  # Fix to 10 seconds

    # 2. Create waveform tensor for MODEL INPUT
    # CRITICAL: PANNs expects raw audio waveform with shape (batch_size, samples)
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
    Returns (audio_np, sr, spectrogram_tensor, predictions_list | error_dict)
    """
    model, labels = load_panns_model()
    
    if model is None or labels is None:
        return None, None, None, {"error": "AI Model or labels failed to load."}

    try:
        audio_np, sr, waveform_tensor, spectrogram_tensor = preprocess_audio(audio_path)
        
        # CRITICAL: Pass raw waveform to the model, not the spectrogram!
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