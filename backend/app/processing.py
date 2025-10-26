import librosa
import numpy as np

TARGET_SAMPLE_RATE = 16000  # AST model requires 16kHz audio

def load_audio(file_path):
    """
    Loads an audio file from file_path, converts it to mono,
    and resamples it to the target sample rate.
    """
    try:
        # Load audio file
        # sr=None preserves the original sample rate
        waveform, sr = librosa.load(file_path, sr=None, mono=False)

        # Convert to mono if it's stereo
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)

        # Resample to the target sample rate (16kHz for AST)
        if sr != TARGET_SAMPLE_RATE:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            
        return waveform, TARGET_SAMPLE_RATE

    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None