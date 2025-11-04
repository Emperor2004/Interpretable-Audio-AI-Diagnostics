"""
model_loader.py - Centralized model loading and management
This module handles all PANNs model loading, caching, and configuration.
"""

from panns_inference.models import Cnn14
from panns_inference.config import labels
import torch
import os
import requests

# --- Model Configuration Constants ---
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 0
FMAX = 16000
MODEL_FILENAME = 'Cnn14_mAP=0.431.pth'
DOWNLOAD_URL = f'https://huggingface.co/thelou1s/panns-inference/resolve/main/{MODEL_FILENAME}'

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Global Model Cache ---
_model_cache = {
    'model': None,
    'processor': None,
    'id2label': None,
    'label2id': None,
    'labels': None
}

def download_file_if_missing(file_path: str, url: str) -> None:
    """
    Downloads a file from a URL if it does not exist locally.
    
    Args:
        file_path: Local path where the file should be saved
        url: URL to download the file from
        
    Raises:
        RuntimeError: If download fails
    """
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

def _initialize_model():
    """
    Internal function to initialize and load the PANNs model.
    Only called once and cached.
    """
    try:
        print("Initializing PANNs Cnn14 model...")
        
        # 1. Download checkpoint if missing
        download_file_if_missing(MODEL_FILENAME, DOWNLOAD_URL)
        
        # 2. Create label mappings
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        
        # 3. Initialize model structure
        model = Cnn14(
            sample_rate=SAMPLE_RATE,
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            mel_bins=MEL_BINS,
            classes_num=len(labels),
            fmin=FMIN,
            fmax=FMAX
        )
        
        # 4. Load pretrained weights
        checkpoint = torch.load(MODEL_FILENAME, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        # 5. Cache everything
        _model_cache['model'] = model
        _model_cache['processor'] = None  # PANNs doesn't use a processor
        _model_cache['id2label'] = id2label
        _model_cache['label2id'] = label2id
        _model_cache['labels'] = labels
        
        print(f"✓ PANNs Cnn14 model loaded successfully")
        print(f"✓ Loaded {len(labels)} AudioSet labels")
        print(f"✓ Device: {device}")
        
    except Exception as e:
        print(f"✗ Error loading PANNs model: {e}")
        # Reset cache on failure
        _model_cache['model'] = None
        _model_cache['processor'] = None
        _model_cache['id2label'] = None
        _model_cache['label2id'] = None
        _model_cache['labels'] = None
        raise

def get_model():
    """
    Returns the loaded PANNs model.
    Loads the model on first call and caches it.
    
    Returns:
        torch.nn.Module: The loaded PANNs Cnn14 model
    """
    if _model_cache['model'] is None:
        _initialize_model()
    return _model_cache['model']

def get_labels():
    """
    Returns the AudioSet labels list.
    
    Returns:
        list: List of AudioSet label strings
    """
    if _model_cache['labels'] is None:
        _initialize_model()
    return _model_cache['labels']

def get_label_mappings():
    """
    Returns the id2label and label2id mappings.
    
    Returns:
        tuple: (id2label dict, label2id dict)
    """
    if _model_cache['id2label'] is None:
        _initialize_model()
    return _model_cache['id2label'], _model_cache['label2id']

def get_model_components():
    """
    Returns all model components (model, processor, id2label, label2id).
    This is the main function to use for getting everything at once.
    
    Returns:
        tuple: (model, processor, id2label, label2id)
    """
    if _model_cache['model'] is None:
        _initialize_model()
    return (
        _model_cache['model'],
        _model_cache['processor'],
        _model_cache['id2label'],
        _model_cache['label2id']
    )

def get_device():
    """
    Returns the PyTorch device being used (CPU or CUDA).
    
    Returns:
        torch.device: The device
    """
    return device

def get_model_config():
    """
    Returns the model configuration parameters.
    
    Returns:
        dict: Dictionary containing model configuration
    """
    return {
        'sample_rate': SAMPLE_RATE,
        'window_size': WINDOW_SIZE,
        'hop_size': HOP_SIZE,
        'mel_bins': MEL_BINS,
        'fmin': FMIN,
        'fmax': FMAX,
        'num_classes': len(labels),
        'model_name': 'Cnn14',
        'checkpoint': MODEL_FILENAME
    }

def is_model_loaded():
    """
    Check if the model is currently loaded.
    
    Returns:
        bool: True if model is loaded, False otherwise
    """
    return _model_cache['model'] is not None

def unload_model():
    """
    Unloads the model from memory and clears the cache.
    Useful for freeing up GPU/CPU memory.
    """
    if _model_cache['model'] is not None:
        del _model_cache['model']
        _model_cache['model'] = None
        _model_cache['processor'] = None
        _model_cache['id2label'] = None
        _model_cache['label2id'] = None
        _model_cache['labels'] = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Model unloaded from memory")