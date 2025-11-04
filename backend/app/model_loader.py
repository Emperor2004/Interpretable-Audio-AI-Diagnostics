from panns_inference.models import Cnn14
from panns_inference.config import labels # Import 'labels' list directly
import torch
import os
# from transformers import AutoConfig # Not needed

# --- 1. Load the PANNs Cnn14 Model ---
model_name = "PANNs Cnn14 (Pretrained on AudioSet)"

# Initialize all components outside the try block
model = None
processor = None
id2label = {}
label2id = {}

try:
    print(f"Loading PANNs model: {model_name}")
    
    # 1. 'labels' is already the list we imported from config.
    
    # 2. Create the id2label and label2id mappings
    #    We use the imported 'labels' list directly
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # 3. Create the PANNs model
    # FIX: Cnn14() requires configuration parameters when instantiated directly.
    # We supply the default AudioSet configuration parameters here.
    model = Cnn14(
        sample_rate=32000, 
        window_size=1024, 
        hop_size=320, 
        mel_bins=64, 
        fmin=50, 
        fmax=14000, 
        classes_num=527
    )
    
    processor = None # PANNs doesn't use a 'processor'
    model.eval()
    
    print("PANNs Cnn14 model loaded successfully.")
    print(f"Loaded {len(id2label)} AudioSet labels.")

except Exception as e:
    print(f"Error loading PANNs model: {e}")
    # Reset to failed state
    model = None
    processor = None
    id2label = {}
    label2id = {}

def get_model_components():
    """Returns the loaded model, processor, and label mappings."""
    # Returns 4 components for general compatibility
    return model, processor, id2label, label2id
