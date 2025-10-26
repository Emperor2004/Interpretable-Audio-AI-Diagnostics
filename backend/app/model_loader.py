from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch
import os
import dotenv

# Use a SOTA Audio Spectrogram Transformer (AST)
# This model is fine-tuned on AudioSet and can identify 527 sound classes,
# including "Cough", "Wheezing", "Speech", "Snoring", etc.
# MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
dotenv.load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "MIT/ast-finetuned-audioset-10-10-0.4593")

# This is a singleton pattern. These will be loaded once and reused.
try:
    processor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()  # Set model to evaluation mode
    
    # Get the mapping from ID to human-readable label
    id2label = model.config.id2label

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an internet connection to download the model.")
    processor = None
    model = None
    id2label = {}

def get_model_components():
    """Returns the loaded model, processor, and labels."""
    return model, processor, id2label