from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch
import os
from transformers import AutoConfig

# --- 1. Define the ResNet50 CNN model ---
# This class implements the "ResNet50 Architecture" from your diagram.

class AudioResNet50(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        # Load a pre-trained ResNet50
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # --- Adapt for Audio (1-channel Spectrogram) ---
        # Original 'conv1' expects 3-channel (RGB) images.
        # We replace it with a new layer that accepts 1-channel.
        
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            1, # 1 input channel (spectrogram)
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        # Initialize the new layer's weights by averaging the old ones
        self.base_model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        # --- Adapt the Classifier Head ---
        # Replace the final FC layer (1000 classes) with our 2-class output
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_labels)

    def forward(self, x):
        # Add a channel dimension if it's missing: (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.base_model(x)

# --- 2. Load the Model ---
model_name = "Custom Audio ResNet50 (Untrained)"
model_path_for_labels = "./model" # Path to your OLD 97% model

try:
    print(f"Loading new ResNet50 model for GRAD-CAM: {model_name}")
    
    # Load the labels ("coughing", "sneezing") from your *old* model's config
    config = AutoConfig.from_pretrained(model_path_for_labels)
    id2label = config.id2label
    
    # Create our new ResNet50 model
    model = AudioResNet50(num_labels=len(id2label))
    
    processor = None 
    model.eval()
    print("ResNet50 model loaded successfully.")
    print(f"Loaded labels: {list(id2label.values())}")

except Exception as e:
    print(f"Error loading model: {e}")
    processor = None
    model = None
    id2label = {}

def get_model_components():
    """Returns the loaded model, processor, and labels."""
    return model, processor, id2label