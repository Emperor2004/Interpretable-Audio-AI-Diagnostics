"""
xai.py - Explainable AI (Grad-CAM) visualization
This module generates Grad-CAM visualizations for PANNs predictions.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from typing import Optional

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import librosa
import librosa.display 

# Import from centralized model loader
try:
    from . import model_loader
    SAMPLE_RATE = model_loader.SAMPLE_RATE
    HOP_SIZE = model_loader.HOP_SIZE
    MEL_BINS = model_loader.MEL_BINS
    device = model_loader.device
except ImportError as e:
    print(f"Error importing model_loader: {e}")
    SAMPLE_RATE = 32000
    HOP_SIZE = 320
    MEL_BINS = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def panns_reshape_transform(tensor):
    """
    Reshapes the feature map output from the PANNs model's ConvBlocks.
    Input shape is typically (B, C, 1, H, W) for audio treated as 1D.
    Output shape should be (B, C, H, W).
    
    Args:
        tensor: Feature map from convolutional layer
        
    Returns:
        torch.Tensor: Reshaped feature map
    """
    if tensor.dim() == 5:
        tensor = tensor.squeeze(2)
    return tensor

class PANNsModelWrapper(torch.nn.Module):
    """
    Wrapper for PANNs model to make it compatible with Grad-CAM.
    
    This wrapper:
    1. Takes spectrogram input (already processed by spectrogram_extractor and logmel_extractor)
    2. Returns only the clipwise_output tensor (not a dictionary)
    
    PANNs models return dictionaries, but Grad-CAM expects tensor outputs.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        """
        Forward pass through PANNs model starting from preprocessed spectrogram.
        
        Args:
            x: Preprocessed spectrogram tensor (batch, 1, time, freq)
            
        Returns:
            torch.Tensor: Clipwise output predictions
        """
        # Input x should be shape (batch, 1, time, freq) from logmel_extractor
        # Pass through batch normalization
        x = x.transpose(1, 3)  # (batch, freq, time, 1)
        x = self.model.bn0(x)
        x = x.transpose(1, 3)  # (batch, 1, time, freq)
        
        # Pass through all conv blocks
        x = self.model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.model.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        # Global pooling
        x = torch.mean(x, dim=3)
        
        # Aggregate across time
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        # Dropout and final fully connected layer
        x = torch.nn.functional.dropout(x, p=0.5, training=self.model.training)
        clipwise_output = torch.sigmoid(self.model.fc_audioset(x))
        
        return clipwise_output

def generate_grad_cam(waveform_tensor: torch.Tensor, spectrogram_tensor: torch.Tensor, 
                      target_index: int) -> Optional[str]:
    """
    Generates Grad-CAM visualization for the predicted class.

    Args:
        waveform_tensor: Raw audio waveform tensor (1, samples) - used to generate fresh spectrogram
        spectrogram_tensor: Pre-computed spectrogram (1, 1, H, W) for visualization overlay
        target_index: The index of the predicted class to explain

    Returns:
        Optional[str]: Base64-encoded PNG image string of the CAM overlay, or None on failure
    """
    
    # Get model from centralized loader
    try:
        model = model_loader.get_model()
    except Exception as e:
        print(f"Error loading model for Grad-CAM: {e}")
        return None
    
    if model is None:
        return None
    
    # Pass the waveform through the model's spectrogram extractor
    # to get the proper input format for Grad-CAM
    with torch.no_grad():
        # Extract spectrogram using the model's built-in extractor
        x = model.spectrogram_extractor(waveform_tensor)
        x = model.logmel_extractor(x)
        # Output shape: (batch, 1, time, freq)
        spec_for_gradcam = x
    
    # Wrap the model to work with Grad-CAM
    wrapped_model = PANNsModelWrapper(model)
    
    # Target the final convolutional layer for highest-level semantic features
    target_layers = [model.conv_block6.conv2]
    
    # Define the target: the output of the specified class index
    targets = [ClassifierOutputTarget(target_index)]

    try:
        print(f"Debug: Spectrogram for Grad-CAM shape: {spec_for_gradcam.shape}")
        
        # Initialize GradCAM with the wrapped model
        cam = GradCAM(
            model=wrapped_model, 
            target_layers=target_layers,
            reshape_transform=panns_reshape_transform
        )
        
        # Generate the CAM heatmap
        grayscale_cam = cam(input_tensor=spec_for_gradcam, targets=targets)[0, :]

        # ------------------- Visualization -------------------

        # Use the pre-computed spectrogram for visualization
        spectrogram_np = spectrogram_tensor.squeeze().cpu().numpy()
        
        # Setup plotting
        plt.ioff()
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        
        # Plot Spectrogram (Base Image)
        librosa.display.specshow(spectrogram_np, 
                                 sr=SAMPLE_RATE, 
                                 hop_length=HOP_SIZE, 
                                 x_axis='time', 
                                 y_axis='mel',
                                 ax=ax)

        # Overlay Grad-CAM Heatmap
        # Resize CAM to match spectrogram dimensions if needed
        from scipy.ndimage import zoom
        if grayscale_cam.shape != spectrogram_np.shape:
            zoom_factors = (spectrogram_np.shape[0] / grayscale_cam.shape[0],
                          spectrogram_np.shape[1] / grayscale_cam.shape[1])
            grayscale_cam = zoom(grayscale_cam, zoom_factors, order=1)
        
        ax.imshow(grayscale_cam, cmap='jet', alpha=0.5, 
                  extent=[ax.get_xlim()[0], ax.get_xlim()[1], 
                         ax.get_ylim()[0], ax.get_ylim()[1]], 
                  aspect='auto')

        # Get label name for title
        try:
            labels = model_loader.get_labels()
            label_name = labels[target_index]
            ax.set_title(f'Grad-CAM: {label_name}')
        except:
            ax.set_title(f'Grad-CAM for Class Index {target_index}')
        
        plt.tight_layout()

        # Convert Plot to Base64 String
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Ensure resources are cleaned up
        if 'cam' in locals():
            del cam
        try:
            if 'fig' in locals():
                plt.close(fig)
        except:
            pass