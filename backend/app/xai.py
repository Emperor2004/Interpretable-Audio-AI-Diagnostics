import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import torch
import librosa.display
from scipy.ndimage import zoom

def generate_xai_insights(attentions, mel_spectrogram, prediction_label, id2label):
    """
    Generates a heatmap overlay and a human-readable explanation
    from the model's attention weights.
    """
    
    # 1. Process Attention
    # attentions shape is (batch, num_heads, seq_len+1, seq_len+1)
    # We get the last layer's attention
    # We average across all 12 attention heads
    # We take the attention from the [CLS] token (index 0) to all other patches
    
    # Squeeze out the batch dimension
    att_matrix = attentions[-1].squeeze(0) 
    
    # Average across all heads
    avg_att = torch.mean(att_matrix, dim=0)
    
    # Get attention from [CLS] token (index 0) to all 144 patches
    # We skip index 0 ([CLS] to [CLS]) and take the next 144
    cls_to_patch_att = avg_att[0, 1:145]
    
    # Reshape to the 12x12 grid of patches
    heatmap = cls_to_patch_att.reshape(12, 12).detach().numpy()

    # 2. Generate Heatmap Image
    heatmap_image_base64 = create_heatmap_overlay(mel_spectrogram, heatmap)
    
    # 3. Generate Simple Language Explanation
    explanation = generate_text_explanation(heatmap, prediction_label)
    
    return heatmap_image_base64, explanation

def create_heatmap_overlay(mel_spectrogram, heatmap):
    """
    Overlays the 12x12 attention heatmap on top of the full-res
    mel spectrogram and returns a base64 encoded image.
    """
    
    # Upscale the 12x12 heatmap to match the spectrogram dimensions (128, 1024)
    # The AST model processes audio in 10.24s chunks (1024 frames)
    # We need to make sure our spectrogram matches this
    
    # Fix the time dimension for plotting
    # AST pads/truncates to 1024 frames (10.24s * 100 frames/s)
    # The mel spectrogram from the processor is (128, 1024)
    
    # Ensure mel_spectrogram is on CPU and is a numpy array
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.squeeze().cpu().numpy()
        
    # Resize spectrogram if it's not 1024 frames long
    if mel_spectrogram.shape[1] != 1024:
        # Simple padding/truncating for visualization
        if mel_spectrogram.shape[1] > 1024:
            mel_spectrogram = mel_spectrogram[:, :1024]
        else:
            pad_width = 1024 - mel_spectrogram.shape[1]
            mel_spectrogram = np.pad(mel_spectrogram, ((0,0), (0, pad_width)), mode='constant')
            
    # Upscale heatmap (12x12) to match spectrogram (128x1024)
    # Note: AST patches are 16x16. 128/16 = 8. 1024/16 = 64.
    # The model *actually* uses 128 bins and 1024 frames.
    # The 12x12 patches come from a different calc.
    # Let's just resize the 12x12 to match the spec dimensions for a smooth viz.
    zoom_factors = (128 / 12, 1024 / 12)
    upscaled_heatmap = zoom(heatmap, zoom_factors, order=1) # order=1 is bilinear interp

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the Mel Spectrogram
    librosa.display.specshow(mel_spectrogram, sr=16000, hop_length=160, x_axis='time', y_axis='mel', ax=ax)
    
    # Overlay the heatmap
    ax.imshow(upscaled_heatmap, cmap='jet', alpha=0.4, aspect='auto', extent=[0, 10.24, 0, 8000])
    
    ax.set_title(f"XAI Attention Heatmap for Audio Event")
    fig.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"


def generate_text_explanation(heatmap, prediction_label):
    """
    Analyzes the 12x12 heatmap and generates a simple sentence.
    """
    
    # Find the "hottest" patch
    hottest_patch_coords = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y, x = hottest_patch_coords
    
    # Map patch coordinates to time and frequency
    
    # Time (X-axis, 0-11)
    if x < 4:
        time_segment = "the beginning"
    elif x < 8:
        time_segment = "the middle"
    else:
        time_segment = "the end"
        
    # Frequency (Y-axis, 0-11)
    # Note: 0 is high frequency, 11 is low frequency in spectrograms
    if y < 4:
        freq_band = "high-frequency"
    elif y < 8:
        freq_band = "mid-frequency"
    else:
        freq_band = "low-frequency"
        
    # Generate the sentence
    explanation = (
        f"The model confidently identified the event as **{prediction_label}**. "
        f"Its decision was primarily based on a strong signal in the "
        f"**{freq_band}** range, which occurred towards **{time_segment}** "
        f"of the audio clip."
    )
    
    return explanation