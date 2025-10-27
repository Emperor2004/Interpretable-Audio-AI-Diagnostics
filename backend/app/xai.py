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
    
    # Ensure mel_spectrogram is on CPU and is a numpy array
    if isinstance(mel_spectrogram, torch.Tensor):
        mel_spectrogram = mel_spectrogram.squeeze().cpu().numpy()
        
    spec_height, spec_width = mel_spectrogram.shape
    
    # Upscale heatmap (12x12) to match spectrogram
    zoom_factors = (spec_height / 12, spec_width / 12)
    upscaled_heatmap = zoom(heatmap, zoom_factors, order=1) 

    # Normalize the heatmap for better visualization
    # This scales all values to be between 0 and 1
    normalized_heatmap = (upscaled_heatmap - np.min(upscaled_heatmap)) / (np.max(upscaled_heatmap) - np.min(upscaled_heatmap))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the Mel Spectrogram
    S_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    spec_img = librosa.display.specshow(S_db, sr=16000, hop_length=160, x_axis='time', y_axis='mel', ax=ax)
    
    # Get the coordinate extent from the axes *after* plotting the specshow
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Create the extent list in the correct order [left, right, bottom, top]
    im_extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    
    # Use a more intuitive 'hot' colormap
    # Use the normalized_heatmap**2 as the alpha channel.
    # This makes low-attention areas (e.g., 0.1) almost fully transparent (alpha=0.01)
    # and high-attention areas (e.g., 0.9) mostly opaque (alpha=0.81)
    ax.imshow(
        normalized_heatmap, 
        cmap='hot', 
        alpha=normalized_heatmap**2,  # This is the key change
        aspect='auto', 
        extent=im_extent, 
        origin='lower'
    )

    # Add a color bar to show frequency mapping
    fig.colorbar(spec_img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"XAI Attention Heatmap for Audio Event")
    fig.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesB()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode as base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

def generate_text_explanation(heatmap, prediction_label):
    """
    Analyzes the 12x12 heatmap and generates a simple sentence
    in HTML.
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
        
    # Generate the sentence using HTML tags instead of markdown
    explanation = (
        f"The model confidently identified the event as <strong>{prediction_label}</strong>. "
        f"Its decision was primarily based on a strong signal in the "
        f"<strong>{freq_band}</strong> range, which occurred towards <strong>{time_segment}</strong> "
        f"of the audio clip."
    )
    
    return explanation