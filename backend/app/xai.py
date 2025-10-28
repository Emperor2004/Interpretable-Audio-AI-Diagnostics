import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import torch
import librosa
import librosa.display

def generate_xai_insights(attentions, mel_spectrogram, waveform, sr, prediction_label, id2label):
    """
    Generates a heatmap and explanation from the model's attention weights.
    """
    
    # 1. Process Attention
    att_matrix = attentions[-1].squeeze(0) 
    avg_att = torch.mean(att_matrix, dim=0)
    cls_to_patch_att = avg_att[0, 1:145]
    heatmap = cls_to_patch_att.reshape(12, 12).detach().numpy()
    
    # --- DEBUGGING LOG ---
    print("\n--- DEBUG: Raw 12x12 Heatmap Values ---")
    print(f"Heatmap Min: {np.min(heatmap)}")
    print(f"Heatmap Max: {np.max(heatmap)}")
    print(f"Heatmap Mean: {np.mean(heatmap)}")
    print("----------------------------------------\n")

    # 2. Generate Waveform Plot
    heatmap_image_base64 = create_waveform_plot(waveform, sr, heatmap)
    
    # 3. Generate Simple Language Explanation
    explanation = generate_text_explanation(heatmap, prediction_label)
    
    return heatmap_image_base64, explanation

def create_waveform_plot(waveform, sr, heatmap):
    """
    Plots the audio waveform and highlights time segments (patches)
    that received high attention from the model.
    """
    
    # --- 1. Create the base plot ---
    fig, ax = plt.subplots(figsize=(10, 3)) # Waveforms are shorter
    
    # Plot the waveform
    librosa.display.waveshow(waveform, sr=sr, ax=ax, color='blue', alpha=0.6)
    
    ax.set_title("Waveform with XAI Attention Highlights")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # --- 2. Calculate time segments ---
    duration = librosa.get_duration(y=waveform, sr=sr)
    num_time_patches = 12
    patch_duration = duration / num_time_patches
    
    # --- 3. Find and highlight "hot" time segments ---
    # Find the average attention for the whole 12x12 grid
    heatmap_mean = np.mean(heatmap)
    
    # Find the average attention for each *time slice* (column)
    # heatmap.mean(axis=0) averages all 12 frequency patches for each time patch
    time_slice_attention = heatmap.mean(axis=0) 
    
    for x in range(num_time_patches):
        # If this time slice's average attention is high...
        if time_slice_attention[x] > heatmap_mean:
            
            # ...highlight this segment on the plot
            start_time = x * patch_duration
            end_time = (x + 1) * patch_duration
            
            # axvspan creates a vertical highlight box
            ax.axvspan(start_time, end_time, color='red', alpha=0.3)

    fig.tight_layout()
    
    # --- 4. Save to Base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
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
    y, x = hottest_patch_coords # y=0, x=0 in your case
    
    # Map patch coordinates to time and frequency
    if x < 4:
        time_segment = "the beginning"
    elif x < 8:
        time_segment = "the middle"
    else:
        time_segment = "the end"
        
    if y < 4:
        freq_band = "high-frequency"
    elif y < 8:
        freq_band = "mid-frequency"
    else:
        freq_band = "low-frequency"
        
    # Generate the sentence
    explanation = (
        f"The model confidently identified the event as <strong>{prediction_label}</strong>. "
        f"Its decision was primarily based on a strong signal in the "
        f"<strong>{freq_band}</strong> range, which occurred towards <strong>{time_segment}</strong> "
        f"of the audio clip."
    )
    
    return explanation