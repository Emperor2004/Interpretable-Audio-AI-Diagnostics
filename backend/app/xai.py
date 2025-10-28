import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import librosa
import librosa.display
from collections import Counter # Import Counter to count events

def generate_xai_insights(waveform, sr, hot_timestamps, top_prediction_label):
    """
    Generates a waveform plot and a text explanation based on
    the list of detected event timestamps.
    """
    
    # 1. Generate Waveform Plot (now with color-coding)
    heatmap_image_base64 = create_waveform_plot(waveform, sr, hot_timestamps)
    
    # 2. Generate Simple Language Explanation (now with counting)
    explanation = generate_text_explanation(hot_timestamps, top_prediction_label)
    
    return heatmap_image_base64, explanation

def create_waveform_plot(waveform, sr, hot_timestamps):
    """
    Plots the audio waveform and highlights all detected
    time segments, color-coded by their label.
    """
    
    # --- NEW: Define colors for each symptom ---
    SYMPTOM_COLORS = {
        "coughing": "red",
        "sneezing": "orange",
        "default": "gray" # Fallback color
    }
    
    # --- 1. Create the base plot ---
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(waveform, sr=sr, ax=ax, color='blue', alpha=0.6)
    
    ax.set_title("Waveform with Detected Symptom Highlights")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # --- 2. Highlight all "hot" segments ---
    # We now also get the label for each event
    for start_time, end_time, label in hot_timestamps:
        
        # Get the correct color for this label
        color = SYMPTOM_COLORS.get(label, SYMPTOM_COLORS["default"])
        
        # axvspan creates a vertical highlight box
        ax.axvspan(start_time, end_time, color=color, alpha=0.4, label=label)

    # --- 3. Create a clean legend ---
    # This avoids duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    if unique_labels:
        ax.legend(unique_labels.values(), unique_labels.keys())

    fig.tight_layout()
    
    # --- 4. Save to Base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"


def generate_text_explanation(hot_timestamps, top_prediction_label):
    """
    Generates a simple sentence based on the *count*
    of each detected event type.
    """
    
    # Get the label from each tuple: [('cough'), ('cough'), ('sneeze'), ...]
    all_event_labels = [label for _, _, label in hot_timestamps]
    
    # Count the occurrences of each: {'coughing': 2, 'sneezing': 1}
    event_counts = Counter(all_event_labels)
    
    if not event_counts:
        return "The model did not detect any target symptoms (e.g., coughing, sneezing) in this clip."
    
    # --- NEW: Build a list of results ---
    explanation_parts = ["The model detected:"]
    explanation_parts.append("<ul>")
    
    for label, count in event_counts.items():
        # Make the label title-cased
        pretty_label = label.replace("_", " ").title()
        
        # Add a line item, e.g., "<li><strong>2 Coughing event(s)</strong></li>"
        explanation_parts.append(
            f"<li><strong>{count} {pretty_label} event(s)</strong></li>"
        )
        
    explanation_parts.append("</ul>")
    
    return " ".join(explanation_parts)