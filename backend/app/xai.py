import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import librosa
import librosa.display
from collections import Counter # Import Counter to count events

def generate_xai_insights(waveform, sr, hot_timestamps, top_prediction_label):
    """
    Generates a waveform/spectrogram plot and a text explanation.
    """
    
    # 1. Generate the new two-part plot
    heatmap_image_base64 = create_dual_plot(waveform, sr, hot_timestamps)
    
    # 2. Generate the text explanation (this function is unchanged)
    explanation = generate_text_explanation(hot_timestamps, top_prediction_label)
    
    return heatmap_image_base64, explanation

# --- RENAMED and HEAVILY MODIFIED this function ---
def create_dual_plot(waveform, sr, hot_timestamps):
    """
    Plots the audio waveform on top and the spectrogram on the bottom.
    Highlights all detected time segments on both plots.
    """
    
    SYMPTOM_COLORS = {
        "coughing": "red",
        "sneezing": "orange",
        "Crackle": "red",     # Color for your new model
        "Wheeze": "orange", # Color for your new model
        "default": "gray"
    }
    
    # --- NEW: Create a figure with 2 subplots, stacked vertically ---
    # We make the figure taller (figsize=(10, 6))
    # We use sharex=True so they share the same time axis
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # --- 1. Plot 1: The Waveform (on the top plot, axes[0]) ---
    librosa.display.waveshow(waveform, sr=sr, ax=axes[0], color='blue', alpha=0.6)
    axes[0].set_title("Waveform and Spectrogram with Detected Symptoms")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel(None) # Remove x-axis label from top plot

    # --- 2. Plot 2: The Spectrogram (on the bottom plot, axes[1]) ---
    # Calculate the Mel Spectrogram
    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=sr), ref=np.max)
    # Display the spectrogram
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_ylabel("Mel Frequency")
    axes[1].set_xlabel("Time (s)")

    # --- 3. Highlight all "hot" segments on BOTH plots ---
    for start_time, end_time, label in hot_timestamps:
        
        color = SYMPTOM_COLORS.get(label, SYMPTOM_COLORS["default"])
        
        # Add highlight box to the top plot (waveform)
        axes[0].axvspan(start_time, end_time, color=color, alpha=0.4, label=label)
        # Add highlight box to the bottom plot (spectrogram)
        axes[1].axvspan(start_time, end_time, color=color, alpha=0.4)

    # --- 4. Create a clean legend on the top plot ---
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    if unique_labels:
        axes[0].legend(unique_labels.values(), unique_labels.keys())

    # --- 5. Finalize and Save ---
    fig.tight_layout() # Adjusts plots to prevent overlap
    
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
    of each detected event type. (This function is unchanged)
    """
    
    all_event_labels = [label for _, _, label in hot_timestamps]
    event_counts = Counter(all_event_labels)
    
    if not event_counts:
        # --- UPDATE: Add new labels to the example ---
        return "The model did not detect any target symptoms (e.g., coughing, sneezing, crackle, wheeze) in this clip."
    
    explanation_parts = ["The model detected:"]
    explanation_parts.append("<ul>")
    
    for label, count in event_counts.items():
        pretty_label = label.replace("_", " ").title()
        explanation_parts.append(
            f"<li><strong>{count} {pretty_label} event(s)</strong></li>"
        )
        
    explanation_parts.append("</ul>")
    
    return " ".join(explanation_parts)