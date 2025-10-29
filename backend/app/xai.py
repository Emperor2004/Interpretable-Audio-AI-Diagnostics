import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import librosa
import librosa.display
from collections import Counter
import torch
import torch.nn.functional as F

# --- Import GRAD-CAM ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .model_loader import get_model_components

model, _, _ = get_model_components()

def generate_xai_insights(waveform, sr, hot_timestamps, all_chunk_predictions, best_chunk_spectrogram, best_chunk_pred_id):
    """
    Generates all XAI visualizations and explanations.
    """
    
    detection_plot_b64 = create_dual_plot(waveform, sr)
    
    xai_heatmap_b64 = None
    if best_chunk_spectrogram is not None and model is not None and best_chunk_pred_id != -1:
        try:
            xai_heatmap_b64 = create_grad_cam_heatmap(best_chunk_spectrogram, best_chunk_pred_id)
        except Exception as e:
            print(f"Error generating GRAD-CAM heatmap: {e}")
            xai_heatmap_b64 = None
    
    explanation_dict = generate_text_explanation(hot_timestamps, all_chunk_predictions)
    
    return detection_plot_b64, xai_heatmap_b64, explanation_dict


def create_grad_cam_heatmap(chunk_spectrogram, pred_id):
    """
    Runs GRAD-CAM on the provided 224x224 spectrogram chunk.
    """
    if model is None:
        return None

    # --- 1. Prepare inputs for GRAD-CAM ---
    # Add batch and channel dimensions: (1, 1, 224, 224)
    input_tensor = chunk_spectrogram.unsqueeze(0).unsqueeze(0)

    # --- 2. Set up GRAD-CAM ---
    # Target the last conv layer in ResNet50, as per your diagram
    target_layer = model.base_model.layer4[-1] 
    
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer], 
        use_cuda=torch.cuda.is_available()
    )
    
    # Target the specific class ID
    targets = [ClassifierOutputTarget(pred_id)]

    # --- 3. Generate the heatmap (Grad-CAM steps 1-4) ---
    grayscale_cam = cam(
        input_tensor=input_tensor, 
        targets=targets
    )
    grayscale_cam = grayscale_cam[0, :] # Get first heatmap
    
    # --- 4. Normalize spectrogram for visualization ---
    spec_min = chunk_spectrogram.min()
    spec_max = chunk_spectrogram.max()
    spec_img = (chunk_spectrogram - spec_min) / (spec_max - spec_min)
    spec_img_rgb = np.stack([spec_img.numpy()] * 3, axis=-1)

    # --- 5. Overlay heatmap (Visualization Module) ---
    visualization = show_cam_on_image(
        spec_img_rgb, 
        grayscale_cam, 
        use_rgb=True,
        image_weight=0.6 # This is the 60/40 blend
    )
    
    # --- 6. Plot the final image ---
    fig, ax = plt.subplots(figsize=(10, 5)) # Adjusted size
    ax.imshow(visualization)
    ax.set_title("GRAD-CAM Heatmap (for first non-silent chunk)")
    ax.set_xlabel("Time (Frames)")
    ax.set_ylabel("Mel Bins (Frequency)")
    fig.tight_layout()

    # --- 7. Convert to Base64 ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"


def create_dual_plot(waveform, sr):
    """
    Plots the full audio waveform and its spectrogram.
    NOTE: Spectrogram is now 224-bins to be consistent.
    """
    SYMPTOM_COLORS = {
        "coughing": "red",
        "sneezing": "orange",
        "default": "gray"
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # --- Plot 1: Waveform ---
    librosa.display.waveshow(waveform, sr=sr, ax=axes[0], color='blue', alpha=0.6)
    axes[0].set_title("Symptom Detection Plot")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel(None)
    
    # --- Plot 2: Spectrogram (now 224 bins) ---
    S_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=224, n_fft=2048, hop_length=512), 
        ref=np.max
    )
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_ylabel("Mel Frequency (224 Bins)")
    axes[1].set_xlabel("Time (s)")

    # --- Highlighting logic is now removed ---
    # We are just showing the full plot, highlights are in the text
    
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

# --- This function is unchanged ---
def generate_text_explanation(hot_timestamps, all_chunk_predictions):
    target_counts = Counter([label for _, _, label in hot_timestamps])
    other_labels = Counter()
    target_symptoms_set = {"coughing", "sneezing"} 
    for chunk in all_chunk_predictions:
        label = chunk['label']
        confidence = chunk['confidence']
        if confidence > 0.50 and label not in target_symptoms_set:
            other_labels[label] += 1
    primary_symptoms = [
        {"label": label.replace("_", " ").title(), "count": count}
        for label, count in target_counts.items()
    ]
    other_sounds = [
        {"label": label.replace("_", " ").title(), "count": count}
        for label, count in other_labels.most_common(3)
    ]
    return {
        "primary_symptoms": primary_symptoms,
        "other_sounds": other_sounds
    }