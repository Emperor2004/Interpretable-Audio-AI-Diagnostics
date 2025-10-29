import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import shutil
import os
import librosa

from . import model_loader
from . import processing
from . import xai

app = FastAPI(title="Interpretable Audio AI API")

# --- CORS Middleware (Unchanged) ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, _, id2label = model_loader.get_model_components()

if model is None:
    print("FATAL: Model could not be loaded. API will not be functional.")

@app.get("/")
def read_root():
    return {"message": "Interpretable Audio AI API is running."}

# --- Updated Spectrogram helper function ---
def waveform_to_spectrogram(waveform, sr, n_mels=224, target_width=224):
    """
    Converts a waveform to a 224x224 Mel Spectrogram for ResNet50.
    """
    S = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sr, 
        n_mels=n_mels,  # 224 bins
        n_fft=2048,
        hop_length=512
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # --- Resize to a fixed image size ---
    if S_db.shape[1] < target_width:
        pad_width = target_width - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        S_db = S_db[:, :target_width]
    
    return torch.tensor(S_db, dtype=torch.float32)
# ---

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    CHUNK_SEC = 2.5 # Need more time for a 224-width spectrogram
    STEP_SEC = 1.0   
    CONF_THRESHOLD = 0.0 
    TARGET_SYMPTOMS = ["coughing", "sneezing"] 
    
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        waveform, sr = processing.load_audio(temp_file_path)
        if waveform is None:
            raise HTTPException(status_code=400, detail="Could not process audio file.")
            
        chunk_samples = int(CHUNK_SEC * sr)
        step_samples = int(STEP_SEC * sr)
        total_samples = len(waveform)
        
        hot_timestamps = []
        all_chunk_predictions = [] 
        
        best_chunk_spectrogram = None
        best_chunk_pred_id = -1
        
        for start in range(0, total_samples - chunk_samples, step_samples):
            end = start + chunk_samples
            chunk_waveform = waveform[start:end]
            
            # --- Convert chunk to 224x224 Spectrogram ---
            chunk_spectrogram = waveform_to_spectrogram(chunk_waveform, sr)
            
            inputs = chunk_spectrogram.unsqueeze(0) 
            
            with torch.no_grad():
                logits = model(inputs)
                
            confidence, index = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            pred_id = index.item()
            pred_label = id2label.get(pred_id, "Unknown")
            pred_conf = confidence.item()
            
            all_chunk_predictions.append({"label": pred_label, "confidence": pred_conf})
            
            # --- This is the fix for the blank heatmap ---
            # Save the FIRST *NON-SILENT* chunk for Grad-CAM
            if best_chunk_spectrogram is None and chunk_waveform.any():
                best_chunk_spectrogram = chunk_spectrogram
                best_chunk_pred_id = pred_id
            
            if pred_label in TARGET_SYMPTOMS and pred_conf >= CONF_THRESHOLD:
                start_time = start / sr
                end_time = end / sr
                hot_timestamps.append((start_time, end_time, pred_label))

        # --- Event merging logic (UNCHANGED) ---
        merged_timestamps = []
        if hot_timestamps:
            hot_timestamps.sort(key=lambda x: x[0])
            current_start, current_end, current_label = hot_timestamps[0]
            for next_start, next_end, next_label in hot_timestamps[1:]:
                if next_start <= current_end and next_label == current_label:
                    current_end = max(current_end, next_end)
                else:
                    merged_timestamps.append((current_start, current_end, current_label))
                    current_start, current_end, current_label = next_start, next_end, next_label
            merged_timestamps.append((current_start, current_end, current_label))
        # ---

        if merged_timestamps:
            top_prediction_label = merged_timestamps[0][2] 
            top_prediction_conf = 100.0
        else:
            top_prediction_label = "other"
            top_prediction_conf = 100.0

        # --- Generate XAI Insights (Grad-CAM) ---
        detection_plot_b64, xai_heatmap_b64, explanation_dict = xai.generate_xai_insights(
            waveform,
            sr,
            merged_timestamps,
            all_chunk_predictions,
            best_chunk_spectrogram,
            best_chunk_pred_id 
        )
        
        return {
            "top_prediction": {
                "label": top_prediction_label,
                "confidence": top_prediction_conf
            },
            "all_predictions": [],
            "xai_detection_plot": detection_plot_b64,
            "xai_attention_heatmap": xai_heatmap_b64,
            "xai_explanation": explanation_dict
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)