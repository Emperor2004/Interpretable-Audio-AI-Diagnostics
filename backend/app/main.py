import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import shutil
import os

# Import our custom modules
from . import model_loader
from . import processing
from . import xai

app = FastAPI(title="Interpretable Audio AI API")

# Allow CORS for our Next.js app
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model, processor, and labels on startup
model, processor, id2label = model_loader.get_model_components()

if model is None:
    print("FATAL: Model could not be loaded. API will not be functional.")

@app.get("/")
def read_root():
    return {"message": "Interpretable Audio AI API is running."}

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Main endpoint to upload and analyze an audio file using a
    sliding window to find all *target sound events*.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # --- 1. Define Sliding Window Parameters ---
    CHUNK_SEC = 1.5      # Analyze 1.5-second chunks
    STEP_SEC = 0.5       # Move the window 0.5 seconds at a time
    CONF_THRESHOLD = 0.80 # Only count if > 80% confident
    
    # --- NEW: Define all symptoms we want to find ---
    TARGET_SYMPTOMS = ["coughing", "sneezing"]
    
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # --- 2. Load the *entire* audio file ---
        waveform, sr = processing.load_audio(temp_file_path)
        if waveform is None:
            raise HTTPException(status_code=400, detail="Could not process audio file.")
            
        # Convert chunk/step times to audio samples
        chunk_samples = int(CHUNK_SEC * sr)
        step_samples = int(STEP_SEC * sr)
        total_samples = len(waveform)
        
        all_predictions = []     # Store predictions from all chunks
        # Store (start_time, end_time, label)
        hot_timestamps = []
        
        # --- 3. Slide the window across the audio ---
        for start in range(0, total_samples - chunk_samples, step_samples):
            end = start + chunk_samples
            chunk = waveform[start:end]
            
            # Prepare features for this chunk
            inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
            
            # Run Inference
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            
            # Get top prediction for this chunk
            confidence, index = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            pred_id = index.item()
            pred_label = id2label.get(pred_id, "Unknown")
            pred_conf = confidence.item()
            
            # --- 4. Check for *any* target symptom ---
            # --- (This is the updated logic) ---
            if pred_label in TARGET_SYMPTOMS and pred_conf >= CONF_THRESHOLD:
                start_time = start / sr
                end_time = end / sr
                # Add the label so we know *what* was found
                hot_timestamps.append((start_time, end_time, pred_label))

        # --- 5. Determine Overall Prediction ---
        if hot_timestamps:
            # Use the label of the *first* detected event as the "top" one
            top_prediction_label = hot_timestamps[0][2] 
            top_prediction_conf = 100.0 # Placeholder
        else:
            top_prediction_label = "other"
            top_prediction_conf = 100.0

        # --- 6. Generate XAI Insights (Waveform + Text) ---
        # We pass the raw, un-merged list of events to xai.py
        heatmap_base64, explanation = xai.generate_xai_insights(
            waveform,
            sr,
            hot_timestamps, # Pass the list of (start, end, label) tuples
            top_prediction_label
        )
        
        return {
            "top_prediction": {
                "label": top_prediction_label,
                "confidence": top_prediction_conf
            },
            "all_predictions": [],
            "xai_heatmap_image": heatmap_base64,
            "xai_explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)