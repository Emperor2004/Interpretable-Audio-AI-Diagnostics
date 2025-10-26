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
    Main endpoint to upload and analyze an audio file.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # Create a temporary directory to store the file
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Load and process audio using Librosa
        waveform, sr = processing.load_audio(temp_file_path)
        if waveform is None:
            raise HTTPException(status_code=400, detail="Could not process audio file.")
            
        # 2. Prepare features for the AST model
        # The processor creates the mel spectrogram and handles padding/truncation
        inputs = processor(
            waveform, 
            sampling_rate=sr, 
            return_tensors="pt"
        )
        
        mel_spectrogram = inputs.input_values[0] # Get the spectrogram for viz

        # 3. Run Inference
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        logits = outputs.logits
        attentions = outputs.attentions # This is a tuple of all layer attentions
        
        # 4. Get Top-1 Prediction
        # We get the top 5 just to show more info if needed
        top_k = 5
        confidences, indices = torch.topk(torch.softmax(logits, dim=-1), k=top_k)
        
        predictions = []
        for i in range(top_k):
            pred_id = indices[0][i].item()
            predictions.append({
                "label": id2label.get(pred_id, "Unknown"),
                "confidence": round(confidences[0][i].item() * 100, 2)
            })
            
        top_prediction_label = predictions[0]["label"]
        
        # 5. Generate XAI Insights (Heatmap + Text)
        heatmap_base64, explanation = xai.generate_xai_insights(
            attentions, 
            mel_spectrogram, 
            top_prediction_label,
            id2label
        )
        
        return {
            "top_prediction": predictions[0],
            "all_predictions": predictions,
            "xai_heatmap_image": heatmap_base64,
            "xai_explanation": explanation
        }

    except Exception as e:
        return HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)