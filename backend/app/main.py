import os
import shutil
import tempfile
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from . import processing
from . import xai

# --- FastAPI Setup ---

# Initialize FastAPI application
app = FastAPI(title="PANNs Audio XAI Backend")

# Define CORS settings to allow frontend access
origins = [
    "*", # Allow all origins for local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Models ---

class AnalysisResponse(BaseModel):
    """Schema for the successful analysis response."""
    predictions: List[Dict[str, Any]]
    cam_base64: str
    prediction_time: float = 0.0

# --- Lifecycle Hooks ---

@app.on_event("startup")
def startup_event():
    """Load model and setup dependencies on startup."""
    # Ensure model loads eagerly and prints status
    processing.load_panns_model()

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "PANNs XAI Backend Running"}

@app.post("/analyze_audio", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, runs PANNs prediction, and generates Grad-CAM visualization.
    """
    
    # 1. Save uploaded file to a temporary location
    try:
        # FastAPI's recommended way to handle files for external processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file temporarily: {e}")
    finally:
        # Ensure the UploadFile content is consumed
        await file.close()

    try:
        # 2. Run full analysis (load, preprocess, predict)
        audio_np, sr, spectrogram_tensor, analysis_result = processing.get_full_analysis(temp_file_path)

        # CRITICAL FIX: Handle error dictionary returned by processing.py
        if isinstance(analysis_result, dict) and "error" in analysis_result:
            # Raise an HTTPException with the specific error message
            raise HTTPException(status_code=500, detail=analysis_result["error"])

        # analysis_result is now the successful list of predictions
        predictions = analysis_result
        
        # Find the index of the top predicted class for CAM generation
        if not predictions:
            raise HTTPException(status_code=400, detail="No predictions could be generated for the provided audio.")
            
        target_index = predictions[0]['index']

        # 3. Generate Grad-CAM image (base64 encoded)
        # CRITICAL FIX: Create waveform tensor and pass both waveform and spectrogram
        waveform_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(processing.device)
        
        print(f"Debug: Creating waveform tensor for Grad-CAM with shape: {waveform_tensor.shape}")
        print(f"Debug: Spectrogram tensor shape: {spectrogram_tensor.shape}")
        print(f"Debug: Target index: {target_index}")
        
        cam_base64 = xai.generate_grad_cam(
            waveform_tensor=waveform_tensor,
            spectrogram_tensor=spectrogram_tensor,
            target_index=target_index
        )
        
        if cam_base64 is None:
            raise HTTPException(status_code=500, detail="Failed to generate Grad-CAM visualization.")
            
        # 4. Return results
        return AnalysisResponse(
            predictions=predictions,
            cam_base64=cam_base64,
            prediction_time=0.0 # Placeholder for actual time if measured
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions (e.g., 400 or specific 500s we raised above)
        raise
    except Exception as e:
        # Catch any unexpected Python exceptions
        print(f"Unexpected error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during analysis: {str(e)}")
        
    finally:
        # 5. Cleanup temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- XAI Module Import Check ---
# Note: The Grad-CAM model logic is in xai.py
# The target layer for Cnn14 is 'model.conv_block4.conv'