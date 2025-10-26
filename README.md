# Interpretable Audio AI Diagnostics
## Project Title:

Beyond the Black Box: A Case Study in Interpretable Audio AI for Clinical Diagnostics

## Primary Domains:

- Healthcare: The primary application domain.

- Audio: The core data modality.

- Explainable AI (XAI): The core methodology and novelty.

## The Case Study (Problem Statement):

### The Problem: 

Deep learning models can now detect diseases from audio (e.g., coughs, vocal tremors) with high accuracy. However, they are "black boxes." A doctor will not trust an AI that just outputs "Diseased (95%)" without a justification. This lack of trust and interpretability is the single biggest barrier to adopting these powerful tools in a clinical setting.

### Our Case: 

Can we build a prototype system that not only classifies audio with State-of-the-Art (SOTA) accuracy but also explains why it made its decision in a way that is verifiable and easy for a human to understand?

## Our Proposed Solution:

We are building a full-stack web application that serves as an interpretable diagnostic aid.

A user (like a doctor or researcher) can upload an audio file. Our system will then:

1. Analyze the audio using a SOTA Transformer model.

2. Predict the most likely sound event (e.g., "Cough," "Speech," "Wheezing").

3. Explain its prediction by providing two outputs:

    - A Visual Heatmap (XAI): An overlay on the audio's spectrogram that visually highlights exactly which parts of the sound (in time and frequency) the model "listened" to.

    - A Plain-Language Summary: An auto-generated sentence that translates the complex heatmap into a simple, human-readable explanation (e.g., "The model focused on a high-frequency event at the beginning of the clip.").

This solves the "black box" problem by making the AI's reasoning transparent.

## Technology Stack:

- Backend: FastAPI (Python)

    - Why: High-performance, asynchronous, and perfect for serving ML models. Easy to create API endpoints.

- Frontend: Next.js (React / TypeScript)

    - Why: Modern, fast, and easy to build a professional-looking user interface.

- AI Model: Audio Spectrogram Transformer (AST) from Hugging Face (MIT/ast-finetuned-audioset-10-10-0.4593).

    - Why: This is a SOTA model pre-trained by MIT researchers on AudioSet, a massive dataset with 527 sound classes. We don't need to train it; we are using it for inference. Its attention mechanism is the key to our XAI.

- Audio Processing: librosa (Python)

    - Why: The industry standard for loading, resampling, and converting audio into spectrograms.

- Plotting (for XAI): matplotlib & numpy

    - Why: To generate the heatmap image from the model's attention weights.

## High-Level Architecture

1. Frontend (Next.js @ localhost:3000): User uploads a .wav file. axios sends this file via a POST request to our backend.

2. Backend (FastAPI @ localhost:8000):

    - The /analyze_audio endpoint receives the file.

    - librosa loads and resamples the audio to 16kHz.

    - The ASTFeatureExtractor (processor) converts the audio into a Mel spectrogram.

    - The ASTForAudioClassification (model) runs inference, outputting logits (the prediction) and attentions (the XAI data).

3. XAI Module (xai.py):

    - This is our custom logic. It takes the attentions tensor from the model.

    - It averages the attention heads of the final layer to get a 12x12 attention grid.

    - It uses matplotlib to overlay this small grid as a heatmap on the full-resolution spectrogram.

    - It analyzes the "hottest" part of the grid to generate the English explanation.

4. Response: The backend sends a JSON object back to the frontend containing:

    - The top prediction (e.g., "Cough").

    - The confidence score.

    - The heatmap image (as a base64 string).

    - The plain-language explanation.

5. Frontend (Again): React state updates, and the results are dynamically displayed to the user.

## Key Features (Deliverables)

- File Upload Interface: A clean web page to upload audio files.

- SOTA Classification: Accurate, real-time audio classification using a Transformer model.

- Visual XAI Heatmap: A spectrogram visualization showing the model's "focus."

- Textual XAI Summary: A simple, auto-generated sentence explaining the model's reasoning.

- RESTful API: A well-defined FastAPI backend that separates the AI logic from the UI.

## What We Are Not Doing (Project Scope)

- We are NOT training a model. Training this model takes weeks. We are engineers, not data scientists for this project. Our job is to use the SOTA model and build a novel, value-added application around it.

- We are NOT building a new dataset. We will use 2-3 sample .wav files (e.g., a cough, our voice) for the live demo to prove the system works end-to-end.