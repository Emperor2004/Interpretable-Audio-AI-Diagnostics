# Audio Classification with Explainable AI (XAI)

## Beyond the Black Box: Interpretable Audio Classification for Clinical Applications

A full-stack web application that combines state-of-the-art audio classification with explainable AI visualization, making deep learning model decisions transparent and verifiable for clinical and research applications.

---

## 🎯 Project Overview

### The Problem

Deep learning models can detect sounds and classify audio events with remarkable accuracy. However, they operate as "black boxes" - outputting predictions without explaining their reasoning. In critical domains like healthcare (e.g., detecting disease from coughs or vocal patterns), this lack of interpretability prevents adoption. A clinician cannot trust an AI that simply says "Disease Detected (95%)" without justification.

### Our Solution

We've built a prototype system that not only classifies audio with high accuracy but also **explains why** it made each decision through:

1. **Visual Grad-CAM Heatmaps**: Overlays on the audio spectrogram highlighting exactly which time-frequency regions influenced the model's prediction
2. **SOTA Classification**: Real-time audio event detection using PANNs CNN14, trained on 527 AudioSet categories
3. **Interactive Web Interface**: Professional UI for uploading audio, viewing predictions, and exploring explanations

---

## 🏗️ Architecture

### Technology Stack

#### Backend (FastAPI + PyTorch)
- **FastAPI**: High-performance, asynchronous Python web framework for serving ML models
- **PyTorch**: Deep learning framework for model inference
- **PANNs (CNN14)**: Pre-trained Convolutional Neural Network from PANNs (Pretrained Audio Neural Networks) library
- **librosa**: Industry-standard audio processing library
- **pytorch-grad-cam**: Gradient-weighted Class Activation Mapping for explainability

#### Frontend (Next.js + TypeScript)
- **Next.js 14**: Modern React framework with App Router
- **TypeScript**: Type-safe development
- **Axios**: HTTP client for API communication
- **Tailwind CSS**: Utility-first styling

### Model: PANNs CNN14

- **Architecture**: Convolutional Neural Network with 6 conv blocks
- **Training Data**: AudioSet (2M+ audio clips, 527 sound event classes)
- **Performance**: mAP = 0.431 on AudioSet evaluation
- **Input**: 10-second audio clips at 32kHz sample rate
- **Output**: Multi-label classification across 527 categories

---

## 🔬 How It Works

### 1. Audio Upload & Preprocessing
```
User uploads audio file (.wav, .mp3, .ogg, .flac)
         ↓
Backend receives file via FastAPI endpoint
         ↓
librosa loads and resamples to 32kHz
         ↓
Audio fixed to 10-second duration
         ↓
Mel spectrogram generated (64 mel bins)
```

### 2. Model Inference
```
Waveform tensor → PANNs CNN14 Model
         ↓
Forward pass through 6 convolutional blocks
         ↓
Global pooling + Fully connected layer
         ↓
Sigmoid activation → 527 class probabilities
         ↓
Top 5 predictions extracted
```

### 3. Explainable AI (Grad-CAM)
```
Target class identified (highest probability)
         ↓
Grad-CAM initialized on conv_block6 (final conv layer)
         ↓
Gradients computed w.r.t. target class
         ↓
Feature maps weighted by gradients
         ↓
Heatmap generated showing influential regions
         ↓
Heatmap overlaid on mel spectrogram
         ↓
Base64-encoded PNG returned to frontend
```

### 4. Results Display
```
Frontend receives:
  - Top 5 predictions with probabilities
  - Grad-CAM heatmap (base64 image)
  - Processing metadata
         ↓
React state updates → UI renders results
         ↓
User sees:
  - Classification rankings with confidence bars
  - Audio player for playback
  - Grad-CAM visualization with interpretation guide
  - Analysis metadata
```

---

## 📁 Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & endpoints
│   │   ├── model_loader.py      # Centralized model loading & caching
│   │   ├── processing.py        # Audio preprocessing & inference
│   │   └── xai.py               # Grad-CAM visualization generation
│   └── requirements.txt         # Python dependencies
│
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx         # Main UI component
│   │       ├── layout.tsx       # App layout
│   │       └── globals.css      # Global styles
│   ├── package.json             # Node dependencies
│   └── next.config.js           # Next.js configuration
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (3.10 recommended)
- **Node.js 18+** (for Next.js)
- **Git** (for cloning)

### Backend Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Create Python virtual environment**
```bash
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the FastAPI server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install dependencies**
```bash
npm install
# or
yarn install
```

3. **Start the development server**
```bash
npm run dev
# or
yarn dev
```

The frontend will be available at `http://localhost:3000`

---

## 🎬 Usage

1. **Open the application** in your browser at `http://localhost:3000`

2. **Upload an audio file** by clicking the file input or dragging and dropping

3. **Click "Analyze Audio"** to start the analysis

4. **View results**:
   - **Top 5 Predictions**: See the most likely sound categories with confidence scores
   - **Audio Player**: Listen to your uploaded audio
   - **Grad-CAM Heatmap**: Visual explanation showing which spectrogram regions influenced the prediction
   - **Interpretation Guide**: Understand how to read the heatmap

---

## 🧪 Key Components Explained

### Backend Modules

#### `model_loader.py`
**Purpose**: Centralized model management with caching

**Key Features**:
- Lazy loading: Model loads on first request
- Automatic checkpoint download from HuggingFace
- Device management (CPU/GPU)
- Label mapping (527 AudioSet classes)
- Configuration management

**Key Functions**:
- `get_model()`: Returns cached model instance
- `get_labels()`: Returns AudioSet label list
- `get_model_config()`: Returns model hyperparameters

#### `processing.py`
**Purpose**: Audio preprocessing and inference

**Key Features**:
- Audio loading and resampling to 32kHz
- Mel spectrogram generation (64 bins, 1024 FFT, 320 hop)
- Waveform tensor preparation for model
- Prediction formatting (top-k results)

**Key Functions**:
- `preprocess_audio()`: Converts audio file to model-ready tensors
- `get_full_analysis()`: End-to-end pipeline from file to predictions

#### `xai.py`
**Purpose**: Explainable AI visualization using Grad-CAM

**Key Features**:
- Grad-CAM implementation for audio CNNs
- Target layer: `conv_block6.conv2` (highest-level features)
- Attention map generation and resizing
- Heatmap overlay on mel spectrogram
- Base64 encoding for web transfer

**Key Classes**:
- `PANNsModelWrapper`: Makes PANNs compatible with pytorch-grad-cam
- Custom reshape transform for audio feature maps

**Key Functions**:
- `generate_grad_cam()`: Creates heatmap visualization for target class

#### `main.py`
**Purpose**: FastAPI application and API endpoints

**Key Endpoints**:
- `GET /`: Health check
- `POST /analyze_audio`: Main analysis endpoint

**Features**:
- CORS middleware for frontend communication
- Temporary file handling for uploads
- Comprehensive error handling
- Resource cleanup (temp files)

### Frontend Component

#### `page.tsx`
**Purpose**: Main React component for user interface

**Key Features**:
- File upload with drag-and-drop support
- Audio playback controls
- Real-time loading states
- Error handling and display
- Responsive design with Tailwind CSS

**State Management**:
- `file`: Uploaded file object
- `audioURL`: Object URL for audio playback
- `result`: API response with predictions and heatmap
- `isLoading`: Loading state
- `error`: Error message display

---

## 🔍 Explainable AI: Grad-CAM Explained

### What is Grad-CAM?

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is a technique that produces visual explanations for decisions from CNN-based models.

### How It Works for Audio

1. **Forward Pass**: Audio passes through the model, producing predictions
2. **Gradient Computation**: Gradients of the target class are computed with respect to the final convolutional layer
3. **Weight Calculation**: Feature maps are weighted by their importance to the prediction
4. **Heatmap Generation**: A spatial heatmap shows which regions activated most strongly
5. **Overlay**: The heatmap is overlaid on the mel spectrogram for interpretation

### Interpreting the Heatmap

- **Red/Orange regions**: Areas that strongly influenced the prediction
- **Blue/Cool regions**: Areas with minimal influence
- **Time axis (horizontal)**: When in the audio the important features occurred
- **Frequency axis (vertical)**: Which frequency bands were most relevant

**Example**: For a "Cough" classification, you might see red highlights around 1-3kHz during the cough event, showing the model focused on the characteristic frequency content of coughing.

---

## 📊 Model Performance

### PANNs CNN14 Metrics
- **mAP (mean Average Precision)**: 0.431 on AudioSet
- **Classes**: 527 sound event categories
- **Training Data**: ~2 million audio clips from AudioSet
- **Architecture**: 6 convolutional blocks with batch normalization

### Sample Categories
- Music (instruments, genres)
- Human sounds (speech, cough, laughter)
- Animal sounds (dog bark, bird chirp)
- Environmental sounds (rain, wind, traffic)
- Mechanical sounds (engine, tools)

---

## 🔧 Configuration

### Backend Configuration
Edit constants in `backend/app/model_loader.py`:

```python
SAMPLE_RATE = 32000      # Audio sample rate (Hz)
WINDOW_SIZE = 1024       # FFT window size
HOP_SIZE = 320           # FFT hop length
MEL_BINS = 64            # Number of mel frequency bins
FMIN = 0                 # Minimum frequency
FMAX = 16000             # Maximum frequency (Nyquist at 32kHz)
```

### Frontend Configuration
Edit `frontend/src/app/page.tsx`:

```typescript
// Change backend URL (line 53)
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

---

## 🎓 Educational Value

This project demonstrates:

1. **Full-Stack ML Engineering**: Integration of deep learning with web technologies
2. **Production ML Patterns**: Model loading, caching, error handling, API design
3. **Explainable AI**: Practical XAI implementation for audio domain
4. **Audio Signal Processing**: Spectrograms, mel-frequency analysis
5. **Modern Web Development**: React, TypeScript, async/await patterns
6. **API Design**: RESTful endpoints, file uploads, base64 encoding

---

## 🚧 Known Limitations

1. **Fixed Audio Length**: Audio is truncated/padded to 10 seconds
2. **Single File Processing**: No batch upload support
3. **No Authentication**: Open API without user management
4. **Limited Error Recovery**: Some edge cases may not be handled gracefully
5. **No Model Comparison**: Only PANNs CNN14 available (not AST as originally planned)

---

## 🔮 Future Enhancements

- [ ] **Multi-Model Support**: Add AST (Audio Spectrogram Transformer) for comparison
- [ ] **Batch Processing**: Upload and analyze multiple files
- [ ] **User Authentication**: Secure access with login system
- [ ] **Database Integration**: Store analysis history
- [ ] **Export Results**: Download reports as PDF/CSV
- [ ] **Advanced XAI**: Add attention rollout for Transformer models
- [ ] **Real-Time Analysis**: Record audio directly in browser
- [ ] **Medical Fine-Tuning**: Specialize for clinical audio (coughs, breathing)
- [ ] **Model Confidence Calibration**: Improve probability estimates
- [ ] **Audio Augmentation**: Show how noise affects predictions

---

## 📚 References

### Papers
- Kong et al. (2020). "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
- Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Gemmeke et al. (2017). "Audio Set: An ontology and human-labeled dataset for audio events"

### Libraries
- [PANNs Inference](https://github.com/qiuqiangkong/panns_inference)
- [PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
- [librosa](https://librosa.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)

---

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

**Areas for contribution**:
- Additional XAI methods (LIME, SHAP)
- Model optimization (quantization, ONNX)
- UI/UX improvements
- Testing framework
- Docker containerization
- Documentation enhancements

---

## 📄 License

This project is for educational purposes. Model weights are subject to their respective licenses (PANNs uses Apache 2.0).

---

## 👥 Authors

Built as a case study in Interpretable AI for Audio Classification.

**Contact**: [Your contact information]

---

## 🙏 Acknowledgments

- **Kong et al.** for the PANNs architecture and pretrained weights
- **MIT AudioSet** for the comprehensive sound event dataset
- **Hugging Face** community for model hosting infrastructure
- **pytorch-grad-cam** maintainers for the excellent XAI library

---

**Note**: This README reflects the current implementation using PANNs CNN14. The original project proposal mentioned AST (Audio Spectrogram Transformer), but the actual implementation uses PANNs for better compatibility and performance.