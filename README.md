# Deepfake Detector

Multi-layer deepfake detection tool using behavioral and visual analysis techniques.  
Can we tell what's real anymore? This project explores modern approaches to detecting AI-generated video manipulation through a combination of behavioral patterns, texture analysis, frequency domain analysis, and temporal consistency checking.

---

## 📊 Overview

This is an **educational proof-of-concept** demonstrating modern deepfake detection approaches.  
It analyzes videos using four distinct methods:

### 🧠 Behavioral Pattern Analysis
Humans are beautifully inconsistent. We blink irregularly, make tiny unconscious movements, breathe naturally. Deepfakes often lack this organic chaos.

**Tracks:**  
- Blink patterns  
- Eye movements  
- Head micro-movements  
- Natural motion variance  

### 🎨 Texture Consistency Analysis
GAN-generated faces leave subtle artifacts—unnatural smoothness, weird noise patterns, texture inconsistencies.

**Detects:**  
- Gradient patterns  
- Texture variance  
- Local consistency  
- Facial smoothness anomalies  

### 📊 Frequency Domain Analysis (FFT)
Real videos have characteristic frequency distributions. Deepfakes often have suspicious spectral signatures invisible to the human eye.

**Analyzes:**  
- High-frequency artifacts  
- Spectral anomalies  
- Frequency distribution patterns  

### ⏱️ Temporal Stability Assessment
Videos are sequences of frames that must be consistent. Deepfakes can have identity drift, lighting jumps, or temporal glitches.

**Measures:**  
- Frame-to-frame consistency  
- Embedding drift  
- Temporal coherence  

---

## 🚀 Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/Dushani-Ekanayake/deepfake-detecter.git
cd deepfake-detecter
Install dependencies:

pip install -r requirements.txt


Run the detector:

python deepfake_detector.py


Enter video path when prompted:

📁 Enter video file path: C:\Users\User\Desktop\test_video.mp4

Requirements

Python 3.8 or higher

OpenCV

NumPy

SciPy

All dependencies are listed in requirements.txt.

💻 Usage
Basic Usage
python deepfake_detector.py


The tool will prompt you for a video file path. You can:

Type the full path to your video file

Drag and drop the video file into the terminal (automatically pastes path)

Supported Video Formats

MP4

AVI

MOV

MKV

Any format supported by OpenCV

Example Workflow

Record or download a test video

Run the detector

Review the suspicion score and detailed breakdown

Check category-specific findings

Use the report to understand which aspects seem suspicious

📁 Project Structure
deepfake-detecter/
├── deepfake_detector.py    # Main detection script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── TESTING_GUIDE.md        # Guide for testing the detector
└── LICENSE                 # MIT License

🧪 Testing & Validation

Current Testing Status:
⚠️ Limited validation - This is a proof-of-concept requiring extensive testing

Recommended Testing Approach:

Collect Test Dataset

5-10 authentic videos (real people)

5-10 deepfake videos (public datasets)

Ensure varied lighting, angles, quality

Public Deepfake Datasets

FaceForensics++

Celeb-DF

DFDC

Run Systematic Tests
python deepfake_detector.py


Document suspicion score, verdict, false positives/negatives

See TESTING_GUIDE.md for detailed testing instructions

Expected Performance:

Accuracy: 60–70% on mixed-quality content

Better with: Older/obvious deepfakes, clear authentic videos

Struggles with: State-of-the-art deepfakes, poor quality footage

⚠️ Critical Limitations & Scope

What This Tool IS:
✅ Educational proof-of-concept demonstrating detection principles
✅ Learning artifact for computer vision and analysis techniques
✅ Multi-layered approach showing modern detection methods
✅ Undergraduate research project exploring the deepfake problem

What This Tool IS NOT:
❌ NOT production-ready
❌ NOT forensically sound
❌ NOT highly accurate (~30–40% error rate expected)
❌ NOT a replacement for professional analysis tools

Technical Limitations:

No neural network detection: classical CV methods only

Simple feature extraction: basic texture/frequency analysis

Limited dataset testing

Significant false positives/negatives

Requires clear footage (poor lighting/angles reduce accuracy)

Single-face only

No real-time capability

Why These Limitations Exist:
This project prioritizes learning and demonstration over accuracy. True robust deepfake detection requires:

Large labeled datasets (10,000+ videos)

Deep learning models (ResNet, EfficientNet, etc.)

Extensive training and validation

Significant computational resources

🔧 How It Works
Analysis Pipeline
Video Input → Face Detection → Multi-Layer Analysis → Suspicion Score
                                      ├─ Behavioral (15% weight)
                                      ├─ Texture (25% weight)
                                      ├─ Frequency (25% weight)
                                      └─ Temporal (20% weight)

Technical Methods

1. Behavioral Analysis:

Eye Aspect Ratio (EAR) for blink detection

Statistical analysis of blink intervals

Head pose variance tracking

Face position and size consistency

2. Texture Analysis:

Sobel gradient analysis for texture patterns

Laplacian variance for smoothness detection

Local texture consistency measurement

GAN artifact detection

3. Frequency Domain:

2D Fast Fourier Transform (FFT)

High-frequency energy ratio analysis

Spectral anomaly detection

Frequency distribution patterns

4. Temporal Stability:

Histogram-based face embeddings

Frame-to-frame drift calculation

Temporal consistency measurement

Identity stability tracking

🛠️ Built With

Python 3.8+

OpenCV

NumPy

SciPy

No deep learning frameworks required—demonstrates what’s possible with classical techniques.

📖 Related Article

Read the full technical article:
"Fighting Deepfakes in 2026: How We Can Actually Tell What's Real"
[Link to Medium article]

Covers:

Why current detection methods struggle

Blockchain-based verification

Invisible digital watermarking

AI-powered behavioral and visual analysis

The future of video verification

🔮 Future Improvements

Immediate Enhancements (Feasible):

Web interface using Streamlit/Gradio

Batch processing for multiple videos

Export reports to PDF/JSON

Video comparison mode (real vs suspected fake)

Improved visualization of results

Research Extensions (Advanced):

CNN-based feature extraction (ResNet, EfficientNet)

Facial landmark detection (dlib/MediaPipe)

Attention mechanisms for spatial analysis

LSTM for temporal pattern learning

Training on FaceForensics++ dataset

Multi-model ensemble approach

Production Requirements (Not in Scope):

Large-scale dataset training (100K+ videos)

Model optimization and quantization

API deployment with authentication

Continuous learning pipeline

Professional forensic validation

