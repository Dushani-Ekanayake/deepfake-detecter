<img width="1117" height="1008" alt="Screenshot 2026-01-15 151156" src="https://github.com/user-attachments/assets/09c650ce-0c28-4a82-a91d-fecb2b5ff188" />
<img width="1034" height="943" alt="Screenshot 2026-01-15 151204" src="https://github.com/user-attachments/assets/1a5b5e6b-f9ae-4cba-850d-54a429b6e4e2" />
<img width="1098" height="784" alt="Screenshot 2026-01-15 151212" src="https://github.com/user-attachments/assets/130dd6c4-570b-4e74-b277-905db0bbe008" />

Deepfake Detector

Multi-layer deepfake detection tool using behavioral and visual analysis techniques.
Can we tell what's real anymore? This project explores modern approaches to detecting AI-generated video manipulation through a combination of behavioral patterns, texture analysis, frequency domain analysis, and temporal consistency checking.

📊 Overview

This is an educational proof-of-concept demonstrating modern deepfake detection approaches. It analyzes videos using four distinct methods:

🧠 Behavioral Pattern Analysis

Humans are beautifully inconsistent. We blink irregularly, make tiny unconscious movements, breathe naturally. Deepfakes often lack this organic chaos.
Tracks: Blink patterns, eye movements, head micro-movements, natural motion variance

🎨 Texture Consistency Analysis

GAN-generated faces leave subtle artifacts—unnatural smoothness, weird noise patterns, texture inconsistencies.
Detects: Gradient patterns, texture variance, local consistency, facial smoothness anomalies

📊 Frequency Domain Analysis (FFT)

Real videos have characteristic frequency distributions. Deepfakes often have suspicious spectral signatures invisible to the human eye.
Analyzes: High-frequency artifacts, spectral anomalies, frequency distribution patterns

⏱️ Temporal Stability Assessment

Videos are sequences of frames that must be consistent. Deepfakes can have identity drift, lighting jumps, or temporal glitches.
Measures: Frame-to-frame consistency, embedding drift, temporal coherence

🚀 Quick Start

Installation

Clone the repository:

bashgit clone https://github.com/Dushani-Ekanayake/deepfake-detecter.git
cd deepfake-detecter

Install dependencies:

bashpip install -r requirements.txt

Run the detector:

bashpython deepfake_detector.py

Enter video path when prompted:

📁 Enter video file path: C:\Users\User\Desktop\test_video.mp4
Requirements

Python 3.8 or higher
OpenCV
NumPy
SciPy

All dependencies are listed in requirements.txt

💻 Usage
Basic Usage
bashpython deepfake_detector.py
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
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── TESTING_GUIDE.md        # Guide for testing the detector
└── LICENSE                 # MIT License

🧪 Testing & Validation

Current Testing Status
⚠️ Limited validation - This is a proof-of-concept requiring extensive testing
Recommended Testing Approach
To properly validate this tool:

Collect Test Dataset:

5-10 authentic videos (real people)
5-10 deepfake videos (from public datasets)
Ensure varied lighting, angles, quality


Public Deepfake Datasets:

FaceForensics++: https://github.com/ondyari/FaceForensics
Celeb-DF: http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html
DFDC: https://ai.facebook.com/datasets/dfdc/


Run Systematic Tests:

bash   python deepfake_detector.py
   # Document: suspicion score, verdict, false positives/negatives
See TESTING_GUIDE.md for detailed testing instructions.
Expected Performance

Accuracy: 60-70% on mixed-quality content
Better with: Older/obvious deepfakes, clear authentic videos
Struggles with: State-of-the-art deepfakes, poor quality footage


⚠️ Critical Limitations & Scope

What This Tool IS:

✅ Educational proof-of-concept demonstrating detection principles
✅ Learning artifact for computer vision and analysis techniques
✅ Multi-layered approach showing modern detection methods
✅ Undergraduate research project exploring the deepfake problem

What This Tool IS NOT:

❌ NOT production-ready - Do not use for real-world decisions
❌ NOT forensically sound - Not suitable for legal/official use
❌ NOT highly accurate - Expect ~30-40% error rate
❌ NOT a replacement for professional analysis tools
Technical Limitations:

No neural network detection: Uses classical CV methods only
Simple feature extraction: Basic texture/frequency analysis
Limited dataset testing: Not validated on large-scale benchmarks
False positives/negatives: Significant error rate expected
Requires clear footage: Poor lighting/angles reduce accuracy
Single-face only: Not optimized for multi-person videos
No real-time capability: Processing can be slow for long videos

Why These Limitations Exist:
This project prioritizes learning and demonstration over accuracy. Building a truly robust deepfake detector requires:

Large labeled datasets (10,000+ videos)
Deep learning models (ResNet, EfficientNet, etc.)
Extensive training and validation
Significant computational resources
Professional research team

For an educational project, we focus on understanding the principles rather than achieving state-of-the-art results.

🔧 How It Works

Analysis Pipeline

Video Input → Face Detection → Multi-Layer Analysis → Suspicion Score
                                      ├─ Behavioral (15% weight)
                                      ├─ Texture (25% weight)
                                      ├─ Frequency (25% weight)
                                      └─ Temporal (20% weight)
Technical Methods
1. Behavioral Analysis:

Eye Aspect Ratio (EAR) calculation for blink detection
Statistical analysis of blink intervals (std deviation)
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

Python 3.8+ - Core programming language
OpenCV - Computer vision and video processing
NumPy - Mathematical operations and array processing
SciPy - Signal processing (FFT) and scientific computing

No deep learning frameworks required—demonstrates what's possible with classical techniques.

📖 Related Article

Read the full technical article explaining the deepfake detection problem and multi-layer verification approach:
"Fighting Deepfakes in 2026: How We Can Actually Tell What's Real"
[Link to Medium article]

The article covers:

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
 Facial landmark detection with dlib/MediaPipe
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

