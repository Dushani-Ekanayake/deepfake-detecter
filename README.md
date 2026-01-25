# 🔍 Deepfake Detector


<p align="center">
  <strong>Multi-layer deepfake detection tool using behavioral and visual analysis techniques.</strong>
</p>

<p align="center">
  Can we tell what's real anymore? This project explores modern approaches to detecting AI-generated video manipulation through a combination of behavioral patterns, texture analysis, frequency domain analysis, and temporal consistency checking.
</p>

---

## 📊 Overview

This is an **educational proof-of-concept** demonstrating modern deepfake detection approaches. It analyzes videos using four distinct methods:

| Method | Description |
|--------|-------------|
| 🧠 **Behavioral Pattern Analysis** | Tracks blink patterns, eye movements, head micro-movements, natural motion variance |
| 🎨 **Texture Consistency Analysis** | Detects gradient patterns, texture variance, local consistency, facial smoothness anomalies |
| 📊 **Frequency Domain Analysis** | Analyzes high-frequency artifacts, spectral anomalies, frequency distribution patterns |
| ⏱️ **Temporal Stability Assessment** | Measures frame-to-frame consistency, embedding drift, temporal coherence |

---


---

## 🚀 Quick Start

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Dushani-Ekanayake/deepfake-detecter.git
cd deepfake-detecter
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the detector:**
```bash
python deepfake_detector.py
```

**4. Enter video path when prompted:**
```
📁 Enter video file path: C:\Users\User\Desktop\test_video.mp4
```

### Requirements

- Python 3.8 or higher
- OpenCV
- NumPy
- SciPy

All dependencies are listed in `requirements.txt`

---

## 💻 Usage

### Basic Usage

```bash
python deepfake_detector.py
```

The tool will prompt you for a video file path. You can:
- Type the full path to your video file
- Drag and drop the video file into the terminal (automatically pastes path)

### Supported Video Formats

- MP4, AVI, MOV, MKV
- Any format supported by OpenCV

### Example Workflow

1. Record or download a test video
2. Run the detector
3. Review the suspicion score and detailed breakdown
4. Check category-specific findings
5. Use the report to understand which aspects seem suspicious

---

## 📁 Project Structure

```
deepfake-detecter/
├── deepfake_detector.py    # Main detection script
├── requirements.txt         # Python dependencies

```

---

## 🧪 Testing & Validation

### Current Testing Status

⚠️ **Limited validation** - This is a proof-of-concept requiring extensive testing

### Recommended Testing Approach

**1. Collect Test Dataset:**
- 5-10 authentic videos (real people)
- 5-10 deepfake videos (from public datasets)
- Ensure varied lighting, angles, quality

**2. Public Deepfake Datasets:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)
- [DFDC](https://ai.facebook.com/datasets/dfdc/)

**3. Run Systematic Tests:**
```bash
python deepfake_detector.py
# Document: suspicion score, verdict, false positives/negatives
```


### Expected Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 60-70% on mixed-quality content |
| **Better with** | Older/obvious deepfakes, clear authentic videos |
| **Struggles with** | State-of-the-art deepfakes, poor quality footage |

---

## ⚠️ Critical Limitations & Scope

### What This Tool IS ✅

- ✅ **Educational proof-of-concept** demonstrating detection principles
- ✅ **Learning artifact** for computer vision and analysis techniques
- ✅ **Multi-layered approach** showing modern detection methods
- ✅ **Research project** exploring the deepfake problem

### What This Tool IS NOT ❌

- ❌ **NOT production-ready** - Do not use for real-world decisions
- ❌ **NOT forensically sound** - Not suitable for legal/official use
- ❌ **NOT highly accurate** - Expect ~30-40% error rate
- ❌ **NOT a replacement** for professional analysis tools

### Technical Limitations

- **No neural network detection**: Uses classical CV methods only
- **Simple feature extraction**: Basic texture/frequency analysis
- **Limited dataset testing**: Not validated on large-scale benchmarks
- **False positives/negatives**: Significant error rate expected
- **Requires clear footage**: Poor lighting/angles reduce accuracy
- **Single-face only**: Not optimized for multi-person videos
- **No real-time capability**: Processing can be slow for long videos

> **Why These Limitations Exist:** This project prioritizes **learning and demonstration** over accuracy. Building a truly robust deepfake detector requires large labeled datasets (10,000+ videos), deep learning models, extensive training, and significant computational resources.

---

## 🔧 How It Works

### Analysis Pipeline

```
Video Input → Face Detection → Multi-Layer Analysis → Suspicion Score
                                      ├─ Behavioral (15% weight)
                                      ├─ Texture (25% weight)
                                      ├─ Frequency (25% weight)
                                      └─ Temporal (20% weight)
```

### Technical Methods

<details>
<summary><strong>1. Behavioral Analysis</strong></summary>

- Eye Aspect Ratio (EAR) calculation for blink detection
- Statistical analysis of blink intervals (std deviation)
- Head pose variance tracking
- Face position and size consistency
</details>

<details>
<summary><strong>2. Texture Analysis</strong></summary>

- Sobel gradient analysis for texture patterns
- Laplacian variance for smoothness detection
- Local texture consistency measurement
- GAN artifact detection
</details>

<details>
<summary><strong>3. Frequency Domain</strong></summary>

- 2D Fast Fourier Transform (FFT)
- High-frequency energy ratio analysis
- Spectral anomaly detection
- Frequency distribution patterns
</details>

<details>
<summary><strong>4. Temporal Stability</strong></summary>

- Histogram-based face embeddings
- Frame-to-frame drift calculation
- Temporal consistency measurement
- Identity stability tracking
</details>

---

## 🛠️ Built With

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Computer vision and video processing |
| **NumPy** | Mathematical operations and array processing |
| **SciPy** | Signal processing (FFT) and scientific computing |

> No deep learning frameworks required—demonstrates what's possible with classical techniques.

---

## 📖 Related Article

Read the full technical article explaining the deepfake detection problem and multi-layer verification approach:

**"Fighting Deepfakes : How We Can Actually Tell What's Real"**

📝 [Read on Medium](#) *https://dushaniekanayake.medium.com/fighting-deepfakes-how-we-can-actually-tell-whats-real-3d8aba2a6035*

**Article covers:**
- Why current detection methods struggle
- Blockchain-based verification
- Invisible digital watermarking
- AI-powered behavioral and visual analysis
- The future of video verification

---

## 🔮 Future Improvements

### Immediate Enhancements (Feasible)
- [ ] Web interface using Streamlit/Gradio
- [ ] Batch processing for multiple videos
- [ ] Export reports to PDF/JSON
- [ ] Video comparison mode (real vs suspected fake)
- [ ] Improved visualization of results

### Research Extensions (Advanced)
- [ ] CNN-based feature extraction (ResNet, EfficientNet)
- [ ] Facial landmark detection with dlib/MediaPipe
- [ ] Attention mechanisms for spatial analysis
- [ ] LSTM for temporal pattern learning
- [ ] Training on FaceForensics++ dataset

### Production Requirements (Not in Scope)
- [ ] Large-scale dataset training (100K+ videos)
- [ ] Model optimization and quantization
- [ ] API deployment with authentication
- [ ] Continuous learning pipeline
- [ ] Professional forensic validation



**Usage Terms:**
- ✅ Free to use for educational purposes
- ✅ Free to modify and adapt
- ✅ Free to share with attribution
- ❌ Not for commercial use without understanding limitations
- ❌ Not for legal/forensic applications

---

## ⚠️ Disclaimer

> **IMPORTANT:** This tool is for educational and research purposes only.

**Key Points:**
- Results should NOT be used as definitive proof of manipulation
- Always verify important videos through multiple methods
- Do not rely on this tool for legal, forensic, or high-stakes decisions
- The tool has significant limitations and known error rates
- Professional deepfake analysis requires specialized tools and expertise

By using this tool, you acknowledge that you understand its educational nature and limitations.

---


## 📚 Additional Resources

**Learn More About Deepfakes:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics) - Benchmark dataset
- [Deepfake Detection Challenge](https://ai.facebook.com/datasets/dfdc/) - Facebook's DFDC
- [Celeb-DF](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html) - Celebrity deepfakes

**Detection Research:**
- "Detecting Face Synthesis Using Convolutional Neural Networks"
- "The Eyes Tell All: Detecting Political Orientation from Eye Movement Data"
- "Exposing Deep Fakes Using Inconsistent Head Poses"

**Tools & Frameworks:**
- [Sensity AI](https://sensity.ai/) - Professional deepfake detection
- [Microsoft Video Authenticator](https://www.microsoft.com/en-us/ai/video-authenticator)

---

---

**📊 Project Stats**
<img width="1034" height="943" alt="Screenshot 2026-01-15 151204" src="https://github.com/user-attachments/assets/1cd0aaa7-bb0f-4e87-abdd-eec84d49b723" />
<img width="1117" height="1008" alt="Screenshot 2026-01-15 151156" src="https://github.com/user-attachments/assets/879126ad-bf0e-4d12-b438-a56136b97fe4" />



</div>
