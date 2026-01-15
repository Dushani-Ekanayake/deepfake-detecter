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


