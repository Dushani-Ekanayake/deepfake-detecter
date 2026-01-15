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
