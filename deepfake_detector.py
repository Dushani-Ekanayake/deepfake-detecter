"""
Deepfake Detector - Behavioral & Visual Analysis
Multi-layered approach combining behavioral patterns and visual artifacts
Educational proof-of-concept for deepfake detection research
"""

import cv2
import numpy as np
from scipy.spatial import distance
from scipy import fftpack
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

class DeepfakeDetector:
    def __init__(self):
        # Initialize OpenCV face and eye detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Behavioral metrics storage
        self.blink_history = deque(maxlen=100)
        self.eye_open_history = deque(maxlen=30)
        self.face_position_history = deque(maxlen=50)
        self.face_size_history = deque(maxlen=50)
        
        # Visual analysis storage
        self.texture_scores = []
        self.frequency_anomalies = []
        self.embedding_drifts = []
        
        # Tracking variables
        self.previous_eye_count = 2
        self.previous_face_embedding = None
        self.blink_counter = 0
        self.total_blinks = 0
        self.frames_processed = 0
        
        # Suspicion scores by category (0-100 scale)
        self.suspicion_scores = {
            'blink_pattern': 0,
            'movement_analysis': 0,
            'texture_consistency': 0,
            'frequency_artifacts': 0,
            'temporal_stability': 0
        }
        
        # Detection confidence tracking
        self.confidence_metrics = {
            'face_detected_frames': 0,
            'eyes_detected_frames': 0,
            'analysis_completeness': 0
        }
    
    def detect_faces_and_eyes(self, frame):
        """Detect faces and eyes in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        eyes_data = []
        face_data = None
        
        if len(faces) > 0:
            self.confidence_metrics['face_detected_frames'] += 1
            face_data = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face_data
            
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            
            if len(eyes) > 0:
                self.confidence_metrics['eyes_detected_frames'] += 1
            
            for (ex, ey, ew, eh) in eyes:
                eyes_data.append((x + ex, y + ey, ew, eh))
        
        return face_data, eyes_data, gray
    
    def analyze_texture_consistency(self, frame, face_rect):
        """
        Analyze face texture for inconsistencies
        Deepfakes often have subtle texture artifacts from GAN generation
        """
        if face_rect is None:
            return
        
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance using Laplacian
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Calculate local binary pattern variance (texture consistency)
        # Simple approximation: compare neighboring pixel differences
        dx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        texture_score = np.std(gradient_magnitude)
        
        self.texture_scores.append(texture_score)
        
        # Deepfakes often have either too smooth or too noisy textures
        if len(self.texture_scores) > 20:
            avg_texture = np.mean(self.texture_scores[-20:])
            texture_std = np.std(self.texture_scores[-20:])
            
            # Suspicious if texture is too uniform (overly smooth)
            if texture_std < 5.0:
                self.suspicion_scores['texture_consistency'] += 0.5
            # Or if variance is abnormally high (GAN artifacts)
            elif texture_variance < 50 or texture_variance > 1000:
                self.suspicion_scores['texture_consistency'] += 0.3
    
    def analyze_frequency_artifacts(self, frame, face_rect):
        """
        Frequency domain analysis using FFT
        Deepfakes often have detectable artifacts in frequency domain
        """
        if face_rect is None:
            return
        
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Apply 2D FFT
        fft = fftpack.fft2(gray_face)
        fft_shift = fftpack.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Analyze high-frequency components
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        
        # Extract high-frequency ring (outer region)
        mask = np.zeros((rows, cols))
        r_outer = min(rows, cols) // 2
        r_inner = r_outer // 2
        y_grid, x_grid = np.ogrid[:rows, :cols]
        mask_area = ((x_grid - ccol)**2 + (y_grid - crow)**2 >= r_inner**2) & \
                    ((x_grid - ccol)**2 + (y_grid - crow)**2 <= r_outer**2)
        mask[mask_area] = 1
        
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
            self.frequency_anomalies.append(high_freq_ratio)
            
            # Deepfakes often have unusual high-frequency distributions
            if len(self.frequency_anomalies) > 10:
                avg_ratio = np.mean(self.frequency_anomalies[-10:])
                # Suspicious if too much or too little high-frequency content
                if avg_ratio < 0.15 or avg_ratio > 0.35:
                    self.suspicion_scores['frequency_artifacts'] += 0.4
    
    def analyze_temporal_stability(self, frame, face_rect):
        """
        Analyze frame-to-frame consistency
        Deepfakes can have temporal inconsistencies in face embeddings
        """
        if face_rect is None:
            return
        
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return
        
        # Resize to standard size for comparison
        face_resized = cv2.resize(face_region, (64, 64))
        
        # Create simple embedding using histogram
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        embedding = cv2.calcHist([gray_face], [0], None, [32], [0, 256]).flatten()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-7)  # Normalize
        
        if self.previous_face_embedding is not None:
            # Calculate embedding drift
            drift = np.linalg.norm(embedding - self.previous_face_embedding)
            self.embedding_drifts.append(drift)
            
            # Check for unusual temporal jumps
            if len(self.embedding_drifts) > 15:
                drift_std = np.std(self.embedding_drifts[-15:])
                avg_drift = np.mean(self.embedding_drifts[-15:])
                
                # Suspicious if drift is too high (inconsistent) or too low (too stable)
                if drift_std > 0.3 or avg_drift < 0.01:
                    self.suspicion_scores['temporal_stability'] += 0.3
        
        self.previous_face_embedding = embedding
    
    def analyze_blink_pattern(self, num_eyes):
        """Analyze blink patterns based on eye detection"""
        self.eye_open_history.append(num_eyes)
        
        # Blink detection
        if self.previous_eye_count == 2 and num_eyes < 2:
            self.blink_counter += 1
        elif self.previous_eye_count < 2 and num_eyes == 2:
            if self.blink_counter > 0:
                self.total_blinks += 1
                self.blink_history.append(time.time())
                self.blink_counter = 0
        
        self.previous_eye_count = num_eyes
        
        # Analyze blink rate
        if len(self.blink_history) > 10:
            time_span = self.blink_history[-1] - self.blink_history[0]
            if time_span > 5:
                blink_rate = (len(self.blink_history) / time_span) * 60
                
                # Calculate suspicion score for blink rate
                if blink_rate < 10:
                    self.suspicion_scores['blink_pattern'] += 0.5  # Too few blinks
                elif blink_rate > 30:
                    self.suspicion_scores['blink_pattern'] += 0.5  # Too many blinks
                
                # Check blink consistency
                if len(self.blink_history) > 5:
                    intervals = [self.blink_history[i] - self.blink_history[i-1] 
                                for i in range(1, len(self.blink_history))]
                    std_dev = np.std(intervals)
                    
                    if std_dev < 0.3:  # Too regular
                        self.suspicion_scores['blink_pattern'] += 0.4
                    elif std_dev > 5.0:  # Too random
                        self.suspicion_scores['blink_pattern'] += 0.3
    
    def analyze_movement_patterns(self, face_rect):
        """Analyze natural face movement patterns"""
        if face_rect is not None:
            x, y, w, h = face_rect
            center_x = x + w/2
            center_y = y + h/2
            size = w * h
            
            self.face_position_history.append((center_x, center_y))
            self.face_size_history.append(size)
            
            # Check for unnatural stillness
            if len(self.face_position_history) > 20:
                positions = np.array(self.face_position_history)
                variance = np.var(positions, axis=0)
                total_variance = variance[0] + variance[1]
                
                # Too still
                if total_variance < 50:
                    self.suspicion_scores['movement_analysis'] += 0.4
                
                # Check size consistency
                if len(self.face_size_history) > 20:
                    size_variance = np.var(self.face_size_history)
                    size_std = np.std(self.face_size_history)
                    
                    # Unnatural stability or too much variation
                    if size_std < 100:
                        self.suspicion_scores['movement_analysis'] += 0.3
                    elif size_std > 5000:
                        self.suspicion_scores['movement_analysis'] += 0.2
    
    def calculate_overall_suspicion(self):
        """
        Calculate overall suspicion score (0-100%)
        0% = Definitely authentic
        100% = Definitely fake
        """
        # Normalize each category score
        normalized_scores = {}
        max_vals = {
            'blink_pattern': 200,
            'movement_analysis': 150,
            'texture_consistency': 100,
            'frequency_artifacts': 80,
            'temporal_stability': 100
        }
        
        for key, value in self.suspicion_scores.items():
            normalized = min(100, (value / max_vals[key]) * 100)
            normalized_scores[key] = normalized
        
        # Weighted average (modern features get more weight)
        weights = {
            'blink_pattern': 0.15,
            'movement_analysis': 0.15,
            'texture_consistency': 0.25,
            'frequency_artifacts': 0.25,
            'temporal_stability': 0.20
        }
        
        overall_suspicion = sum(
            normalized_scores[key] * weights[key] 
            for key in normalized_scores.keys()
        )
        
        # Calculate confidence in detection
        face_detection_rate = self.confidence_metrics['face_detected_frames'] / max(1, self.frames_processed)
        eye_detection_rate = self.confidence_metrics['eyes_detected_frames'] / max(1, self.frames_processed)
        
        analysis_confidence = (face_detection_rate * 0.6 + eye_detection_rate * 0.4) * 100
        
        return overall_suspicion, normalized_scores, analysis_confidence
    
    def process_video(self, video_path):
        """Process video and analyze for deepfake indicators"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("Analyzing video for deepfake indicators...")
        print("=" * 70)
        print(f"Video Properties:")
        print(f"   • FPS: {fps:.1f}")
        print(f"   • Total Frames: {total_frames}")
        print(f"   • Duration: {total_frames/fps:.1f}s")
        print("=" * 70)
        print("\n Running Multi-Layer Analysis:")
        print("   ✓ Behavioral Pattern Analysis")
        print("   ✓ Texture Consistency Check")
        print("   ✓ Frequency Domain Analysis (FFT)")
        print("   ✓ Temporal Stability Assessment")
        print("\n" + "=" * 70)
        
        self.frames_processed = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frames_processed += 1
            
            # Detect faces and eyes
            face_data, eyes_data, gray = self.detect_faces_and_eyes(frame)
            num_eyes = len(eyes_data)
            
            # Run all analysis methods
            self.analyze_blink_pattern(num_eyes)
            self.analyze_movement_patterns(face_data)
            self.analyze_texture_consistency(frame, face_data)
            self.analyze_frequency_artifacts(frame, face_data)
            self.analyze_temporal_stability(frame, face_data)
            
            # Display progress
            if self.frames_processed % 30 == 0:
                progress = (self.frames_processed / total_frames) * 100 if total_frames > 0 else 0
                print(f" Progress: {progress:.1f}% ({self.frames_processed}/{total_frames} frames)", end='\r')
        
        print(f"\n\n✅ Analysis complete! Processed {self.frames_processed} frames.\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate detailed analysis report with suspicion scores"""
        overall_suspicion, category_scores, confidence = self.calculate_overall_suspicion()
        authenticity_score = 100 - overall_suspicion
        
        print("=" * 70)
        print("DEEPFAKE DETECTION REPORT")
        print("=" * 70)
        
        # Main verdict
        print(f"\n OVERALL SUSPICION SCORE: {overall_suspicion:.1f}%")
        print(f" AUTHENTICITY SCORE: {authenticity_score:.1f}%")
        print(f" ANALYSIS CONFIDENCE: {confidence:.1f}%")
        
        # Verdict with color coding
        if overall_suspicion < 25:
            verdict = " LIKELY AUTHENTIC"
            explanation = "Strong indicators of genuine human behavior and visual consistency."
        elif overall_suspicion < 50:
            verdict = " LOW SUSPICION"
            explanation = "Minor anomalies detected, but likely authentic with low concern."
        elif overall_suspicion < 70:
            verdict = " MODERATE SUSPICION"
            explanation = "Multiple anomalies detected. Manual review strongly recommended."
        else:
            verdict = "HIGH SUSPICION"
            explanation = "Significant indicators of manipulation detected across multiple metrics."
        
        print(f"\n VERDICT: {verdict}")
        print(f"{explanation}")
        
        # Category breakdown
        print(f"\n DETAILED SUSPICION BREAKDOWN:")
        print(f"   • Blink Pattern Analysis:      {category_scores['blink_pattern']:>6.1f}%")
        print(f"   • Movement Analysis:           {category_scores['movement_analysis']:>6.1f}%")
        print(f"   • Texture Consistency:         {category_scores['texture_consistency']:>6.1f}%")
        print(f"   • Frequency Artifacts (FFT):   {category_scores['frequency_artifacts']:>6.1f}%")
        print(f"   • Temporal Stability:          {category_scores['temporal_stability']:>6.1f}%")
        
        # Behavioral metrics
        print(f"\n BEHAVIORAL METRICS:")
        print(f"   • Total Blinks: {self.total_blinks}")
        
        if len(self.blink_history) > 1:
            time_span = self.blink_history[-1] - self.blink_history[0]
            if time_span > 0:
                blink_rate = (len(self.blink_history) / time_span) * 60
                print(f"   • Blink Rate: {blink_rate:.1f} blinks/min (normal: 15-20)")
        
        print(f"   • Frames Analyzed: {self.frames_processed}")
        print(f"   • Face Detection Rate: {confidence:.1f}%")
        
        # Key findings
        print(f"\n KEY FINDINGS:")
        findings = []
        
        if category_scores['blink_pattern'] > 50:
            findings.append(" Abnormal blinking patterns detected")
        if category_scores['movement_analysis'] > 50:
            findings.append(" Unnatural movement characteristics")
        if category_scores['texture_consistency'] > 50:
            findings.append(" Texture inconsistencies in facial region")
        if category_scores['frequency_artifacts'] > 50:
            findings.append(" Suspicious frequency-domain artifacts")
        if category_scores['temporal_stability'] > 50:
            findings.append(" Temporal inconsistencies detected")
        
        if not findings:
            findings.append(" No major anomalies detected")
            findings.append("Behavioral patterns appear natural")
            findings.append("Visual consistency maintained")
        
        for finding in findings:
            print(f"   {finding}")
        
        # Limitations and disclaimer
        print(f"\n" + "=" * 70)
        print("  IMPORTANT LIMITATIONS:")
        print("=" * 70)
        print("""
This is an EDUCATIONAL proof-of-concept tool with known limitations:

✗ NOT suitable for forensic or legal use
✗ NOT a replacement for professional deepfake analysis
✗ Limited by simple feature extraction methods
✗ No neural network-based detection
✗ Requires clear, well-lit video with visible face
✗ May produce false positives/negatives

✓ Useful for understanding deepfake detection concepts
✓ Demonstrates multi-layered analysis approach
✓ Educational tool for learning computer vision
✓ Shows modern detection techniques (FFT, texture analysis)

RECOMMENDATION: Always verify important videos through:
• Multiple detection tools
• Professional forensic analysis
• Original source verification
• Cross-reference with other evidence
        """)
        print("=" * 70)
        
        return {
            'suspicion_score': overall_suspicion,
            'authenticity_score': authenticity_score,
            'confidence': confidence,
            'verdict': verdict,
            'category_scores': category_scores,
            'total_blinks': self.total_blinks,
            'frames_processed': self.frames_processed
        }


def main():
    """Main execution function"""
    print("=" * 70)
    print("DEEPFAKE DETECTOR v2.0 - Multi-Layer Analysis")
    print("=" * 70)
    print("""
Educational proof-of-concept for deepfake detection research

ANALYSIS METHODS:
  🔸 Behavioral Pattern Analysis (blink patterns, movements)
  🔸 Texture Consistency Check (GAN artifact detection)
  🔸 Frequency Domain Analysis (FFT-based)
  🔸 Temporal Stability Assessment (frame-to-frame consistency)

⚠️  This is NOT production-ready software. See limitations in output.
    """)
    
    try:
        detector = DeepfakeDetector()
    except Exception as e:
        print(f" Error initializing detector: {e}")
        return
    
    video_path = input("📁 Enter video file path: ").strip()
    video_path = video_path.strip('"').strip("'")
    
    if not video_path:
        print("\n USAGE: python deepfake_detector.py")
        print("   Then enter the full path to your video file.")
        print("   Example: C:\\Users\\User\\Desktop\\test_video.mp4\n")
        return
    
    try:
        print(f"\n Loading: {video_path}\n")
        results = detector.process_video(video_path)
        
        print(f"\n✅ Analysis complete!")
        print(f" Suspicion Score: {results['suspicion_score']:.1f}%")
        print(f" Authenticity Score: {results['authenticity_score']:.1f}%\n")
        
    except FileNotFoundError:
        print(f"\n Video file not found: {video_path}")
    except ValueError as e:
        print(f"\n Error: {e}")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print("   Check that video format is supported (MP4, AVI, MOV, MKV)")


if __name__ == "__main__":
    main()