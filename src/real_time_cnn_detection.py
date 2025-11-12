"""
Sign Language Detection - Real-time CNN Detection
Author: Aravind
Description: Real-time sign language detection using trained CNN model
"""

import cv2
import numpy as np
import json
import time
from collections import deque
import tensorflow as tf

class SignLanguageCNNDetector:
    """Real-time sign language detector using CNN model"""
    
    def __init__(self, model_path='models/sign_language_cnn.h5',
                 classes_path='models/class_names.json'):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained CNN model
            classes_path: Path to class names JSON
        """
        print("Loading CNN model...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = json.load(f)
        
        print(f"Model loaded! Recognizing {len(self.class_names)} gestures")
        print(f"Classes: {', '.join(self.class_names)}")
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        
        # Colors
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_PURPLE = (255, 0, 255)
        
        # ROI (Region of Interest) settings
        self.roi_size = 300
        
    def preprocess_frame(self, roi):
        """
        Preprocess ROI for model prediction
        
        Args:
            roi: Region of interest from frame
            
        Returns:
            Preprocessed image ready for model
        """
        # Resize to 28x28
        img = cv2.resize(roi, (28, 28))
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Reshape for model input
        img = img.reshape(1, 28, 28, 1)
        
        return img
    
    def predict_gesture(self, roi):
        """
        Predict gesture from ROI
        
        Args:
            roi: Region of interest
            
        Returns:
            Tuple of (gesture_label, confidence)
        """
        # Preprocess
        img = self.preprocess_frame(roi)
        
        # Predict
        prediction = self.model.predict(img, verbose=0)[0]
        
        # Get top prediction
        class_idx = np.argmax(prediction)
        confidence = prediction[class_idx]
        gesture = self.class_names[class_idx]
        
        return gesture, confidence
    
    def smooth_prediction(self, gesture, confidence):
        """Smooth predictions using temporal buffer"""
        self.prediction_buffer.append((gesture, confidence))
        
        # Get most common prediction with high confidence
        if len(self.prediction_buffer) >= 3:
            high_conf_predictions = [(g, c) for g, c in self.prediction_buffer if c > 0.7]
            
            if high_conf_predictions:
                gestures = [g for g, c in high_conf_predictions]
                most_common = max(set(gestures), key=gestures.count)
                avg_conf = np.mean([c for g, c in high_conf_predictions if g == most_common])
                return most_common, avg_conf
        
        return gesture, confidence
    
    def draw_ui(self, frame, gesture, confidence, fps, roi_top_left):
        """Draw user interface"""
        h, w, _ = frame.shape
        
        # Draw ROI rectangle
        x, y = roi_top_left
        cv2.rectangle(frame, (x, y), (x + self.roi_size, y + self.roi_size), 
                     self.COLOR_GREEN, 3)
        
        # ROI instruction
        cv2.putText(frame, 'Place hand here', 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_GREEN, 2)
        
        # Top panel - semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, 'Sign Language CNN Detector', 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_YELLOW, 2)
        
        # Prediction
        if gesture and confidence > 0.5:
            color = self.COLOR_GREEN if confidence > 0.8 else self.COLOR_YELLOW
            cv2.putText(frame, f'Sign: {gesture}', 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Confidence
            cv2.putText(frame, f'Confidence: {confidence*100:.1f}%', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_WHITE, 2)
        else:
            cv2.putText(frame, 'Show sign in green box', 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_RED, 2)
        
        # FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', 
                   (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_YELLOW, 2)
        
        # Instructions
        cv2.putText(frame, 'ESC: Exit | S: Screenshot | R: Reset', 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 2)
        
        # Watermark
        cv2.putText(frame, 'Built by Aravind', 
                   (w - 210, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_PURPLE, 2)
    
    def run(self):
        """Start real-time detection"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("Sign Language CNN Detection - Real-time Mode")
        print("="*60)
        print("Instructions:")
        print("  - Place your hand in the green box")
        print("  - Make a clear sign gesture")
        print("  - ESC to exit")
        print("="*60)
        
        prev_time = time.time()
        fps = 0
        screenshot_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            # Define ROI (center of frame)
            roi_x = (w - self.roi_size) // 2
            roi_y = (h - self.roi_size) // 2
            roi = frame[roi_y:roi_y + self.roi_size, roi_x:roi_x + self.roi_size]
            
            # Predict gesture
            gesture, confidence = self.predict_gesture(roi)
            gesture, confidence = self.smooth_prediction(gesture, confidence)
            
            # Draw UI
            self.draw_ui(frame, gesture, confidence, fps, (roi_x, roi_y))
            
            # Display
            cv2.imshow('Sign Language CNN Detection by Aravind', frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f'screenshot_cnn_{screenshot_count}.png'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                self.prediction_buffer.clear()
                print("Prediction buffer reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Detection stopped. Thank you!")
        print("="*60)

if __name__ == '__main__':
    import os
    
    if not os.path.exists('models/sign_language_cnn.h5'):
        print("\nError: Model not found!")
        print("Please train the model first:")
        print("  python src/train_kaggle_model.py")
    else:
        try:
            detector = SignLanguageCNNDetector()
            detector.run()
        except Exception as e:
            print(f"\nError: {e}")
            print("Please ensure the model is trained correctly.")
