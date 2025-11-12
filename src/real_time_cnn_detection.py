"""
Sign Language Detection - Real-time CNN Detection (Debugging/Logging Version)
Author: Aravind
"""

import cv2
import numpy as np
import json
import time
from collections import deque
import tensorflow as tf

class SignLanguageCNNDetector:
    def __init__(self, model_path='models/sign_language_cnn.h5',
                 classes_path='models/class_names.json'):
        print("Loading CNN model...")
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"Model loaded! Recognizing {len(self.class_names)} gestures")
        print(f"Classes: {', '.join(self.class_names)}")
        self.prediction_buffer = deque(maxlen=7)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_PURPLE = (255, 0, 255)
        self.roi_size = 300
    def preprocess_frame(self, roi, debug=False):
        img = cv2.resize(roi, (28, 28))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        img_norm = img.astype('float32') / 255.0
        if debug:
            print(f"[DEBUG] ROI mean: {img_norm.mean():.4f}, min: {img_norm.min():.4f}, max: {img_norm.max():.4f}")
            cv2.imshow("ROI_RAW", roi)
            cv2.imshow("PREPROCESSED", img)
        return img_norm.reshape(1, 28, 28, 1)
    def predict_gesture(self, roi):
        img = self.preprocess_frame(roi, debug=True)
        prediction = self.model.predict(img, verbose=0)[0]
        top3_idx = np.argsort(prediction)[-3:][::-1]
        top3_predictions = [(self.class_names[idx], prediction[idx]) for idx in top3_idx]
        class_idx = top3_idx[0]
        confidence = prediction[class_idx]
        gesture = self.class_names[class_idx]
        return gesture, confidence, top3_predictions
    def smooth_prediction(self, gesture, confidence):
        self.prediction_buffer.append((gesture, confidence))
        if len(self.prediction_buffer) >= 5:
            high_conf_predictions = [(g, c) for g, c in self.prediction_buffer if c > 0.6]
            if high_conf_predictions:
                from collections import Counter
                gestures = [g for g, c in high_conf_predictions]
                gesture_counts = Counter(gestures)
                most_common = gesture_counts.most_common(1)[0][0]
                avg_conf = np.mean([c for g, c in high_conf_predictions if g == most_common])
                return most_common, avg_conf
        return gesture, confidence
    def draw_ui(self, frame, gesture, confidence, fps, roi_top_left, top3_predictions, preprocessed_roi):
        h, w, _ = frame.shape
        x, y = roi_top_left
        cv2.rectangle(frame, (x, y), (x + self.roi_size, y + self.roi_size), self.COLOR_GREEN, 3)
        cv2.putText(frame, 'Place hand here (fill the box)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GREEN, 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, 'Sign Language CNN Detector', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_YELLOW, 2)
        if gesture and confidence > 0.5:
            color = self.COLOR_GREEN if confidence > 0.8 else self.COLOR_YELLOW
            cv2.putText(frame, f'Sign: {gesture}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)
            cv2.putText(frame, f'Confidence: {confidence*100:.1f}%', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_WHITE, 2)
            cv2.putText(frame, f'Top3: {top3_predictions[0][0]}({top3_predictions[0][1]*100:.0f}%) '
                               f'{top3_predictions[1][0]}({top3_predictions[1][1]*100:.0f}%) '
                               f'{top3_predictions[2][0]}({top3_predictions[2][1]*100:.0f}%)', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        else:
            cv2.putText(frame, 'Show sign in green box', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_RED, 2)
            cv2.putText(frame, 'Use plain background & good lighting', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_YELLOW, 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_YELLOW, 2)
        if preprocessed_roi is not None:
            preview = cv2.resize(preprocessed_roi, (100, 100))
            frame[10:110, w-110:w-10] = cv2.cvtColor((preview * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(frame, (w-110, 10), (w-10, 110), self.COLOR_WHITE, 2)
            cv2.putText(frame, 'Processed', (w-110, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        cv2.putText(frame, 'ESC: Exit | S: Screenshot | R: Reset | H: Help', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 2)
        cv2.putText(frame, 'Built by Aravind', (w - 210, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_PURPLE, 2)
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("\n" + "="*60)
        print("Sign Language CNN Detection - Real-time Mode")
        print("="*60)
        print("Instructions:")
        print("  - Place your hand in the green box")
        print("  - Fill the entire box with your hand")
        print("  - Use a plain, contrasting background")
        print("  - Ensure good lighting")
        print("  - Hold gesture steady for 2-3 seconds")
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
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            roi_x = (w - self.roi_size) // 2
            roi_y = (h - self.roi_size) // 2
            roi = frame[roi_y:roi_y + self.roi_size, roi_x:roi_x + self.roi_size]
            gesture, confidence, top3_predictions = self.predict_gesture(roi)
            gesture, confidence = self.smooth_prediction(gesture, confidence)
            preprocessed = self.preprocess_frame(roi).reshape(28, 28)
            self.draw_ui(frame, gesture, confidence, fps, (roi_x, roi_y), top3_predictions, preprocessed)
            cv2.imshow('Sign Language CNN Detection by Aravind', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                screenshot_count += 1
                filename = f'screenshot_cnn_{screenshot_count}.png'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                self.prediction_buffer.clear()
                print("Prediction buffer reset")
            elif key == ord('h') or key == ord('H'):
                print("\nTips for better accuracy:")
                print("  1. Use a plain, dark background")
                print("  2. Ensure good, even lighting")
                print("  3. Fill the green box with your hand")
                print("  4. Hold gesture steady for 2-3 seconds")
                print("  5. Make clear, distinct gestures")
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
