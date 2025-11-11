"""
Sign Language Detection - Data Collection Module
Author: Aravind
Description: Collects hand gesture data using webcam and MediaPipe for training

This script captures hand landmarks in real-time and saves them to a CSV file
for training the gesture classification model.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime


class DataCollector:
    """Collects hand gesture landmark data for training"""
    
    def __init__(self, output_dir='data/processed'):
        """
        Initialize the data collector
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.data = []
        self.current_gesture = None
        self.samples_per_gesture = 0
        self.target_samples = 500
        
        # UI colors
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        
    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmark coordinates from detected hand
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            List of normalized landmark coordinates [x1, y1, z1, x2, y2, z2, ...]
        """
        landmarks = []
        
        # Get wrist position for normalization
        wrist = hand_landmarks.landmark[0]
        
        for landmark in hand_landmarks.landmark:
            # Normalize coordinates relative to wrist
            landmarks.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])
            
        return landmarks
    
    def calculate_angles(self, hand_landmarks):
        """
        Calculate angles between finger segments
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            List of angles in radians
        """
        angles = []
        
        # Finger tip IDs: thumb, index, middle, ring, pinky
        finger_tips = [4, 8, 12, 16, 20]
        finger_mcp = [2, 5, 9, 13, 17]  # Metacarpophalangeal joints
        
        wrist = np.array([hand_landmarks.landmark[0].x, 
                         hand_landmarks.landmark[0].y,
                         hand_landmarks.landmark[0].z])
        
        for tip_id, mcp_id in zip(finger_tips, finger_mcp):
            tip = np.array([hand_landmarks.landmark[tip_id].x,
                           hand_landmarks.landmark[tip_id].y,
                           hand_landmarks.landmark[tip_id].z])
            mcp = np.array([hand_landmarks.landmark[mcp_id].x,
                           hand_landmarks.landmark[mcp_id].y,
                           hand_landmarks.landmark[mcp_id].z])
            
            # Calculate angle
            v1 = mcp - wrist
            v2 = tip - mcp
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)
            
        return angles
    
    def draw_info(self, frame, gesture, count, fps):
        """
        Draw information overlay on frame
        
        Args:
            frame: Video frame
            gesture: Current gesture being collected
            count: Number of samples collected
            fps: Current FPS
        """
        h, w, _ = frame.shape
        
        # Semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, 'Sign Language Data Collection', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_YELLOW, 2)
        
        # Instructions
        if gesture:
            cv2.putText(frame, f'Collecting: {gesture}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_GREEN, 2)
            cv2.putText(frame, f'Samples: {count}/{self.target_samples}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GREEN, 2)
            
            # Progress bar
            progress = int((count / self.target_samples) * (w - 20))
            cv2.rectangle(frame, (10, 100), (10 + progress, 110), self.COLOR_GREEN, -1)
            cv2.rectangle(frame, (10, 100), (w - 10, 110), self.COLOR_GREEN, 2)
        else:
            cv2.putText(frame, 'Press A-Z or 0-9 to start collecting gesture', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_BLUE, 2)
            cv2.putText(frame, 'Press Q to quit', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_BLUE, 2)
        
        # FPS counter
        cv2.putText(frame, f'FPS: {fps:.1f}', 
                   (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_YELLOW, 2)
        
        # Made by watermark
        cv2.putText(frame, 'Built by Aravind', 
                   (w - 200, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
    
    def start(self):
        """Start data collection from webcam"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 60)
        print("Sign Language Data Collection Tool")
        print("=" * 60)
        print("Instructions:")
        print("1. Position your hand clearly in front of the camera")
        print("2. Press A-Z or 0-9 to start collecting that gesture")
        print("3. Perform the gesture naturally with variations")
        print("4. Press Q to quit and save data")
        print("=" * 60)
        
        prev_time = time.time()
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            # Draw hand landmarks and collect data
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=self.COLOR_GREEN, thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=self.COLOR_BLUE, thickness=2)
                    )
                    
                    # Collect data if gesture is active
                    if self.current_gesture and self.samples_per_gesture < self.target_samples:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        angles = self.calculate_angles(hand_landmarks)
                        
                        # Combine features
                        features = landmarks + angles
                        features.append(self.current_gesture)
                        
                        self.data.append(features)
                        self.samples_per_gesture += 1
                        
                        # Check if target reached
                        if self.samples_per_gesture >= self.target_samples:
                            print(f"✓ Completed collecting {self.target_samples} samples for '{self.current_gesture}'")
                            self.current_gesture = None
                            self.samples_per_gesture = 0
            
            # Draw UI
            self.draw_info(frame, self.current_gesture, self.samples_per_gesture, fps)
            
            # Display frame
            cv2.imshow('Data Collection - Sign Language Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key != 255:  # Any key pressed
                char = chr(key).upper()
                if char.isalnum() and not self.current_gesture:
                    self.current_gesture = char
                    self.samples_per_gesture = 0
                    print(f"Started collecting gesture: '{char}'")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.save_data()
    
    def save_data(self):
        """Save collected data to CSV file"""
        if not self.data:
            print("No data collected!")
            return
        
        # Create column names
        landmark_cols = []
        for i in range(21):
            landmark_cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
        angle_cols = ['thumb_angle', 'index_angle', 'middle_angle', 'ring_angle', 'pinky_angle']
        columns = landmark_cols + angle_cols + ['label']
        
        # Create DataFrame
        df = pd.DataFrame(self.data, columns=columns)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'landmarks_{timestamp}.csv'
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        print("\n" + "=" * 60)
        print(f"Data saved successfully!")
        print(f"File: {filepath}")
        print(f"Total samples: {len(self.data)}")
        print(f"Gestures: {df['label'].unique().tolist()}")
        print(f"Samples per gesture: {df['label'].value_counts().to_dict()}")
        print("=" * 60)


if __name__ == '__main__':
    collector = DataCollector()
    collector.start()
