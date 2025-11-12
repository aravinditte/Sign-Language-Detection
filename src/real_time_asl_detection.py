"""
Real-time ASL Alphabet CNN Detection
Author: Aravind
Usage: python src/real_time_asl_detection.py
"""
import cv2
import numpy as np
import tensorflow as tf
import json
import os

MODEL_PATH = 'models/asl_alphabet_cnn.h5'
CLASSES_PATH = 'models/asl_classes.json'
IMG_SIZE = 64
ROI_SIZE = 280

# Load model & class labels
def load_model_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Invert dict: int->label (sorted by index)
    idx_to_label = [None]*len(class_indices)
    for k,v in class_indices.items():
        idx_to_label[v] = k
    return model, idx_to_label

def preprocess(frame_roi):
    img = cv2.resize(frame_roi, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def main():
    model, idx_to_label = load_model_classes()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("\n[INFO] Running real-time ASL Alphabet detection...\n")
    print("[INFO] Press ESC to quit | S to save frame.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        x1 = (w - ROI_SIZE) // 2
        y1 = (h - ROI_SIZE) // 2
        x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE
        roi = frame[y1:y2, x1:x2]
        inp = preprocess(roi)
        preds = model.predict(inp, verbose=0)[0]
        idx = np.argmax(preds)
        confidence = preds[idx]
        prediction = idx_to_label[idx] if idx < len(idx_to_label) else "?"
        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
        # Display prediction
        label = f"Predicted: {prediction} | Conf: {confidence*100:.1f}%"
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 2)
        # Show preview of ROI used
        roi_small = cv2.resize(roi, (120, 120))
        frame[10:130, w-130:w-10] = roi_small
        cv2.putText(frame, 'ROI', (w-110, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),1)
        cv2.imshow('ASL Alphabet Detection (CNN)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC exit
            break
        elif key == ord('s'):
            fname = f'snapshot_{prediction}.png'
            cv2.imwrite(fname, frame)
            print(f"[INFO] Snapshot saved: {fname}")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")

if __name__ == '__main__':
    if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
        print("[ERROR] Trained model or class labels not found. Please train the model first.")
        print("  python src/train_asl_cnn.py")
        exit(1)
    main()
