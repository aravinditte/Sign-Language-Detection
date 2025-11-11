# Sign Language Detection System

A comprehensive real-time sign language detection system using MediaPipe, OpenCV, and deep learning to recognize and classify hand gestures.

**Author:** Aravind  
**Repository:** [github.com/aravinditte/Sign-Language-Detection](https://github.com/aravinditte/Sign-Language-Detection)

## 🚀 Features

- **Real-time Hand Tracking**: Utilizes MediaPipe's hand landmark detection for accurate tracking
- **Neural Network Classification**: Deep learning model trained on custom gesture data
- **Data Collection Tool**: Built-in interface to collect and label training data
- **High Accuracy**: Optimized architecture with dropout and batch normalization
- **Smooth Predictions**: Temporal smoothing for stable real-time recognition
- **Interactive UI**: Clean interface with confidence scores and FPS counter
- **Extensible**: Easy to add new gestures and retrain

## 📋 Requirements

- Python 3.8+
- Webcam/Camera
- 4GB+ RAM recommended

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/aravinditte/Sign-Language-Detection.git
cd Sign-Language-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Project Structure

```
Sign-Language-Detection/
├── src/
│   ├── data_collection.py      # Data collection tool
│   ├── train_model.py           # Model training script
│   ├── real_time_detection.py  # Real-time detection
│   └── utils.py                 # Utility functions
├── data/
│   └── processed/               # Collected landmark data
├── models/
│   ├── gesture_classifier.h5    # Trained model
│   └── label_encoder.pkl        # Label encoder
├── config/
│   └── config.yaml              # Configuration file
├── docs/
│   ├── IMPLEMENTATION.md        # Detailed implementation guide
│   └── COMPREHENSIVE_GUIDE.md   # Complete project guide
├── requirements.txt
├── README.md
└── LICENSE
```

## 🎯 Quick Start

### 1. Collect Training Data

```bash
python src/data_collection.py
```

- Press A-Z or 0-9 to start collecting a gesture
- Perform the gesture naturally with variations
- Collect 500+ samples per gesture
- Press Q to quit

### 2. Train the Model

```bash
python src/train_model.py
```

This will train a neural network on your collected data and save the model.

### 3. Run Real-time Detection

```bash
python src/real_time_detection.py
```

**Controls:**
- **ESC** - Exit
- **S** - Screenshot
- **R** - Reset predictions

## 🧠 How It Works

### Hand Landmark Detection

MediaPipe detects 21 key points on the hand:
- Wrist (1 point)
- Thumb (4 points)
- Index, Middle, Ring, Pinky (4 points each)

Each landmark provides x, y, z coordinates in 3D space.

### Feature Engineering

From the 21 landmarks, we extract:
- **63 normalized coordinates** (x, y, z for each landmark)
- **5 angles** between finger segments
- **Relative positions** to wrist

This creates a 68-dimensional feature vector.

### Neural Network Architecture

```
Input (68 features)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Output(num_classes) + Softmax
```

### Real-time Inference

1. Capture webcam frame
2. Detect hand with MediaPipe
3. Extract 21 landmarks
4. Compute feature vector
5. Predict with neural network
6. Apply temporal smoothing
7. Display result

## 📊 Performance

- **Training Accuracy**: ~95-98%
- **Real-time FPS**: 25-30 FPS
- **Inference Time**: ~10-15ms per frame
- **Hand Detection**: ~20-30ms per frame

## 🔧 Configuration

Customize settings in `config/config.yaml`:

```yaml
camera:
  width: 1280
  height: 720
  fps: 30

mediapipe:
  max_num_hands: 1
  min_detection_confidence: 0.7

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## 🎨 Extending the System

### Add New Gestures

1. Run data collection
2. Collect 500+ samples for new gesture
3. Retrain model
4. Deploy updated model

### Improve Accuracy

- Collect more diverse data (lighting, angles, backgrounds)
- Increase samples per gesture (1000+)
- Add data augmentation
- Try deeper architectures
- Experiment with LSTM for temporal gestures

### Integration Ideas

- **Text-to-Speech**: Add voice output for recognized gestures
- **Mobile App**: Port to Android/iOS with TensorFlow Lite
- **Web Interface**: Create web app with TensorFlow.js
- **Two-handed Gestures**: Support both hands
- **Sentence Formation**: Combine gestures into words/sentences

## 🐛 Troubleshooting

### Camera Issues
- Check camera index (try 0, 1, 2)
- Verify camera permissions
- Test with other applications

### Poor Detection
- Ensure good lighting
- Use plain background
- Position hand clearly in frame
- Collect more training data

### Slow Performance
- Lower camera resolution
- Reduce MediaPipe complexity
- Close other applications
- Use GPU acceleration

## 📈 Performance Tips

### Speed Optimization
- Model quantization for faster inference
- Frame skipping (process every nth frame)
- Lower input resolution
- Use TensorFlow Lite

### Accuracy Optimization
- Data augmentation during training
- Ensemble methods (multiple models)
- Temporal smoothing (average predictions)
- Transfer learning from pre-trained models

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **OpenCV** for computer vision tools
- **TensorFlow** for deep learning framework
- Sign language community for inspiration

## 📞 Contact

**Aravind** - Full Stack Developer

- GitHub: [@aravinditte](https://github.com/aravinditte)
- Repository: [Sign-Language-Detection](https://github.com/aravinditte/Sign-Language-Detection)

## 📚 References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)

---

**Built with ❤️ by Aravind**

## 🎯 Future Enhancements

- [ ] Dynamic gesture recognition (hand movements over time)
- [ ] Multi-language sign language support
- [ ] Mobile application (iOS/Android)
- [ ] Web-based interface
- [ ] Cloud API deployment
- [ ] Two-handed gesture support
- [ ] Word and sentence prediction
- [ ] Integration with communication apps
- [ ] Performance analytics dashboard
- [ ] Gesture customization interface