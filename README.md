# Sign Language Detection System

A comprehensive real-time sign language detection system using MediaPipe, OpenCV, and deep learning (CNN) to recognize and classify hand gestures from the Kaggle Sign Language MNIST dataset.

**Author:** Aravind  
**Repository:** [github.com/aravinditte/Sign-Language-Detection](https://github.com/aravinditte/Sign-Language-Detection)

## Features

- **CNN-based Classification**: Deep learning model trained on 27,455 sign language images
- **Kaggle Dataset Integration**: Uses Sign Language MNIST dataset (24 ASL letters)
- **Real-time Detection**: Live webcam detection with 95%+ accuracy
- **High Performance**: ~25-30 FPS on standard hardware
- **Professional UI**: Clean interface with confidence scores and FPS counter
- **Easy Setup**: Simple 3-step process to get started

## Requirements

- Python 3.8+
- Webcam/Camera
- 4GB+ RAM (8GB recommended)
- GPU (optional, speeds up training)

## Quick Start (3 Steps)

### Step 1: Setup Environment

```bash
# Clone repository
git clone https://github.com/aravinditte/Sign-Language-Detection.git
cd Sign-Language-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get Dataset

**Option A - Automatic (Recommended):**

1. Get Kaggle API token from https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\YourName\.kaggle\` (Windows)
3. Run: `python src/download_dataset.py`

**Option B - Manual:**

1. Download from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. Extract and place CSV files in `data/kaggle/`

### Step 3: Train & Run

```bash
# Train the CNN model
python src/train_kaggle_model.py

# Run real-time detection
python src/real_time_cnn_detection.py
```

## Project Structure

```
Sign-Language-Detection/
|-- data/
|   +-- kaggle/                    # Kaggle dataset
|       |-- sign_mnist_train.csv   # 27,455 training samples
|       +-- sign_mnist_test.csv    # 7,172 test samples
|
|-- models/
|   |-- sign_language_cnn.h5       # Trained CNN model
|   |-- class_names.json           # Class labels
|   |-- confusion_matrix_cnn.png   # Performance visualizations
|   +-- training_history_cnn.png
|
|-- src/
|   |-- download_dataset.py        # Download Kaggle dataset
|   |-- train_kaggle_model.py      # Train CNN model
|   |-- real_time_cnn_detection.py # Real-time detection
|   |-- data_collection.py         # Custom data collection
|   |-- train_model.py             # Train on custom data
|   +-- utils.py                   # Utilities
|
|-- requirements.txt
|-- README.md
|-- SETUP_GUIDE.md                 # Detailed setup instructions
+-- LICENSE
```

## How It Works

### Dataset: Sign Language MNIST

- **Source**: Kaggle (datamunge/sign-language-mnist)
- **Size**: 34,627 images total
- **Format**: 28x28 grayscale images in CSV format
- **Classes**: 24 letters (A-Y, excluding J and Z which require motion)
- **Split**: 27,455 training + 7,172 test samples

### CNN Architecture

```
Input (28x28x1 grayscale image)
    |
    v
Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    |
    v
Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    |
    v
Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
    |
    v
Flatten
    |
    v
Dense(256) + BatchNorm + Dropout(0.5)
    |
    v
Dense(128) + BatchNorm + Dropout(0.5)
    |
    v
Dense(24) + Softmax
```

### Training Process

1. **Load Data**: Read CSV files and reshape to 28x28 images
2. **Preprocess**: Normalize pixels to [0,1], one-hot encode labels
3. **Split**: 85% train, 15% validation from training set
4. **Train**: CNN with Adam optimizer, early stopping, learning rate reduction
5. **Evaluate**: Test on 7,172 held-out samples
6. **Save**: Best model based on validation accuracy

### Real-time Detection

1. Capture frame from webcam
2. Extract region of interest (ROI)
3. Resize to 28x28 and convert to grayscale
4. Normalize and predict with CNN
5. Apply temporal smoothing (average last 5 predictions)
6. Display result with confidence score

## Performance

### Model Metrics
- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 93-98%
- **Test Accuracy**: 95-98%
- **Training Time**: 10-15 min (GPU) / 30-60 min (CPU)

### Real-time Performance
- **FPS**: 25-30 frames per second
- **Latency**: 10-20ms per prediction
- **Accuracy**: 90-95% in good lighting conditions

## Supported Gestures

**24 ASL Letters**: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

**Note**: J and Z are excluded as they require motion (dynamic gestures)

## Usage Guide

### Training the Model

```bash
python src/train_kaggle_model.py
```

**Options to modify in the script:**
- `epochs`: Number of training iterations (default: 50)
- `batch_size`: Samples per batch (default: 128)
- `learning_rate`: Initial learning rate (default: 0.001)

### Real-time Detection

```bash
python src/real_time_cnn_detection.py
```

**Controls:**
- **ESC**: Exit application
- **S**: Take screenshot
- **R**: Reset prediction buffer

**Tips for best results:**
- Use good lighting
- Position hand clearly in green box
- Make distinct, clear gestures
- Hold gesture steady for 1-2 seconds

## Troubleshooting

### Dataset Issues
- **"Dataset not found"**: Ensure CSV files are in `data/kaggle/` directory
- **Kaggle API error**: Check kaggle.json placement and permissions
- **Download fails**: Try manual download option

### Training Issues
- **Out of memory**: Reduce batch_size to 64 or 32
- **Slow training**: Use GPU or reduce epochs
- **Low accuracy**: Increase epochs or adjust learning rate

### Detection Issues
- **Camera not found**: Check camera index (try 0, 1, 2)
- **Low FPS**: Close other applications, use GPU
- **Poor accuracy**: Improve lighting, clear background

## Advanced Configuration

### Improve Model Accuracy

1. **Data Augmentation**: Add rotation, scaling, brightness variations
2. **Deeper Network**: Add more convolutional layers
3. **Transfer Learning**: Use pre-trained models (VGG16, ResNet50)
4. **Hyperparameter Tuning**: Optimize learning rate, batch size

### Extend Functionality

- **Add J and Z**: Implement motion tracking with MediaPipe
- **Word Formation**: String letters to form words
- **Text-to-Speech**: Add voice output using pyttsx3
- **Mobile App**: Deploy with TensorFlow Lite
- **Web Interface**: Create Flask/Django web app

## Performance Tips

### Training Optimization
- Use GPU for 3-5x speedup
- Enable mixed precision training
- Use larger batch sizes (if memory allows)
- Implement learning rate scheduling

### Inference Optimization
- Reduce ROI size for faster processing
- Skip frames (process every 2nd or 3rd frame)
- Use TensorFlow Lite for mobile deployment
- Implement multi-threading

## Alternative Approach: Custom Dataset

You can also collect your own dataset:

```bash
# Collect custom data
python src/data_collection.py

# Train on custom data
python src/train_model.py

# This approach uses MediaPipe landmarks instead of CNN
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- **Dataset**: Sign Language MNIST by tecperson (Kaggle)
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **Kaggle**: Dataset hosting and community

## Contact

**Aravind** - Full Stack Developer

- GitHub: [@aravinditte](https://github.com/aravinditte)
- Repository: [Sign-Language-Detection](https://github.com/aravinditte/Sign-Language-Detection)

## References

- [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Keras API Reference](https://keras.io/api/)

## Quick Reference Commands

```bash
# Setup
pip install -r requirements.txt

# Download dataset
python src/download_dataset.py

# Train model
python src/train_kaggle_model.py

# Real-time detection
python src/real_time_cnn_detection.py

# Custom data collection (alternative)
python src/data_collection.py
python src/train_model.py
```

---

**Built with dedication by Aravind**

## Future Enhancements

- Dynamic gesture recognition (J and Z letters)
- Multi-language sign language support
- Mobile application (Android/iOS)
- Web-based interface with live demo
- Cloud API deployment
- Two-handed gesture support
- Word and sentence prediction
- Integration with communication apps
- Voice synthesis for accessibility
- Gesture customization interface