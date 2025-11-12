# Complete Setup Guide - Sign Language Detection
**Author: Aravind**

## Quick Start - Kaggle Dataset Approach

This guide will help you set up and run the Sign Language Detection system using the Kaggle dataset.

---

## Prerequisites

### 1. System Requirements
- Python 3.8 or higher
- Webcam (for real-time detection)
- 4GB+ RAM
- GPU (optional, but recommended for faster training)

### 2. Software Requirements
- VS Code (recommended) or any Python IDE
- Git (for cloning repository)
- Kaggle account (for dataset download)

---

## Step-by-Step Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/aravinditte/Sign-Language-Detection.git
cd Sign-Language-Detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup Kaggle API (for automatic download)

#### Option A: Automatic Download with Kaggle API

1. **Get Kaggle API Token:**
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New API Token"
   - Download `kaggle.json`

2. **Place kaggle.json:**
   
   **Windows:**
   ```bash
   mkdir C:\Users\YourUsername\.kaggle
   copy kaggle.json C:\Users\YourUsername\.kaggle\kaggle.json
   ```
   
   **macOS/Linux:**
   ```bash
   mkdir ~/.kaggle
   mv kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Install Kaggle package:**
   ```bash
   pip install kaggle
   ```

4. **Download dataset:**
   ```bash
   python src/download_dataset.py
   ```

#### Option B: Manual Download

1. Go to: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
2. Click "Download" (may need to accept terms)
3. Extract the ZIP file
4. Create directory: `data/kaggle/`
5. Place these files in `data/kaggle/`:
   - `sign_mnist_train.csv`
   - `sign_mnist_test.csv`

---

## Training the Model

### Step 5: Train CNN Model

Run the training script:

```bash
python src/train_kaggle_model.py
```

**What happens:**
- Loads training data (27,455 samples)
- Loads test data (7,172 samples)
- Builds CNN architecture
- Trains for 50 epochs (with early stopping)
- Saves best model to `models/sign_language_cnn.h5`
- Generates confusion matrix and training curves

**Expected Output:**
```
Training samples: 27455
Test samples: 7172
Train set: (23336, 28, 28, 1)
Validation set: (4119, 28, 28, 1)
Test set: (7172, 28, 28, 1)

Test Accuracy: ~95-99%
```

**Training Time:**
- With GPU: ~10-15 minutes
- With CPU: ~30-60 minutes

---

## Running Real-time Detection

### Step 6: Test Real-time Detection

```bash
python src/real_time_cnn_detection.py
```

**How to Use:**
1. Position your hand in the green box
2. Make a clear sign gesture (A-Y, excluding J and Z)
3. The system will display the predicted sign and confidence
4. Press ESC to exit

**Controls:**
- **ESC** - Exit application
- **S** - Take screenshot
- **R** - Reset prediction buffer

---

## Project Structure

```
Sign-Language-Detection/
|-- data/
|   +-- kaggle/                    # Kaggle dataset files
|       |-- sign_mnist_train.csv
|       +-- sign_mnist_test.csv
|
|-- models/
|   |-- sign_language_cnn.h5       # Trained CNN model
|   |-- class_names.json           # Class label names
|   |-- confusion_matrix_cnn.png   # Evaluation plots
|   +-- training_history_cnn.png
|
|-- src/
|   |-- download_dataset.py        # Download Kaggle dataset
|   |-- train_kaggle_model.py      # Train CNN model
|   |-- real_time_cnn_detection.py # Real-time detection
|   |-- data_collection.py         # Custom data collection
|   |-- train_model.py             # Train on custom data
|   +-- utils.py                   # Utility functions
|
|-- requirements.txt
|-- README.md
|-- SETUP_GUIDE.md
+-- LICENSE
```

---

## Troubleshooting

### Issue: "Dataset not found"
**Solution:**
- Ensure CSV files are in `data/kaggle/` directory
- Check file names: `sign_mnist_train.csv` and `sign_mnist_test.csv`
- Try manual download if automatic fails

### Issue: "Model not found" when running detection
**Solution:**
- Train the model first: `python src/train_kaggle_model.py`
- Check if `models/sign_language_cnn.h5` exists

### Issue: Low accuracy during training
**Solution:**
- Increase epochs (default is 50, try 100)
- Adjust batch size (default 128, try 64 or 256)
- Check if GPU is being used (significant speed improvement)

### Issue: Camera not detected
**Solution:**
- Check camera index in code (try 0, 1, or 2)
- Verify camera permissions
- Test camera with other applications

### Issue: Slow real-time detection
**Solution:**
- Lower camera resolution in code
- Use GPU acceleration
- Reduce prediction buffer size
- Close other applications

---

## Model Performance

### Expected Metrics:
- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 93-98%
- **Test Accuracy**: 95-98%
- **Inference Time**: 10-20ms per frame
- **Real-time FPS**: 25-30 FPS

### Supported Gestures:
A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

**Note:** J and Z are excluded as they require motion (not static gestures)

---

## Advanced Configuration

### Modify Training Parameters

Edit `src/train_kaggle_model.py`:

```python
# Training settings
epochs = 50          # Increase for more training
batch_size = 128     # Decrease if GPU memory issues
learning_rate = 0.001 # Default Adam learning rate

# Model architecture
# Modify in build_cnn_model() function
```

### Modify Detection Settings

Edit `src/real_time_cnn_detection.py`:

```python
# ROI size (detection box)
self.roi_size = 300  # Default 300x300 pixels

# Prediction smoothing
self.prediction_buffer = deque(maxlen=5)  # Average last 5 predictions

# Confidence threshold
if confidence > 0.5:  # Adjust threshold (0.0-1.0)
```

---

## Next Steps

### Improve Model Performance:
1. **Data Augmentation**: Add rotation, scaling, brightness variations
2. **Deeper Network**: Add more convolutional layers
3. **Transfer Learning**: Use pre-trained models (VGG16, ResNet)
4. **Ensemble Methods**: Combine multiple models

### Extend Functionality:
1. **Add J and Z**: Implement motion detection
2. **Word Formation**: String letters to form words
3. **Text-to-Speech**: Add voice output
4. **Mobile App**: Deploy to Android/iOS
5. **Web Interface**: Create web-based demo

---

## Resources

### Documentation:
- TensorFlow: https://www.tensorflow.org/tutorials
- OpenCV: https://docs.opencv.org/
- Kaggle Dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

### Community:
- GitHub Issues: Report bugs and request features
- Stack Overflow: Technical questions
- Kaggle Discussions: Dataset-specific questions

---

## Support

If you encounter any issues:

1. Check this guide's Troubleshooting section
2. Review error messages carefully
3. Check GitHub Issues for similar problems
4. Create a new issue with:
   - Error message
   - Python version
   - OS version
   - Steps to reproduce

---

## Credits

**Developer:** Aravind

**Dataset:** Sign Language MNIST (Kaggle)

**Technologies:**
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

**Built with dedication by Aravind**

GitHub: https://github.com/aravinditte/Sign-Language-Detection
