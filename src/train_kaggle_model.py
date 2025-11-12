"""
Sign Language Detection - Kaggle Dataset Training
Author: Aravind
Description: Train CNN model on Sign Language MNIST dataset from Kaggle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

class SignLanguageCNNTrainer:
    """Train CNN on Sign Language MNIST dataset"""
    
    def __init__(self, train_path='data/kaggle/sign_mnist_train.csv', 
                 test_path='data/kaggle/sign_mnist_test.csv',
                 model_dir='models'):
        """
        Initialize trainer
        
        Args:
            train_path: Path to training CSV
            test_path: Path to test CSV
            model_dir: Directory to save models
        """
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.history = None
        
        # Class labels (A-Z excluding J and Z which require motion)
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        
    def load_data(self):
        """
        Load and preprocess data from CSV files
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("="*60)
        print("Loading Sign Language MNIST Dataset")
        print("="*60)
        
        # Load training data
        print("\nLoading training data...")
        train_df = pd.read_csv(self.train_path)
        print(f"Training samples: {len(train_df)}")
        
        # Load test data
        print("Loading test data...")
        test_df = pd.read_csv(self.test_path)
        print(f"Test samples: {len(test_df)}")
        
        # Separate features and labels
        X_train_full = train_df.drop('label', axis=1).values
        y_train_full = train_df['label'].values
        
        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values
        
        # Reshape to 28x28 images
        X_train_full = X_train_full.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Normalize pixel values to [0, 1]
        X_train_full = X_train_full.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.15, 
            random_state=42,
            stratify=y_train_full
        )
        
        # Convert labels to categorical (one-hot encoding)
        num_classes = len(self.class_names)
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test_categorical = to_categorical(y_test, num_classes)
        
        print("\nData preprocessing complete!")
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.class_names}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test_categorical, y_test
    
    def build_cnn_model(self, input_shape=(28, 28, 1), num_classes=24):
        """
        Build CNN architecture for sign language recognition
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=50, batch_size=128):
        """
        Train the CNN model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("\n" + "="*60)
        print("Training Sign Language CNN Classifier")
        print("="*60)
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test_categorical, y_test = self.load_data()
        
        # Build model
        self.model = self.build_cnn_model()
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'best_sign_language_cnn.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train model
        print("\n" + "="*60)
        print("Training started...")
        print("="*60)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate on test set
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Detailed classification report
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.class_names
        ))
        
        # Generate visualizations
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_training_history()
        self.plot_sample_predictions(X_test, y_test, y_pred)
        
        return self.history
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Sign Language CNN Classifier', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.model_dir, 'confusion_matrix_cnn.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.model_dir, 'training_history_cnn.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
        plt.close()
    
    def plot_sample_predictions(self, X_test, y_test, y_pred, num_samples=15):
        """Plot sample predictions"""
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            
            true_label = self.class_names[y_test[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            color = 'green' if y_test[idx] == y_pred[idx] else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions - Sign Language CNN', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.model_dir, 'sample_predictions_cnn.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to: {save_path}")
        plt.close()
    
    def save_model(self, model_name='sign_language_cnn.h5'):
        """Save trained model"""
        if self.model is None:
            print("No model to save! Train the model first.")
            return
        
        model_path = os.path.join(self.model_dir, model_name)
        self.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save class names
        import json
        classes_path = os.path.join(self.model_dir, 'class_names.json')
        with open(classes_path, 'w') as f:
            json.dump(self.class_names, f)
        print(f"Class names saved to: {classes_path}")
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print("\nBuilt by Aravind")

if __name__ == '__main__':
    # Check if dataset exists
    if not os.path.exists('data/kaggle/sign_mnist_train.csv'):
        print("\nDataset not found!")
        print("Please run: python src/download_dataset.py")
        print("Or download manually from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    else:
        # Initialize trainer
        trainer = SignLanguageCNNTrainer()
        
        # Train model
        trainer.train(epochs=50, batch_size=128)
        
        # Save model
        trainer.save_model('sign_language_cnn.h5')
