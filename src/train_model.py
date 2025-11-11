"""
Sign Language Detection - Model Training Module
Author: Aravind
Description: Trains a neural network classifier on collected gesture data

This script loads landmark data, preprocesses it, trains a deep learning model,
and evaluates its performance.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class ModelTrainer:
    """Trains gesture classification model"""
    
    def __init__(self, data_path='data/processed/landmarks.csv', model_dir='models'):
        """
        Initialize the model trainer
        
        Args:
            data_path: Path to CSV file with landmark data
            model_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load data from CSV and preprocess"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        print(f"Total samples: {len(df)}")
        print(f"Gestures: {df['label'].unique().tolist()}")
        print(f"Samples per gesture:\n{df['label'].value_counts()}")
        
        # Separate features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, input_shape, num_classes):
        """
        Build neural network architecture
        
        Args:
            input_shape: Shape of input features
            num_classes: Number of gesture classes
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            
            # First hidden layer
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=100, batch_size=32):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("\n" + "="*60)
        print("Training Sign Language Gesture Classifier")
        print("="*60)
        
        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data()
        
        # Build model
        input_shape = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        self.model = self.build_model(input_shape, num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
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
            os.path.join(self.model_dir, 'best_model.h5'),
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
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Detailed classification report
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_training_history()
        
        return self.history
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix - Sign Language Gesture Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = os.path.join(self.model_dir, 'confusion_matrix.png')
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
        
        save_path = os.path.join(self.model_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
        plt.close()
    
    def save_model(self, model_name='gesture_classifier.h5'):
        """Save trained model and label encoder"""
        if self.model is None:
            print("No model to save! Train the model first.")
            return
        
        # Save model
        model_path = os.path.join(self.model_dir, model_name)
        self.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to: {encoder_path}")
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)


if __name__ == '__main__':
    # Initialize trainer
    trainer = ModelTrainer(data_path='data/processed/landmarks.csv')
    
    # Train model
    trainer.train(epochs=100, batch_size=32)
    
    # Save model
    trainer.save_model('gesture_classifier.h5')
