"""
ASL Alphabet CNN Trainer
Author: Aravind
Description: Train a fast CNN on real ASL Alphabet images (folders A-Z)
"""
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def get_data_generators(base_dir, img_size=64, batch_size=32, val_split=0.15):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode='nearest')
    train_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'asl_alphabet_train/asl_alphabet_train'),
        target_size=(img_size,img_size),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(base_dir, 'asl_alphabet_train/asl_alphabet_train'),
        target_size=(img_size,img_size),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True)
    return train_gen, val_gen

def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

def plot_training(history, out_dir='models'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'asl_cnn_training_plot.png'))
    plt.close()

def main():
    base_dir = 'data/asl_alphabet'
    img_size = 64
    batch_size = 32
    train_gen, val_gen = get_data_generators(base_dir, img_size, batch_size)
    print('Classes:', train_gen.class_indices)
    input_shape = (img_size, img_size, 3)
    num_classes = train_gen.num_classes
    model = build_cnn(input_shape, num_classes)
    model.summary()
    os.makedirs('models', exist_ok=True)
    es = EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)
    ckpt = ModelCheckpoint('models/asl_alphabet_cnn.h5',save_best_only=True,monitor='val_accuracy')
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=[es,ckpt]
    )
    plot_training(history)
    class_indices = train_gen.class_indices
    import json
    with open('models/asl_classes.json', 'w') as f:
        json.dump(class_indices, f)
    print('Done! Model saved to models/asl_alphabet_cnn.h5')

if __name__ == '__main__':
    main()
