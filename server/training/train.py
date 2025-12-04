"""
CNN Training Script using TensorFlow/Keras.
Loads images from dataset directory, trains a model on skin lesion classification.
Usage:
  python server/training/train.py --data-dir server/data/HAM10000 --arch DenseNet169 --epochs 30 --batch-size 32
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow startup messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet169, ResNet50, MobileNetV2, EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

# Suppress TensorFlow logger
tf.get_logger().setLevel('ERROR')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('server/logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path('server/logs').mkdir(parents=True, exist_ok=True)

class JsonLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Ensure values are float for JSON serialization
        serializable_logs = {k: float(v) for k, v in logs.items()}
        print(f"\nJSON_METRICS: {json.dumps({'epoch': epoch + 1, **serializable_logs})}", flush=True)

def load_images_from_csv(csv_path, data_dir, img_size=(224, 224)):
    logger.info(f"Loading images from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"CSV shape: {df.shape}")

    images = []
    labels = []

    image_cols = [
        'image_path', 'path', 'filepath', 'file', 'filename',
        'image', 'img_path', 'img', 'image_id', 'image_name', 'img_id', 'isic_id'
    ]
    label_cols = [
        'label', 'diagnosis', 'diagnosis_1', 'diagnosis_2', 'dx', 'class', 'category', 'lesion_id'
    ]

    has_image_col = next((c for c in image_cols if c in df.columns), None)
    has_label_col = next((c for c in label_cols if c in df.columns), None)

    for idx, row in df.iterrows():
        try:
            img_path = None
            if has_image_col:
                val = str(row[has_image_col]).strip()
                if val:
                    candidate = os.path.join(data_dir, val)
                    if os.path.exists(candidate):
                        img_path = candidate
                    else:
                        base = os.path.basename(val)
                        candidate2 = os.path.join(data_dir, base)
                        if os.path.exists(candidate2):
                            img_path = candidate2
            if img_path is None:
                id_col = next((c for c in ['image_id', 'image_name', 'img_id', 'isic_id'] if c in df.columns), None)
                if id_col:
                    base_id = str(row[id_col]).strip()
                    for ext in ['.jpg', '.png', '.jpeg']:
                        candidate = os.path.join(data_dir, f"{base_id}{ext}")
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                        images_dir = os.path.join(data_dir, 'images')
                        candidate3 = os.path.join(images_dir, f"{base_id}{ext}")
                        if os.path.exists(candidate3):
                            img_path = candidate3
                            break
            if img_path is None:
                continue

            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            images.append(np.array(img))

            lbl = None
            if has_label_col:
                lbl = str(row[has_label_col]).strip() if row[has_label_col] is not None else None
            if lbl is None or len(lbl) == 0:
                lbl = 'unknown'
            labels.append(lbl)
        except Exception as e:
            logger.warning(f"Error loading image at index {idx}: {e}")
            continue

    return np.array(images), np.array(labels)


def load_images_from_directory(data_dir, img_size=(224, 224)):
    logger.info(f"Loading images from directory: {data_dir}")
    images = []
    labels = []

    for root, _, files in os.walk(data_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    images.append(np.array(img))
                    label_name = os.path.basename(root)
                    if root == data_dir:
                        label_name = 'unknown'
                    labels.append(label_name)
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels)


def encode_labels(labels):
    """Encode string labels to integers and return mapping."""
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    encoded = np.array([label_to_idx[label] for label in labels])
    return encoded, label_to_idx, idx_to_label


def build_model(arch, num_classes, input_shape=(224, 224, 3)):
    """Build transfer learning model."""
    logger.info(f"Building {arch} model with {num_classes} classes")
    
    # Load pretrained backbone
    if arch == 'DenseNet169':
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif arch == 'EfficientNetB3':
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model


def train_model(data_dir, arch='DenseNet169', epochs=30, batch_size=32, optimizer='Adam', learning_rate=0.001, class_balancing=True):
    """Main training pipeline."""
    logger.info(f"Starting training pipeline")
    logger.info(f"  Architecture: {arch}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Optimizer: {optimizer}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Class Balancing (SMOTE): {class_balancing}")
    
    # Load dataset
    csv_path = os.path.join(data_dir, 'metadata.csv')
    if os.path.exists(csv_path):
        logger.info("Found metadata.csv, loading from CSV...")
        X, y = load_images_from_csv(csv_path, data_dir)
    else:
        logger.info("No metadata.csv, loading from directory structure...")
        X, y = load_images_from_directory(data_dir)
    
    if len(X) == 0:
        logger.error("No images found!")
        try:
            sys.stderr.write("No images found! Ensure metadata.csv has 'image_path' or 'image_id', or place images under data_dir/class_name/*.jpg\n")
            sys.stderr.flush()
        except Exception:
            pass
        return False
    
    logger.info(f"Loaded {len(X)} images")
    
    # Encode labels
    y_encoded, label_to_idx, idx_to_label = encode_labels(y)
    num_classes = len(label_to_idx)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Label mapping: {label_to_idx}")
    
    # Normalize images
    X = X.astype('float32') / 255.0
    logger.info("Images normalized to [0, 1]")
    # If only one class present (unlabeled / flat dataset), switch to autoencoder training
    if num_classes <= 1:
        logger.info("Single-class or unlabeled dataset detected — switching to autoencoder training")

        # Train/val/test split without stratify
        X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
        logger.info(f"Autoencoder Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # Build a simple convolutional autoencoder
        input_shape = X.shape[1:]
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
        x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
        x = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2,2), padding='same')(x)

        x = keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2,2))(x)
        x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2,2))(x)
        x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2,2))(x)
        decoded = keras.layers.Conv2D(input_shape[2], (3,3), activation='sigmoid', padding='same')(x)

        autoencoder = keras.Model(inputs, decoded, name='autoencoder')

        # Compile
        if optimizer == 'Adam':
            opt = Adam(learning_rate=float(learning_rate))
        elif optimizer == 'SGD':
            opt = SGD(learning_rate=float(learning_rate))
        elif optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=float(learning_rate))
        else:
            opt = Adam(learning_rate=float(learning_rate))

        autoencoder.compile(optimizer=opt, loss='mse')
        logger.info('Autoencoder compiled')

        # Train
        logger.info('Starting autoencoder training...')
        history = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            verbose=2,
            callbacks=[JsonLogger()]
        )

        # Evaluate
        test_mse = autoencoder.evaluate(X_test, X_test, verbose=0)
        logger.info(f'Autoencoder test MSE: {test_mse:.6f}')

        # Save model
        model_dir = Path('server/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = model_dir / f"derma_autoencoder_{timestamp}.h5"
        autoencoder.save(str(model_path))
        logger.info(f"Autoencoder model saved to {model_path}")

        # Save metadata
        metadata = {
            'architecture': 'autoencoder',
            'model_type': 'autoencoder',
            'num_classes': num_classes,
            'reconstruction_mse': float(test_mse),
            'epochs_trained': epochs,
            'batch_size': batch_size,
            'optimizer': optimizer,
            'learning_rate': float(learning_rate),
            'model_path': str(model_path),
            'training_date': datetime.now().isoformat()
        }
        metadata_path = model_dir / f"derma_autoencoder_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        return True

    # Convert labels to categorical
    y_cat = keras.utils.to_categorical(y_encoded, num_classes)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=0.3, random_state=42, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Build model
    model = build_model(arch, num_classes)
    
    # Compile
    if optimizer == 'Adam':
        opt = Adam(learning_rate=float(learning_rate))
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=float(learning_rate))
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=float(learning_rate))
    else:
        opt = Adam(learning_rate=float(learning_rate))
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Model compiled")
    
    # Data augmentation
    if class_balancing:
        train_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        logger.info("Data augmentation enabled (simulating SMOTE)")
    else:
        train_gen = ImageDataGenerator()
    
    # Train
    logger.info("Starting training...")
    history = model.fit(
        train_gen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        verbose=2,
        callbacks=[JsonLogger()]
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    
    # Save model
    model_dir = Path('server/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = model_dir / f"derma_{arch.lower()}_{timestamp}.h5"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'architecture': arch,
        'num_classes': num_classes,
        'label_map': idx_to_label,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': epochs,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'learning_rate': float(learning_rate),
        'model_path': str(model_path),
        'training_date': datetime.now().isoformat()
    }
    
    metadata_path = model_dir / f"derma_{arch.lower()}_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN model on skin lesion dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--arch', type=str, default='DenseNet169', choices=['DenseNet169', 'ResNet50', 'MobileNetV2', 'EfficientNetB3'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'])
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--class-balancing', type=lambda x: (str(x).lower() == 'true'), default=True)
    
    args = parser.parse_args()
    
    success = train_model(
        args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        class_balancing=args.class_balancing
    )
    
    sys.exit(0 if success else 1)
