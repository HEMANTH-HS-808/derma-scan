# DermaScan Pro - Backend Integration Guide

Complete guide for integrating the deep learning model backend with the React frontend.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Preparation Pipeline](#data-preparation-pipeline)
3. [Model Training Process](#model-training-process)
4. [Backend Server Setup](#backend-server-setup)
5. [API Integration](#api-integration)
6. [Deployment Guide](#deployment-guide)

---

## System Architecture

### Frontend-Backend Communication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (Vite)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Login → Clinician/Researcher Role Selection           │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                      ↓               │
│  ┌──────────────────┐              ┌───────────────────────┐   │
│  │ Clinician View   │              │  Researcher/Admin Lab │   │
│  │ • Image Upload   │              │ • Model Config        │   │
│  │ • Analysis       │              │ • Train/Validate      │   │
│  │ • Results Report │              │ • Metrics Monitoring  │   │
│  └──────────────────┘              └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          ↓
        ┌──────────────────────────────────────┐
        │  HTTP/REST API (Python FastAPI)     │
        │  ┌────────────────────────────────┐ │
        │  │ POST /api/analyze              │ │
        │  │ POST /api/train                │ │
        │  │ GET /api/training-status       │ │
        │  │ POST /api/predict              │ │
        │  └────────────────────────────────┘ │
        └──────────────────────────────────────┘
           ↓
        ┌──────────────────────────────────────┐
        │   ML BACKEND (TensorFlow/Keras)     │
        │  ┌────────────────────────────────┐ │
        │  │ CNN Models (DenseNet/ResNet)   │ │
        │  │ Image Preprocessing Pipeline   │ │
        │  │ Training/Inference Engine      │ │
        │  └────────────────────────────────┘ │
        └──────────────────────────────────────┘
           ↓
        ┌──────────────────────────────────────┐
        │     DATA LAYER (Database/Files)     │
        │  • Trained Models (.h5/.pb)        │
        │  • Training Logs & Metrics         │
        │  • Diagnostic Results Cache        │
        └──────────────────────────────────────┘
```

---

## Data Preparation Pipeline

### 1. Dataset Requirements

**Recommended Datasets:**
- **HAM10000**: 10,000 training images, 7 skin disease classes
- **ISIC 2019**: 25,000 images, 8 classes (recommended for production)
- **Custom Clinical Dataset**: Your own labeled images

**Image Specifications:**
- **Format**: JPEG, PNG, or DICOM
- **Size**: Minimum 224×224 pixels (recommended 512×512)
- **Color Space**: RGB or BGR
- **Quality**: Dermatoscopic or high-quality photographs

**Class Distribution (HAM10000):**
| Class | Count | Label |
|-------|-------|-------|
| Melanoma | 1,113 | mel |
| Melanocytic Nevus | 6,705 | nv |
| Basal Cell Carcinoma | 514 | bcc |
| Actinic Keratosis | 327 | akiec |
| Benign Keratosis | 1,099 | bkl |
| Dermatofibroma | 115 | df |
| Vascular Lesion | 142 | vasc |

### 2. Image Preprocessing Pipeline

```python
import cv2
import numpy as np
from PIL import Image

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for dermatological images
    """
    
    @staticmethod
    def resize_image(image_path, target_size=(224, 224)):
        """
        Step 1: Input & Resize
        Normalize image dimensions to model input size
        """
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        return img_resized
    
    @staticmethod
    def dull_razor_hair_removal(image):
        """
        Step 2: Noise Reduction - Hair Removal
        Implements the "Dull Razor" algorithm for digital hair removal
        
        Algorithm Steps:
        1. Apply grayscale conversion
        2. Detect dark pixels (likely hair)
        3. Apply morphological operations
        4. Inpaint detected regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological closing to detect hair
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Create mask for hair-like pixels
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Dilate to expand hair regions
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Inpaint hair regions
        inpainted = cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)
        return inpainted
    
    @staticmethod
    def segmentation_otsu(image):
        """
        Step 3: Segmentation - Lesion Isolation
        Uses Otsu's thresholding to isolate lesion from healthy skin
        
        Otsu's Method:
        - Automatically determines optimal threshold value
        - Minimizes within-class variance
        - Separates foreground (lesion) from background (skin)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask with largest contour (lesion)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        return segmented, mask
    
    @staticmethod
    def normalize_image(image):
        """
        Step 4: Normalization
        Normalize pixel values to 0-1 range using ImageNet stats
        """
        # Convert to float32
        img_float = image.astype(np.float32) / 255.0
        
        # ImageNet normalization statistics (if using pre-trained weights)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Normalize each channel
        for i in range(3):
            img_float[:, :, i] = (img_float[:, :, i] - mean[i]) / std[i]
        
        return img_float
    
    @staticmethod
    def full_pipeline(image_path, target_size=(224, 224)):
        """
        Complete preprocessing pipeline
        """
        # Step 1: Resize
        img = PreprocessingPipeline.resize_image(image_path, target_size)
        
        # Step 2: Hair Removal
        img = PreprocessingPipeline.dull_razor_hair_removal(img)
        
        # Step 3: Segmentation
        img, mask = PreprocessingPipeline.segmentation_otsu(img)
        
        # Step 4: Normalize
        img = PreprocessingPipeline.normalize_image(img)
        
        return img, mask
```

### 3. Class Imbalance Handling - SMOTE

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np

def handle_class_imbalance(X_train, y_train, sampling_strategy='auto'):
    """
    Synthetic Minority Over-sampling Technique (SMOTE)
    
    Problem: Melanoma is rare (~10% of cases)
    Solution: Generate synthetic minority samples
    
    Parameters:
    - X_train: Training features (images flattened or feature vectors)
    - y_train: Training labels
    - sampling_strategy: 'auto' or dict specifying target ratios
    
    Returns:
    - X_resampled: Balanced training data
    - y_resampled: Balanced labels
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
    
    # Reshape images for SMOTE (flatten if needed)
    X_shape = X_train.shape
    if len(X_shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Reshape back
    if len(X_shape) > 2:
        X_resampled = X_resampled.reshape(X_resampled.shape[0], *X_shape[1:])
    
    return X_resampled, y_resampled
```

---

## Model Training Process

### Transfer Learning CNN Architecture

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet169, ResNet50, MobileNetV2

class DermaScanModel:
    """
    Transfer learning-based CNN for skin lesion classification
    """
    
    def __init__(self, architecture='DenseNet169', num_classes=7, input_size=(224, 224, 3)):
        """
        Initialize model with specified backbone architecture
        
        Available architectures:
        - DenseNet169: Dense connections, optimal for medical imaging
        - ResNet50: Standard residual network, good baseline
        - MobileNetV2: Lightweight, suitable for edge deployment
        """
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = None
    
    def build_transfer_learning_model(self):
        """
        Build model using transfer learning
        
        Steps:
        1. Load pre-trained backbone (ImageNet weights)
        2. Freeze initial layers
        3. Add custom classification layers
        4. Compile model
        """
        
        # Step 1: Load pre-trained backbone
        if self.architecture == 'DenseNet169':
            base_model = DenseNet169(
                input_shape=self.input_size,
                weights='imagenet',
                include_top=False  # Remove top classification layer
            )
        elif self.architecture == 'ResNet50':
            base_model = ResNet50(
                input_shape=self.input_size,
                weights='imagenet',
                include_top=False
            )
        elif self.architecture == 'MobileNetV2':
            base_model = MobileNetV2(
                input_shape=self.input_size,
                weights='imagenet',
                include_top=False
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Step 2: Freeze initial layers (keep learned features)
        base_model.trainable = False
        
        # Step 3: Add custom classification layers
        model = keras.Sequential([
            layers.Input(shape=self.input_size),
            base_model,
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers for classification
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer (7 skin disease classes)
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Step 4: Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def unfreeze_layers(self, num_layers=50):
        """
        Unfreeze final layers for fine-tuning
        Allows the model to adapt to medical imaging domain
        """
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all but last num_layers
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
```

### Training Function

```python
def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    """
    Train the model
    
    Parameters:
    - model: Compiled Keras model
    - X_train: Training images (batch, 224, 224, 3)
    - y_train: Training labels (one-hot encoded)
    - X_val: Validation images
    - y_val: Validation labels
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    
    Returns:
    - history: Training history object containing metrics
    """
    
    # Define callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        
        # Save best model
        keras.callbacks.ModelCheckpoint(
            'derma_best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

### Model Evaluation

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model on test set
    
    Metrics calculated:
    - Accuracy: Overall correctness
    - Sensitivity (Recall): True positive rate for each class
    - Specificity: True negative rate
    - Precision: Positive predictive value
    - AUC-ROC: Area under ROC curve
    """
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Classification Report
    report = classification_report(
        y_true_labels, y_pred_labels,
        target_names=class_names
    )
    
    print("Classification Report:")
    print(report)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_labels == y_true_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # AUC-ROC (for binary or one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
        print(f"AUC-ROC Score: {auc:.4f}")
    except:
        print("AUC-ROC not available for this configuration")
    
    return cm, report, accuracy
```

---

## Backend Server Setup

### Python Environment Setup

```bash
# Create virtual environment
python -m venv derma_env

# Activate environment
# Windows:
derma_env\Scripts\activate
# macOS/Linux:
source derma_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow==2.13.0
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pillow==10.1.0
numpy==1.24.3
opencv-python==4.8.1.78
scikit-image==0.21.0
scikit-learn==1.3.2
imblearn==0.0
pydantic==2.5.0
aiofiles==23.2.1
```

### FastAPI Backend Server

```python
# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras
import asyncio
from datetime import datetime

app = FastAPI(title="DermaScan Pro API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
try:
    model = keras.models.load_model('models/derma_trained_model.h5')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

CLASS_NAMES = ['melanoma', 'nevus', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
CLASS_LABELS = {
    'melanoma': 'Malignant Melanoma',
    'nevus': 'Melanocytic Nevus',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'vasc': 'Vascular Lesion'
}

def preprocess_image(image_bytes):
    """Preprocess image using complete pipeline"""
    # Load image
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    
    # Convert to BGR if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    # Resize
    image = cv2.resize(image, (224, 224))
    
    # Hair removal (simplified Dull Razor)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    image = cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return np.expand_dims(image, axis=0)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image and return diagnosis
    
    Returns:
    - disease: Predicted disease class
    - confidence: Confidence score (0-100)
    - risk_score: Risk score (1-10)
    - clinical_indicators: ABCD rule scores
    - processing_time: Time taken for analysis
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = datetime.now()
        
        # Read and preprocess image
        image_data = await file.read()
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(np.max(prediction[0]) * 100)
        class_idx = np.argmax(prediction[0])
        disease = CLASS_NAMES[class_idx]
        
        # Calculate ABCD indicators (mock calculation for demo)
        prediction_scores = prediction[0]
        abcd = {
            'A': float(prediction_scores[class_idx]),
            'B': float(prediction_scores[(class_idx + 1) % len(CLASS_NAMES)]),
            'C': float(prediction_scores[(class_idx + 2) % len(CLASS_NAMES)]),
            'D': float(prediction_scores[(class_idx + 3) % len(CLASS_NAMES)])
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "disease": disease,
            "label": CLASS_LABELS.get(disease, disease),
            "confidence": round(confidence, 2),
            "risk_score": 5 + class_idx,  # Mock risk calculation
            "clinical_indicators": abcd,
            "processing_time": f"{processing_time:.2f}s",
            "model_version": "v2.1-DenseNet169"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/training-status")
async def training_status():
    """Get current training status"""
    return {
        "training_active": False,
        "current_epoch": 0,
        "total_epochs": 30,
        "current_accuracy": 0.0,
        "current_loss": 0.0
    }

@app.post("/api/predict-batch")
async def predict_batch(files: list = File(...)):
    """Batch prediction endpoint for multiple images"""
    results = []
    for file in files:
        result = await analyze_image(file)
        results.append(result)
    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run Backend Server

```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## API Integration

### Frontend API Service (React)

```javascript
// src/services/apiService.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIService {
  async analyzeImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Analysis error:', error);
      throw error;
    }
  }
  
  async getTrainingStatus() {
    const response = await fetch(`${API_BASE_URL}/api/training-status`);
    return await response.json();
  }
  
  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return await response.json();
  }
}

export default new APIService();
```

### Update Frontend to Use Real API

```javascript
// In UserDashboard component
import apiService from './services/apiService';

const processImage = async () => {
  setAnalyzing(true);
  try {
    const result = await apiService.analyzeImage(imageFile);
    setResult(result);
  } catch (error) {
    console.error('Analysis failed:', error);
    // Show error to user
  }
  setAnalyzing(false);
};
```

---

## Deployment Guide

### Docker Deployment

```dockerfile
# Dockerfile for backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy trained model
COPY models/ /app/models/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build:
      context: ./derma-scan
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
      - ./backend/logs:/app/logs
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  model_cache:
```

### Cloud Deployment (AWS Example)

```yaml
# AWS Lambda function for serverless deployment
# (Requires AWS SAM or Serverless Framework)

AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  DermaScanFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_handler.main
      Runtime: python3.10
      Timeout: 60
      MemorySize: 3008
      EphemeralStorage:
        Size: 10240
      CodeUri: backend/
      Events:
        AnalyzeAPI:
          Type: Api
          Properties:
            Path: /analyze
            Method: POST
            RestApiId: !Ref DermaScanAPI
      Environment:
        Variables:
          MODEL_PATH: /mnt/models/derma_model.h5

  DermaScanAPI:
    Type: AWS::ApiGatewayV2::Api
    Properties:
      Name: DermaScanAPI
      ProtocolType: HTTP
      CorsConfiguration:
        AllowOrigins:
          - '*'
        AllowMethods:
          - POST
          - GET
        AllowHeaders:
          - '*'
```

---

## Performance Optimization

### Model Quantization

```python
# Reduce model size for deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('derma_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Caching Strategy

```python
# Cache predictions for identical images
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_prediction(image_hash):
    """Return cached prediction if image already analyzed"""
    pass

def compute_image_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()
```

---

## Monitoring & Logging

```python
# Logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('derma_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log API requests
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response
```

---

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure model file path is correct
   - Verify TensorFlow version compatibility
   - Check disk space for model file

2. **Image Processing Error**
   - Verify image format is supported
   - Check image dimensions
   - Ensure preprocessing pipeline is compatible

3. **GPU Memory Error**
   - Reduce batch size
   - Use model quantization
   - Enable memory optimization

---

## Next Steps

1. Prepare your dataset (HAM10000 or ISIC)
2. Set up Python environment and install dependencies
3. Train the model using the provided code
4. Deploy backend server
5. Update React frontend API endpoints
6. Test end-to-end workflow
7. Deploy to production

For detailed implementation examples, refer to the code snippets above.

