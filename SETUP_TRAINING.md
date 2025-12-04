# DermaScan Pro - Complete Setup & Training Guide

Your derma scanning app now has **real CNN training** fully wired end-to-end. The uploaded dataset will actually train a model using TensorFlow/Keras.

---

## Quick Start (5 minutes)

### Terminal 1: Start the FastAPI backend

```powershell
cd "d:\derma\derma scan\derma-scan"

# Create Python virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r server/requirements.txt

# Start FastAPI server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Terminal 2: Start the React frontend

```powershell
cd "d:\derma\derma scan\derma-scan"

# npm dev already configured
npm run dev
```

You should see:
```
  ➜  Local:   http://localhost:5173/
```

---

## How to Train Your Model

### 1. Open the app in browser
- Go to http://localhost:5173
- Click **"Launch Diagnostic Tool"**

### 2. Login as Researcher
- Click **"Researcher"** tab
- Click **"Access System"**
- You'll see the **Admin Lab** button in the navbar

### 3. Upload Your Dataset

In the left panel under **"Upload Dataset"**:

**Option A: Upload Images + CSV**
- Prepare a CSV file `metadata.csv` with columns:
  ```
  image_id,image_path,label
  HAM_0000001,images/HAM_0000001.jpg,melanoma
  HAM_0000002,images/HAM_0000002.jpg,benign
  ...
  ```
- Select the CSV + all image files in the file picker
- Click **"Upload to Backend"** (you'll see progress bar)

**Option B: Upload Images in Folder Structure**
- Organize images by class:
  ```
  dataset/
    ├── melanoma/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── benign/
    │   ├── img1.jpg
    │   └── ...
  ```
- Select all files
- Click **"Upload to Backend"**

**Option C: Use Pre-loaded Datasets**
- Keep default "HAM10000" selected (if you have sample data in `server/data/HAM10000/`)

### 4. Configure Training

In the **"Neural Network Config"** panel:

| Setting | Options | Recommendation |
|---------|---------|-----------------|
| Architecture | DenseNet169, ResNet50, MobileNetV2, EfficientNetB3 | **DenseNet169** (best accuracy) |
| Optimizer | Adam, SGD, RMSprop | **Adam** |
| Learning Rate | 0.0001, 0.001, 0.01 | **0.001** |
| Epochs | 1-100 | **30** (for quick test) or **100** (for production) |
| Batch Size | 16, 32, 64 | **32** |
| SMOTE Class Balancing | Toggle | **On** (recommended) |

### 5. Start Training

- Click **"Start Training"** button
- Watch the console logs update in real-time
- Training will:
  - Load images from your uploaded dataset
  - Preprocess: resize to 224×224, normalize, augment
  - Train on 70% of images, validate on 20%, test on 10%
  - Save the trained model to `server/models/`

### 6. Monitor Progress

- **Job ID**: Displayed at top right of metrics panel
- **Progress**: Shows 0-100%
- **Console Logs**: Real-time TensorFlow output
- **Status**: queued → running → completed/failed

Training typically takes:
- **GPU (NVIDIA, RTX 2060+)**: ~30 seconds per epoch
- **CPU**: ~2-5 minutes per epoch

---

## What Happens Behind the Scenes

### Frontend (React - http://localhost:5173)
1. User uploads files → saved to browser state
2. Clicks "Upload to Backend" → POST to `/api/upload-dataset`
3. Clicks "Start Training" → POST to `/api/start-training`
4. Polls `/api/job/{job_id}/status` every 2 seconds
5. Updates UI with progress, logs, job status

### Backend (FastAPI - http://localhost:8000)
1. **POST /api/upload-dataset**
   - Receives multipart files
   - Saves to `server/data/<dataset_name>/`
   - Returns file list and location

2. **POST /api/start-training**
   - Creates job record with UUID
   - Queues training in background
   - Returns `job_id` immediately (non-blocking)

3. **Background Training Process** (subprocess)
   - Runs `python server/training/train.py`
   - Loads images from `server/data/<dataset_name>/`
   - **Data Pipeline**:
     - Detects CSV or directory structure
     - Loads images, resizes to 224×224
     - Encodes labels to integers
     - Splits: 70% train, 20% val, 10% test
   - **Model Training**:
     - Builds transfer learning model (ImageNet pretrained)
     - Applies data augmentation (rotation, shift, zoom, flip)
     - Trains for N epochs
     - Validates after each epoch
     - Evaluates on test set
   - **Artifact Storage**:
     - Saves model as `.h5` (Keras format)
     - Saves metadata JSON with accuracy, loss, hyperparams
     - Location: `server/models/derma_<arch>_<timestamp>.h5`
   - **Logging**:
     - Writes to `server/logs/training.log`
     - Updates job status in `server/jobs/<job_id>.json`

4. **GET /api/job/{job_id}/status**
   - Returns current job status (queued/running/completed/failed)
   - Progress (0-100)
   - Message (current epoch, loss, accuracy)
   - Model path (when complete)

---

## Dataset Format Examples

### Example 1: CSV + Images (Recommended)

**Directory Structure:**
```
server/data/HAM10000/
├── metadata.csv
├── images/
│   ├── HAM_0000001.jpg
│   ├── HAM_0000002.jpg
│   └── ...
└── ...
```

**CSV Format** (`metadata.csv`):
```csv
image_id,image_path,label
HAM_0000001,images/HAM_0000001.jpg,mel
HAM_0000002,images/HAM_0000002.jpg,nv
HAM_0000003,images/HAM_0000003.jpg,bcc
...
```

### Example 2: Directory Structure

```
server/data/HAM10000/
├── mel/              (melanoma class)
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── nv/               (benign mole class)
│   ├── img_101.jpg
│   ├── img_102.jpg
│   └── ...
├── bcc/              (basal cell carcinoma class)
│   ├── img_201.jpg
│   └── ...
└── ...
```

---

## Troubleshooting

### Error: "Backend error: 404"
- Make sure FastAPI server is running on port 8000
- Check Terminal 1: `uvicorn server.app:app --reload ...`

### Error: "No images found!"
- Verify images are in `server/data/<dataset_name>/`
- Check image file extensions: `.jpg`, `.png`, `.jpeg`
- If using CSV, verify paths in metadata.csv are correct

### Training stuck at "queued"
- Check `server/logs/training.log` for errors
- Ensure TensorFlow/CUDA installed correctly: `pip list | grep tensorflow`
- Try manually running: `python server/training/train.py --data-dir server/data/HAM10000 --epochs 5`

### Out of Memory (OOM) error
- Reduce batch size: 16 or 8 instead of 32
- Reduce epochs: 10 instead of 30
- Use smaller model: MobileNetV2 instead of DenseNet169
- Close other GPU processes

### Very slow training (CPU only, not using GPU)
- TensorFlow will use CPU if GPU not available (much slower, ~1-5 min/epoch)
- To use GPU:
  - Install NVIDIA GPU drivers
  - Install CUDA 11.0+ and cuDNN
  - Reinstall TensorFlow: `pip install --upgrade tensorflow[and-cuda]`
  - Verify: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## File Structure After Training

```
derma-scan/
├── server/
│   ├── app.py                        # FastAPI application
│   ├── requirements.txt              # Python packages
│   ├── training/
│   │   └── train.py                 # Training script
│   ├── data/
│   │   └── HAM10000/                # Uploaded dataset
│   │       ├── metadata.csv
│   │       ├── image_001.jpg
│   │       └── ...
│   ├── models/                       # Trained models (OUTPUT)
│   │   ├── derma_densenet169_20251203_180000.h5
│   │   ├── derma_densenet169_20251203_180000_metadata.json
│   │   └── ...
│   ├── logs/                         # Training logs
│   │   └── training.log
│   └── jobs/                         # Job tracking
│       ├── a1b2c3d4.json
│       └── ...
├── src/
│   ├── App.jsx                       # React app (UPDATED)
│   └── ...
├── .venv/                            # Python virtual environment
├── node_modules/
├── package.json
└── vite.config.js
```

---

## API Reference

### Upload Dataset
```
POST http://localhost:8000/api/upload-dataset
Content-Type: multipart/form-data

Query Params:
  datasetName: "HAM10000"

Files:
  files: [image1.jpg, image2.jpg, ..., metadata.csv]

Response:
{
  "status": "success",
  "dataset_name": "HAM10000",
  "dataset_path": "server/data/HAM10000",
  "files_saved": ["image1.jpg", "image2.jpg", "metadata.csv"],
  "message": "Uploaded 3 file(s)"
}
```

### Start Training
```
POST http://localhost:8000/api/start-training
Content-Type: application/json

Body:
{
  "dataset_name": "HAM10000",
  "architecture": "DenseNet169",
  "epochs": 30,
  "batch_size": 32,
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "class_balancing": true
}

Response:
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "message": "Training job started"
}
```

### Get Job Status
```
GET http://localhost:8000/api/job/{job_id}/status

Response:
{
  "job_id": "a1b2c3d4",
  "status": "running",
  "progress": 45.0,
  "message": "Epoch 15/30 complete. Loss: 0.5234, Accuracy: 85.67%",
  "created_at": "2025-12-03T18:00:00",
  "started_at": "2025-12-03T18:00:05",
  "completed_at": null,
  "model_path": null
}
```

Status values: `queued` → `running` → `completed` or `failed`

---

## Next Steps / Advanced Usage

1. **Deploy Model for Inference**
   - Load saved `.h5` model: `tf.keras.models.load_model('server/models/derma_*.h5')`
   - Add REST endpoint: `POST /api/predict` (takes image, returns class + confidence)
   - Use in UserDashboard for real predictions instead of random results

2. **Production Deployment**
   - Use Gunicorn/Nginx instead of `uvicorn --reload`
   - Move data/models to cloud storage (AWS S3, Google Cloud)
   - Use Celery for distributed training jobs
   - Add authentication, rate limiting, quotas
   - Monitor with Prometheus, log with ELK

3. **Improve Accuracy**
   - Use larger dataset (>5000 images)
   - Train for more epochs (50-100)
   - Use data augmentation
   - Experiment with architectures (EfficientNetB7)
   - Fine-tune pretrained models longer

4. **Add Model Serving**
   - TensorFlow Serving (for inference at scale)
   - TorchServe (if switching to PyTorch)
   - ONNX export for cross-platform compatibility

---

## Summary

✅ **Dataset Upload**: Choose files → upload to backend → saved locally  
✅ **Real Training**: TensorFlow/Keras on your dataset → actual CNN trained  
✅ **Progress Tracking**: Job polling → real-time logs and status  
✅ **Model Storage**: Trained `.h5` model + metadata saved to disk  
✅ **End-to-End Flow**: Upload → Configure → Train → Monitor → Model Ready

Your trained model is saved in `server/models/` and ready to deploy!

Questions? Check `server/README.md` for more details.
