# DermaScan Backend Server

FastAPI server for handling dataset uploads and CNN model training.

## Quick Start

### 1. Install Python dependencies

```powershell
# Navigate to project root
cd "d:\derma\derma scan\derma-scan"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r server/requirements.txt
```

### 2. Start the backend server

```powershell
# From project root with .venv activated
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

### 3. Use the React frontend

- In another terminal, start the React dev server:
  ```powershell
  cd "d:\derma\derma scan\derma-scan"
  npm run dev
  ```
- Open http://localhost:5173 in your browser
- Login as Researcher (Admin Lab)
- Upload dataset in the "Upload Dataset" section
- Click "Start Training" to train the CNN model

## API Endpoints

### Upload Dataset
```
POST /api/upload-dataset
```
Accepts multipart form data with files (CSV, images, archives).
- Query param: `datasetName` (e.g., "HAM10000")
- Files: image files (.jpg, .png), CSV metadata, or archives (.zip, .tar, .tgz)

**Response:**
```json
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
POST /api/start-training
```
Request body:
```json
{
  "dataset_name": "HAM10000",
  "architecture": "DenseNet169",
  "epochs": 30,
  "batch_size": 32,
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "class_balancing": true
}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "queued",
  "message": "Training job started"
}
```

### Get Job Status
```
GET /api/job/{job_id}/status
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "running",
  "progress": 45,
  "message": "Epoch 15/30",
  "created_at": "2025-12-03T18:00:00",
  "started_at": "2025-12-03T18:00:05",
  "completed_at": null,
  "model_path": null
}
```

Job status values: `queued`, `running`, `completed`, `failed`

### List Datasets
```
GET /api/datasets
```

### List Trained Models
```
GET /api/models
```

## Dataset Format

### Option 1: CSV + Images
- CSV file: `metadata.csv` with columns:
  - `image_id`: unique identifier (e.g., "HAM_0000001")
  - `image_path`: relative path to image (e.g., "images/HAM_0000001.jpg")
  - `label`: class label (e.g., "melanoma", "benign", etc.)

- Images: organized in `images/` subdirectory or flat directory

### Option 2: Directory Structure
- Images organized by class:
  ```
  server/data/HAM10000/
    ├── melanoma/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── benign/
    │   ├── img1.jpg
    │   └── ...
    └── ...
  ```

The training script will auto-detect format and load accordingly.

## Training Script

**Location:** `server/training/train.py`

**Features:**
- Loads images from CSV or directory structure
- Encodes labels to integers
- Normalizes images to [0, 1]
- Splits into train (70%), validation (20%), test (10%)
- Builds transfer learning model (DenseNet169, ResNet50, MobileNetV2, EfficientNetB3)
- Data augmentation (rotation, shift, flip, zoom)
- Compiles with specified optimizer and learning rate
- Trains on GPU (if available) or CPU
- Saves model as `.h5` file
- Saves metadata JSON with training results

**Run manually:**
```powershell
python server/training/train.py --data-dir server/data/HAM10000 --arch DenseNet169 --epochs 30 --batch-size 32
```

## Supported Architectures

- **DenseNet169**: Dense connections, optimal for medical imaging (recommended)
- **ResNet50**: Residual connections, balanced performance
- **MobileNetV2**: Lightweight, edge deployment ready
- **EfficientNetB3**: Scaling optimized, high efficiency

## Hardware Requirements

- **CPU**: Intel i5+ or equivalent (slow, ~minutes/epoch)
- **GPU**: NVIDIA GTX 1080+ with CUDA 11.0+ (fast, ~seconds/epoch)
- **RAM**: 8 GB minimum (16+ recommended for batch size 64+)
- **Disk**: 5+ GB for models, logs, and datasets

## Troubleshooting

### "No module named tensorflow"
Make sure virtual environment is activated and requirements are installed:
```powershell
pip install -r server/requirements.txt
```

### "No images found!"
Check that:
1. Images are in `server/data/<dataset_name>/` or subdirectories
2. Images have valid extensions: .jpg, .jpeg, .png
3. CSV (if used) has correct columns and image paths are correct

### Training is very slow
- Use GPU if available (install CUDA, cuDNN)
- Reduce batch size to fit in memory
- Use MobileNetV2 for faster training (lower accuracy)
- Reduce image resolution (modify in `train.py`)

### Out of Memory (OOM)
- Reduce batch size (e.g., 16 or 8)
- Reduce epochs
- Use smaller model (MobileNetV2 instead of DenseNet169)
- Ensure no other GPU processes running

## Production Deployment

For production, consider:
1. Use Redis for job queue instead of in-memory dict
2. Add database for job/model tracking
3. Use Celery or APScheduler for distributed training
4. Add authentication and rate limiting
5. Set up model serving (TF Serving, TorchServe)
6. Use containerization (Docker) and orchestration (Kubernetes)
7. Add monitoring and logging (Prometheus, ELK)

## File Structure

```
derma-scan/
├── server/
│   ├── app.py                    # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── training/
│   │   └── train.py             # CNN training script
│   ├── data/                     # Dataset storage
│   │   ├── HAM10000/
│   │   │   ├── metadata.csv
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── models/                   # Trained models
│   │   ├── derma_densenet169_20251203_180000.h5
│   │   ├── derma_densenet169_20251203_180000_metadata.json
│   │   └── ...
│   ├── logs/                     # Training logs
│   │   └── training.log
│   └── jobs/                     # Job status tracking
│       ├── a1b2c3d4.json
│       └── ...
├── src/
│   ├── App.jsx                   # React app with UI
│   └── ...
├── package.json
└── vite.config.js
```

## Notes

- All uploaded files are stored in `server/data/<dataset_name>/`
- All trained models are stored in `server/models/` with metadata
- Training runs in a background subprocess (async)
- Job status is tracked in memory and persisted to JSON files
- Logs are written to `server/logs/training.log` and stdout
