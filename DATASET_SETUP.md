# Dataset Setup Guide

## Problem

The current test dataset has only **8 images** (7 Benign, 1 Malignant), which is insufficient for production-grade medical predictions.

**Issues with small datasets:**
- ❌ Not enough data for meaningful training
- ❌ Stratified splitting fails with small class counts
- ❌ Model will overfit
- ❌ Poor generalization to new images

**Production requirements:**
- ✅ Minimum 1,000+ images per class
- ✅ Balanced class distribution
- ✅ Diverse image quality and angles

---

## Solution Options

### Option 1: Download HAM10000 Dataset (Recommended)

**HAM10000** is a gold-standard dermoscopy dataset with **10,015 images** across 7 diagnostic categories.

#### Quick Start (No API Key Required)

```bash
# Install dependencies
pip install requests tqdm pandas

# Download from Harvard Dataverse (free, no auth)
python download_dataset.py --source dataverse
```

This will:
- Download ~1.5GB of dermoscopy images
- Extract to `server/data/HAM10000/`
- Set up metadata automatically

#### Alternative: Using Kaggle

```bash
# Install Kaggle API
pip install kaggle

# Set up Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Place kaggle.json in ~/.kaggle/

# Download dataset
python download_dataset.py --source kaggle
```

---

### Option 2: Expand Current Dataset (Quick Testing)

If you just want to test the training pipeline quickly, you can augment the existing 8 images:

```bash
python download_dataset.py --source expand --output server/data/expanded
```

This will:
- Create **1,000 images** from your 8 originals using augmentation
- Apply rotations, flips, zooms, brightness changes
- Good for **testing only**, not production

> [!WARNING]
> Augmented data is NOT suitable for medical production use. It's only for testing the training pipeline.

---

### Option 3: Manual Download

If the script doesn't work, download manually:

1. **HAM10000**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. **ISIC 2019**: https://challenge.isic-archive.com/data/

Extract to `server/data/HAM10000/` or `server/data/ISIC2019/`

---

## Dataset Information

### HAM10000 Dataset

| Property | Details |
|----------|---------|
| **Total Images** | 10,015 |
| **Image Size** | 600×450 to 6000×4500 pixels |
| **Format** | JPG |
| **Classes** | 7 diagnostic categories |
| **Source** | Dermatology clinics |

**Class Distribution:**
- Melanocytic nevi (nv): 6,705 images
- Melanoma (mel): 1,113 images
- Benign keratosis (bkl): 1,099 images
- Basal cell carcinoma (bcc): 514 images
- Actinic keratosis (akiec): 327 images
- Vascular lesions (vasc): 142 images
- Dermatofibroma (df): 115 images

---

## Training with New Dataset

### Step 1: Verify Dataset

```bash
# Check dataset was downloaded
ls server/data/HAM10000/

# Should see:
# - *.jpg files (10,015 images)
# - metadata.csv
```

### Step 2: Train Model

**Via UI:**
1. Navigate to http://localhost:5173/
2. Login as Researcher (`admin_lab` / `password`)
3. Go to Admin Lab
4. Select Dataset: `HAM10000`
5. Configure:
   - Architecture: DenseNet169
   - Epochs: 30
   - Batch Size: 32
   - Class Balancing: On
6. Click "Start Training"

**Via API:**
```bash
python -c "
import requests
r = requests.post('http://localhost:8000/api/start-training', json={
    'dataset_name': 'HAM10000',
    'architecture': 'DenseNet169',
    'epochs': 30,
    'batch_size': 32,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'class_balancing': True
})
print(r.json())
"
```

### Step 3: Monitor Progress

```bash
# Get job ID from response, then check status
python -c "
import requests
r = requests.get('http://localhost:8000/api/job/YOUR_JOB_ID/status')
print(r.json())
"
```

---

## Expected Training Times

| Dataset | Epochs | Batch Size | CPU Time | GPU Time |
|---------|--------|------------|----------|----------|
| Custom (8 images) | 2 | 2 | 1-2 min | 30 sec |
| Expanded (1,000 images) | 10 | 32 | 20-30 min | 5-10 min |
| HAM10000 (10,015 images) | 30 | 32 | 4-6 hours | 30-60 min |

---

## Troubleshooting

### Download Failed

**Error**: Connection timeout

**Solution**: Try alternative source or manual download

```bash
# Try dataverse instead of kaggle
python download_dataset.py --source dataverse

# Or download manually from links above
```

### Not Enough Memory

**Error**: OOM (Out of Memory) during training

**Solution**: Reduce batch size

```json
{
  "batch_size": 16  // Instead of 32
}
```

### Training Takes Too Long

**Solution**: Use smaller dataset or GPU

```bash
# Option 1: Use expanded dataset (1,000 images)
python download_dataset.py --source expand

# Option 2: Reduce epochs for testing
{
  "epochs": 10  // Instead of 30
}
```

---

## Production Recommendations

For actual medical deployment:

1. ✅ Use **HAM10000** or **ISIC 2019** dataset
2. ✅ Train for **30+ epochs**
3. ✅ Use **GPU** for faster training (NVIDIA CUDA)
4. ✅ Implement **cross-validation** (5-fold)
5. ✅ Track **AUC-ROC** metrics
6. ✅ Conduct **clinical validation** study
7. ✅ Get **regulatory approval** (FDA, CE)

---

## Next Steps

After dataset setup:

1. **Train model** with new larger dataset
2. **Evaluate** on test set (check accuracy, AUC)
3. **Deploy** trained model to production
4. **Monitor** model performance in real-world use
5. **Retrain** periodically with new data

---

## References

- HAM10000 Dataset: https://doi.org/10.1038/sdata.2018.161
- ISIC Archive: https://www.isic-archive.com/
- Dermatology AI Research: https://arxiv.org/abs/1710.05006
