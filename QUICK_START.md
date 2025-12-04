# DermaScan Pro - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### For Clinicians (First-Time Users)

#### Step 1: Access the Application
1. Open your browser
2. Go to `http://localhost:5173` (or your deployment URL)
3. You'll see the DermaScan Pro home page

#### Step 2: Login as Clinician
1. Click "Launch Diagnostic Tool"
2. Select **"Clinician"** role
3. Keep default credentials:
   - Username: `dr_user`
   - Password: `password`
4. Click "Access System"

#### Step 3: Upload Skin Lesion Image
1. Click "Upload Lesion Image"
2. Select a clear, well-lit dermatological image
3. Lesion should be centered in the frame
4. Image format: JPG, PNG

#### Step 4: Analyze
1. Click "Analyze Lesion" button
2. Watch the 4-step processing pipeline:
   - ✓ Input & Resize (224×224)
   - ✓ Noise Reduction (Hair removal)
   - ✓ Segmentation (Lesion isolation)
   - ✓ AI Classification

#### Step 5: Review Results
1. **Severity Badge**: Shows risk level (Critical/Moderate/Low/Benign)
2. **AI Confidence**: 80-99% accuracy score
3. **ABCD Analysis**: Click each parameter to see details
   - A = Asymmetry
   - B = Border irregularity
   - C = Color variation
   - D = Diameter
4. **Treatment Recommendations**: Scroll for medical protocols
5. **Feedback**: Mark if diagnosis was accurate/incorrect

**Pro Tip:** Always consult with a qualified dermatologist for final diagnosis!

---

### For Researchers (Model Training)

#### Step 1: Access Admin Lab
1. Login as **"Researcher"** role
2. Default credentials:
   - Username: `admin_lab`
   - Password: `password`
3. Click "Access System"
4. You'll see "Admin Lab" in navigation

#### Step 2: Configure Model
1. **Dataset**: Select HAM10000 or ISIC 2019
2. **Architecture**: Choose model type:
   - DenseNet169 (Recommended - best accuracy)
   - ResNet50 (Balanced)
   - MobileNetV2 (Fast)
3. **Class Balancing**: Toggle SMOTE on/off
4. **Optimizer**: Adam (default recommended)
5. **Learning Rate**: 0.001 (default)
6. **Epochs**: 30 (adjustable 1-100)
7. **Batch Size**: 32

#### Step 3: Start Training
1. Click "Start Training" button
2. Watch real-time metrics:
   - Accuracy increasing (green line)
   - Loss decreasing (gray line)
   - Console showing epoch details

#### Step 4: Monitor Progress
- **Progress Bar**: Shows training completion
- **Convergence Chart**: Visualizes accuracy vs loss
- **Console Logs**: Real-time training output
- **Metrics**: Shows current epoch performance

**Training Time:** ~30 seconds per epoch in simulation mode

---

## 📊 Understanding ABCD Rule

### What is ABCD?

The ABCD rule is a **clinical diagnostic algorithm** for melanoma screening:

| Parameter | Meaning | Score Interpretation |
|-----------|---------|---------------------|
| **A** | **Asymmetry** | 0.0 = Symmetric ✓ → 1.0 = Asymmetric ⚠️ |
| **B** | **Border** | 0.0 = Smooth ✓ → 1.0 = Irregular ⚠️ |
| **C** | **Color** | 0.0 = Uniform ✓ → 1.0 = Multiple colors ⚠️ |
| **D** | **Diameter** | 0.0 = <6mm ✓ → 1.0 = >6mm ⚠️ |

### Reading the Results

- **Green bar** (0.0-0.3): Low risk - benign feature
- **Yellow bar** (0.3-0.6): Moderate - requires monitoring
- **Red bar** (0.6-1.0): High risk - concerning feature

**Total ABCD Score:** Average of all 4 parameters × 100

---

## 🔍 Disease Classifications

### Quick Reference

```
MELANOMA
├─ Severity: CRITICAL ⚠️
├─ Risk Score: 9/10
├─ Features: Asymmetry, irregular borders, multiple colors
└─ Action: Urgent dermatology consultation

BASAL CELL CARCINOMA (BCC)
├─ Severity: MODERATE
├─ Risk Score: 6/10
├─ Features: Pearly bump, translucent appearance
└─ Action: Medical evaluation recommended

BENIGN KERATOSIS
├─ Severity: LOW
├─ Risk Score: 2/10
├─ Features: Waxy, stuck-on appearance
└─ Action: Monitor for changes

MELANOCYTIC NEVUS (Mole)
├─ Severity: BENIGN
├─ Risk Score: 1/10
├─ Features: Uniform color, smooth borders
└─ Action: Routine monitoring

ACTINIC KERATOSIS
├─ Severity: MODERATE
├─ Risk Score: 5/10
├─ Features: Scaly patch, sun-exposed areas
└─ Action: Sun protection, field treatment
```

---

## 🏥 Clinical Workflow

### Typical Clinician Session

```
1. LOGIN (30 seconds)
   ↓
2. UPLOAD IMAGE (1-2 minutes)
   - Patient skin lesion photo
   - Well-lit, clear image
   ↓
3. ANALYZE (3-5 seconds)
   - Watch preprocessing pipeline
   - See real-time progress
   ↓
4. REVIEW RESULTS (5-10 minutes)
   - Check severity badge
   - Study ABCD parameters
   - Read clinical description
   ↓
5. DOCUMENT (2-3 minutes)
   - Mark feedback (accurate/incorrect)
   - Save/print report
   ↓
6. CONSULT (patient-dependent)
   - Refer to dermatologist if needed
   - Schedule follow-up
   - Document in medical record
```

---

## 🔬 Research Workflow

### Typical Researcher Session

```
1. LOGIN (30 seconds)
   ↓
2. CONFIGURE (2-3 minutes)
   - Select dataset
   - Choose architecture
   - Set hyperparameters
   ↓
3. TRAIN (30-60 seconds in demo)
   - Click "Start Training"
   - Monitor progress bar
   - Watch convergence metrics
   ↓
4. EVALUATE (1-2 minutes)
   - Review accuracy/loss graphs
   - Check console logs
   - Note final metrics
   ↓
5. OPTIMIZE (ongoing)
   - Adjust learning rate
   - Try different architecture
   - Tune hyperparameters
   ↓
6. DEPLOY (production)
   - Save best model
   - Export for deployment
   - Integrate with backend
```

---

## 💾 Image Requirements

### For Best Analysis Results

**Dimensions:**
- Minimum: 224×224 pixels
- Recommended: 512×512 or higher
- Aspect ratio: Square or slightly rectangular

**Quality:**
- Clear focus on lesion
- High resolution (3MP+)
- Good natural lighting
- No shadows or glare

**Composition:**
- Lesion centered in frame
- Minimal background visible
- No bandages or coverings
- No makeup or ointments
- Include scale reference if possible

**File Format:**
- JPG (most compatible)
- PNG (with transparency)
- Not supported: BMP, GIF, WebP

---

## ⚙️ System Requirements

### Browser
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Computer
- RAM: 2GB minimum (4GB recommended)
- Disk: 500MB for application
- Internet: Stable connection for API calls

### For Backend (Optional)
- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- 4GB+ RAM

---

## 🐛 Troubleshooting

### Image Won't Upload
**Problem:** "File upload failed"
- **Solution 1:** Check file size (<10MB)
- **Solution 2:** Verify image format (JPG/PNG)
- **Solution 3:** Try different browser
- **Solution 4:** Clear browser cache

### Slow Analysis
**Problem:** Analysis taking >10 seconds
- **Solution 1:** Check internet connection
- **Solution 2:** Verify server is running
- **Solution 3:** Try with smaller image
- **Solution 4:** Restart browser

### Login Failed
**Problem:** "Authentication error"
- **Solution 1:** Verify credentials
  - Clinician: `dr_user` / `password`
  - Researcher: `admin_lab` / `password`
- **Solution 2:** Check server connection
- **Solution 3:** Clear browser cookies
- **Solution 4:** Try incognito mode

### Training Won't Start
**Problem:** "Training failed to initialize"
- **Solution 1:** Verify dataset is selected
- **Solution 2:** Check model architecture
- **Solution 3:** Try different browser
- **Solution 4:** Refresh page and retry

---

## 🔗 Connecting Backend

### Setting Up Python Backend

```bash
# 1. Create virtual environment
python -m venv derma_env
source derma_env/bin/activate  # or: derma_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place trained model
cp derma_model.h5 models/

# 4. Start server
uvicorn main:app --reload
```

### Connecting Frontend

Update `.env` file:
```
REACT_APP_API_URL=http://localhost:8000
```

Then restart React dev server:
```bash
npm run dev
```

---

## 📚 Documentation

### For Users
- **USAGE_GUIDE.md**: Complete user manual
- **IMPLEMENTATION_SUMMARY.md**: Feature overview

### For Developers
- **BACKEND_INTEGRATION_GUIDE.md**: Technical documentation
- Includes: Data prep, model training, API setup, deployment

---

## 🚨 Important Notes

### Medical Disclaimer
⚠️ **This tool is for DIAGNOSTIC ASSISTANCE ONLY**
- Always consult a qualified dermatologist
- Do not rely solely on AI analysis
- This is NOT a medical diagnosis
- For clinical decisions, seek professional medical advice

### Data Privacy
✓ Images processed locally in demo mode
✓ No data stored without permission
✓ Follow HIPAA compliance in production
✓ Implement encryption for sensitive data

### Model Limitations
- Training on demo dataset (may show unrealistic metrics)
- Transfer learning from ImageNet (natural images)
- Requires real medical dataset for production
- Regular retraining with new data recommended

---

## 🎯 Next Steps

### For Clinicians
1. [x] Access the application
2. [x] Practice with test images
3. [x] Learn ABCD rule interpretation
4. [ ] Integrate into clinical workflow
5. [ ] Provide feedback to improve model

### For Researchers
1. [x] Access Admin Lab
2. [x] Experiment with configurations
3. [x] Monitor training metrics
4. [ ] Prepare real dataset
5. [ ] Deploy production model
6. [ ] Set up continuous monitoring

---

## 📞 Support

### Common Questions

**Q: Can I use this for official diagnosis?**
A: No, always consult a qualified dermatologist for diagnosis.

**Q: How accurate is the AI?**
A: Demo mode shows ~95% accuracy. Real-world performance depends on dataset quality.

**Q: Can I train with my own data?**
A: Yes, see BACKEND_INTEGRATION_GUIDE.md for dataset preparation.

**Q: How do I deploy to production?**
A: Follow Docker deployment guide in BACKEND_INTEGRATION_GUIDE.md

---

## 🎓 Learning Resources

- Melanoma detection: https://www.cancer.org/cancer/melanoma-skin-cancer/
- ABCD Rule: https://dermnetnz.org/topics/dermoscopy-basic-principles/
- Deep Learning in Medical Imaging: https://arxiv.org/abs/2102.06747
- Transfer Learning: https://towardsdatascience.com/transfer-learning-with-python-

---

**Last Updated:** December 3, 2025
**Version:** 2.0.0

