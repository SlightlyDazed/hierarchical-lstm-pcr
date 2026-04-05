# Pipeline Documentation

## Complete Workflow

This document explains each stage of the pipeline and which script to use.

---

## Stage 1: Data Preprocessing

**Goal**: Convert raw DICOM images to 16-bit PNG and extract 1143 features per DCE phase.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1.1 | `preprocessing/dicom_to_png.py` | DICOM folders | 16-bit PNG images |
| 1.2 | `preprocessing/feature_extraction.py` | PNG images | 1143-dim feature vectors |
| 1.3 | `preprocessing/duke_features.py` | Duke PNG | Duke feature CSV |

**Features extracted (1143 total)**:
- DenseNet-121 (ImageNet): 1024 dimensions
- PyRadiomics (shape, texture, GLCM): ~100 dimensions  
- Spatial features: 19 dimensions

---

## Stage 2: Phase LSTM Pre-training

**Goal**: Learn DCE contrast dynamics (wash-in/wash-out) via autoencoder reconstruction.

| Script | Dataset | Output |
|--------|---------|--------|
| `training/pretrain.py` | Duke (n=922) | `phase_lstm_encoder_pretrained.pt` |

**Key hyperparameters**:
- Epochs: 1500 (early stop ~622)
- Batch size: 32
- Learning rate: 0.001 (OneCycleLR)
- Mixed precision (AMP) + EMA

**Validation loss achieved**: 0.566

---

## Stage 3: Integrated Model Training

**Goal**: Train full hierarchical LSTM with 5-fold stratified CV.

| Script | Dataset | Output |
|--------|---------|--------|
| `training/train_cv.py` | I-SPY 2 (n=199) | `integrated_lstm_best.pt` |

**Architecture**:
1. Frozen Phase LSTM Encoder (from Stage 2)
2. Temporal LSTM (trainable)
3. Clinical feature fusion
4. MLP classifier

**Key hyperparameters**:
- Epochs: 50
- Patience: 15 (early stopping)
- Learning rate: 0.001
- Loss: FocalLoss (class weight 2:1)

---

## Stage 4: Evaluation & Calibration

**Goal**: Generate evidence and apply post-hoc isotonic calibration.

| Script | Input | Output |
|--------|-------|--------|
| `evaluation.py` | Trained model | Figures + JSON metrics |

**Outputs**:
- ROC curves (per fold + mean)
- Confusion matrices
- Calibration analysis
- Phase attention heatmaps

**Calibration improvement**:
- Brier score: 0.277 → 0.201 (-27.2%)
- Probability separation: +684%

---

## Data Requirements

### I-SPY 2 (Training)
```
ISPY2_ALL/
├── ISPY2-{patient_id}/
│   ├── T0/Ph0-Ph5/*.png    # Baseline
│   ├── T1/Ph0-Ph5/*.png    # Early treatment
│   ├── T2/Ph0-Ph5/*.png    # Mid treatment
│   └── T3/Ph0-Ph5/*.png    # Pre-surgery
└── clinical_data.csv
```

### Duke (Pre-training)
```
Duke_MRI/
├── {patient_id}/
│   └── Ph0-Ph5/*.png       # Single timepoint
└── clinical_annotations.csv
```

---

## Reproducibility Checklist

- [x] DICOM → PNG conversion completed
- [x] Features extracted (1143-dim per phase)
- [x] Phase LSTM pre-trained on Duke (val_loss=0.566)
- [x] 5-Fold CV on I-SPY 2 (mean AUC=0.704)
- [x] Isotonic calibration applied (Brier=0.201)
- [x] Evidence figures generated
