# Hierarchical LSTM for pCR Prediction in Breast Cancer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **A hierarchical LSTM architecture for predicting pathological complete response (pCR) in breast cancer using longitudinal DCE-MRI**

**Paper**: *Hierarchical LSTM for Pathological Complete Response Prediction in Breast Cancer Using Longitudinal DCE-MRI: A Proof-of-Concept Study*

**Authors**: Alexander Laurente, Diego Pantoja, Fabricio León, Paola Sagastegui  
**Affiliation**: School of Biomedical Engineering, Universidad Nacional Mayor de San Marcos

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.704 ± 0.065 |
| **Sensitivity** | 66.7% |
| **Specificity** | 75.9% |
| **NPV** | 82.1% |
| **Brier Score (Calibrated)** | 0.201 |

---

## 🔄 Pipeline Overview

The complete pipeline consists of 4 stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                        STAGE 1: PREPROCESSING                   │
│  DICOM → 16-bit PNG → DenseNet-121 + Radiomics (1143 features) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: PHASE LSTM PRE-TRAINING             │
│    Masked autoencoder on Duke dataset (n=922, single-timepoint) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 3: INTEGRATED TRAINING                │
│      5-Fold CV on I-SPY 2 (n=199, longitudinal T0-T3)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       STAGE 4: EVALUATION                       │
│         Isotonic calibration + Evidence generation              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
hierarchical-lstm-pcr/
│
├── code/                          # All source code
│   ├── preprocessing/             # Stage 1: Data preparation
│   │   ├── dicom_to_png.py       # DICOM → 16-bit PNG conversion
│   │   ├── feature_extraction.py # DenseNet-121 + Radiomics (1143 features)
│   │   └── duke_features.py      # Duke dataset feature extraction
│   ├── models/                    # Neural network architectures
│   │   ├── phase_lstm.py         # Phase LSTM encoder (intra-timepoint)
│   │   └── integrated_lstm.py    # Full model + FocalLoss + Dataset
│   ├── training/                  # Stages 2-3: Model training
│   │   ├── pretrain.py           # Phase LSTM autoencoder pre-training
│   │   ├── train_cv.py           # 5-Fold stratified cross-validation
│   │   └── finetune.py           # pCR fine-tuning
│   ├── utils/                     # Utilities
│   │   ├── data_loader.py        # Phase-level dataset loader
│   │   ├── normalization.py      # FeatureNormalizer (StandardScaler)
│   │   └── unified_dataset.py    # Duke + I-SPY 2 unified loader
│   └── evaluation.py             # Generate evidence & figures
│
├── checkpoints/                   # Pre-trained model weights
│   ├── phase_lstm_encoder_pretrained.pt  # Phase encoder (1.66M params)
│   ├── integrated_lstm_best.pt           # Best integrated model
│   ├── phase_lstm_normalizer.pkl         # Feature normalizer
│   └── isotonic_calibrator.pkl           # Post-hoc calibrator
│
├── data/                          # Pre-extracted features
│   ├── ispy2_features.pkl        # I-SPY 2 (n=199, 1143 features/phase)
│   ├── duke_features.pkl         # Duke (n=922, 1143 features/phase)
│   └── duke_clinical.csv         # Duke clinical annotations
│
├── results/                       # Experimental results
│   ├── figures/                  # Paper figures (7 PNGs)
│   ├── cv_results.json           # 5-fold CV metrics
│   ├── calibration_results.json
│   └── isotonic_calibration_results.json
│
├── docs/                          # Documentation
│   ├── PIPELINE.md               # Pipeline workflow
│   └── TECHNICAL_REPORT.md       # Technical details
│
├── paper/                         # Paper text
│   └── ieee_paper.txt
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/hierarchical-lstm-pcr.git
cd hierarchical-lstm-pcr
pip install -r requirements.txt
```

### Stage 1: Preprocessing

```bash
# Convert DICOM to 16-bit PNG
python code/preprocessing/dicom_to_png.py \
    --input_dir /path/to/dicom \
    --output_dir /path/to/png

# Extract features (1143-dim: DenseNet + Radiomics)
python code/preprocessing/feature_extraction.py \
    --data_dir /path/to/png \
    --output_csv features.csv
```

### Stage 2: Phase LSTM Pre-training

```bash
python code/training/pretrain.py \
    --epochs 1500 \
    --batch-size 32 \
    --lr 0.001 \
    --use-amp \
    --use-ema
```

### Stage 3: 5-Fold Cross-Validation

```bash
python code/training/train_cv.py \
    --epochs 50 \
    --lr 0.001 \
    --patience 15 \
    --encoder-path checkpoints/phase_lstm_encoder_pretrained.pt
```

### Stage 4: Evaluation

```bash
python code/evaluation.py \
    --model-path checkpoints/integrated_lstm_best.pt \
    --output-dir results/
```

---

## 🏛️ Architecture

```
Input: Patient with 4 timepoints × 6 phases × 1143 features
                           ↓
┌──────────────────────────────────────────┐
│     PHASE LSTM ENCODER (Pre-trained)     │
│  • Input: 6 phases × 1143 features       │
│  • Bidirectional LSTM (hidden=128)       │
│  • Temporal attention over phases        │
│  • Output: 256-dim phase embedding       │
└──────────────────────────────────────────┘
                           ↓
        [Repeat for each of 4 timepoints]
                           ↓
┌──────────────────────────────────────────┐
│           TEMPORAL LSTM                   │
│  • Input: 4 timepoints × 256-dim         │
│  • Bidirectional LSTM (hidden=64)        │
│  • Output: 128-dim temporal context      │
└──────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────┐
│    CLINICAL FUSION + CLASSIFIER          │
│  • Concat: temporal (128) + clinical(12) │
│  • 2-layer MLP → pCR probability         │
└──────────────────────────────────────────┘
```

---

## 📚 Data

### I-SPY 2 Trial (n=199)
- 4 longitudinal timepoints (T0, T1, T2, T3)
- 6 DCE phases per timepoint
- Access: [TCIA](https://wiki.cancerimagingarchive.net/display/Public/ISPY2)

### Duke Breast MRI (n=922)
- Single timepoint, used for pre-training
- Access: [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903)

---

## 📖 Citation

```bibtex
@article{laurente2025hierarchical,
  title={Hierarchical LSTM for Pathological Complete Response Prediction 
         in Breast Cancer Using Longitudinal DCE-MRI},
  author={Laurente, Alexander and Pantoja, Diego and León, Fabricio 
          and Sagastegui, Paola},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2025}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)
