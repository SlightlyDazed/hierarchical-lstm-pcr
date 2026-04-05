"""
FASE 6.3: Integrated Phase LSTM + Temporal LSTM for pCR Prediction
====================================================================
Integra el Phase LSTM pre-entrenado con el Temporal LSTM para predicción de pCR.

MEJORAS:
- Train/Val/Test split (75%/20%/5%)
- Métricas completas en test set  
- Pre-trained Phase Encoder (Val Loss 0.566)
- Clinical features (Age, HR, HER2, Subtype)
- Class balancing

Para ejecutar:
    python fase6_integrated_lstm.py --epochs 100 --lr 0.001

Author: Alexander
Date: 2025-12-23
"""

import os
import sys
import argparse
import pickle
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import train_test_split

# Local imports
from fase6_phase_dataset_flat import load_ispy2_data
from fase6_phase_lstm_pretrain import PhaseLSTMEncoder
from fase6_normalization import FeatureNormalizer


# =============================================================================
# Focal Loss for Class Imbalance
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight: Optional class weights
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def optimize_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, dict]:
    """Find optimal threshold using Youden's J statistic.
    
    J = Sensitivity + Specificity - 1 = TPR - FPR
    
    Returns:
        optimal_threshold: Best threshold for classification
        metrics_at_threshold: Dict with sensitivity, specificity at optimal point
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr  # Youden's J statistic
    best_idx = np.argmax(j_scores)
    
    optimal_threshold = thresholds[best_idx]
    
    # Calculate metrics at optimal threshold
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]
    
    return optimal_threshold, {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'youden_j': float(j_scores[best_idx])
    }

# Config
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2")
OUTPUT_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results" / "integrated_lstm"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_ENCODER = OUTPUT_DIR / "phase_lstm_encoder_pretrained.pt"
PRETRAINED_NORMALIZER = OUTPUT_DIR / "phase_lstm_normalizer.pkl"


# =============================================================================
# Clinical Feature Extraction
# =============================================================================
def extract_clinical_features(clinical_df: pd.DataFrame, patient_id: str) -> np.ndarray:
    """Extract 12 clinical features for a patient."""
    row = clinical_df[clinical_df['PatientID'] == patient_id]
    features = np.zeros(12, dtype=np.float32)
    
    if len(row) == 0:
        features[-1] = 1.0  # Missing indicator
        return features
    
    row = row.iloc[0]
    
    # Age (normalized 0-1)
    age = row.get('Age', 50)
    features[0] = (float(age) - 20) / 70.0 if pd.notna(age) else 0.5
    
    # HR (binary)
    hr = row.get('HR', -1)
    features[1] = float(hr) if pd.notna(hr) and hr >= 0 else 0.5
    
    # HER2 (binary)
    her2 = row.get('HER2', -1)
    features[2] = float(her2) if pd.notna(her2) and her2 >= 0 else 0.5
    
    # Subtype one-hot (4 features)
    subtype = row.get('Subtype', '')
    if pd.notna(subtype):
        subtype_map = {'TNBC': 0, 'HR+/HER2-': 1, 'HR+/HER2+': 2, 'HR-/HER2+': 3}
        idx = subtype_map.get(str(subtype), -1)
        if idx >= 0:
            features[3 + idx] = 1.0
    
    # Treatment intervals
    for i, col in enumerate(['T0_to_T1_days', 'T1_to_T2_days', 'T2_to_T3_days']):
        val = row.get(col, 0)
        features[7 + i] = min(float(val) / 90.0, 2.0) if pd.notna(val) and val > 0 else 0
    
    # Timepoints count
    tp_count = row.get('Timepoints_count', 4)
    features[10] = float(tp_count) / 4.0 if pd.notna(tp_count) else 1.0
    
    return features


# =============================================================================
# Dataset
# =============================================================================
class ISPY2TemporalDataset(Dataset):
    """ISPY2 dataset with temporal sequences + clinical features."""
    
    def __init__(
        self,
        patient_ids: List[str],
        ispy2_features: Dict,
        ispy2_clinical: pd.DataFrame,
        feature_dim: int = 1143,
        num_phases: int = 6,
        num_timepoints: int = 4
    ):
        self.ispy2_features = ispy2_features
        self.ispy2_clinical = ispy2_clinical
        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.num_timepoints = num_timepoints
        
        # Filter patients with valid pCR labels
        self.valid_patients = []
        for pid in patient_ids:
            if pid in ispy2_features:
                row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
                if len(row) > 0 and 'pCR' in row.columns:
                    pcr = row.iloc[0].get('pCR', -1)
                    if pd.notna(pcr) and int(pcr) >= 0:
                        self.valid_patients.append(pid)
    
    def __len__(self) -> int:
        return len(self.valid_patients)
    
    def __getitem__(self, idx: int) -> Dict:
        patient_id = self.valid_patients[idx]
        patient_data = self.ispy2_features[patient_id]
        
        features = np.zeros((self.num_timepoints, self.num_phases, self.feature_dim), dtype=np.float32)
        mask = np.zeros((self.num_timepoints, self.num_phases), dtype=np.float32)
        timepoint_mask = np.zeros(self.num_timepoints, dtype=np.float32)
        
        timepoints = ['T0', 'T1', 'T2', 'T3']
        phases = ['Ph0', 'Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5']
        
        for t_idx, tp in enumerate(timepoints):
            if tp not in patient_data:
                continue
            
            tp_data = patient_data[tp]
            has_valid_phase = False
            
            for p_idx, ph in enumerate(phases[:self.num_phases]):
                if ph not in tp_data:
                    continue
                
                phase_data = tp_data[ph]
                densenet = np.array(phase_data.get('densenet_features', []), dtype=np.float32)
                radiomics = np.array(phase_data.get('radiomics_features', []), dtype=np.float32)
                spatial = np.array(phase_data.get('spatial_features', []), dtype=np.float32)
                combined = np.concatenate([densenet, radiomics, spatial])
                
                if len(combined) > 0:
                    if len(combined) < self.feature_dim:
                        padded = np.zeros(self.feature_dim, dtype=np.float32)
                        padded[:len(combined)] = combined
                        features[t_idx, p_idx] = padded
                    else:
                        features[t_idx, p_idx] = combined[:self.feature_dim]
                    mask[t_idx, p_idx] = 1.0
                    has_valid_phase = True
            
            if has_valid_phase:
                timepoint_mask[t_idx] = 1.0
        
        clinical = extract_clinical_features(self.ispy2_clinical, patient_id)
        row = self.ispy2_clinical[self.ispy2_clinical['PatientID'] == patient_id]
        pcr = int(row.iloc[0]['pCR'])
        
        return {
            'patient_id': patient_id,
            'features': torch.FloatTensor(features),
            'phase_mask': torch.FloatTensor(mask),
            'timepoint_mask': torch.FloatTensor(timepoint_mask),
            'clinical': torch.FloatTensor(clinical),
            'pCR': pcr
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        'patient_ids': [b['patient_id'] for b in batch],
        'features': torch.stack([b['features'] for b in batch]),
        'phase_mask': torch.stack([b['phase_mask'] for b in batch]),
        'timepoint_mask': torch.stack([b['timepoint_mask'] for b in batch]),
        'clinical': torch.stack([b['clinical'] for b in batch]),
        'pCR': torch.tensor([b['pCR'] for b in batch]),
    }


# =============================================================================
# Model: Integrated Phase LSTM + Temporal LSTM
# =============================================================================
class IntegratedLSTM(nn.Module):
    """
    Integrated model using:
    - Pre-trained Phase Encoder (FROZEN)
    - Temporal LSTM over timepoints
    - Clinical features fusion
    - pCR classification head
    """
    
    def __init__(
        self,
        phase_encoder: PhaseLSTMEncoder,
        clinical_dim: int = 12,
        temporal_hidden: int = 64,
        fusion_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        # Phase Encoder (optionally frozen)
        self.phase_encoder = phase_encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.phase_encoder.parameters():
                param.requires_grad = False
        
        # Temporal LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=phase_encoder.output_dim,  # 256
            hidden_size=temporal_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Clinical encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        # Fusion
        fusion_input = temporal_hidden * 2 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # pCR head
        self.pcr_head = nn.Sequential(
            nn.Linear(fusion_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        phase_mask: torch.Tensor,
        timepoint_mask: torch.Tensor,
        clinical: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T=4, P=6, F=1143)
            phase_mask: (B, T, P)
            timepoint_mask: (B, T)
            clinical: (B, 12)
        Returns:
            logits: (B, 2)
        """
        B, T, P, F = x.shape
        
        # Encode each timepoint with Phase Encoder
        x_flat = x.view(B * T, P, F)
        mask_flat = phase_mask.view(B * T, P)
        
        if self.freeze_encoder:
            with torch.no_grad():
                phase_encodings, _ = self.phase_encoder(x_flat, mask_flat)
        else:
            phase_encodings, _ = self.phase_encoder(x_flat, mask_flat)
        
        temporal_seq = phase_encodings.view(B, T, -1)
        
        # Temporal LSTM
        temporal_out, _ = self.temporal_lstm(temporal_seq)
        
        # Temporal Attention
        attn_scores = self.temporal_attention(temporal_out)
        mask_exp = timepoint_mask.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(mask_exp == 0, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1)
        temporal_context = torch.sum(temporal_out * attn_weights, dim=1)
        
        # Clinical encoding
        clinical_encoded = self.clinical_encoder(clinical)
        
        # Fusion
        fused = torch.cat([temporal_context, clinical_encoded], dim=-1)
        fused = self.fusion(fused)
        
        # pCR prediction
        logits = self.pcr_head(fused)
        
        return logits


# =============================================================================
# Training and Evaluation
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device, normalizer):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        features = batch['features'].to(device)
        phase_mask = batch['phase_mask'].to(device)
        tp_mask = batch['timepoint_mask'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['pCR'].to(device)
        
        # Normalize
        B, T, P, F = features.shape
        features_flat = features.view(-1, F)
        features_norm = normalizer.transform(features_flat)
        features = features_norm.view(B, T, P, F)
        
        optimizer.zero_grad()
        logits = model(features, phase_mask, tp_mask, clinical)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, normalizer, return_predictions=False):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_labels = []
    all_patient_ids = []
    
    for batch in loader:
        features = batch['features'].to(device)
        phase_mask = batch['phase_mask'].to(device)
        tp_mask = batch['timepoint_mask'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['pCR'].to(device)
        
        B, T, P, F = features.shape
        features_flat = features.view(-1, F)
        features_norm = normalizer.transform(features_flat)
        features = features_norm.view(B, T, P, F)
        
        logits = model(features, phase_mask, tp_mask, clinical)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_patient_ids.extend(batch['patient_ids'])
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
    }
    
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    else:
        metrics['auc'] = 0.5
    
    if return_predictions:
        metrics['predictions'] = {
            'patient_ids': all_patient_ids,
            'probs': all_probs.tolist(),
            'preds': all_preds.tolist(),
            'labels': all_labels.tolist()
        }
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
        metrics['classification_report'] = classification_report(all_labels, all_preds, output_dict=True)
    
    return metrics


# =============================================================================
# Main Training
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Integrated Phase+Temporal LSTM for pCR')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set ratio (15%)')
    parser.add_argument('--val-size', type=float, default=0.15, help='Val set ratio (15%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 FASE 6.3: Integrated Phase LSTM + Temporal LSTM")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load pre-trained components
    print("\n📂 Loading pre-trained components...")
    phase_encoder = PhaseLSTMEncoder(input_dim=1143, hidden_dim=128, num_layers=2, dropout=0.4)
    phase_encoder.load_state_dict(torch.load(PRETRAINED_ENCODER, map_location='cpu'))
    print("  ✅ Phase encoder loaded (Val Loss 0.566)")
    
    with open(PRETRAINED_NORMALIZER, 'rb') as f:
        normalizer = pickle.load(f)
    print("  ✅ Normalizer loaded")
    
    # Load ISPY2 data
    print("\n📂 Loading ISPY2 data...")
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    # Get valid patients with pCR labels
    valid_patients = []
    for pid in ispy2_features.keys():
        row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
        if len(row) > 0 and 'pCR' in row.columns:
            pcr = row.iloc[0].get('pCR', -1)
            if pd.notna(pcr) and int(pcr) >= 0:
                valid_patients.append(pid)
    
    print(f"  Total patients with pCR: {len(valid_patients)}")
    
    # Get labels for stratification
    labels = [ispy2_clinical[ispy2_clinical['PatientID']==p].iloc[0]['pCR'] for p in valid_patients]
    labels = np.array(labels)
    
    # Split: Train / Val+Test
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        valid_patients, labels, 
        test_size=args.val_size + args.test_size, 
        random_state=args.seed, 
        stratify=labels
    )
    
    # Split temp into Val / Test
    test_ratio_adjusted = args.test_size / (args.val_size + args.test_size)
    val_ids, test_ids, val_labels, test_labels = train_test_split(
        temp_ids, temp_labels,
        test_size=test_ratio_adjusted,
        random_state=args.seed,
        stratify=temp_labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_ids)} patients (pCR+: {sum(train_labels)}, pCR-: {len(train_labels)-sum(train_labels)})")
    print(f"  Val:   {len(val_ids)} patients (pCR+: {sum(val_labels)}, pCR-: {len(val_labels)-sum(val_labels)})")
    print(f"  Test:  {len(test_ids)} patients (pCR+: {sum(test_labels)}, pCR-: {len(test_labels)-sum(test_labels)})")
    
    # Create datasets
    train_ds = ISPY2TemporalDataset(train_ids, ispy2_features, ispy2_clinical)
    val_ds = ISPY2TemporalDataset(val_ids, ispy2_features, ispy2_clinical)
    test_ds = ISPY2TemporalDataset(test_ids, ispy2_features, ispy2_clinical)
    
    print(f"\n  After filtering - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    
    # Create model
    print("\n🧠 Creating Integrated LSTM model...")
    model = IntegratedLSTM(
        phase_encoder=phase_encoder,
        clinical_dim=12,
        temporal_hidden=64,
        fusion_hidden=64,
        num_layers=1,
        dropout=0.3,
        freeze_encoder=True
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable:,} params")
    print(f"  Frozen:    {frozen:,} params (Phase Encoder)")
    
    # Class weights (sklearn-style: balanced)
    # weight_i = n_samples / (n_classes * count_i)
    pcr_counts = np.bincount([train_ds[i]['pCR'] for i in range(len(train_ds))])
    n_samples = len(train_ds)
    n_classes = len(pcr_counts)
    pcr_weights = n_samples / (n_classes * pcr_counts + 1e-6)
    class_weights = torch.tensor(pcr_weights, dtype=torch.float32, device=device)
    print(f"  pCR distribution: {dict(enumerate(pcr_counts))}")
    print(f"  Class weights (balanced): [{pcr_weights[0]:.3f}, {pcr_weights[1]:.3f}]")
    print(f"    pCR- weight: {pcr_weights[0]:.3f} (majority class)")
    print(f"    pCR+ weight: {pcr_weights[1]:.3f} (minority class)")
    
    # Use Focal Loss for better handling of class imbalance
    criterion = FocalLoss(alpha=1.0, gamma=1.0, weight=class_weights)
    print(f"  Using Focal Loss (alpha=1.0, gamma=1.0)")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.05  # Increased regularization
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=0.05)")
    
    # Training loop
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print("-" * 70)
    
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, normalizer)
        val_metrics = evaluate(model, val_loader, criterion, device, normalizer)
        
        val_auc = val_metrics['auc']
        scheduler.step(val_auc)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_auc)
        history['val_acc'].append(val_metrics['accuracy'])
        
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'integrated_lstm_best.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, "
                  f"Val[AUC={val_auc:.4f}, Acc={val_metrics['accuracy']:.4f}] "
                  f"{'✅ BEST' if is_best else ''}")
        
        if patience_counter >= args.patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    model.load_state_dict(torch.load(OUTPUT_DIR / 'integrated_lstm_best.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device, normalizer, return_predictions=True)
    
    # Get optimal threshold using Youden's J on validation set
    val_metrics_full = evaluate(model, val_loader, criterion, device, normalizer, return_predictions=True)
    val_probs = np.array(val_metrics_full['predictions']['probs'])
    val_labels = np.array(val_metrics_full['predictions']['labels'])
    optimal_threshold, threshold_metrics = optimize_threshold(val_probs, val_labels)
    
    # Recalculate test predictions with optimal threshold
    test_probs = np.array(test_metrics['predictions']['probs'])
    test_labels = np.array(test_metrics['predictions']['labels'])
    test_preds_optimal = (test_probs >= optimal_threshold).astype(int)
    
    # Recalculate metrics with optimal threshold
    cm_optimal = confusion_matrix(test_labels, test_preds_optimal)
    tn, fp, fn, tp = cm_optimal.ravel()
    
    print(f"\n🏆 Best Validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"\n🎯 Optimal Threshold: {optimal_threshold:.4f} (Youden's J)")
    print(f"   Sensitivity (val): {threshold_metrics['sensitivity']:.4f}")
    print(f"   Specificity (val): {threshold_metrics['specificity']:.4f}")
    
    print(f"\n📈 Test Set Metrics (threshold={optimal_threshold:.4f}):")
    print(f"   AUC:         {test_metrics['auc']:.4f}")
    print(f"   Accuracy:    {accuracy_score(test_labels, test_preds_optimal):.4f}")
    print(f"   Sensitivity: {tp/(tp+fn) if (tp+fn) > 0 else 0:.4f}")
    print(f"   Specificity: {tn/(tn+fp) if (tn+fp) > 0 else 0:.4f}")
    print(f"   PPV:         {tp/(tp+fp) if (tp+fp) > 0 else 0:.4f}")
    print(f"   NPV:         {tn/(tn+fn) if (tn+fn) > 0 else 0:.4f}")
    
    print(f"\n📊 Confusion Matrix (threshold={optimal_threshold:.4f}):")
    print(f"   [[TN={cm_optimal[0,0]:3d}  FP={cm_optimal[0,1]:3d}]")
    print(f"    [FN={cm_optimal[1,0]:3d}  TP={cm_optimal[1,1]:3d}]]")
    
    print(f"\n📊 Comparison with threshold=0.5:")
    cm = np.array(test_metrics['confusion_matrix'])
    print(f"   Old: [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]  Specificity={cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    print(f"         [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_auc': float(best_val_auc),
        'optimal_threshold': float(optimal_threshold),
        'threshold_metrics': threshold_metrics,
        'test_metrics': {k: v for k, v in test_metrics.items() if k != 'predictions'},
        'test_metrics_optimal': {
            'auc': test_metrics['auc'],
            'accuracy': float(accuracy_score(test_labels, test_preds_optimal)),
            'sensitivity': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
            'specificity': float(tn/(tn+fp)) if (tn+fp) > 0 else 0,
            'ppv': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
            'npv': float(tn/(tn+fn)) if (tn+fn) > 0 else 0,
            'confusion_matrix': cm_optimal.tolist()
        },
        'history': history,
        'config': vars(args),
        'split': {
            'train': len(train_ds),
            'val': len(val_ds),
            'test': len(test_ds)
        }
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(RESULTS_DIR / f'results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(RESULTS_DIR / f'predictions_{timestamp}.json', 'w') as f:
        json.dump(test_metrics.get('predictions', {}), f, indent=2)
    
    print(f"\n💾 Results saved to: {RESULTS_DIR}")
    print(f"💾 Model saved to: {OUTPUT_DIR / 'integrated_lstm_best.pt'}")
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
