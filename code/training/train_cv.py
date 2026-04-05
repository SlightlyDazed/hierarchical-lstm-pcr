"""
FASE 6.4: 5-Fold Cross-Validation + Visualizations
===================================================
Entrena el modelo con 5-Fold CV estratificado y genera:
- Curvas ROC por fold y promedio
- Confusion matrices
- Attention heatmaps (alternativa a GradCAM para LSTM)
- Métricas agregadas con intervalos de confianza

Para ejecutar:
    python fase6_cv_training.py --epochs 50 --folds 5

Author: Alexander
Date: 2025-12-23
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, auc
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from fase6_phase_dataset_flat import load_ispy2_data
from fase6_phase_lstm_pretrain import PhaseLSTMEncoder
from fase6_normalization import FeatureNormalizer
from fase6_integrated_lstm import (
    IntegratedLSTM, ISPY2TemporalDataset, collate_fn,
    FocalLoss, optimize_threshold
)

# Config
PROJECT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2")
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results" / "cv_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_ENCODER = MODELS_DIR / "phase_lstm_encoder_pretrained.pt"
PRETRAINED_NORMALIZER = MODELS_DIR / "phase_lstm_normalizer.pkl"


def set_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def train_epoch(model, loader, criterion, optimizer, device, normalizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
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
        
        optimizer.zero_grad()
        logits = model(features, phase_mask, tp_mask, clinical)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, normalizer):
    """Evaluate and return predictions."""
    model.eval()
    total_loss = 0.0
    all_probs = []
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
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_patient_ids.extend(batch['patient_ids'])
    
    return {
        'loss': total_loss / len(loader),
        'probs': np.array(all_probs),
        'labels': np.array(all_labels),
        'patient_ids': all_patient_ids
    }


@torch.no_grad()
def get_attention_weights(model, loader, device, normalizer):
    """Extract attention weights for visualization."""
    model.eval()
    all_phase_attn = []
    all_temporal_attn = []
    all_labels = []
    all_patient_ids = []
    
    for batch in loader:
        features = batch['features'].to(device)
        phase_mask = batch['phase_mask'].to(device)
        tp_mask = batch['timepoint_mask'].to(device)
        clinical = batch['clinical'].to(device)
        labels = batch['pCR']
        
        B, T, P, F = features.shape
        features_flat = features.view(-1, F)
        features_norm = normalizer.transform(features_flat)
        features = features_norm.view(B, T, P, F)
        
        # Get phase attention from encoder
        x_flat = features.view(B * T, P, F)
        mask_flat = phase_mask.view(B * T, P)
        
        with torch.no_grad():
            _, phase_attn = model.phase_encoder(x_flat, mask_flat)
        
        phase_attn = phase_attn.view(B, T, P).cpu().numpy()
        all_phase_attn.extend(phase_attn)
        all_labels.extend(labels.numpy())
        all_patient_ids.extend(batch['patient_ids'])
    
    return {
        'phase_attention': np.array(all_phase_attn),
        'labels': np.array(all_labels),
        'patient_ids': all_patient_ids
    }


def plot_roc_curves(fold_results: List[Dict], output_dir: Path):
    """Plot ROC curves for all folds + mean."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(fold_results)))
    
    for i, result in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(result['labels'], result['probs'])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate for mean calculation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        ax.plot(fpr, tpr, lw=1.5, alpha=0.6, color=colors[i],
                label=f'Fold {i+1} (AUC = {roc_auc:.3f})')
    
    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='darkred', lw=3,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    # Confidence interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='pink', alpha=0.3,
                    label='± 1 std. dev.')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('5-Fold Cross-Validation ROC Curves\npCR Prediction', fontsize=16)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'cv_roc_curves.png')
    plt.close()
    print(f"  ✅ Saved: cv_roc_curves.png")
    
    return mean_auc, std_auc


def plot_confusion_matrices(fold_results: List[Dict], thresholds: List[float], output_dir: Path):
    """Plot confusion matrices for all folds."""
    set_style()
    n_folds = len(fold_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    all_labels = []
    all_preds = []
    
    for i, (result, threshold) in enumerate(zip(fold_results, thresholds)):
        preds = (result['probs'] >= threshold).astype(int)
        cm = confusion_matrix(result['labels'], preds)
        
        all_labels.extend(result['labels'])
        all_preds.extend(preds)
        
        # Normalize for display
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['pCR-', 'pCR+'],
                    yticklabels=['pCR-', 'pCR+'],
                    annot_kws={'size': 14})
        
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        axes[i].set_title(f'Fold {i+1}\nSpec={spec:.2f}, Sens={sens:.2f}', fontsize=12)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Aggregate confusion matrix
    cm_total = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Greens', ax=axes[5],
                xticklabels=['pCR-', 'pCR+'],
                yticklabels=['pCR-', 'pCR+'],
                annot_kws={'size': 14})
    
    tn, fp, fn, tp = cm_total.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = (tn + tp) / (tn + fp + fn + tp)
    
    axes[5].set_title(f'AGGREGATE (n={len(all_labels)})\nSpec={spec:.2f}, Sens={sens:.2f}, Acc={acc:.2f}', fontsize=12)
    axes[5].set_xlabel('Predicted')
    axes[5].set_ylabel('Actual')
    
    plt.suptitle('5-Fold Cross-Validation Confusion Matrices', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_confusion_matrices.png')
    plt.close()
    print(f"  ✅ Saved: cv_confusion_matrices.png")
    
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'spec': spec, 'sens': sens, 'acc': acc}


def plot_attention_heatmap(attention_data: Dict, output_dir: Path):
    """Plot phase attention heatmaps grouped by pCR status."""
    set_style()
    
    phase_attn = attention_data['phase_attention']  # (N, T, P)
    labels = attention_data['labels']
    
    # Average by pCR status
    pcr_pos_mask = labels == 1
    pcr_neg_mask = labels == 0
    
    if pcr_pos_mask.sum() > 0 and pcr_neg_mask.sum() > 0:
        pcr_pos_attn = phase_attn[pcr_pos_mask].mean(axis=0)  # (T, P)
        pcr_neg_attn = phase_attn[pcr_neg_mask].mean(axis=0)  # (T, P)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        phase_labels = ['Ph0\n(Pre)', 'Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5\n(Late)']
        tp_labels = ['T0\n(Baseline)', 'T1\n(Early)', 'T2\n(Mid)', 'T3\n(Pre-Surg)']
        
        # pCR- heatmap
        im1 = axes[0].imshow(pcr_neg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.35)
        axes[0].set_xticks(range(6))
        axes[0].set_xticklabels(phase_labels)
        axes[0].set_yticks(range(4))
        axes[0].set_yticklabels(tp_labels)
        axes[0].set_title(f'pCR- (Non-responders, n={pcr_neg_mask.sum()})', fontsize=14)
        axes[0].set_xlabel('DCE Phase')
        axes[0].set_ylabel('Treatment Timepoint')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')
        
        # pCR+ heatmap
        im2 = axes[1].imshow(pcr_pos_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.35)
        axes[1].set_xticks(range(6))
        axes[1].set_xticklabels(phase_labels)
        axes[1].set_yticks(range(4))
        axes[1].set_yticklabels(tp_labels)
        axes[1].set_title(f'pCR+ (Responders, n={pcr_pos_mask.sum()})', fontsize=14)
        axes[1].set_xlabel('DCE Phase')
        plt.colorbar(im2, ax=axes[1], label='Attention Weight')
        
        # Difference heatmap
        diff = pcr_pos_attn - pcr_neg_attn
        im3 = axes[2].imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
        axes[2].set_xticks(range(6))
        axes[2].set_xticklabels(phase_labels)
        axes[2].set_yticks(range(4))
        axes[2].set_yticklabels(tp_labels)
        axes[2].set_title('Difference (pCR+ - pCR-)', fontsize=14)
        axes[2].set_xlabel('DCE Phase')
        plt.colorbar(im3, ax=axes[2], label='Δ Attention')
        
        plt.suptitle('Phase Attention Analysis by pCR Status', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'cv_attention_heatmap.png')
        plt.close()
        print(f"  ✅ Saved: cv_attention_heatmap.png")


def plot_metrics_summary(fold_metrics: List[Dict], output_dir: Path):
    """Plot summary metrics across folds."""
    set_style()
    
    metrics = ['auc', 'accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']
    metric_names = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.12
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(fold_metrics)))
    
    for i, fold_metric in enumerate(fold_metrics):
        values = [fold_metric[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=f'Fold {i+1}', color=colors[i], alpha=0.8)
    
    # Mean line
    means = []
    stds = []
    for m in metrics:
        vals = [fm[m] for fm in fold_metrics]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    
    ax.errorbar(x + width * 2, means, yerr=stds, fmt='ko-', markersize=10, lw=2,
                label='Mean ± Std', capsize=5)
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('5-Fold Cross-Validation Metrics Summary', fontsize=16)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(x[i] + width * 2, mean + std + 0.03, f'{mean:.2f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_metrics_summary.png')
    plt.close()
    print(f"  ✅ Saved: cv_metrics_summary.png")
    
    return dict(zip(metrics, zip(means, stds)))


def main():
    parser = argparse.ArgumentParser(description='5-Fold CV for pCR Prediction')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per fold')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 FASE 6.4: 5-Fold Cross-Validation + Visualizations")
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
    with open(PRETRAINED_NORMALIZER, 'rb') as f:
        normalizer = pickle.load(f)
    print("  ✅ Normalizer loaded")
    
    # Load ISPY2 data
    print("\n📂 Loading ISPY2 data...")
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    # Get valid patients with pCR labels
    valid_patients = []
    labels = []
    for pid in ispy2_features.keys():
        row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
        if len(row) > 0 and 'pCR' in row.columns:
            pcr = row.iloc[0].get('pCR', -1)
            if pd.notna(pcr) and int(pcr) >= 0:
                valid_patients.append(pid)
                labels.append(int(pcr))
    
    valid_patients = np.array(valid_patients)
    labels = np.array(labels)
    
    print(f"  Total patients: {len(valid_patients)}")
    print(f"  pCR+: {labels.sum()}, pCR-: {len(labels) - labels.sum()}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    fold_metrics = []
    fold_thresholds = []
    all_attention_data = {'phase_attention': [], 'labels': [], 'patient_ids': []}
    
    print(f"\n🚀 Starting {args.folds}-Fold Cross-Validation...")
    print("-" * 70)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_patients, labels)):
        print(f"\n📊 Fold {fold + 1}/{args.folds}")
        
        train_pids = valid_patients[train_idx]
        val_pids = valid_patients[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        print(f"   Train: {len(train_pids)} (pCR+: {train_labels.sum()})")
        print(f"   Val:   {len(val_pids)} (pCR+: {val_labels.sum()})")
        
        # Create datasets
        train_ds = ISPY2TemporalDataset(train_pids.tolist(), ispy2_features, ispy2_clinical)
        val_ds = ISPY2TemporalDataset(val_pids.tolist(), ispy2_features, ispy2_clinical)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)
        
        # Create fresh model for each fold
        phase_encoder = PhaseLSTMEncoder(input_dim=1143, hidden_dim=128, num_layers=2, dropout=0.4)
        phase_encoder.load_state_dict(torch.load(PRETRAINED_ENCODER, map_location='cpu'))
        
        model = IntegratedLSTM(
            phase_encoder=phase_encoder,
            clinical_dim=12,
            temporal_hidden=64,
            fusion_hidden=64,
            freeze_encoder=True
        ).to(device)
        
        # Class weights
        pcr_counts = np.bincount(train_labels)
        n_samples = len(train_labels)
        n_classes = len(pcr_counts)
        pcr_weights = n_samples / (n_classes * pcr_counts + 1e-6)
        class_weights = torch.tensor(pcr_weights, dtype=torch.float32, device=device)
        
        criterion = FocalLoss(alpha=1.0, gamma=1.0, weight=class_weights)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, normalizer)
            val_result = evaluate(model, val_loader, criterion, device, normalizer)
            
            val_auc = roc_auc_score(val_result['labels'], val_result['probs']) \
                      if len(np.unique(val_result['labels'])) > 1 else 0.5
            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"   Early stop at epoch {epoch + 1}")
                break
        
        # Load best model and evaluate
        model.load_state_dict(best_state)
        model.to(device)
        
        val_result = evaluate(model, val_loader, criterion, device, normalizer)
        
        # Optimize threshold
        optimal_threshold, _ = optimize_threshold(val_result['probs'], val_result['labels'])
        fold_thresholds.append(optimal_threshold)
        
        # Calculate metrics with optimal threshold
        preds = (val_result['probs'] >= optimal_threshold).astype(int)
        cm = confusion_matrix(val_result['labels'], preds)
        tn, fp, fn, tp = cm.ravel()
        
        fold_metric = {
            'auc': best_val_auc,
            'accuracy': accuracy_score(val_result['labels'], preds),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'threshold': optimal_threshold
        }
        
        fold_results.append(val_result)
        fold_metrics.append(fold_metric)
        
        print(f"   Best AUC: {best_val_auc:.4f}, Threshold: {optimal_threshold:.4f}")
        print(f"   Sens: {fold_metric['sensitivity']:.4f}, Spec: {fold_metric['specificity']:.4f}")
        
        # Collect attention data
        attn_data = get_attention_weights(model, val_loader, device, normalizer)
        all_attention_data['phase_attention'].extend(attn_data['phase_attention'])
        all_attention_data['labels'].extend(attn_data['labels'])
        all_attention_data['patient_ids'].extend(attn_data['patient_ids'])
    
    # Convert attention data
    all_attention_data['phase_attention'] = np.array(all_attention_data['phase_attention'])
    all_attention_data['labels'] = np.array(all_attention_data['labels'])
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    mean_auc, std_auc = plot_roc_curves(fold_results, RESULTS_DIR)
    aggregate_cm = plot_confusion_matrices(fold_results, fold_thresholds, RESULTS_DIR)
    plot_attention_heatmap(all_attention_data, RESULTS_DIR)
    mean_metrics = plot_metrics_summary(fold_metrics, RESULTS_DIR)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ 5-FOLD CROSS-VALIDATION COMPLETE")
    print("=" * 70)
    
    print(f"\n📈 Aggregate Metrics (n={len(valid_patients)}):")
    print(f"   AUC:         {mean_auc:.4f} ± {std_auc:.4f}")
    for metric, (mean, std) in mean_metrics.items():
        print(f"   {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
    
    print(f"\n📊 Aggregate Confusion Matrix:")
    print(f"   [[TN={aggregate_cm['tn']:3d}  FP={aggregate_cm['fp']:3d}]")
    print(f"    [FN={aggregate_cm['fn']:3d}  TP={aggregate_cm['tp']:3d}]]")
    
    # Save results
    results = {
        'n_patients': len(valid_patients),
        'n_folds': args.folds,
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'mean_metrics': {k: {'mean': float(v[0]), 'std': float(v[1])} for k, v in mean_metrics.items()},
        'aggregate_cm': aggregate_cm,
        'fold_metrics': fold_metrics,
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_DIR / 'cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {RESULTS_DIR}")
    print("   - cv_roc_curves.png")
    print("   - cv_confusion_matrices.png")
    print("   - cv_attention_heatmap.png")
    print("   - cv_metrics_summary.png")
    print("   - cv_results.json")


if __name__ == '__main__':
    main()
