"""
FASE 6: Generate Evidence and Visualizations
=============================================
Genera todas las evidencias necesarias para demostrar resultados:
- Training curves (loss, AUC)
- ROC curves
- Confusion matrices
- Attention heatmaps
- Predictions CSV

Para ejecutar:
    python generate_evidence.py
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    precision_recall_curve, average_precision_score,
    classification_report
)
from sklearn.model_selection import train_test_split

# Local imports
from fase6_phase_dataset_flat import load_ispy2_data
from fase6_phase_lstm_pretrain import PhaseLSTMEncoder
from fase6_normalization import FeatureNormalizer
from fase6_integrated_lstm import (
    IntegratedLSTM, ISPY2TemporalDataset, 
    collate_fn, extract_clinical_features
)

# Config
PROJECT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2")
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results" / "evidence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_ENCODER = MODELS_DIR / "phase_lstm_encoder_pretrained.pt"
PRETRAINED_NORMALIZER = MODELS_DIR / "phase_lstm_normalizer.pkl"
INTEGRATED_MODEL = MODELS_DIR / "integrated_lstm_best.pt"


def set_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def load_training_history():
    """Load all training histories."""
    histories = {}
    
    # Phase pretrain history
    pretrain_history = PROJECT_DIR / "models/phase_pretrain/training_history_run0.csv"
    if pretrain_history.exists():
        histories['pretrain'] = pd.read_csv(pretrain_history)
    
    # Integrated LSTM history
    integrated_results = list((PROJECT_DIR / "results/integrated_lstm").glob("results_*.json"))
    if integrated_results:
        latest = sorted(integrated_results)[-1]
        with open(latest) as f:
            data = json.load(f)
            histories['integrated'] = pd.DataFrame(data['history'])
    
    # Temporal LSTM clinical results
    temporal_results = MODELS_DIR / "temporal_lstm_clinical_results.pkl"
    if temporal_results.exists():
        with open(temporal_results, 'rb') as f:
            data = pickle.load(f)
            histories['temporal_clinical'] = pd.DataFrame(data['history'])
    
    return histories


def plot_training_curves(histories, output_dir):
    """Generate training curve plots."""
    set_style()
    
    # 1. Phase Pre-training Loss Curve
    if 'pretrain' in histories:
        df = histories['pretrain']
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(df) + 1)
        ax.plot(epochs, df['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(epochs, df['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Phase LSTM Pre-training: Reconstruction Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_idx = df['val_loss'].idxmin()
        best_epoch = best_idx + 1
        best_loss = df.loc[best_idx, 'val_loss']
        ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {best_loss:.4f}\n(epoch {best_epoch})', 
                    xy=(best_epoch, best_loss), xytext=(best_epoch+50, best_loss+0.02),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.savefig(output_dir / 'pretrain_loss_curve.png')
        plt.close()
        print(f"  ✅ Saved: pretrain_loss_curve.png")
    
    # 2. Integrated LSTM Training Curves
    if 'integrated' in histories:
        df = histories['integrated']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(df.index + 1, df['train_loss'], label='Train', linewidth=2)
        axes[0].plot(df.index + 1, df['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Cross-Entropy Loss')
        axes[0].set_title('Integrated LSTM: Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(df.index + 1, df['val_auc'], label='Val AUC', linewidth=2, color='green')
        axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Random (0.5)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('Integrated LSTM: Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0.4, 1.0])
        
        # Mark best
        best_idx = df['val_auc'].idxmax()
        best_auc = df.loc[best_idx, 'val_auc']
        axes[1].axvline(best_idx + 1, color='red', linestyle='--', alpha=0.5)
        axes[1].annotate(f'Best: {best_auc:.4f}', 
                        xy=(best_idx + 1, best_auc), xytext=(best_idx + 5, best_auc - 0.05),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'integrated_training_curves.png')
        plt.close()
        print(f"  ✅ Saved: integrated_training_curves.png")
    
    # 3. Comparison of Models
    if 'integrated' in histories and 'temporal_clinical' in histories:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df1 = histories['temporal_clinical']
        df2 = histories['integrated']
        
        ax.plot(df1.index + 1, df1['val_auc'], label='Temporal LSTM + Clinical (baseline)', 
                linewidth=2, linestyle='--')
        ax.plot(df2.index + 1, df2['val_auc'], label='Integrated Phase + Temporal + Clinical', 
                linewidth=2)
        
        ax.axhline(y=0.5, color='gray', linestyle=':', label='Random', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation AUC-ROC')
        ax.set_title('Model Comparison: pCR Prediction Performance')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        
        plt.savefig(output_dir / 'model_comparison.png')
        plt.close()
        print(f"  ✅ Saved: model_comparison.png")


def generate_predictions_and_roc(output_dir):
    """Generate predictions, ROC curve, and save to CSV."""
    print("\n📊 Generating predictions and ROC curve...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    phase_encoder = PhaseLSTMEncoder(input_dim=1143, hidden_dim=128, num_layers=2, dropout=0.4)
    phase_encoder.load_state_dict(torch.load(PRETRAINED_ENCODER, map_location='cpu'))
    
    model = IntegratedLSTM(
        phase_encoder=phase_encoder,
        clinical_dim=12,
        temporal_hidden=64,
        fusion_hidden=64,
        freeze_encoder=True
    ).to(device)
    
    if INTEGRATED_MODEL.exists():
        model.load_state_dict(torch.load(INTEGRATED_MODEL, map_location='cpu'))
    model.eval()
    
    # Load normalizer
    with open(PRETRAINED_NORMALIZER, 'rb') as f:
        normalizer = pickle.load(f)
    
    # Load data
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    valid_patients = []
    for pid in ispy2_features.keys():
        row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
        if len(row) > 0 and 'pCR' in row.columns:
            pcr = row.iloc[0].get('pCR', -1)
            if pd.notna(pcr) and int(pcr) >= 0:
                valid_patients.append(pid)
    
    labels = [ispy2_clinical[ispy2_clinical['PatientID']==p].iloc[0]['pCR'] for p in valid_patients]
    
    # Create full dataset for predictions
    dataset = ISPY2TemporalDataset(valid_patients, ispy2_features, ispy2_clinical)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    
    all_pids = []
    all_probs = []
    all_preds = []
    all_labels = []
    
    for batch in loader:
        try:
            features = batch['features'].to(device)
            phase_mask = batch['phase_mask'].to(device)
            tp_mask = batch['timepoint_mask'].to(device)
            clinical = batch['clinical'].to(device)
            labels_batch = batch['pCR']
            
            B, T, P, F = features.shape
            features_flat = features.view(-1, F)
            features_norm = normalizer.transform(features_flat)
            features = features_norm.view(B, T, P, F)
            
            with torch.no_grad():
                logits = model(features, phase_mask, tp_mask, clinical)
            
            if not isinstance(logits, torch.Tensor):
                print(f"  ⚠️ Skipping batch - invalid output")
                continue
                
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            
            all_pids.extend(batch['patient_ids'])
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
        except Exception as e:
            print(f"  ⚠️ Error in batch: {e}")
            continue
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'PatientID': all_pids,
        'True_pCR': all_labels,
        'Predicted_pCR': all_preds,
        'Probability_pCR': all_probs,
        'Correct': [int(t == p) for t, p in zip(all_labels, all_preds)]
    })
    
    # Add clinical info
    for pid in predictions_df['PatientID']:
        row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
        if len(row) > 0:
            predictions_df.loc[predictions_df['PatientID'] == pid, 'HR'] = row.iloc[0].get('HR', -1)
            predictions_df.loc[predictions_df['PatientID'] == pid, 'HER2'] = row.iloc[0].get('HER2', -1)
            predictions_df.loc[predictions_df['PatientID'] == pid, 'Subtype'] = row.iloc[0].get('Subtype', 'Unknown')
    
    predictions_df.to_csv(output_dir / 'predictions_all_patients.csv', index=False)
    print(f"  ✅ Saved: predictions_all_patients.csv ({len(predictions_df)} patients)")
    
    # Generate ROC Curve
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    
    axes[1].plot(recall, precision, color='green', lw=2,
                 label=f'PR curve (AP = {ap:.3f})')
    axes[1].fill_between(recall, precision, alpha=0.3, color='green')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_pr_curves.png')
    plt.close()
    print(f"  ✅ Saved: roc_pr_curves.png")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['pCR-', 'pCR+'],
                yticklabels=['pCR-', 'pCR+'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix (n={len(all_labels)})\nAUC = {roc_auc:.3f}', fontsize=14)
    
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    print(f"  ✅ Saved: confusion_matrix.png")
    
    # Classification Report
    report = classification_report(all_labels, all_preds, 
                                   target_names=['pCR-', 'pCR+'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / 'classification_report.csv')
    print(f"  ✅ Saved: classification_report.csv")
    
    # Save metrics summary
    metrics = {
        'n_patients': len(all_labels),
        'n_pcr_positive': int(sum(all_labels)),
        'n_pcr_negative': int(len(all_labels) - sum(all_labels)),
        'auc_roc': float(roc_auc),
        'average_precision': float(ap),
        'accuracy': float(np.mean([t == p for t, p in zip(all_labels, all_preds)])),
        'confusion_matrix': cm.tolist()
    }
    
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Saved: metrics_summary.json")
    
    return predictions_df, metrics


def generate_attention_heatmap(output_dir):
    """Generate attention visualization (Phase and Temporal)."""
    print("\n🔍 Generating attention heatmaps...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model for inference with attention extraction
    phase_encoder = PhaseLSTMEncoder(input_dim=1143, hidden_dim=128, num_layers=2, dropout=0.4)
    phase_encoder.load_state_dict(torch.load(PRETRAINED_ENCODER, map_location='cpu'))
    phase_encoder.to(device)
    phase_encoder.eval()
    
    # Load normalizer and data
    with open(PRETRAINED_NORMALIZER, 'rb') as f:
        normalizer = pickle.load(f)
    
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    # Get a sample patient
    valid_patients = []
    for pid in ispy2_features.keys():
        row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
        if len(row) > 0 and 'pCR' in row.columns:
            pcr = row.iloc[0].get('pCR', -1)
            if pd.notna(pcr) and int(pcr) >= 0:
                valid_patients.append(pid)
    
    dataset = ISPY2TemporalDataset(valid_patients[:10], ispy2_features, ispy2_clinical)
    
    # Collect attention weights
    all_phase_attn = []
    patient_ids = []
    pcr_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample['features'].unsqueeze(0).to(device)  # (1, T, P, F)
            phase_mask = sample['phase_mask'].unsqueeze(0).to(device)
            
            B, T, P, F = features.shape
            
            # Get phase attention for each timepoint
            phase_attns = []
            for t in range(T):
                feat_t = features[0, t:t+1]  # (1, P, F)
                mask_t = phase_mask[0, t:t+1]  # (1, P)
                
                # Normalize
                feat_norm = normalizer.transform(feat_t.view(-1, F)).view(1, P, F)
                
                # Get attention
                _, attn = phase_encoder(feat_norm, mask_t)
                phase_attns.append(attn.squeeze().cpu().numpy())
            
            all_phase_attn.append(np.array(phase_attns))
            patient_ids.append(sample['patient_id'])
            pcr_labels.append(sample['pCR'])
    
    # Average attention per pCR status
    pcr_pos_attn = np.mean([a for a, l in zip(all_phase_attn, pcr_labels) if l == 1], axis=0)
    pcr_neg_attn = np.mean([a for a, l in zip(all_phase_attn, pcr_labels) if l == 0], axis=0)
    
    # Plot
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    phase_labels = ['Ph0\n(Pre)', 'Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5\n(Late)']
    timepoint_labels = ['T0\n(Baseline)', 'T1\n(Early)', 'T2\n(Mid)', 'T3\n(Pre-Surg)']
    
    im1 = axes[0].imshow(pcr_neg_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels(phase_labels)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(timepoint_labels)
    axes[0].set_title('Phase Attention: pCR- (Non-responders)', fontsize=12)
    axes[0].set_xlabel('DCE Phase')
    axes[0].set_ylabel('Treatment Timepoint')
    plt.colorbar(im1, ax=axes[0], label='Attention Weight')
    
    im2 = axes[1].imshow(pcr_pos_attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels(phase_labels)
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(timepoint_labels)
    axes[1].set_title('Phase Attention: pCR+ (Responders)', fontsize=12)
    axes[1].set_xlabel('DCE Phase')
    plt.colorbar(im2, ax=axes[1], label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase_attention_heatmap.png')
    plt.close()
    print(f"  ✅ Saved: phase_attention_heatmap.png")
    
    # Save attention data
    attention_df = pd.DataFrame({
        'PatientID': patient_ids,
        'pCR': pcr_labels
    })
    attention_df.to_csv(output_dir / 'attention_samples.csv', index=False)


def generate_dataset_statistics(output_dir):
    """Generate dataset statistics CSV."""
    print("\n📈 Generating dataset statistics...")
    
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    # Patient statistics
    stats = {
        'total_patients': len(ispy2_features),
        'patients_with_pcr': 0,
        'pcr_positive': 0,
        'pcr_negative': 0,
        'hr_positive': 0,
        'her2_positive': 0,
        'tnbc': 0
    }
    
    patient_data = []
    
    for pid in ispy2_features.keys():
        row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
        
        patient_info = {
            'PatientID': pid,
            'pCR': -1,
            'HR': -1,
            'HER2': -1,
            'Subtype': 'Unknown',
            'T0_phases': 0,
            'T1_phases': 0,
            'T2_phases': 0,
            'T3_phases': 0,
            'total_phases': 0
        }
        
        if len(row) > 0:
            pcr = row.iloc[0].get('pCR', -1)
            if pd.notna(pcr) and int(pcr) >= 0:
                patient_info['pCR'] = int(pcr)
                stats['patients_with_pcr'] += 1
                if int(pcr) == 1:
                    stats['pcr_positive'] += 1
                else:
                    stats['pcr_negative'] += 1
            
            hr = row.iloc[0].get('HR', -1)
            if pd.notna(hr) and int(hr) == 1:
                stats['hr_positive'] += 1
                patient_info['HR'] = 1
            elif pd.notna(hr):
                patient_info['HR'] = int(hr)
            
            her2 = row.iloc[0].get('HER2', -1)
            if pd.notna(her2) and int(her2) == 1:
                stats['her2_positive'] += 1
                patient_info['HER2'] = 1
            elif pd.notna(her2):
                patient_info['HER2'] = int(her2)
            
            subtype = row.iloc[0].get('Subtype', 'Unknown')
            if pd.notna(subtype):
                patient_info['Subtype'] = subtype
                if subtype == 'TNBC':
                    stats['tnbc'] += 1
        
        # Count phases per timepoint
        pdata = ispy2_features[pid]
        for tp in ['T0', 'T1', 'T2', 'T3']:
            if tp in pdata:
                n_phases = len([p for p in pdata[tp] if p.startswith('Ph')])
                patient_info[f'{tp}_phases'] = n_phases
                patient_info['total_phases'] += n_phases
        
        patient_data.append(patient_info)
    
    # Save patient data
    patient_df = pd.DataFrame(patient_data)
    patient_df.to_csv(output_dir / 'dataset_patients.csv', index=False)
    print(f"  ✅ Saved: dataset_patients.csv ({len(patient_df)} patients)")
    
    # Save summary statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / 'dataset_summary.csv', index=False)
    print(f"  ✅ Saved: dataset_summary.csv")
    
    # Distribution plot
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # pCR distribution
    pcr_counts = patient_df[patient_df['pCR'] >= 0]['pCR'].value_counts().sort_index()
    axes[0, 0].bar(['pCR- (0)', 'pCR+ (1)'], pcr_counts.values, color=['salmon', 'lightgreen'])
    axes[0, 0].set_title('pCR Distribution')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(pcr_counts.values):
        axes[0, 0].annotate(str(v), xy=(i, v), ha='center', va='bottom')
    
    # Subtype distribution
    subtype_counts = patient_df['Subtype'].value_counts()
    axes[0, 1].bar(subtype_counts.index, subtype_counts.values, color='steelblue')
    axes[0, 1].set_title('Molecular Subtype Distribution')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Phases per patient
    axes[1, 0].hist(patient_df['total_phases'], bins=20, color='purple', alpha=0.7)
    axes[1, 0].set_title('Total DCE Phases per Patient')
    axes[1, 0].set_xlabel('Number of Phases')
    axes[1, 0].set_ylabel('Count')
    
    # Timepoint availability
    tp_data = [
        (patient_df['T0_phases'] > 0).sum(),
        (patient_df['T1_phases'] > 0).sum(),
        (patient_df['T2_phases'] > 0).sum(),
        (patient_df['T3_phases'] > 0).sum()
    ]
    axes[1, 1].bar(['T0', 'T1', 'T2', 'T3'], tp_data, color='teal')
    axes[1, 1].set_title('Patients with Data per Timepoint')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_distributions.png')
    plt.close()
    print(f"  ✅ Saved: dataset_distributions.png")
    
    return stats


def main():
    print("=" * 70)
    print("📊 GENERATING EVIDENCE AND VISUALIZATIONS")
    print("=" * 70)
    print(f"\nOutput directory: {RESULTS_DIR}")
    
    # 1. Load training histories
    print("\n📂 Loading training histories...")
    histories = load_training_history()
    print(f"  Found: {list(histories.keys())}")
    
    # 2. Generate training curves
    print("\n📈 Generating training curves...")
    plot_training_curves(histories, RESULTS_DIR)
    
    # 3. Generate predictions and ROC
    predictions_df, metrics = generate_predictions_and_roc(RESULTS_DIR)
    
    # 4. Generate attention heatmaps
    generate_attention_heatmap(RESULTS_DIR)
    
    # 5. Generate dataset statistics
    stats = generate_dataset_statistics(RESULTS_DIR)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ EVIDENCE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 All outputs saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n📊 Key Metrics:")
    print(f"  - Patients: {metrics['n_patients']}")
    print(f"  - pCR+: {metrics['n_pcr_positive']}, pCR-: {metrics['n_pcr_negative']}")
    print(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  - Average Precision: {metrics['average_precision']:.4f}")


if __name__ == '__main__':
    main()
