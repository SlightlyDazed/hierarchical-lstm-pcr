"""
FASE 6.2: Phase LSTM Fine-tuning for pCR Prediction
=====================================================

Fine-tuning del encoder pre-entrenado para predicción supervisada de pCR.

Esta es la FASE 2 del entrenamiento jerárquico:
1. FASE 1: Pre-train Phase LSTM con autoencoder (COMPLETADO - Val Loss 0.566)
2. FASE 2: Fine-tune con clasificación pCR (ESTE SCRIPT)
3. FASE 3: Integrar con Temporal LSTM

Para ejecutar:
    python fase6_finetune_pcr.py --epochs 100 --lr 0.0001 --freeze-epochs 10

Author: Alexander
Date: 2025-12-23
"""

import os
import sys
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Local imports
from fase6_phase_dataset_flat import (
    PhaseFlatDataset,
    collate_phase_flat,
    load_duke_data,
    load_ispy2_data
)
from fase6_phase_lstm_pretrain import PhaseLSTMEncoder
from fase6_normalization import FeatureNormalizer

# ═════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2")
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = MODELS_DIR / "phase_finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TENSORBOARD_DIR = PROJECT_DIR / "runs" / "phase_finetune"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Pre-trained model paths
PRETRAINED_ENCODER = MODELS_DIR / "phase_lstm_encoder_pretrained.pt"
PRETRAINED_NORMALIZER = MODELS_DIR / "phase_lstm_normalizer.pkl"


# ═════════════════════════════════════════════════════════════════════
# pCR CLASSIFIER MODEL
# ═════════════════════════════════════════════════════════════════════
class PhaseLSTMClassifier(nn.Module):
    """
    Phase LSTM with classification head for pCR prediction.
    
    Architecture:
        Pre-trained Encoder (frozen initially) → Classification Head → pCR
    """
    
    def __init__(
        self,
        encoder: PhaseLSTMEncoder,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.encoder = encoder
        encoder_dim = encoder.output_dim  # hidden_dim * 2 = 256
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier with Xavier/Kaiming."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 6, 1143) - phase features
            mask: (B, 6) - validity mask
        
        Returns:
            logits: (B, num_classes)
            encoding: (B, hidden_dim*2) - for analysis
        """
        # Encode phases
        encoding, attn_weights = self.encoder(x, mask)
        
        # Classify
        logits = self.classifier(encoding)
        
        return logits, encoding
    
    def freeze_encoder(self):
        """Freeze encoder weights for initial training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("🔓 Encoder unfrozen")


# ═════════════════════════════════════════════════════════════════════
# TRAINER
# ═════════════════════════════════════════════════════════════════════
class PCRTrainer:
    """Trainer for pCR classification with pre-trained encoder."""
    
    def __init__(
        self,
        model: PhaseLSTMClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: FeatureNormalizer,
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.device = device
        self.config = config
        
        # Optimizer (different LR for encoder and classifier)
        encoder_params = list(model.encoder.parameters())
        classifier_params = list(model.classifier.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': config['lr'] * 0.1},  # Lower LR for encoder
            {'params': classifier_params, 'lr': config['lr']}
        ], weight_decay=config['weight_decay'])
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss with class weights for imbalanced data
        if config.get('class_weights') is not None:
            weights = torch.FloatTensor(config['class_weights']).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Mixed Precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(TENSORBOARD_DIR / f"run_{timestamp}")
        
        # Tracking
        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['pCR'].to(self.device)
            
            # Skip samples without pCR label
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            features = features[valid_mask]
            mask = mask[valid_mask]
            labels = labels[valid_mask]
            
            # Normalize
            features = self.normalizer.transform(features)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    logits, _ = self.model(features, mask)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(features, mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item() * len(labels)
            
            # Predictions
            probs = F.softmax(logits.detach(), dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Metrics
        avg_loss = total_loss / len(all_labels) if all_labels else 0
        acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        try:
            auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
        except:
            auc = 0
        
        return avg_loss, auc, acc
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['pCR'].to(self.device)
            
            # Skip samples without pCR label
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            
            features = features[valid_mask]
            mask = mask[valid_mask]
            labels = labels[valid_mask]
            
            # Normalize
            features = self.normalizer.transform(features)
            
            if self.use_amp:
                with autocast('cuda'):
                    logits, _ = self.model(features, mask)
                    loss = self.criterion(logits, labels)
            else:
                logits, _ = self.model(features, mask)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            
            # Predictions
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Metrics
        avg_loss = total_loss / len(all_labels) if all_labels else 0
        acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        try:
            auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
        except:
            auc = 0
        
        return avg_loss, auc, acc
    
    def train(self) -> Dict:
        """Main training loop with staged unfreezing."""
        logger.info(f"🚀 Starting pCR fine-tuning for {self.config['epochs']} epochs")
        logger.info(f"   Freeze epochs: {self.config['freeze_epochs']}")
        logger.info(f"   LR: {self.config['lr']}")
        
        # Start with frozen encoder
        if self.config['freeze_epochs'] > 0:
            self.model.freeze_encoder()
        
        for epoch in range(self.config['epochs']):
            # Unfreeze encoder after freeze_epochs
            if epoch == self.config['freeze_epochs'] and self.config['freeze_epochs'] > 0:
                self.model.unfreeze_encoder()
            
            # Train
            train_loss, train_auc, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_auc, val_acc = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[1]['lr']  # Classifier LR
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('AUC/train', train_auc, epoch)
            self.writer.add_scalar('AUC/val', val_auc, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Logging
            if (epoch + 1) % self.config['log_interval'] == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:3d}/{self.config['epochs']}: "
                    f"Train[L={train_loss:.4f}, AUC={train_auc:.3f}] "
                    f"Val[L={val_loss:.4f}, AUC={val_auc:.3f}] "
                    f"LR={current_lr:.2e}"
                )
            
            # Check for best model (based on AUC)
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_checkpoint('best')
                logger.info(f"   ✅ New best! Val AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"\n⏹️  Early stopping at epoch {epoch+1}")
                logger.info(f"   Best Val AUC: {self.best_auc:.4f} (epoch {self.best_epoch})")
                break
        
        self.writer.close()
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.best_epoch if name == 'best' else len(self.history['train_loss']),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, OUTPUT_DIR / f'pcr_classifier_{name}.pt')
        
        if name == 'best':
            logger.info(f"   💾 Best model saved")
    
    def save_history(self):
        """Save training history."""
        df = pd.DataFrame(self.history)
        df.to_csv(OUTPUT_DIR / 'finetune_history.csv', index=False)
        
        summary = {
            'best_epoch': self.best_epoch,
            'best_val_auc': float(self.best_auc),
            'final_train_auc': float(self.history['train_auc'][-1]),
            'final_val_auc': float(self.history['val_auc'][-1]),
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        }
        
        with open(OUTPUT_DIR / 'finetune_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Phase LSTM Fine-tuning for pCR Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--freeze-epochs', type=int, default=10, 
                        help='Epochs to freeze encoder (0 to skip)')
    
    # Options
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--log-interval', type=int, default=5, help='Log every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'freeze_epochs': args.freeze_epochs,
        'use_amp': args.use_amp,
        'log_interval': args.log_interval
    }
    
    print("=" * 70)
    print("🎯 FASE 2: Phase LSTM Fine-tuning for pCR Prediction")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load pre-trained encoder
    logger.info("\n📂 Loading pre-trained encoder...")
    if not PRETRAINED_ENCODER.exists():
        raise FileNotFoundError(f"Pre-trained encoder not found: {PRETRAINED_ENCODER}")
    
    # Create encoder and load weights
    encoder = PhaseLSTMEncoder(
        input_dim=1143,
        hidden_dim=128,
        num_layers=2,
        dropout=0.4
    )
    encoder.load_state_dict(torch.load(PRETRAINED_ENCODER, map_location='cpu'))
    logger.info(f"   ✅ Loaded: {PRETRAINED_ENCODER.name}")
    
    # Load normalizer
    if not PRETRAINED_NORMALIZER.exists():
        raise FileNotFoundError(f"Normalizer not found: {PRETRAINED_NORMALIZER}")
    
    with open(PRETRAINED_NORMALIZER, 'rb') as f:
        normalizer = pickle.load(f)
    logger.info(f"   ✅ Loaded: {PRETRAINED_NORMALIZER.name}")
    
    # Load data
    logger.info("\n📂 Loading data...")
    duke_features, duke_clinical = load_duke_data()
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    # Create dataset (only samples with valid pCR labels)
    all_ids = list(duke_features.keys()) + list(ispy2_features.keys())
    
    # Split
    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.2, random_state=args.seed
    )
    
    train_ds = PhaseFlatDataset(
        train_ids, duke_features, duke_clinical, ispy2_features, ispy2_clinical
    )
    val_ds = PhaseFlatDataset(
        val_ids, duke_features, duke_clinical, ispy2_features, ispy2_clinical
    )
    
    # Count valid pCR samples
    train_pcr_counts = {'total': 0, 'pcr_pos': 0, 'pcr_neg': 0}
    for i in range(len(train_ds)):
        sample = train_ds[i]
        if sample['pCR'] >= 0:
            train_pcr_counts['total'] += 1
            if sample['pCR'] == 1:
                train_pcr_counts['pcr_pos'] += 1
            else:
                train_pcr_counts['pcr_neg'] += 1
    
    val_pcr_counts = {'total': 0, 'pcr_pos': 0, 'pcr_neg': 0}
    for i in range(len(val_ds)):
        sample = val_ds[i]
        if sample['pCR'] >= 0:
            val_pcr_counts['total'] += 1
            if sample['pCR'] == 1:
                val_pcr_counts['pcr_pos'] += 1
            else:
                val_pcr_counts['pcr_neg'] += 1
    
    logger.info(f"   Train: {train_pcr_counts['total']} samples with pCR labels")
    logger.info(f"     pCR+: {train_pcr_counts['pcr_pos']}, pCR-: {train_pcr_counts['pcr_neg']}")
    logger.info(f"   Val: {val_pcr_counts['total']} samples with pCR labels")
    logger.info(f"     pCR+: {val_pcr_counts['pcr_pos']}, pCR-: {val_pcr_counts['pcr_neg']}")
    
    # Class weights for imbalanced data
    total = train_pcr_counts['pcr_pos'] + train_pcr_counts['pcr_neg']
    if total > 0:
        weight_neg = total / (2 * train_pcr_counts['pcr_neg']) if train_pcr_counts['pcr_neg'] > 0 else 1.0
        weight_pos = total / (2 * train_pcr_counts['pcr_pos']) if train_pcr_counts['pcr_pos'] > 0 else 1.0
        config['class_weights'] = [weight_neg, weight_pos]
        logger.info(f"   Class weights: neg={weight_neg:.2f}, pos={weight_pos:.2f}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_phase_flat, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_phase_flat, num_workers=4, pin_memory=True
    )
    
    # Create classifier model
    model = PhaseLSTMClassifier(
        encoder=encoder,
        num_classes=2,
        dropout=0.5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n🧠 Model: {n_params:,} params ({n_trainable:,} trainable)")
    
    # Train
    trainer = PCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        device=device,
        config=config
    )
    
    history = trainer.train()
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("✅ FINE-TUNING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"   Best Val AUC: {trainer.best_auc:.4f} (epoch {trainer.best_epoch})")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print(f"\n🔍 TensorBoard:")
    print(f"   tensorboard --logdir {TENSORBOARD_DIR}")


if __name__ == '__main__':
    main()
