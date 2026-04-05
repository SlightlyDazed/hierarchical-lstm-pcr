"""
FASE 6: PRODUCTION-GRADE Phase LSTM Pre-training
==================================================

Entrenamiento robusto con:
- OneCycleLR + Warmup
- Mixed Precision (AMP)
- EMA + SWA
- Gradient Accumulation
- Data Augmentation
- TensorBoard logging
- Multi-run averaging

Para ejecutar:
    python fase6_pretrain_production.py --epochs 300 --batch-size 32 --runs 3

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
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import train_test_split

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Local imports
from fase6_phase_dataset_flat import (
    PhaseFlatDataset,
    collate_phase_flat,
    load_duke_data,
    load_ispy2_data
)
from fase6_phase_lstm_pretrain import PhaseLSTMAutoencoder
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
OUTPUT_DIR = PROJECT_DIR / "models" / "phase_pretrain"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TENSORBOARD_DIR = PROJECT_DIR / "runs" / "phase_pretrain"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═════════════════════════════════════════════════════════════════════
class PhaseAugmentation:
    """Data augmentation for phase features."""
    
    def __init__(self, noise_std=0.01, dropout_prob=0.1, mixup_alpha=0.0):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha
    
    def __call__(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to features.
        
        Args:
            features: (B, P, F) or (P, F)
            mask: (B, P) or (P,)
        
        Returns:
            Augmented features
        """
        # Gaussian noise
        if self.noise_std > 0 and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
        
        # Feature dropout (random zeroing)
        if self.dropout_prob > 0 and torch.rand(1).item() > 0.5:
            feat_mask = torch.rand_like(features) > self.dropout_prob
            features = features * feat_mask.float()
        
        return features
    
    def mixup(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor, 
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, float]:
        """
        Mixup augmentation between two samples.
        
        Args:
            features1, features2: (P, F) or (B, P, F)
            alpha: Beta distribution parameter
        
        Returns:
            Mixed features and lambda value
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        
        mixed = lam * features1 + (1 - lam) * features2
        return mixed, lam


# ═════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ═════════════════════════════════════════════════════════════════════
class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow (EMA) weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


# ═════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ═════════════════════════════════════════════════════════════════════
class PhasePretrainer:
    """Production-grade trainer for Phase LSTM pre-training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: FeatureNormalizer,
        device: torch.device,
        config: Dict,
        run_id: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.device = device
        self.config = config
        self.run_id = run_id
        
        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Calculate total steps for OneCycleLR
        steps_per_epoch = len(train_loader) // config['accumulation_steps']
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        total_steps = config['epochs'] * steps_per_epoch
        
        # Scheduler: OneCycleLR with warmup and cosine annealing
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,  # Initial LR = max_lr / 25
            final_div_factor=10000.0  # Final LR = initial_lr / 10000
        )
        
        # Loss: MSE per element for masked loss
        self.criterion = nn.MSELoss(reduction='none')
        
        # Mixed Precision (AMP)
        self.use_amp = config['use_amp'] and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # EMA
        self.ema = EMA(model, decay=config['ema_decay']) if config['use_ema'] else None
        
        # SWA (Stochastic Weight Averaging)
        self.use_swa = config.get('use_swa', False)
        if self.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_start = int(config['epochs'] * 0.75)  # Start SWA at 75% of training
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=config['lr'] * 0.1)
        
        # Data Augmentation
        self.augment = PhaseAugmentation(
            noise_std=config['noise_std'],
            dropout_prob=config['feature_dropout'],
            mixup_alpha=config.get('mixup_alpha', 0.0)
        ) if config['use_augmentation'] else None
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"run{run_id}_lr{config['lr']}_bs{config['batch_size']}_{timestamp}"
        self.writer = SummaryWriter(TENSORBOARD_DIR / run_name)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'grad_norm': []
        }
        
        # Global step for TensorBoard
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: Average training loss
            avg_grad_norm: Average gradient norm
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)  # (B, P, F)
            mask = batch['mask'].to(self.device)          # (B, P)
            
            # Normalize features
            features = self.normalizer.transform(features)
            
            # Data augmentation
            if self.augment and self.model.training:
                features = self.augment(features, mask)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast('cuda'):
                    reconstructed, encoding = self.model(features, mask)
                    
                    # Masked MSE loss
                    mask_exp = mask.unsqueeze(-1)  # (B, P, 1)
                    loss = self.criterion(reconstructed * mask_exp, features * mask_exp)
                    loss = loss.sum() / (mask.sum() * features.shape[-1])  # Normalize
                    loss = loss / self.config['accumulation_steps']
                
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    # Unscale for gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    total_grad_norm += grad_norm.item()
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # EMA update
                    if self.ema:
                        self.ema.update()
                    
                    # Scheduler step (OneCycleLR steps per batch)
                    if not (self.use_swa and epoch >= self.swa_start):
                        self.scheduler.step()
                    
                    self.global_step += 1
            else:
                # Standard forward pass (no AMP)
                reconstructed, encoding = self.model(features, mask)
                
                mask_exp = mask.unsqueeze(-1)
                loss = self.criterion(reconstructed * mask_exp, features * mask_exp)
                loss = loss.sum() / (mask.sum() * features.shape[-1])
                loss = loss / self.config['accumulation_steps']
                
                loss.backward()
                
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                    total_grad_norm += grad_norm.item()
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.ema:
                        self.ema.update()
                    
                    if not (self.use_swa and epoch >= self.swa_start):
                        self.scheduler.step()
                    
                    self.global_step += 1
            
            total_loss += loss.item() * self.config['accumulation_steps']
            num_batches += 1
        
        # SWA update
        if self.use_swa and epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        
        avg_loss = total_loss / num_batches
        num_accum_steps = num_batches // self.config['accumulation_steps']
        avg_grad_norm = total_grad_norm / max(num_accum_steps, 1)
        
        return avg_loss, avg_grad_norm
    
    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """
        Validate on validation set.
        
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        
        # Apply EMA weights if enabled
        if self.ema:
            self.ema.apply_shadow()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Normalize
            features = self.normalizer.transform(features)
            
            # Forward pass
            if self.use_amp:
                with autocast('cuda'):
                    reconstructed, _ = self.model(features, mask)
            else:
                reconstructed, _ = self.model(features, mask)
            
            # Masked loss
            mask_exp = mask.unsqueeze(-1)
            loss = self.criterion(reconstructed * mask_exp, features * mask_exp)
            loss = loss.sum() / (mask.sum() * features.shape[-1])
            
            total_loss += loss.item()
            num_batches += 1
        
        # Restore original weights
        if self.ema:
            self.ema.restore()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            history: Training history dictionary
        """
        logger.info(f"🚀 Starting training for {self.config['epochs']} epochs")
        logger.info(f"   LR: {self.config['lr']}, Batch Size: {self.config['batch_size']}")
        logger.info(f"   Gradient Accumulation: {self.config['accumulation_steps']}")
        logger.info(f"   Mixed Precision: {self.use_amp}")
        logger.info(f"   EMA: {self.config['use_ema']} (decay={self.config['ema_decay']})")
        logger.info(f"   Data Augmentation: {self.config['use_augmentation']}")
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss, grad_norm = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            # Current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            self.history['grad_norm'].append(grad_norm)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Gradient_Norm', grad_norm, epoch)
            
            # Console logging
            if (epoch + 1) % self.config['log_interval'] == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1:4d}/{self.config['epochs']}: "
                    f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                    f"LR={current_lr:.2e}, GradNorm={grad_norm:.2f}"
                )
            
            # Check for best model
            if val_loss < self.best_val_loss - self.config['min_delta']:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save best checkpoint
                self.save_checkpoint('best')
                logger.info(f"   ✅ New best model! Val Loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
            
            # Periodic checkpoint
            if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"\n⏹️  Early stopping at epoch {epoch+1}")
                logger.info(f"   Best Val Loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})")
                break
        
        # Final SWA BatchNorm update
        if self.use_swa:
            logger.info("📊 Updating SWA BatchNorm statistics...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            self.save_checkpoint('swa_final')
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.best_epoch if name == 'best' else len(self.history['train_loss']),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        if self.ema:
            checkpoint['ema_state'] = self.ema.state_dict()
        
        if self.use_swa and name == 'swa_final':
            checkpoint['swa_model_state'] = self.swa_model.state_dict()
        
        checkpoint_path = OUTPUT_DIR / f'checkpoint_{name}_run{self.run_id}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save encoder separately for downstream tasks
        if name == 'best':
            encoder_path = OUTPUT_DIR / f'phase_lstm_encoder_pretrained_run{self.run_id}.pt'
            torch.save(self.model.encoder.state_dict(), encoder_path)
            logger.info(f"   💾 Encoder saved to: {encoder_path.name}")
    
    def save_history(self):
        """Save training history to CSV and JSON."""
        # CSV format
        df = pd.DataFrame(self.history)
        df.to_csv(OUTPUT_DIR / f'training_history_run{self.run_id}.csv', index=False)
        
        # JSON format
        with open(OUTPUT_DIR / f'training_history_run{self.run_id}.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Summary
        summary = {
            'run_id': self.run_id,
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'final_train_loss': float(self.history['train_loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1]),
            'total_epochs': len(self.history['train_loss']),
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()}
        }
        
        with open(OUTPUT_DIR / f'training_summary_run{self.run_id}.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Phase LSTM Production-Grade Pre-training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, 
                        help='Maximum learning rate (for OneCycleLR)')
    parser.add_argument('--weight-decay', type=float, default=0.001, 
                        help='Weight decay for AdamW')
    parser.add_argument('--patience', type=int, default=75, 
                        help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=1e-6, 
                        help='Minimum improvement for early stopping')
    
    # Optimization techniques
    parser.add_argument('--accumulation-steps', type=int, default=1, 
                        help='Gradient accumulation steps (effective batch = batch_size * steps)')
    parser.add_argument('--grad-clip', type=float, default=1.0, 
                        help='Gradient clipping max norm')
    parser.add_argument('--use-amp', action='store_true', 
                        help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--use-ema', action='store_true', 
                        help='Use Exponential Moving Average (EMA)')
    parser.add_argument('--ema-decay', type=float, default=0.9995, 
                        help='EMA decay rate')
    parser.add_argument('--use-swa', action='store_true', 
                        help='Use Stochastic Weight Averaging (SWA)')
    
    # Data augmentation
    parser.add_argument('--use-augmentation', action='store_true', 
                        help='Enable data augmentation')
    parser.add_argument('--noise-std', type=float, default=0.01, 
                        help='Gaussian noise standard deviation')
    parser.add_argument('--feature-dropout', type=float, default=0.1, 
                        help='Feature dropout probability')
    parser.add_argument('--mixup-alpha', type=float, default=0.0, 
                        help='Mixup alpha (0 = disabled)')
    
    # Model architecture (OPTIMIZED defaults)
    parser.add_argument('--hidden-dim', type=int, default=256, 
                        help='LSTM hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3, 
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, 
                        help='Model dropout rate')
    
    # Logging and checkpoints
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='Log every N epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=50, 
                        help='Save checkpoint every N epochs')
    
    # Multi-run for robust results
    parser.add_argument('--runs', type=int, default=1, 
                        help='Number of training runs (for averaging)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (incremented for each run)')
    
    args = parser.parse_args()
    
    # Build config dictionary
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'min_delta': args.min_delta,
        'accumulation_steps': args.accumulation_steps,
        'grad_clip': args.grad_clip,
        'use_amp': args.use_amp,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'use_swa': args.use_swa,
        'use_augmentation': args.use_augmentation,
        'noise_std': args.noise_std,
        'feature_dropout': args.feature_dropout,
        'mixup_alpha': args.mixup_alpha,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'log_interval': args.log_interval,
        'checkpoint_interval': args.checkpoint_interval
    }
    
    # Print header
    print("=" * 70)
    print("🚀 FASE 1 PRODUCTION: Phase LSTM Pre-training")
    print("=" * 70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    logger.info("\n📂 Loading data...")
    duke_features, duke_clinical = load_duke_data()
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    # Combine patient IDs
    all_ids = list(duke_features.keys()) + list(ispy2_features.keys())
    logger.info(f"   Total patients: Duke={len(duke_features)}, ISPY2={len(ispy2_features)}")
    
    # Multi-run training
    all_results = []
    
    for run in range(args.runs):
        run_seed = args.seed + run
        
        if args.runs > 1:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"🔄 RUN {run + 1}/{args.runs} (seed={run_seed})")
            logger.info(f"{'=' * 70}")
        
        # Set seeds for reproducibility
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        
        # Train/val split
        train_ids, val_ids = train_test_split(
            all_ids, 
            test_size=0.2, 
            random_state=run_seed
        )
        
        # Create datasets
        train_ds = PhaseFlatDataset(
            train_ids, 
            duke_features, duke_clinical,
            ispy2_features, ispy2_clinical
        )
        val_ds = PhaseFlatDataset(
            val_ids, 
            duke_features, duke_clinical,
            ispy2_features, ispy2_clinical
        )
        
        logger.info(f"   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
        # Fit normalizer on training data
        logger.info("⚖️  Fitting normalizer on training data...")
        all_feats = []
        for i in range(len(train_ds)):
            sample = train_ds[i]
            feats = sample['features'].numpy()
            mask = sample['mask'].numpy()
            if mask.sum() > 0:
                all_feats.append(feats[mask > 0])
        
        all_feats = np.concatenate(all_feats, axis=0)
        logger.info(f"   Normalizer fitted on {all_feats.shape[0]} phase samples")
        
        normalizer = FeatureNormalizer(method='robust_v2')  # IMPROVED normalizer
        normalizer.fit(all_feats)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_phase_flat,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_phase_flat,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = PhaseLSTMAutoencoder(
            input_dim=1143,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Model parameters: {n_params:,}")
        
        # Create trainer and train
        trainer = PhasePretrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            normalizer=normalizer,
            device=device,
            config=config,
            run_id=run
        )
        
        history = trainer.train()
        
        # Store results
        all_results.append({
            'run': run + 1,
            'seed': run_seed,
            'best_val_loss': trainer.best_val_loss,
            'best_epoch': trainer.best_epoch,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        })
    
    # Multi-run summary
    if args.runs > 1:
        print("\n" + "=" * 70)
        print("📊 MULTI-RUN SUMMARY")
        print("=" * 70)
        
        best_losses = [r['best_val_loss'] for r in all_results]
        mean_loss = np.mean(best_losses)
        std_loss = np.std(best_losses)
        
        print(f"\nMean Best Val Loss: {mean_loss:.6f} ± {std_loss:.6f}")
        print("\nPer-run results:")
        for r in all_results:
            print(f"  Run {r['run']}: Best={r['best_val_loss']:.6f} (epoch {r['best_epoch']})")
        
        # Find best run
        best_run_idx = np.argmin(best_losses)
        best_run = all_results[best_run_idx]
        print(f"\n🏆 Best run: #{best_run['run']} with Val Loss = {best_run['best_val_loss']:.6f}")
        
        # Save multi-run summary
        multirun_summary = {
            'num_runs': args.runs,
            'mean_best_val_loss': float(mean_loss),
            'std_best_val_loss': float(std_loss),
            'best_run': best_run,
            'all_runs': all_results,
            'config': config
        }
        
        with open(OUTPUT_DIR / 'multirun_summary.json', 'w') as f:
            json.dump(multirun_summary, f, indent=2)
    
    # Save normalizer (from last run or best run)
    normalizer_path = OUTPUT_DIR / 'phase_lstm_normalizer.pkl'
    with open(normalizer_path, 'wb') as f:
        pickle.dump(normalizer, f)
    logger.info(f"\n💾 Normalizer saved to: {normalizer_path}")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"📁 Models saved to: {OUTPUT_DIR}")
    print(f"📈 TensorBoard logs: {TENSORBOARD_DIR}")
    print(f"\n🔍 To view training curves:")
    print(f"   tensorboard --logdir {TENSORBOARD_DIR}")


if __name__ == '__main__':
    main()
