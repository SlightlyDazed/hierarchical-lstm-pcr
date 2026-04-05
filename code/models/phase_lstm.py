"""
FASE 6: Phase LSTM Pre-training Models
=======================================
Modelos para pre-entrenar el Phase LSTM usando autoencoder.

PhaseLSTMEncoder: Encode phases → representation
PhaseLSTMAutoencoder: Encoder + Decoder for reconstruction

Uso:
    from fase6_phase_lstm_pretrain import PhaseLSTMEncoder, PhaseLSTMAutoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PhaseAttention(nn.Module):
    """Attention mechanism to aggregate sequence of hidden states."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, SeqLen, Hidden)
            mask: (B, SeqLen) - 1 for valid, 0 for padding
        
        Returns:
            context: (B, Hidden)
            attn_weights: (B, SeqLen, 1)
        """
        attn_scores = self.attn(hidden_states)  # (B, SeqLen, 1)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, SeqLen, 1)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e4)  # FP16 compatible
            
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, SeqLen, 1)
        
        # Weighted sum: (B, Hidden)
        context = torch.sum(hidden_states * attn_weights, dim=1)
        
        return context, attn_weights


class PhaseLSTMEncoder(nn.Module):
    """
    Phase LSTM Encoder that learns representations of DCE phases.
    
    Input: (B, 6 phases, feature_dim)
    Output: (B, hidden_dim * 2)  # Bidirectional
    """
    def __init__(
        self, 
        input_dim: int = 1143,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. Feature Projection (normalize scale)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Phase LSTM (bidirectional)
        self.phase_lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Attention aggregation
        self.attention = PhaseAttention(hidden_dim * 2)
        
        # Output dimension
        self.output_dim = hidden_dim * 2
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 6, feature_dim) - phases
            mask: (B, 6) - validity mask (1=valid, 0=padding)
        
        Returns:
            phase_repr: (B, hidden_dim*2) - aggregated representation
            attn_weights: (B, 6, 1) - attention weights
        """
        # Project features
        x_proj = self.projection(x)  # (B, 6, 512)
        
        # LSTM
        lstm_out, _ = self.phase_lstm(x_proj)  # (B, 6, hidden_dim*2)
        
        # Attention aggregation
        phase_repr, attn_weights = self.attention(lstm_out, mask)
        
        return phase_repr, attn_weights
    
    def encode_sequence(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns the full LSTM output sequence (for Temporal LSTM input).
        
        Args:
            x: (B, 6, feature_dim)
            mask: (B, 6)
        
        Returns:
            sequence: (B, 6, hidden_dim*2)
        """
        x_proj = self.projection(x)
        lstm_out, _ = self.phase_lstm(x_proj)
        return lstm_out


class PhaseLSTMAutoencoder(nn.Module):
    """
    Autoencoder for pre-training Phase LSTM (FASE 1).
    
    Encoder: (B, 6, 1143) → (B, hidden_dim*2)
    Decoder: (B, hidden_dim*2) → (B, 6, 1143)
    """
    def __init__(
        self, 
        input_dim: int = 1143,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_phases = 6
        
        # Encoder
        self.encoder = PhaseLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder: reconstruct phases from encoding
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.num_phases * input_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 6, 1143)
            mask: (B, 6)
        
        Returns:
            reconstructed: (B, 6, 1143)
            encoding: (B, hidden_dim*2)
        """
        B = x.shape[0]
        
        # Encode
        encoding, _ = self.encoder(x, mask)  # (B, hidden_dim*2)
        
        # Decode
        recon_flat = self.decoder(encoding)  # (B, 6*1143)
        reconstructed = recon_flat.view(B, self.num_phases, self.input_dim)
        
        return reconstructed, encoding
    
    def get_encoder(self) -> PhaseLSTMEncoder:
        """Get the encoder module for FASE 2."""
        return self.encoder


class ContrastivePhaseLSTM(nn.Module):
    """
    Alternative: Contrastive learning for Phase LSTM pre-training.
    Uses triplet loss with positive pairs from same patient.
    """
    def __init__(
        self, 
        input_dim: int = 1143,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        projection_dim: int = 64
    ):
        super().__init__()
        
        self.encoder = PhaseLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 6, 1143)
            mask: (B, 6)
        
        Returns:
            projection: (B, projection_dim) - for contrastive loss
            encoding: (B, hidden_dim*2) - representation
        """
        encoding, _ = self.encoder(x, mask)
        projection = self.projection_head(encoding)
        projection = F.normalize(projection, dim=-1)  # L2 normalize
        
        return projection, encoding


# =============================================================================
# Test
# =============================================================================
def test_models():
    print("Testing Phase LSTM Pre-training Models...")
    
    B, P, feat_dim = 4, 6, 1143
    x = torch.randn(B, P, feat_dim)
    mask = torch.ones(B, P)
    mask[0, 4:] = 0  # Simulate some padding
    
    # Test Encoder
    print("\n1. PhaseLSTMEncoder")
    encoder = PhaseLSTMEncoder(input_dim=feat_dim)
    repr_out, attn = encoder(x, mask)
    print(f"   Input: {x.shape}")
    print(f"   Output: {repr_out.shape}")
    print(f"   Attention: {attn.shape}")
    
    # Test Autoencoder
    print("\n2. PhaseLSTMAutoencoder")
    autoencoder = PhaseLSTMAutoencoder(input_dim=feat_dim)
    recon, enc = autoencoder(x, mask)
    print(f"   Input: {x.shape}")
    print(f"   Encoding: {enc.shape}")
    print(f"   Reconstructed: {recon.shape}")
    
    # Reconstruction loss
    loss = F.mse_loss(recon * mask.unsqueeze(-1), x * mask.unsqueeze(-1))
    print(f"   Recon Loss: {loss.item():.4f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Test Contrastive
    print("\n3. ContrastivePhaseLSTM")
    contrastive = ContrastivePhaseLSTM(input_dim=feat_dim)
    proj, enc = contrastive(x, mask)
    print(f"   Input: {x.shape}")
    print(f"   Projection: {proj.shape}")
    print(f"   Encoding: {enc.shape}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_models()
