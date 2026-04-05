"""
Feature Normalization Module
==============================
Proporciona clase para normalizar features de MRI manejando outliers.
Usa RobustScaler seguido de Log1p para valores muy grandes (FirstOrder).
"""

import torch
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

class FeatureNormalizer:
    def __init__(self, method='robust'):
        self.method = method
        self.scaler = None
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            
    def fit(self, features: np.ndarray):
        """
        Fit scaler on training data.
        
        Args:
            features: (N_samples, Feature_dim) array
            Flattened phases/timepoints into samples for fitting statistics.
        """
        # Handle high magnitude features (FirstOrder > 10^6)
        # We apply log1p to compress range before scaling if variance is extreme
        self.high_var_mask = np.var(features, axis=0) > 1e6
        
        # Fit scaler
        X = features.copy()
        if self.high_var_mask.any():
            X[:, self.high_var_mask] = np.log1p(np.abs(X[:, self.high_var_mask]))
            
        self.scaler.fit(X)
        return self
        
    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to Tensor.
        Input shape: (Batch, ..., Feature_dim)
        """
        device = features.device
        shape = features.shape
        X = features.cpu().numpy().reshape(-1, shape[-1])
        
        # Log high variance features
        if hasattr(self, 'high_var_mask') and self.high_var_mask.any():
            X_log = X.copy()
            X_log[:, self.high_var_mask] = np.log1p(np.abs(X_log[:, self.high_var_mask]))
            X = X_log
            
        # Apply scaler
        X_scaled = self.scaler.transform(X)
        
        # Clip extreme outliers (e.g. > 10 sigma) to prevent instability
        X_scaled = np.clip(X_scaled, -10, 10)
        
        return torch.tensor(X_scaled, dtype=torch.float32).reshape(shape).to(device)

    def fit_transform(self, features: np.ndarray) -> torch.Tensor:
        self.fit(features)
        return self.transform(torch.tensor(features))
