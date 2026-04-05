"""
FASE 6: Phase Dataset Flat - Flatten Temporal for Pre-training
================================================================
Dataset que DESCOMPONE timepoints en samples independientes.

Duke paciente → 1 sample (T0)
ISPY2 paciente → 4 samples (T0, T1, T2, T3)

Cada sample: (6 phases, MAX_FEATURE_DIM)

Uso:
    python fase6_phase_dataset_flat.py
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2")
DATA_DIR = PROJECT_DIR / "data/processed"

# Duke paths
DUKE_CLINICAL_CSV = DATA_DIR / "duke_clinical_unified.csv"
DUKE_FEATURES_PKL = DATA_DIR / "features_cache/duke_features.pkl"

# ISPY2 paths  
ISPY2_FEATURES_PKL = DATA_DIR / "features_cache/patient_features_fixed.pkl"
ISPY2_AUDIT_CSV = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/ISPY2_ALL/audit_results/audit_report.csv")

# Constants
MAX_FEATURE_DIM = 1143  # Max dimension (ISPY2 has 1135, Duke has 1076)
NUM_PHASES = 6  # Max phases (ISPY2 has 6, Duke has 5)


# =============================================================================
# Load Functions
# =============================================================================
def load_duke_data() -> Tuple[Dict, pd.DataFrame]:
    """Load Duke data (features + clinical)."""
    logger.info("Loading Duke data...")
    
    # Clinical
    if not DUKE_CLINICAL_CSV.exists():
        logger.error(f"Not found: {DUKE_CLINICAL_CSV}")
        return {}, pd.DataFrame()
    
    clinical_df = pd.read_csv(DUKE_CLINICAL_CSV)
    logger.info(f"  Clinical: {len(clinical_df)} patients")
    
    # Features
    if not DUKE_FEATURES_PKL.exists():
        logger.warning(f"Not found: {DUKE_FEATURES_PKL}")
        return {}, clinical_df
    
    with open(DUKE_FEATURES_PKL, 'rb') as f:
        features_dict = pickle.load(f)
    
    logger.info(f"  Features: {len(features_dict)} patients")
    
    return features_dict, clinical_df


def load_ispy2_data() -> Tuple[Dict, pd.DataFrame]:
    """Load ISPY2 data (features + clinical)."""
    logger.info("Loading ISPY2 data...")
    
    # Features
    if not ISPY2_FEATURES_PKL.exists():
        logger.error(f"Not found: {ISPY2_FEATURES_PKL}")
        return {}, pd.DataFrame()
    
    with open(ISPY2_FEATURES_PKL, 'rb') as f:
        features_dict = pickle.load(f)
    
    logger.info(f"  Features: {len(features_dict)} patients")
    
    # Clinical
    if not ISPY2_AUDIT_CSV.exists():
        logger.warning(f"Not found: {ISPY2_AUDIT_CSV}")
        return features_dict, pd.DataFrame()
    
    clinical_df = pd.read_csv(ISPY2_AUDIT_CSV)
    logger.info(f"  Clinical: {len(clinical_df)} patients")
    
    return features_dict, clinical_df


# =============================================================================
# PhaseFlatDataset
# =============================================================================
class PhaseFlatDataset(Dataset):
    """
    Dataset que DESCOMPONE timepoints en samples independientes.
    
    Duke paciente → 1 sample (T0)
    ISPY2 paciente → 4 samples (T0, T1, T2, T3)
    
    Cada sample: (6 phases, MAX_FEATURE_DIM)
    """
    def __init__(
        self,
        patient_ids: Optional[List[str]] = None,
        duke_features: Optional[Dict] = None,
        duke_clinical: Optional[pd.DataFrame] = None,
        ispy2_features: Optional[Dict] = None,
        ispy2_clinical: Optional[pd.DataFrame] = None,
        feature_dim: int = MAX_FEATURE_DIM,
        num_phases: int = NUM_PHASES,
        include_duke: bool = True,
        include_ispy2: bool = True
    ):
        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.samples = []
        
        # If no data provided, load it
        if duke_features is None and include_duke:
            duke_features, duke_clinical = load_duke_data()
        if ispy2_features is None and include_ispy2:
            ispy2_features, ispy2_clinical = load_ispy2_data()
        
        # ═══════════════════════════════════════════════════════════
        # DUKE: 1 timepoint → 1 sample por paciente
        # ═══════════════════════════════════════════════════════════
        if include_duke and duke_features:
            duke_ids = patient_ids if patient_ids else list(duke_features.keys())
            duke_ids = [pid for pid in duke_ids if pid.startswith('Breast_MRI')]
            
            for pid in duke_ids:
                if pid not in duke_features:
                    continue
                
                # Get clinical info for pCR (if available)
                pcr = -1
                if duke_clinical is not None and len(duke_clinical) > 0:
                    row = duke_clinical[duke_clinical['patient_id'] == pid]
                    if len(row) > 0:
                        pcr_val = row.iloc[0].get('pCR', -1)
                        pcr = int(pcr_val) if pcr_val >= 0 else -1
                
                # Extract features: Duke has 'phases' dict with 'phase_0', 'phase_1', etc.
                patient_data = duke_features[pid]
                phases_dict = patient_data.get('phases', {})
                
                features = np.zeros((self.num_phases, self.feature_dim), dtype=np.float32)
                mask = np.zeros(self.num_phases, dtype=np.float32)
                
                for phase_name, phase_data in phases_dict.items():
                    # 'phase_0' -> 0
                    phase_idx = int(phase_name.split('_')[1])
                    if phase_idx < self.num_phases:
                        # Duke has 'combined' key with 1076 features
                        duke_feats = np.array(phase_data.get('combined', []), dtype=np.float32)
                        if len(duke_feats) > 0:
                            # Pad to max dimension
                            if len(duke_feats) < self.feature_dim:
                                padded = np.zeros(self.feature_dim, dtype=np.float32)
                                padded[:len(duke_feats)] = duke_feats
                                features[phase_idx] = padded
                            else:
                                features[phase_idx] = duke_feats[:self.feature_dim]
                            mask[phase_idx] = 1.0
                
                # Only add if at least some phases are valid
                if mask.sum() > 0:
                    self.samples.append({
                        'patient_id': pid,
                        'timepoint': 0,
                        'features': torch.FloatTensor(features),
                        'mask': torch.FloatTensor(mask),
                        'source': 'duke',
                        'pCR': pcr
                    })
        
        # ═══════════════════════════════════════════════════════════
        # ISPY2: 4 timepoints → 4 samples por paciente
        # ═══════════════════════════════════════════════════════════
        if include_ispy2 and ispy2_features:
            ispy2_ids = patient_ids if patient_ids else list(ispy2_features.keys())
            ispy2_ids = [pid for pid in ispy2_ids if pid.startswith('ISPY2')]
            
            for pid in ispy2_ids:
                if pid not in ispy2_features:
                    continue
                
                # Get clinical info for pCR
                pcr = -1
                if ispy2_clinical is not None and len(ispy2_clinical) > 0:
                    row = ispy2_clinical[ispy2_clinical['PatientID'] == pid]
                    if len(row) > 0 and 'pCR' in row.columns:
                        pcr_val = row.iloc[0].get('pCR', -1)
                        if pd.notna(pcr_val):
                            pcr = int(pcr_val)
                
                patient_data = ispy2_features[pid]
                timepoints = ['T0', 'T1', 'T2', 'T3']
                phases = ['Ph0', 'Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5']
                
                for t_idx, tp in enumerate(timepoints):
                    if tp not in patient_data:
                        continue
                    
                    tp_data = patient_data[tp]
                    features = np.zeros((self.num_phases, self.feature_dim), dtype=np.float32)
                    mask = np.zeros(self.num_phases, dtype=np.float32)
                    
                    for p_idx, ph in enumerate(phases[:self.num_phases]):
                        if ph not in tp_data:
                            continue
                        
                        phase_data = tp_data[ph]
                        
                        # ISPY2 features: densenet + radiomics + spatial
                        densenet = np.array(phase_data.get('densenet_features', []), dtype=np.float32)
                        radiomics = np.array(phase_data.get('radiomics_features', []), dtype=np.float32)
                        spatial = np.array(phase_data.get('spatial_features', []), dtype=np.float32)
                        
                        # Concatenate
                        combined = np.concatenate([densenet, radiomics, spatial])
                        
                        if len(combined) > 0:
                            # Pad to max dimension
                            if len(combined) < self.feature_dim:
                                padded = np.zeros(self.feature_dim, dtype=np.float32)
                                padded[:len(combined)] = combined
                                features[p_idx] = padded
                            else:
                                features[p_idx] = combined[:self.feature_dim]
                            mask[p_idx] = 1.0
                    
                    # Only add if at least some phases are valid
                    if mask.sum() > 0:
                        self.samples.append({
                            'patient_id': pid,
                            'timepoint': t_idx,
                            'features': torch.FloatTensor(features),
                            'mask': torch.FloatTensor(mask),
                            'source': 'ispy2',
                            'pCR': pcr
                        })
        
        # Summary
        duke_count = sum(1 for s in self.samples if s['source'] == 'duke')
        ispy2_count = sum(1 for s in self.samples if s['source'] == 'ispy2')
        logger.info(f"Total samples: {len(self.samples)}")
        logger.info(f"  Duke samples: {duke_count}")
        logger.info(f"  ISPY2 samples: {ispy2_count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_phase_flat(batch: List[Dict]) -> Dict:
    """
    Collate function for flattened samples.
    All samples have shape (6, 1143) → direct stack.
    """
    return {
        'features': torch.stack([b['features'] for b in batch]),  # (B, 6, 1143)
        'mask': torch.stack([b['mask'] for b in batch]),          # (B, 6)
        'patient_id': [b['patient_id'] for b in batch],
        'timepoint': torch.tensor([b['timepoint'] for b in batch]),
        'source': [b['source'] for b in batch],
        'pCR': torch.tensor([b['pCR'] for b in batch]),
    }


# =============================================================================
# Test
# =============================================================================
def main():
    print("="*60)
    print("FASE 6: Phase Dataset Flat - Test")
    print("="*60)
    
    # Create dataset
    dataset = PhaseFlatDataset()
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test a few samples
    print("\n--- Sample inspection ---")
    for i in [0, len(dataset)//2, len(dataset)-1]:
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Patient: {sample['patient_id']}")
        print(f"  Source: {sample['source']}")
        print(f"  Timepoint: {sample['timepoint']}")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Mask: {sample['mask'].tolist()}")
        print(f"  pCR: {sample['pCR']}")
    
    # Test DataLoader
    print("\n--- DataLoader test ---")
    loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collate_phase_flat
    )
    
    batch = next(iter(loader))
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")
    print(f"Sources: {batch['source']}")
    print(f"pCR: {batch['pCR'].tolist()}")
    
    print("\n✅ Test passed!")


if __name__ == "__main__":
    main()
