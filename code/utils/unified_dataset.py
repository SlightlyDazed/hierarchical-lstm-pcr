"""
FASE 6: Unified Dataset - Duke + ISPY2 Fusion
==============================================
Combina features y labels de Duke e ISPY2 en un dataset unificado
para entrenar el Phase LSTM Multi-Task.

Estrategia de fusión:
- Duke: Single timepoint (diagnóstico), 5 fases DCE
- ISPY2: Multi-timepoint (T0-T3), 6 fases DCE por timepoint

El modelo procesará:
- Duke: Solo Phase LSTM (secuencia de 5 fases)
- ISPY2: Phase LSTM + Temporal LSTM (completo)

Output Labels:
- pCR: Binary (both datasets)
- Molecular Subtype: 4-class (both datasets, better in Duke)
- Kinetic Pattern: 3-class (extracted from DCE curve)

Uso:
    python fase6_unified_dataset.py
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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

# Output
UNIFIED_DATASET_PKL = DATA_DIR / "unified_dataset.pkl"


# =============================================================================
# Constants
# =============================================================================
# Feature dimensions (deben coincidir entre datasets)
DUKE_FEATURE_DIM = 1076   # DenseNet(1024) + Kinetic(8) + FirstOrder(12) + Histogram(32)
ISPY2_FEATURE_DIM = 1143  # Formato existente de ISPY2

# Número de fases por dataset
DUKE_NUM_PHASES = 5   # phase_0 to phase_4
ISPY2_NUM_PHASES = 6  # Ph0 to Ph5

# Número de timepoints
ISPY2_NUM_TIMEPOINTS = 4  # T0, T1, T2, T3


# =============================================================================
# Load and Process Duke Data
# =============================================================================
def load_duke_data() -> Tuple[Dict, pd.DataFrame]:
    """
    Carga datos de Duke (features + clinical).
    
    Returns:
        features_dict, clinical_df
    """
    logger.info("Cargando datos Duke...")
    
    # Clinical
    if not DUKE_CLINICAL_CSV.exists():
        logger.error(f"No se encontró: {DUKE_CLINICAL_CSV}")
        logger.info("Ejecutar primero: python fase6_duke_clinical.py")
        return {}, pd.DataFrame()
    
    clinical_df = pd.read_csv(DUKE_CLINICAL_CSV)
    logger.info(f"  Clinical: {len(clinical_df)} pacientes")
    
    # Features
    if not DUKE_FEATURES_PKL.exists():
        logger.warning(f"No se encontró: {DUKE_FEATURES_PKL}")
        logger.info("Ejecutar primero: python fase6_duke_feature_extraction.py")
        return {}, clinical_df
    
    with open(DUKE_FEATURES_PKL, 'rb') as f:
        features_dict = pickle.load(f)
    
    logger.info(f"  Features: {len(features_dict)} pacientes")
    
    return features_dict, clinical_df


def load_ispy2_data() -> Tuple[Dict, pd.DataFrame]:
    """
    Carga datos de ISPY2 (features + clinical).
    
    Returns:
        features_dict, clinical_df
    """
    logger.info("Cargando datos ISPY2...")
    
    # Features
    if not ISPY2_FEATURES_PKL.exists():
        logger.error(f"No se encontró: {ISPY2_FEATURES_PKL}")
        return {}, pd.DataFrame()
    
    with open(ISPY2_FEATURES_PKL, 'rb') as f:
        features_dict = pickle.load(f)
    
    logger.info(f"  Features: {len(features_dict)} pacientes")
    
    # Clinical
    if not ISPY2_AUDIT_CSV.exists():
        logger.warning(f"No se encontró: {ISPY2_AUDIT_CSV}")
        return features_dict, pd.DataFrame()
    
    clinical_df = pd.read_csv(ISPY2_AUDIT_CSV)
    
    # FIX: Calculate HR correctly (ER+ or PR+)
    # Check if necessary columns exist
    if 'ER' in clinical_df.columns and 'PR' in clinical_df.columns:
        # Ensure numeric 0/1 (handles NaNs as False temporarily for bitwise, or keep NaN)
        # Using fillna(-1) to preserve missing info
        er_vec = pd.to_numeric(clinical_df['ER'], errors='coerce').fillna(-1).astype(int)
        pr_vec = pd.to_numeric(clinical_df['PR'], errors='coerce').fillna(-1).astype(int)
        
        # Logic: If both -1, HR is -1. If one is 1, HR is 1. If both 0, HR 0.
        # This is complex in vector. Let's iterate or use apply.
        def calc_hr(row):
            er = row.get('ER', float('nan'))
            pr = row.get('PR', float('nan'))
            
            # If explicit HR exists and is valid, trust it? Or overwrite? 
            # User request implies overwrite/fixing.
            
            # Treat numeric 1 as Positive
            er_pos = (er == 1) or (er == 'Positive')
            pr_pos = (pr == 1) or (pr == 'Positive')
            
            if er_pos or pr_pos:
                return 1
            
            # If both are definitively negative (0)
            er_neg = (er == 0) or (er == 'Negative')
            pr_neg = (pr == 0) or (pr == 'Negative')
            
            if er_neg and pr_neg:
                return 0
                
            # Else uncertain/missing
            return -1

        clinical_df['HR_calc'] = clinical_df.apply(calc_hr, axis=1)
        
        # Override HR column or fill it
        if 'HR' not in clinical_df.columns:
             clinical_df['HR'] = clinical_df['HR_calc']
        else:
             # Fill NaNs in HR with calculated
             clinical_df['HR'] = clinical_df['HR'].fillna(clinical_df['HR_calc'])
    
    logger.info(f"  Clinical: {len(clinical_df)} pacientes")
    
    return features_dict, clinical_df


# =============================================================================
# Unified Dataset Class
# =============================================================================
class UnifiedBreastMRIDataset(Dataset):
    """
    Dataset unificado para Duke + ISPY2.
    
    Maneja diferencias entre datasets:
    - Duke: single timepoint, 5 phases
    - ISPY2: 4 timepoints, 6 phases each
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        duke_features: Dict,
        duke_clinical: pd.DataFrame,
        ispy2_features: Dict,
        ispy2_clinical: pd.DataFrame,
        feature_dim: int = 1143,  # Usar dimensión de ISPY2 (mayor)
        num_phases: int = 6,
        num_timepoints: int = 4
    ):
        self.patient_ids = patient_ids
        self.duke_features = duke_features
        self.duke_clinical = duke_clinical
        self.ispy2_features = ispy2_features
        self.ispy2_clinical = ispy2_clinical
        
        self.feature_dim = feature_dim
        self.num_phases = num_phases
        self.num_timepoints = num_timepoints
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        patient_id = self.patient_ids[idx]
        
        # Determinar dataset
        if patient_id.startswith('Breast_MRI'):
            return self._get_duke_sample(patient_id)
        else:
            return self._get_ispy2_sample(patient_id)
    
    def _get_duke_sample(self, patient_id: str) -> Dict:
        """Obtiene sample de Duke (single timepoint)."""
        
        # Features: shape (1, num_phases, feature_dim)
        # Duke solo tiene 1 timepoint, padding para 4 si es necesario
        features = np.zeros((1, self.num_phases, self.feature_dim), dtype=np.float32)
        mask = np.zeros((1, self.num_phases), dtype=np.float32)
        
        if patient_id in self.duke_features:
            patient_data = self.duke_features[patient_id]
            
            for phase_name, phase_data in patient_data['phases'].items():
                phase_idx = int(phase_name.split('_')[1])  # 'phase_0' -> 0
                if phase_idx < self.num_phases:
                    # Pad features to match ISPY2 dimension
                    duke_feats = phase_data['combined']
                    if len(duke_feats) < self.feature_dim:
                        padded = np.zeros(self.feature_dim, dtype=np.float32)
                        padded[:len(duke_feats)] = duke_feats
                        features[0, phase_idx] = padded
                    else:
                        features[0, phase_idx] = duke_feats[:self.feature_dim]
                    mask[0, phase_idx] = 1.0
        
        # Labels
        labels = self._get_duke_labels(patient_id)
        
        return {
            'patient_id': patient_id,
            'dataset': 'duke',
            'features': torch.tensor(features),
            'mask': torch.tensor(mask),
            'is_single_timepoint': True,
            **labels
        }
    
    def _get_ispy2_sample(self, patient_id: str) -> Dict:
        """Obtiene sample de ISPY2 (multi-timepoint)."""
        
        # Features: shape (4 timepoints, 6 phases, feature_dim)
        features = np.zeros((self.num_timepoints, self.num_phases, self.feature_dim), dtype=np.float32)
        mask = np.zeros((self.num_timepoints, self.num_phases), dtype=np.float32)
        
        if patient_id in self.ispy2_features:
            patient_data = self.ispy2_features[patient_id]
            
            timepoints = ['T0', 'T1', 'T2', 'T3']
            phases = ['Ph0', 'Ph1', 'Ph2', 'Ph3', 'Ph4', 'Ph5']  # ISPY2 naming (Ph0-Ph5)
            
            for t_idx, tp in enumerate(timepoints):
                if tp in patient_data:
                    for p_idx, ph in enumerate(phases[:self.num_phases]):
                        if ph in patient_data[tp]:
                            phase_data = patient_data[tp][ph]
                            
                            # Combine features from separate arrays
                            if 'combined' in phase_data:
                                feats = np.array(phase_data['combined'], dtype=np.float32)
                            else:
                                # Combine densenet + radiomics + spatial
                                densenet = np.array(phase_data.get('densenet_features', []), dtype=np.float32)
                                radiomics = np.array(phase_data.get('radiomics_features', []), dtype=np.float32)
                                spatial = np.array(phase_data.get('spatial_features', []), dtype=np.float32)
                                feats = np.concatenate([densenet, radiomics, spatial])
                            
                            # Pad or truncate to match feature_dim
                            if len(feats) < self.feature_dim:
                                padded = np.zeros(self.feature_dim, dtype=np.float32)
                                padded[:len(feats)] = feats
                                features[t_idx, p_idx] = padded
                            else:
                                features[t_idx, p_idx] = feats[:self.feature_dim]
                            mask[t_idx, p_idx] = 1.0
        
        # Labels
        labels = self._get_ispy2_labels(patient_id)
        
        return {
            'patient_id': patient_id,
            'dataset': 'ispy2',
            'features': torch.tensor(features),
            'mask': torch.tensor(mask),
            'is_single_timepoint': False,
            **labels
        }
    
    def _get_duke_labels(self, patient_id: str) -> Dict:
        """Obtiene labels de Duke."""
        labels = {
            'pCR': -1,
            'mol_subtype': -1,
            'ER': -1,
            'PR': -1,
            'HER2': -1,
            'kinetic_pattern': -1  # Calculado de features
        }
        
        row = self.duke_clinical[self.duke_clinical['patient_id'] == patient_id]
        if len(row) > 0:
            row = row.iloc[0]
            labels['pCR'] = int(row['pCR']) if row['pCR'] >= 0 else -1
            labels['mol_subtype'] = int(row['mol_subtype_computed']) if row['mol_subtype_computed'] >= 0 else -1
            labels['ER'] = int(row['ER']) if row['ER'] >= 0 else -1
            labels['PR'] = int(row['PR']) if row['PR'] >= 0 else -1
            labels['HER2'] = int(row['HER2']) if row['HER2'] >= 0 else -1
        
        return labels
    
    def _get_ispy2_labels(self, patient_id: str) -> Dict:
        """Obtiene labels de ISPY2."""
        labels = {
            'pCR': -1,
            'mol_subtype': -1,
            'ER': -1,
            'PR': -1,
            'HER2': -1,
            'kinetic_pattern': -1
        }
        
        row = self.ispy2_clinical[self.ispy2_clinical['PatientID'] == patient_id]
        if len(row) > 0:
            row = row.iloc[0]
            
            # pCR
            if 'pCR' in row.index and pd.notna(row['pCR']):
                labels['pCR'] = int(row['pCR'])
            
            # HR (from audit_report.csv)
            if 'HR' in row.index and pd.notna(row['HR']):
                labels['ER'] = int(row['HR'])  # HR = ER for ISPY2
            
            # HER2
            if 'HER2' in row.index and pd.notna(row['HER2']):
                labels['HER2'] = int(row['HER2'])
            
            # Molecular Subtype mapping
            # Subtype in audit: 'HR+/HER2-', 'TNBC', 'HR+/HER2+', 'HR-/HER2+'
            if 'Subtype' in row.index and pd.notna(row['Subtype']):
                subtype_map = {
                    'HR+/HER2-': 0,   # Luminal
                    'Luminal': 0,
                    'HR+/HER2+': 1,   # Luminal HER2+
                    'HR-/HER2+': 2,   # HER2-enriched
                    'HER2-enriched': 2,
                    'TNBC': 3,        # Triple Negative
                    'Triple Negative': 3
                }
                subtype_str = str(row['Subtype'])
                labels['mol_subtype'] = subtype_map.get(subtype_str, -1)
        
        return labels


# =============================================================================
# Collate Function
# =============================================================================
def collate_unified(batch: List[Dict]) -> Dict:
    """
    Collate function que maneja batches mixtos Duke+ISPY2.
    """
    # Separar por tipo
    single_tp = [b for b in batch if b['is_single_timepoint']]
    multi_tp = [b for b in batch if not b['is_single_timepoint']]
    
    result = {
        'patient_ids': [b['patient_id'] for b in batch],
        'datasets': [b['dataset'] for b in batch],
        'is_single_timepoint': torch.tensor([b['is_single_timepoint'] for b in batch]),
        'pCR': torch.tensor([b['pCR'] for b in batch]),
        'mol_subtype': torch.tensor([b['mol_subtype'] for b in batch]),
        'ER': torch.tensor([b['ER'] for b in batch]),
        'PR': torch.tensor([b['PR'] for b in batch]),
        'HER2': torch.tensor([b['HER2'] for b in batch]),
    }
    
    # Stack features y masks (ya tienen mismo shape tras padding)
    result['features'] = torch.stack([b['features'] for b in batch])
    result['masks'] = torch.stack([b['mask'] for b in batch])
    
    return result


# =============================================================================
# Create Dataset Splits
# =============================================================================
def create_unified_splits(
    duke_features: Dict,
    duke_clinical: pd.DataFrame,
    ispy2_features: Dict,
    ispy2_clinical: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[UnifiedBreastMRIDataset, UnifiedBreastMRIDataset, UnifiedBreastMRIDataset]:
    """
    Crea splits train/val/test estratificados por pCR.
    """
    logger.info("Creando splits train/val/test...")
    
    # Recolectar todos los pacientes con pCR válido
    all_patients = []
    all_labels = []
    
    # Duke
    for patient_id in duke_features.keys():
        row = duke_clinical[duke_clinical['patient_id'] == patient_id]
        if len(row) > 0:
            pcr = row.iloc[0]['pCR']
            if pcr >= 0:
                all_patients.append(patient_id)
                all_labels.append(int(pcr))
    
    duke_count = len(all_patients)
    logger.info(f"  Duke con pCR válido: {duke_count}")
    
    # ISPY2
    for patient_id in ispy2_features.keys():
        row = ispy2_clinical[ispy2_clinical['PatientID'] == patient_id]
        if len(row) > 0 and 'pCR' in row.columns:
            pcr = row.iloc[0]['pCR']
            if pd.notna(pcr) and int(pcr) >= 0:
                all_patients.append(patient_id)
                all_labels.append(int(pcr))
    
    ispy2_count = len(all_patients) - duke_count
    logger.info(f"  ISPY2 con pCR válido: {ispy2_count}")
    logger.info(f"  Total: {len(all_patients)}")
    
    # Distribución de pCR
    all_labels = np.array(all_labels)
    logger.info(f"  pCR=0: {np.sum(all_labels == 0)}, pCR=1: {np.sum(all_labels == 1)}")
    
    # Split estratificado
    from sklearn.model_selection import train_test_split
    
    # Split inicial: train+val vs test
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        all_patients, all_labels,
        test_size=test_ratio,
        stratify=all_labels,
        random_state=random_state
    )
    
    # Split: train vs val
    val_size = val_ratio / (1 - test_ratio)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids, train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    logger.info(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Crear datasets
    train_ds = UnifiedBreastMRIDataset(
        train_ids, duke_features, duke_clinical, ispy2_features, ispy2_clinical
    )
    val_ds = UnifiedBreastMRIDataset(
        val_ids, duke_features, duke_clinical, ispy2_features, ispy2_clinical
    )
    test_ds = UnifiedBreastMRIDataset(
        test_ids, duke_features, duke_clinical, ispy2_features, ispy2_clinical
    )
    
    return train_ds, val_ds, test_ds


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("FASE 6: Unified Dataset - Duke + ISPY2 Fusion")
    print("="*60)
    
    # Cargar datos
    duke_features, duke_clinical = load_duke_data()
    ispy2_features, ispy2_clinical = load_ispy2_data()
    
    if not duke_features and not ispy2_features:
        logger.error("No hay features disponibles. Ejecutar extraction primero.")
        return
    
    # Crear splits
    if duke_features and ispy2_features:
        train_ds, val_ds, test_ds = create_unified_splits(
            duke_features, duke_clinical,
            ispy2_features, ispy2_clinical
        )
    else:
        logger.warning("Solo un dataset disponible. Crear splits con datos parciales.")
        return
    
    # Test del DataLoader
    print("\n" + "="*60)
    print("TEST: DataLoader")
    print("="*60)
    
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_unified)
    
    for batch in loader:
        print(f"  Batch size: {len(batch['patient_ids'])}")
        print(f"  Datasets: {batch['datasets']}")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Masks shape: {batch['masks'].shape}")
        print(f"  pCR: {batch['pCR'].tolist()}")
        print(f"  Single timepoint: {batch['is_single_timepoint'].tolist()}")
        break
    
    # Guardar splits
    splits_data = {
        'train_ids': train_ds.patient_ids,
        'val_ids': val_ds.patient_ids,
        'test_ids': test_ds.patient_ids,
        'duke_clinical': duke_clinical,
        'ispy2_clinical': ispy2_clinical
    }
    
    with open(UNIFIED_DATASET_PKL, 'wb') as f:
        pickle.dump(splits_data, f)
    
    print(f"\n✅ Guardado: {UNIFIED_DATASET_PKL}")


if __name__ == "__main__":
    main()
