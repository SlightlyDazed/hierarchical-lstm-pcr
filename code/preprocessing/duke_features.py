"""
FASE 6: Duke Feature Extraction
================================
Extrae features de las imágenes PNG 16-bit de Duke para el Phase LSTM Multi-Task.

Features extraídas por fase DCE:
1. DenseNet-121 features (1024 dims) - Visual features
2. Kinetic features (8 dims) - wash-in, TTP, washout rates
3. First-order statistics (12 dims) - mean, std, skewness, etc.
4. Histogram features (32 dims) - binned intensity distribution

Total: ~1076 features por fase

Input: Duke_PNG/Breast_MRI_XXX/phase_X/
Output: duke_features.pkl

Uso:
    python fase6_duke_feature_extraction.py [--test] [--start 0] [--end 100]
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# Importar dependencias
try:
    import cv2
except ImportError:
    print("Error: pip install opencv-python")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ImportError:
    print("Error: pip install torch torchvision")
    sys.exit(1)

from scipy import stats, ndimage

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DUKE_PNG_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2/data/processed/Duke_PNG")
OUTPUT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2/data/processed/features_cache")
OUTPUT_FILE = OUTPUT_DIR / "duke_features.pkl"

# Feature dimensions
DENSENET_DIM = 1024
KINETIC_DIM = 8
FIRST_ORDER_DIM = 12
HISTOGRAM_DIM = 32
TOTAL_DIM = DENSENET_DIM + KINETIC_DIM + FIRST_ORDER_DIM + HISTOGRAM_DIM  # 1076


# =============================================================================
# DenseNet Feature Extractor
# =============================================================================
class DenseNetExtractor:
    """Extrae features visuales usando DenseNet-121 pretrained."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Cargar DenseNet-121 pretrained
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Remover classifier, quedarse con features
        self.model = nn.Sequential(*list(densenet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transformación para imágenes 16-bit
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae features de una imagen DCE.
        
        Args:
            image: Imagen 16-bit (H, W) o 8-bit
        
        Returns:
            Feature vector (1024,)
        """
        # Convertir 16-bit a 8-bit para procesamiento
        if image.dtype == np.uint16:
            image_8bit = (image / 256).astype(np.uint8)
        else:
            image_8bit = image
        
        # Convertir a RGB (replicar canal)
        if len(image_8bit.shape) == 2:
            image_rgb = np.stack([image_8bit] * 3, axis=-1)
        else:
            image_rgb = image_8bit
        
        # Transformar
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Extraer features
        with torch.no_grad():
            features = self.model(tensor)
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.squeeze().cpu().numpy()
        
        return features


# =============================================================================
# First-Order Statistics
# =============================================================================
def extract_first_order_features(image: np.ndarray) -> np.ndarray:
    """
    Extrae estadísticas de primer orden de la imagen.
    
    Returns:
        Feature vector (12,)
    """
    # Flatten para estadísticas
    flat = image.flatten().astype(np.float64)
    
    # Estadísticas básicas
    features = [
        np.mean(flat),               # 0: Mean
        np.std(flat),                # 1: Standard deviation
        np.min(flat),                # 2: Min
        np.max(flat),                # 3: Max
        np.median(flat),             # 4: Median
        np.percentile(flat, 25),     # 5: Q1
        np.percentile(flat, 75),     # 6: Q3
        stats.skew(flat),            # 7: Skewness
        stats.kurtosis(flat),        # 8: Kurtosis
        np.sum(flat > np.mean(flat) + np.std(flat)),  # 9: Pixels above mean+std
        np.var(flat),                # 10: Variance
        np.ptp(flat),                # 11: Range (peak-to-peak)
    ]
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Histogram Features
# =============================================================================
def extract_histogram_features(image: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """
    Extrae histograma normalizado de intensidades.
    
    Returns:
        Feature vector (n_bins,)
    """
    # Normalizar imagen a [0, 1]
    img_norm = image.astype(np.float64)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    
    # Histograma
    hist, _ = np.histogram(img_norm.flatten(), bins=n_bins, range=(0, 1), density=True)
    
    return hist.astype(np.float32)


# =============================================================================
# Kinetic Features (calculado entre fases)
# =============================================================================
def extract_kinetic_features(phase_images: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extrae features cinéticas de la curva DCE.
    
    Args:
        phase_images: Dict con imágenes por fase {'phase_0': img, 'phase_1': img, ...}
    
    Returns:
        Feature vector (8,)
    """
    # Ordenar fases disponibles
    phases = sorted([k for k in phase_images.keys() if k.startswith('phase_')])
    
    if len(phases) < 2:
        return np.zeros(KINETIC_DIM, dtype=np.float32)
    
    # Calcular intensidades promedio por fase
    intensities = []
    for phase in phases:
        img = phase_images[phase]
        # Calcular intensidad promedio del 10% más brillante (proxy de tumor)
        flat = img.flatten()
        threshold = np.percentile(flat, 90)
        bright_pixels = flat[flat >= threshold]
        intensities.append(np.mean(bright_pixels))
    
    intensities = np.array(intensities)
    
    # Features cinéticas
    features = []
    
    # 1. Wash-in rate: (max - baseline) / baseline
    baseline = intensities[0]
    peak = np.max(intensities)
    features.append((peak - baseline) / (baseline + 1e-8))  # wash_in_rate
    
    # 2. Peak enhancement value
    features.append(peak)
    
    # 3. Time to peak (índice de fase)
    features.append(float(np.argmax(intensities)))
    
    # 4. Wash-out rate: (peak - final) / peak
    final = intensities[-1]
    features.append((peak - final) / (peak + 1e-8))  # wash_out_rate
    
    # 5. Signal enhancement ratio
    features.append((peak - baseline) / (peak + 1e-8))
    
    # 6. Area under curve (aproximación trapezoidal)
    features.append(np.trapz(intensities))
    
    # 7. Slope inicial (phase 0 -> phase 1)
    if len(intensities) >= 2:
        features.append(intensities[1] - intensities[0])
    else:
        features.append(0.0)
    
    # 8. Slope final (penúltima -> última)
    if len(intensities) >= 2:
        features.append(intensities[-1] - intensities[-2])
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Process Single Patient
# =============================================================================
def process_patient(patient_dir: Path, densenet: Optional[DenseNetExtractor] = None) -> Dict:
    """
    Procesa todas las fases de un paciente y extrae features.
    
    Returns:
        Dict con features por fase
    """
    patient_id = patient_dir.name
    patient_features = {
        'patient_id': patient_id,
        'phases': {},
        'kinetic': None,
        'num_phases': 0
    }
    
    # Cargar imágenes de cada fase
    phase_images = {}
    
    for phase_dir in sorted(patient_dir.iterdir()):
        if not phase_dir.is_dir():
            continue
        
        phase_name = phase_dir.name
        
        # Cargar slice central (más representativo del tumor)
        png_files = sorted(phase_dir.glob("*.png"))
        if not png_files:
            continue
        
        # Usar slice central
        central_idx = len(png_files) // 2
        central_slice = png_files[central_idx]
        
        # Leer imagen 16-bit
        image = cv2.imread(str(central_slice), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        
        phase_images[phase_name] = image
        
        # Extraer features de esta fase
        phase_feats = {}
        
        # DenseNet features
        if densenet is not None:
            phase_feats['densenet'] = densenet.extract(image)
        else:
            phase_feats['densenet'] = np.zeros(DENSENET_DIM, dtype=np.float32)
        
        # First-order features
        phase_feats['first_order'] = extract_first_order_features(image)
        
        # Histogram features
        phase_feats['histogram'] = extract_histogram_features(image)
        
        # Concatenar features de esta fase
        phase_feats['combined'] = np.concatenate([
            phase_feats['densenet'],
            np.zeros(KINETIC_DIM, dtype=np.float32),  # Placeholder, se llena después
            phase_feats['first_order'],
            phase_feats['histogram']
        ])
        
        patient_features['phases'][phase_name] = phase_feats
    
    # Extraer features cinéticas (requiere múltiples fases)
    if len(phase_images) >= 2:
        kinetic_feats = extract_kinetic_features(phase_images)
        patient_features['kinetic'] = kinetic_feats
        
        # Añadir kinetic features a cada fase
        for phase_name in patient_features['phases']:
            combined = patient_features['phases'][phase_name]['combined']
            # Insertar kinetic features en la posición correcta
            combined[DENSENET_DIM:DENSENET_DIM+KINETIC_DIM] = kinetic_feats
            patient_features['phases'][phase_name]['combined'] = combined
    
    patient_features['num_phases'] = len(phase_images)
    
    return patient_features


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Extract features from Duke PNG images')
    parser.add_argument('--test', action='store_true', help='Test with first 10 patients')
    parser.add_argument('--start', type=int, default=0, help='Start from patient index')
    parser.add_argument('--end', type=int, default=None, help='End at patient index')
    parser.add_argument('--no-densenet', action='store_true', help='Skip DenseNet extraction')
    args = parser.parse_args()
    
    print("="*60)
    print("FASE 6: Duke Feature Extraction")
    print("="*60)
    
    # Verificar directorio de entrada
    if not DUKE_PNG_DIR.exists():
        logger.error(f"No se encontró: {DUKE_PNG_DIR}")
        logger.info("Primero ejecutar fase6_duke_dicom_to_png.py")
        return
    
    # Listar pacientes procesados
    patient_dirs = sorted([
        d for d in DUKE_PNG_DIR.iterdir() 
        if d.is_dir() and d.name.startswith('Breast_MRI')
    ])
    
    logger.info(f"Pacientes disponibles: {len(patient_dirs)}")
    
    # Filtrar por rango
    if args.test:
        patient_dirs = patient_dirs[:10]
        logger.info("Modo TEST: procesando solo 10 pacientes")
    else:
        if args.end:
            patient_dirs = patient_dirs[args.start:args.end]
        else:
            patient_dirs = patient_dirs[args.start:]
        logger.info(f"Procesando pacientes {args.start} a {args.start + len(patient_dirs)}")
    
    # Inicializar DenseNet extractor
    if not args.no_densenet:
        logger.info("Inicializando DenseNet-121...")
        densenet = DenseNetExtractor()
    else:
        densenet = None
    
    # Procesar pacientes
    all_features = {}
    
    print(f"\nExtrayendo features...")
    for patient_dir in tqdm(patient_dirs, desc="Pacientes"):
        patient_features = process_patient(patient_dir, densenet)
        if patient_features['num_phases'] > 0:
            all_features[patient_features['patient_id']] = patient_features
    
    # Guardar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(all_features, f)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    print(f"Pacientes con features: {len(all_features)}")
    
    # Estadísticas por fase
    phase_counts = {}
    for patient_data in all_features.values():
        for phase in patient_data['phases'].keys():
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    print("\n--- Fases extraídas ---")
    for phase in sorted(phase_counts.keys()):
        print(f"  {phase}: {phase_counts[phase]} pacientes")
    
    print(f"\n  Feature dim por fase: {TOTAL_DIM}")
    print(f"\n✅ Guardado: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
