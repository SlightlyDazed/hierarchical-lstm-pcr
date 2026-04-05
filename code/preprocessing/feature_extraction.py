"""
FASE 4.6: Scientific Feature Extraction
========================================
Implementación de features científicamente validadas para predicción de pCR.

Basado en literatura 2023-2024:
- Pharmacokinetic: Ktrans proxy, enhancement ratios
- GLCM Texture: entropy, contrast, homogeneity  
- Kinetic Curve: wash-in, wash-out, TTP
- Peritumoral: región 3-6mm, ratios tumor/peritumor
- First-order: estadísticas de intensidad

Uso:
    python fase4_scientific_features.py --test      # Test extractores
    python fase4_scientific_features.py --analyze   # Análisis discriminativo
    python fase4_scientific_features.py --extract   # Extraer para todos
    python fase4_scientific_features.py --compare   # Comparar con baseline
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm
from scipy import stats, ndimage
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Intentar importar skimage para GLCM
try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠ skimage no instalado. GLCM features deshabilitadas.")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PROJECT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2")
ISPY2_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/ISPY2_ALL")
AUDIT_CSV = ISPY2_DIR / "audit_results" / "audit_report.csv"

FEATURES_DIR = PROJECT_DIR / "data/processed/features_cache"
OLD_FEATURES_FILE = FEATURES_DIR / "patient_features_fixed.pkl"
SCIENTIFIC_FEATURES_FILE = FEATURES_DIR / "patient_features_scientific.pkl"

# Configuración de features
PERITUMORAL_MARGIN_MM = 5  # Óptimo según literatura: 3-6mm
PIXEL_SPACING_MM = 0.7  # Típico para DCE-MRI mama

PHASES = ["Ph0", "Ph1", "Ph2", "Ph3", "Ph4", "Ph5"]
TIMEPOINTS = ["T0", "T1", "T2", "T3"]


# =============================================================================
# EXTRACTORES CIENTÍFICOS
# =============================================================================

class PharmacokineticExtractor:
    """
    Extrae features farmacocinéticas simplificadas.
    
    Nota: Sin AIF compleja, usamos aproximaciones basadas en 
    curva de enhancement que son robustas para 6 phases.
    
    Features (5):
    - enhancement_ratio: (Smax - S0) / S0
    - late_enhancement_ratio: (Slate - S0) / S0
    - enhancement_difference: Smax - Slate (washout)
    - IAUC: Initial Area Under Curve (primeros 3 phases)
    - ktrans_proxy: Aproximación de transfer constant
    """
    
    def __init__(self):
        self.feature_names = [
            'pk_enhancement_ratio',
            'pk_late_enhancement_ratio', 
            'pk_enhancement_difference',
            'pk_IAUC',
            'pk_ktrans_proxy'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(self, phase_signals: np.ndarray) -> Dict[str, float]:
        """
        Extrae features farmacocinéticas de señales DCE.
        
        Args:
            phase_signals: Array (6,) con señal promedio por phase
            
        Returns:
            Dict con features
        """
        if len(phase_signals) < 6 or np.all(phase_signals == 0):
            return {name: 0.0 for name in self.feature_names}
        
        S0 = phase_signals[0]  # Baseline (pre-contrast)
        
        if S0 == 0:
            return {name: 0.0 for name in self.feature_names}
        
        # Enhancement en cada phase
        enhancement = (phase_signals - S0) / S0
        
        # Peak enhancement
        Smax = np.max(phase_signals[1:])  # Excluir baseline
        peak_idx = np.argmax(phase_signals[1:]) + 1
        
        # Late enhancement (última phase)
        Slate = phase_signals[-1]
        
        # Features
        enhancement_ratio = (Smax - S0) / S0
        late_enhancement_ratio = (Slate - S0) / S0
        enhancement_difference = enhancement_ratio - late_enhancement_ratio
        
        # IAUC (primeros 3 phases post-contraste)
        # Usando regla del trapecio
        IAUC = np.trapz(enhancement[1:4], dx=1.0)
        
        # Ktrans proxy: slope inicial normalizado
        # Aproximación: pendiente de enhancement / tiempo
        if peak_idx > 1:
            ktrans_proxy = enhancement[peak_idx] / peak_idx
        else:
            ktrans_proxy = enhancement[1]
        
        return {
            'pk_enhancement_ratio': float(enhancement_ratio),
            'pk_late_enhancement_ratio': float(late_enhancement_ratio),
            'pk_enhancement_difference': float(enhancement_difference),
            'pk_IAUC': float(IAUC),
            'pk_ktrans_proxy': float(ktrans_proxy)
        }


class GLCMTextureExtractor:
    """
    Extrae features de textura GLCM.
    
    Basado en literatura:
    - Distances: [1, 2] pixels
    - Angles: [0°, 45°, 90°, 135°]
    - Gray levels: 64
    
    Features (7):
    - contrast, dissimilarity, homogeneity
    - energy, correlation, ASM, entropy
    """
    
    def __init__(self, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = distances
        self.angles = angles
        self.feature_names = [
            'glcm_contrast',
            'glcm_dissimilarity', 
            'glcm_homogeneity',
            'glcm_energy',
            'glcm_correlation',
            'glcm_ASM',
            'glcm_entropy'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(self, image_roi: np.ndarray) -> Dict[str, float]:
        """
        Extrae GLCM features del ROI.
        
        Args:
            image_roi: Array 2D (H, W) grayscale
            
        Returns:
            Dict con features
        """
        if not HAS_SKIMAGE:
            return {name: 0.0 for name in self.feature_names}
        
        if image_roi.size == 0 or image_roi.shape[0] < 5 or image_roi.shape[1] < 5:
            return {name: 0.0 for name in self.feature_names}
        
        try:
            # Normalizar a 64 gray levels
            roi_min, roi_max = image_roi.min(), image_roi.max()
            if roi_max > roi_min:
                image_norm = ((image_roi - roi_min) / (roi_max - roi_min) * 63).astype(np.uint8)
            else:
                return {name: 0.0 for name in self.feature_names}
            
            # Calcular GLCM
            glcm = graycomatrix(
                image_norm,
                distances=self.distances,
                angles=self.angles,
                levels=64,
                symmetric=True,
                normed=True
            )
            
            # Extraer features (promediadas sobre distancias y ángulos)
            features = {
                'glcm_contrast': float(graycoprops(glcm, 'contrast').mean()),
                'glcm_dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
                'glcm_homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
                'glcm_energy': float(graycoprops(glcm, 'energy').mean()),
                'glcm_correlation': float(graycoprops(glcm, 'correlation').mean()),
                'glcm_ASM': float(graycoprops(glcm, 'ASM').mean()),
            }
            
            # Entropy manual (no incluida en graycoprops)
            glcm_flat = glcm.flatten()
            glcm_flat = glcm_flat[glcm_flat > 0]
            entropy = -np.sum(glcm_flat * np.log2(glcm_flat))
            features['glcm_entropy'] = float(entropy)
            
            return features
            
        except Exception as e:
            return {name: 0.0 for name in self.feature_names}


class KineticCurveExtractor:
    """
    Extrae features de curva cinética DCE.
    
    Features (6):
    - wash_in_slope: Pendiente inicial
    - peak_enhancement: Máximo enhancement %
    - time_to_peak: Tiempo al pico
    - washout_rate: Tasa de lavado
    - AUC_kinetic: Área bajo curva
    - kinetic_type: 0=persistent, 1=plateau, 2=washout
    """
    
    def __init__(self):
        self.feature_names = [
            'kinetic_wash_in_slope',
            'kinetic_peak_enhancement',
            'kinetic_time_to_peak',
            'kinetic_washout_rate',
            'kinetic_AUC',
            'kinetic_type'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(self, phase_signals: np.ndarray) -> Dict[str, float]:
        """
        Extrae features de curva cinética.
        
        Args:
            phase_signals: Array (6,) con señal promedio por phase
            
        Returns:
            Dict con features
        """
        if len(phase_signals) < 6 or np.all(phase_signals == 0):
            return {name: 0.0 for name in self.feature_names}
        
        S0 = phase_signals[0]
        
        if S0 == 0:
            return {name: 0.0 for name in self.feature_names}
        
        # Enhancement % en cada phase
        enhancement = (phase_signals[1:] - S0) / S0 * 100
        
        # Wash-in slope (Ph0 → Ph1)
        wash_in_slope = enhancement[0] / 1.0  # Por minuto (asumiendo 1 min entre phases)
        
        # Peak enhancement
        peak_enhancement = float(np.max(enhancement))
        peak_idx = int(np.argmax(enhancement))
        
        # Time-to-peak (en phases, indexado desde 1)
        time_to_peak = float(peak_idx + 1)
        
        # Wash-out (late vs peak)
        late_enhancement = enhancement[-1]
        if peak_enhancement > 0:
            washout_rate = (peak_enhancement - late_enhancement) / peak_enhancement * 100
        else:
            washout_rate = 0.0
        
        # Clasificar patrón cinético
        if washout_rate > 20:
            kinetic_type = 2  # Type III: Washout
        elif washout_rate < -10:
            kinetic_type = 0  # Type I: Persistent
        else:
            kinetic_type = 1  # Type II: Plateau
        
        # AUC de curva cinética
        AUC_kinetic = float(np.trapz(enhancement, dx=1.0))
        
        return {
            'kinetic_wash_in_slope': float(wash_in_slope),
            'kinetic_peak_enhancement': float(peak_enhancement),
            'kinetic_time_to_peak': float(time_to_peak),
            'kinetic_washout_rate': float(washout_rate),
            'kinetic_AUC': float(AUC_kinetic),
            'kinetic_type': float(kinetic_type)
        }


class PeritumoralExtractor:
    """
    Extrae features de región peritumoral.
    
    Región óptima según literatura: 3-6mm
    
    Features (15):
    - peri_mean, peri_std, peri_median
    - peri_tumor_ratio_mean, peri_tumor_ratio_std
    - peri_edema_score
    - peri_glcm_* (7 features)
    - peri_heterogeneity
    """
    
    def __init__(self, margin_mm: float = PERITUMORAL_MARGIN_MM, 
                 pixel_spacing_mm: float = PIXEL_SPACING_MM):
        self.margin_pixels = int(margin_mm / pixel_spacing_mm)
        self.glcm_extractor = GLCMTextureExtractor()
        
        self.feature_names = [
            'peri_mean',
            'peri_std',
            'peri_median',
            'peri_tumor_ratio_mean',
            'peri_tumor_ratio_std',
            'peri_edema_score',
            'peri_heterogeneity',
            'peri_entropy'
        ] + [f'peri_{name}' for name in self.glcm_extractor.feature_names]
        
        self.feature_dim = len(self.feature_names)
    
    def extract(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Extrae features peritumorales.
        
        Args:
            image: Imagen completa (H, W)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dict con features
        """
        x1, y1, x2, y2 = [int(b) for b in bbox]
        h, w = image.shape[:2]
        
        # Validar bbox
        if x2 <= x1 or y2 <= y1:
            return {name: 0.0 for name in self.feature_names}
        
        # Región tumoral
        tumor_roi = image[y1:y2, x1:x2]
        
        if tumor_roi.size == 0:
            return {name: 0.0 for name in self.feature_names}
        
        # Dilatar bbox para región peritumoral
        peri_x1 = max(0, x1 - self.margin_pixels)
        peri_y1 = max(0, y1 - self.margin_pixels)
        peri_x2 = min(w, x2 + self.margin_pixels)
        peri_y2 = min(h, y2 + self.margin_pixels)
        
        # Crear máscara para anillo peritumoral (excluye tumor)
        peri_mask = np.zeros((h, w), dtype=bool)
        peri_mask[peri_y1:peri_y2, peri_x1:peri_x2] = True
        peri_mask[y1:y2, x1:x2] = False  # Excluir tumor
        
        # Extraer pixels peritumorales
        peri_pixels = image[peri_mask]
        
        if peri_pixels.size < 10:
            return {name: 0.0 for name in self.feature_names}
        
        # Estadísticas básicas
        tumor_mean = np.mean(tumor_roi)
        tumor_std = np.std(tumor_roi)
        
        peri_mean = float(np.mean(peri_pixels))
        peri_std = float(np.std(peri_pixels))
        peri_median = float(np.median(peri_pixels))
        
        # Ratios tumor/peritumor (importante para predicción)
        peri_tumor_ratio_mean = peri_mean / (tumor_mean + 1e-6)
        peri_tumor_ratio_std = peri_std / (tumor_std + 1e-6)
        
        # Edema score: peritumor más brillante que tumor indica edema
        peri_edema_score = float(peri_mean > 1.1 * tumor_mean)
        
        # Heterogeneidad (coeficiente de variación)
        peri_heterogeneity = peri_std / (peri_mean + 1e-6)
        
        # Entropy de histograma
        hist, _ = np.histogram(peri_pixels, bins=64, density=True)
        hist = hist[hist > 0]
        peri_entropy = -np.sum(hist * np.log2(hist))
        
        features = {
            'peri_mean': peri_mean,
            'peri_std': peri_std,
            'peri_median': peri_median,
            'peri_tumor_ratio_mean': float(peri_tumor_ratio_mean),
            'peri_tumor_ratio_std': float(peri_tumor_ratio_std),
            'peri_edema_score': peri_edema_score,
            'peri_heterogeneity': float(peri_heterogeneity),
            'peri_entropy': float(peri_entropy)
        }
        
        # GLCM peritumoral (usar región expandida)
        peri_region = image[peri_y1:peri_y2, peri_x1:peri_x2]
        glcm_features = self.glcm_extractor.extract(peri_region)
        
        for name, value in glcm_features.items():
            features[f'peri_{name}'] = value
        
        return features


class FirstOrderExtractor:
    """
    Extrae estadísticas de primer orden (intensidad).
    
    Features (12):
    - mean, std, median
    - skewness, kurtosis
    - entropy, energy
    - min, max, range
    - percentile_10, percentile_90
    """
    
    def __init__(self):
        self.feature_names = [
            'fo_mean',
            'fo_std', 
            'fo_median',
            'fo_skewness',
            'fo_kurtosis',
            'fo_entropy',
            'fo_energy',
            'fo_min',
            'fo_max',
            'fo_range',
            'fo_percentile_10',
            'fo_percentile_90'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(self, image_roi: np.ndarray) -> Dict[str, float]:
        """
        Extrae estadísticas de primer orden.
        
        Args:
            image_roi: Array 2D (H, W) ROI
            
        Returns:
            Dict con features
        """
        if image_roi.size == 0:
            return {name: 0.0 for name in self.feature_names}
        
        flat = image_roi.flatten()
        
        # Estadísticas básicas
        fo_mean = float(np.mean(flat))
        fo_std = float(np.std(flat))
        fo_median = float(np.median(flat))
        
        # Momentos
        fo_skewness = float(stats.skew(flat))
        fo_kurtosis = float(stats.kurtosis(flat))
        
        # Entropy de histograma
        hist, _ = np.histogram(flat, bins=64, density=True)
        hist = hist[hist > 0]
        fo_entropy = float(-np.sum(hist * np.log2(hist)))
        
        # Energy (suma de cuadrados normalizada)
        fo_energy = float(np.sum(hist**2))
        
        # Min/Max
        fo_min = float(np.min(flat))
        fo_max = float(np.max(flat))
        fo_range = fo_max - fo_min
        
        # Percentiles
        fo_p10 = float(np.percentile(flat, 10))
        fo_p90 = float(np.percentile(flat, 90))
        
        return {
            'fo_mean': fo_mean,
            'fo_std': fo_std,
            'fo_median': fo_median,
            'fo_skewness': fo_skewness,
            'fo_kurtosis': fo_kurtosis,
            'fo_entropy': fo_entropy,
            'fo_energy': fo_energy,
            'fo_min': fo_min,
            'fo_max': fo_max,
            'fo_range': fo_range,
            'fo_percentile_10': fo_p10,
            'fo_percentile_90': fo_p90
        }


# =============================================================================
# ADDITIONAL EXTRACTORS FROM LITERATURE
# =============================================================================

class FTVExtractor:
    """
    Extrae Functional Tumor Volume (FTV) features.
    
    FTV es la porción del tumor que muestra enhancement significativo.
    Según literatura: FTV predice pCR mejor que volumen total.
    
    Features (5):
    - ftv_ratio: Porción del tumor con enhancement >70%
    - ftv_peak_ratio: Ratio de enhancement máximo
    - enhancing_fraction: Fracción de pixels con enhancement
    - heterogeneity_ftv: Variabilidad del enhancement
    - hotspot_intensity: Intensidad del "hotspot" (top 10% pixels)
    """
    
    def __init__(self, enhancement_threshold: float = 0.70):
        self.enhancement_threshold = enhancement_threshold
        self.feature_names = [
            'ftv_ratio',
            'ftv_peak_ratio',
            'enhancing_fraction',
            'ftv_heterogeneity',
            'hotspot_intensity'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(self, pre_roi: np.ndarray, post_roi: np.ndarray) -> Dict[str, float]:
        """
        Extrae FTV features.
        
        Args:
            pre_roi: ROI pre-contraste (Ph0)
            post_roi: ROI post-contraste (Ph2 o Ph3)
            
        Returns:
            Dict con features
        """
        if pre_roi.size == 0 or post_roi.size == 0:
            return {name: 0.0 for name in self.feature_names}
        
        # Calcular enhancement map
        pre_mean = np.mean(pre_roi)
        if pre_mean == 0:
            return {name: 0.0 for name in self.feature_names}
        
        enhancement_map = (post_roi - pre_roi) / pre_mean
        
        # FTV ratio: fracción con enhancement > threshold
        ftv_mask = enhancement_map > self.enhancement_threshold
        ftv_ratio = np.sum(ftv_mask) / enhancement_map.size
        
        # Peak enhancement ratio
        ftv_peak_ratio = float(np.max(enhancement_map))
        
        # Enhancing fraction (cualquier enhancement positivo)
        enhancing_fraction = np.sum(enhancement_map > 0) / enhancement_map.size
        
        # Heterogeneity del enhancement
        ftv_heterogeneity = float(np.std(enhancement_map[ftv_mask])) if np.sum(ftv_mask) > 0 else 0.0
        
        # Hotspot intensity (promedio de top 10% pixels)
        if enhancement_map.size > 10:
            threshold_90 = np.percentile(enhancement_map, 90)
            hotspot_pixels = enhancement_map[enhancement_map >= threshold_90]
            hotspot_intensity = float(np.mean(hotspot_pixels))
        else:
            hotspot_intensity = float(np.mean(enhancement_map))
        
        return {
            'ftv_ratio': float(ftv_ratio),
            'ftv_peak_ratio': ftv_peak_ratio,
            'enhancing_fraction': float(enhancing_fraction),
            'ftv_heterogeneity': ftv_heterogeneity,
            'hotspot_intensity': hotspot_intensity
        }


class ShapeExtractor:
    """
    Extrae features de forma del tumor (2D).
    
    Según literatura: sphericity correlaciona con pCR.
    
    Features (8):
    - area, perimeter, circularity
    - aspect_ratio, solidity, extent
    - major_axis, minor_axis
    """
    
    def __init__(self):
        self.feature_names = [
            'shape_area',
            'shape_perimeter',
            'shape_circularity',
            'shape_aspect_ratio',
            'shape_solidity',
            'shape_extent',
            'shape_major_axis',
            'shape_minor_axis'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Extrae features de forma del bbox.
        
        Args:
            bbox: (x1, y1, x2, y2)
            image_shape: (H, W) de imagen original
            
        Returns:
            Dict con features
        """
        x1, y1, x2, y2 = [int(b) for b in bbox]
        h_img, w_img = image_shape
        
        # Dimensiones
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return {name: 0.0 for name in self.feature_names}
        
        # Área y perímetro (aproximación rectangular)
        area = width * height
        perimeter = 2 * (width + height)
        
        # Circularidad (4π*área/perímetro²) - 1 para círculo perfecto
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Aspect ratio
        aspect_ratio = width / height
        
        # Asumiendo tumor es elipse inscrita en bbox
        # Solidity ~= área_elipse / área_bbox = π/4 ≈ 0.785
        # Pero para tumores reales, puede variar
        solidity = np.pi / 4  # Aproximación para bbox
        
        # Extent (área / bbox_área en imagen) - normalizado
        extent = area / (h_img * w_img)
        
        # Major/minor axis (normalizado)
        major_axis = max(width, height) / max(h_img, w_img)
        minor_axis = min(width, height) / min(h_img, w_img)
        
        return {
            'shape_area': float(area),
            'shape_perimeter': float(perimeter),
            'shape_circularity': float(circularity),
            'shape_aspect_ratio': float(aspect_ratio),
            'shape_solidity': float(solidity),
            'shape_extent': float(extent),
            'shape_major_axis': float(major_axis),
            'shape_minor_axis': float(minor_axis)
        }


class DeltaFeatureExtractor:
    """
    Extrae features de cambio temporal (Delta features).
    
    Según literatura: cambios entre timepoints predicen pCR mejor que T0 solo.
    
    Compara T0 vs T1/T2/T3 para calcular:
    - Delta enhancement
    - Delta intensidad
    - Delta kinetic pattern
    """
    
    def __init__(self):
        self.feature_names = [
            'delta_enhancement',
            'delta_peak',
            'delta_washout',
            'delta_intensity_mean',
            'delta_intensity_std'
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract(
        self, 
        t0_signals: np.ndarray, 
        t1_signals: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Extrae features de cambio temporal.
        
        Args:
            t0_signals: Señales de phases en T0 (6,)
            t1_signals: Señales de phases en T1 (6,), opcional
            
        Returns:
            Dict con features
        """
        if t1_signals is None or len(t0_signals) < 6 or len(t1_signals) < 6:
            return {name: 0.0 for name in self.feature_names}
        
        # Enhancement en T0 y T1
        def calc_enhancement(signals):
            if signals[0] == 0:
                return 0.0
            return (np.max(signals[1:]) - signals[0]) / signals[0]
        
        def calc_washout(signals):
            if signals[0] == 0:
                return 0.0
            peak = np.max(signals[1:])
            late = signals[-1]
            if peak == 0:
                return 0.0
            return (peak - late) / peak
        
        enh_t0 = calc_enhancement(t0_signals)
        enh_t1 = calc_enhancement(t1_signals)
        
        peak_t0 = np.max(t0_signals[1:])
        peak_t1 = np.max(t1_signals[1:])
        
        washout_t0 = calc_washout(t0_signals)
        washout_t1 = calc_washout(t1_signals)
        
        # Deltas (cambio relativo)
        delta_enhancement = (enh_t1 - enh_t0) / (abs(enh_t0) + 1e-6)
        delta_peak = (peak_t1 - peak_t0) / (peak_t0 + 1e-6)
        delta_washout = washout_t1 - washout_t0
        
        # Cambio en intensidad promedio
        delta_intensity_mean = (np.mean(t1_signals) - np.mean(t0_signals)) / (np.mean(t0_signals) + 1e-6)
        delta_intensity_std = (np.std(t1_signals) - np.std(t0_signals)) / (np.std(t0_signals) + 1e-6)
        
        return {
            'delta_enhancement': float(delta_enhancement),
            'delta_peak': float(delta_peak),
            'delta_washout': float(delta_washout),
            'delta_intensity_mean': float(delta_intensity_mean),
            'delta_intensity_std': float(delta_intensity_std)
        }


# =============================================================================
# EXTRACTOR COMBINADO
# =============================================================================

class ScientificFeatureExtractor:
    """
    Combina todos los extractores científicos.
    
    Total features: ~63 por timepoint
    - Pharmacokinetic: 5
    - GLCM: 7
    - Kinetic: 6
    - Peritumoral: 15
    - First-order: 12
    - FTV: 5
    - Shape: 8
    - Delta: 5
    """
    
    def __init__(self):
        self.pharmacokinetic = PharmacokineticExtractor()
        self.glcm = GLCMTextureExtractor()
        self.kinetic = KineticCurveExtractor()
        self.peritumoral = PeritumoralExtractor()
        self.first_order = FirstOrderExtractor()
        self.ftv = FTVExtractor()
        self.shape = ShapeExtractor()
        self.delta = DeltaFeatureExtractor()
        
        self.feature_dim = (
            self.pharmacokinetic.feature_dim +
            self.glcm.feature_dim +
            self.kinetic.feature_dim +
            self.peritumoral.feature_dim +
            self.first_order.feature_dim +
            self.ftv.feature_dim +
            self.shape.feature_dim +
            self.delta.feature_dim
        )
        
        print(f"✓ ScientificFeatureExtractor inicializado")
        print(f"  Total features: {self.feature_dim}")
    
    def extract_from_phases(
        self, 
        phase_images: Dict[str, np.ndarray],
        phase_bboxes: Dict[str, Tuple[int, int, int, int]]
    ) -> Dict[str, float]:
        """
        Extrae todas las features científicas de un conjunto de phases.
        
        Args:
            phase_images: Dict["Ph0"-"Ph5"] -> imagen (H, W)
            phase_bboxes: Dict["Ph0"-"Ph5"] -> bbox (x1, y1, x2, y2)
            
        Returns:
            Dict con todas las features
        """
        all_features = {}
        
        # 1. Extraer señales de phases (para pharmacokinetic y kinetic)
        phase_signals = []
        for phase in PHASES:
            if phase in phase_images and phase in phase_bboxes:
                img = phase_images[phase]
                x1, y1, x2, y2 = [int(b) for b in phase_bboxes[phase]]
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    phase_signals.append(np.mean(roi))
                else:
                    phase_signals.append(0.0)
            else:
                phase_signals.append(0.0)
        
        phase_signals = np.array(phase_signals)
        
        # 2. Pharmacokinetic features
        pk_features = self.pharmacokinetic.extract(phase_signals)
        all_features.update(pk_features)
        
        # 3. Kinetic curve features
        kinetic_features = self.kinetic.extract(phase_signals)
        all_features.update(kinetic_features)
        
        # 4. Features de textura de phase post-contraste (Ph3 o mejor disponible)
        texture_phase = None
        for ph in ["Ph3", "Ph2", "Ph4", "Ph1"]:
            if ph in phase_images:
                texture_phase = ph
                break
        
        if texture_phase and texture_phase in phase_bboxes:
            img = phase_images[texture_phase]
            x1, y1, x2, y2 = [int(b) for b in phase_bboxes[texture_phase]]
            roi = img[y1:y2, x1:x2]
            bbox = phase_bboxes[texture_phase]
            
            # GLCM
            glcm_features = self.glcm.extract(roi)
            all_features.update(glcm_features)
            
            # First-order
            fo_features = self.first_order.extract(roi)
            all_features.update(fo_features)
            
            # Peritumoral
            peri_features = self.peritumoral.extract(img, bbox)
            all_features.update(peri_features)
        else:
            # Fallback: zeros
            for name in self.glcm.feature_names:
                all_features[name] = 0.0
            for name in self.first_order.feature_names:
                all_features[name] = 0.0
            for name in self.peritumoral.feature_names:
                all_features[name] = 0.0
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista ordenada de nombres de features."""
        return (
            self.pharmacokinetic.feature_names +
            self.kinetic.feature_names +
            self.glcm.feature_names +
            self.first_order.feature_names +
            self.peritumoral.feature_names
        )


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def test_extractors():
    """Test de todos los extractores con datos sintéticos."""
    print("="*70)
    print("TEST: Extractores Científicos")
    print("="*70)
    
    # Crear imagen sintética de tumor
    np.random.seed(42)
    image = np.random.randint(50, 200, size=(256, 256)).astype(np.float32)
    
    # Simular tumor más brillante en centro
    y, x = np.ogrid[:256, :256]
    tumor_mask = ((x - 128)**2 + (y - 128)**2) < 30**2
    image[tumor_mask] += 100
    
    bbox = (100, 100, 156, 156)
    
    # Simular 6 phases con enhancement
    phase_signals = np.array([100, 180, 200, 190, 175, 160])  # Curva tipo washout
    
    # Test cada extractor
    print("\n1. PharmacokineticExtractor")
    pk = PharmacokineticExtractor()
    pk_features = pk.extract(phase_signals)
    for name, value in pk_features.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n2. KineticCurveExtractor")
    kinetic = KineticCurveExtractor()
    kinetic_features = kinetic.extract(phase_signals)
    for name, value in kinetic_features.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n3. GLCMTextureExtractor")
    glcm = GLCMTextureExtractor()
    roi = image[100:156, 100:156]
    glcm_features = glcm.extract(roi)
    for name, value in glcm_features.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n4. FirstOrderExtractor")
    fo = FirstOrderExtractor()
    fo_features = fo.extract(roi)
    for name, value in fo_features.items():
        print(f"   {name}: {value:.4f}")
    
    print("\n5. PeritumoralExtractor")
    peri = PeritumoralExtractor()
    peri_features = peri.extract(image, bbox)
    for name, value in list(peri_features.items())[:8]:  # Primeros 8
        print(f"   {name}: {value:.4f}")
    
    print("\n✅ Todos los extractores funcionan correctamente")
    
    # Test combinado
    print("\n6. ScientificFeatureExtractor (Combinado)")
    combined = ScientificFeatureExtractor()
    print(f"   Total features: {combined.feature_dim}")
    print(f"   Feature names: {len(combined.get_feature_names())}")


def load_patient_images(patient_id: str, timepoint: str = "T0") -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple]]:
    """
    Carga imágenes reales de DCE-MRI para un paciente.
    
    Returns:
        phase_images: Dict[phase] -> imagen
        phase_bboxes: Dict[phase] -> bbox
    """
    import cv2
    
    patient_dir = ISPY2_DIR / patient_id
    if not patient_dir.exists():
        return {}, {}
    
    phase_images = {}
    phase_bboxes = {}
    
    # Patrones para encontrar carpetas
    tp_pattern = f"*{timepoint}-ISPY2MRI{timepoint}*"
    tp_folders = list(patient_dir.glob(tp_pattern))
    
    phase_keywords = {
        "Ph0": ["AX VIBRANT", "AX T1", "pre-contrast", "Ph0", "multiPhase"],
        "Ph1": ["Ph1"], "Ph2": ["Ph2"], "Ph3": ["Ph3"], 
        "Ph4": ["Ph4"], "Ph5": ["Ph5"],
    }
    
    for tp_folder in tp_folders:
        subfolders = [f for f in tp_folder.iterdir() if f.is_dir()]
        
        for phase, keywords in phase_keywords.items():
            if phase in phase_images:
                continue
                
            for subfolder in subfolders:
                folder_name = subfolder.name
                
                for keyword in keywords:
                    if keyword in folder_name:
                        if "Analysis Mask" in folder_name or "SER-" in folder_name:
                            continue
                        if "PE" in folder_name and "Ph" not in folder_name:
                            continue
                            
                        images = sorted(subfolder.glob("*.png"))
                        if images:
                            # Usar slice central
                            middle_idx = len(images) // 2
                            img_path = images[middle_idx]
                            
                            # Cargar imagen
                            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                            if img is not None:
                                # Convertir 16-bit a float
                                if img.dtype == np.uint16:
                                    img = img.astype(np.float32)
                                else:
                                    img = img.astype(np.float32)
                                    
                                phase_images[phase] = img
                                # Usar centro de imagen como bbox por defecto
                                h, w = img.shape[:2]
                                margin = int(min(h, w) * 0.2)
                                phase_bboxes[phase] = (margin, margin, w-margin, h-margin)
                            break
                
                if phase in phase_images:
                    break
    
    return phase_images, phase_bboxes


def analyze_discrimination():
    """Analiza poder discriminativo de features científicas vs baseline."""
    print("\n" + "="*70)
    print("ANÁLISIS: Poder Discriminativo de Features")
    print("="*70)
    
    if not OLD_FEATURES_FILE.exists():
        print(f"❌ No se encontró: {OLD_FEATURES_FILE}")
        return
    
    # Cargar datos existentes
    print(f"\nCargando features de: {OLD_FEATURES_FILE}")
    with open(OLD_FEATURES_FILE, 'rb') as f:
        features_data = pickle.load(f)
    
    audit_df = pd.read_csv(AUDIT_CSV)
    
    # Inicializar extractores
    extractor = ScientificFeatureExtractor()
    
    print("\nExtrayendo features de imágenes reales...")
    
    patient_features = []
    labels = []
    patient_ids_valid = []
    
    sample_patients = list(features_data.keys())[:80]  # Muestra de 80
    
    for patient_id in tqdm(sample_patients, desc="Procesando"):
        # Obtener label
        row = audit_df[audit_df['PatientID'] == patient_id]
        if len(row) == 0 or pd.isna(row.iloc[0]['pCR']):
            continue
        
        pcr = int(row.iloc[0]['pCR'])
        
        # Cargar imágenes reales
        phase_images, phase_bboxes = load_patient_images(patient_id, "T0")
        
        if len(phase_images) < 3:  # Necesitamos al menos 3 phases
            continue
        
        # Extraer signals de phases para pharmacokinetic/kinetic
        phase_signals = []
        for phase in PHASES:
            if phase in phase_images and phase in phase_bboxes:
                img = phase_images[phase]
                x1, y1, x2, y2 = [int(b) for b in phase_bboxes[phase]]
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    phase_signals.append(np.mean(roi))
                else:
                    phase_signals.append(0.0)
            else:
                phase_signals.append(0.0)
        
        phase_signals = np.array(phase_signals)
        
        if np.all(phase_signals == 0):
            continue
        
        # Extraer features
        all_feat = {}
        
        # 1. Pharmacokinetic
        pk_feat = extractor.pharmacokinetic.extract(phase_signals)
        all_feat.update(pk_feat)
        
        # 2. Kinetic
        kinetic_feat = extractor.kinetic.extract(phase_signals)
        all_feat.update(kinetic_feat)
        
        # 3. GLCM, FirstOrder, Peritumoral (de Ph3 o mejor disponible)
        texture_phase = None
        for ph in ["Ph3", "Ph2", "Ph4", "Ph1"]:
            if ph in phase_images:
                texture_phase = ph
                break
        
        if texture_phase:
            img = phase_images[texture_phase]
            bbox = phase_bboxes[texture_phase]
            x1, y1, x2, y2 = [int(b) for b in bbox]
            roi = img[y1:y2, x1:x2]
            
            # GLCM
            glcm_feat = extractor.glcm.extract(roi)
            all_feat.update(glcm_feat)
            
            # First-order
            fo_feat = extractor.first_order.extract(roi)
            all_feat.update(fo_feat)
            
            # Peritumoral
            peri_feat = extractor.peritumoral.extract(img, bbox)
            all_feat.update(peri_feat)
        else:
            # Fallback
            for name in extractor.glcm.feature_names:
                all_feat[name] = 0.0
            for name in extractor.first_order.feature_names:
                all_feat[name] = 0.0
            for name in extractor.peritumoral.feature_names:
                all_feat[name] = 0.0
        
        # Convertir a array
        feature_values = list(all_feat.values())
        
        patient_features.append(feature_values)
        labels.append(pcr)
        patient_ids_valid.append(patient_id)
    
    if len(patient_features) < 10:
        print("❌ Muy pocos pacientes para análisis")
        return
    
    X = np.array(patient_features)
    y = np.array(labels)
    
    # Limpiar NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nDatos: {X.shape[0]} pacientes, {X.shape[1]} features")
    print(f"pCR=0: {np.sum(y==0)}, pCR=1: {np.sum(y==1)}")
    
    # Nombres de features
    feature_names = list(all_feat.keys())
    
    # T-test para cada feature
    significant_features = []
    print(f"\n{'Feature':<35} {'t-stat':>10} {'p-value':>12} {'AUC':>8} {'Sig'}")
    print("-"*75)
    
    from sklearn.metrics import roc_auc_score
    
    for i, name in enumerate(feature_names):
        if X[:, i].std() > 1e-10:
            t_stat, p_val = stats.ttest_ind(X[y==0, i], X[y==1, i])
            
            # AUC individual
            try:
                auc_i = roc_auc_score(y, X[:, i])
                auc_i = max(auc_i, 1-auc_i)  # Asegurar >0.5
            except:
                auc_i = 0.5
            
            sig = "*" if p_val < 0.05 else ""
            if p_val < 0.05:
                significant_features.append((name, p_val, auc_i))
            print(f"{name:<35} {t_stat:>10.3f} {p_val:>12.4f} {auc_i:>8.3f} {sig}")
    
    print(f"\n✓ Features significativas (p<0.05): {len(significant_features)}/{len(feature_names)}")
    
    if significant_features:
        print("\n📊 Top Features Significativas:")
        for name, p_val, auc_i in sorted(significant_features, key=lambda x: x[2], reverse=True)[:10]:
            print(f"   {name}: AUC={auc_i:.3f}, p={p_val:.4f}")
    
    # AUC con LogReg simple usando todas las features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelo con todas las features
    lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
    auc_all = cross_val_score(lr, X_scaled, y, cv=5, scoring='roc_auc')
    
    print(f"\n📊 AUC (todas features): {auc_all.mean():.4f} ± {auc_all.std():.4f}")
    
    # Modelo con top-k features
    for k in [5, 10, 20]:
        if k > X.shape[1]:
            continue
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        auc_k = cross_val_score(lr, X_selected, y, cv=5, scoring='roc_auc')
        print(f"📊 AUC (top-{k} features): {auc_k.mean():.4f} ± {auc_k.std():.4f}")
    
    return X, y, feature_names


def evaluate_phase1_yolo_bbox():
    """
    PHASE 1: Evaluar con YOLO bbox preciso vs bbox genérico.
    
    Usa la información de bbox almacenada en spatial_features de la cache.
    """
    print("\n" + "="*70)
    print("PHASE 1: Evaluación con YOLO Bbox Preciso")
    print("="*70)
    
    if not OLD_FEATURES_FILE.exists():
        print(f"❌ No se encontró: {OLD_FEATURES_FILE}")
        return None
    
    # Cargar datos
    with open(OLD_FEATURES_FILE, 'rb') as f:
        features_data = pickle.load(f)
    
    audit_df = pd.read_csv(AUDIT_CSV)
    extractor = ScientificFeatureExtractor()
    
    print(f"\nPacientes en cache: {len(features_data)}")
    
    patient_features = []
    labels = []
    patient_ids_valid = []
    
    for patient_id in tqdm(features_data.keys(), desc="Phase 1 - YOLO bbox"):
        row = audit_df[audit_df['PatientID'] == patient_id]
        if len(row) == 0 or pd.isna(row.iloc[0]['pCR']):
            continue
        
        pcr = int(row.iloc[0]['pCR'])
        patient_data = features_data[patient_id]
        
        if 'T0' not in patient_data:
            continue
        
        # Extraer bbox de YOLO desde spatial_features
        phase_signals = []
        yolo_bbox = None
        best_phase = None
        
        for phase in PHASES:
            if phase not in patient_data['T0']:
                phase_signals.append(0.0)
                continue
            
            feat = patient_data['T0'][phase]
            if feat['image_path'] == '':
                phase_signals.append(0.0)
                continue
            
            # Obtener spatial features (contiene bbox normalizado)
            spatial = np.array(feat['spatial_features'])
            
            if yolo_bbox is None and spatial[10] > 0.3:  # confidence > 0.3
                yolo_bbox = spatial[:4]  # x1_norm, y1_norm, x2_norm, y2_norm
                best_phase = phase
            
            # Calcular señal promedio (usando DenseNet mean como proxy)
            densenet = np.array(feat['densenet_features'])
            phase_signals.append(np.mean(densenet) * 1000)
        
        phase_signals = np.array(phase_signals)
        
        if yolo_bbox is None or np.all(phase_signals == 0):
            continue
        
        # Extraer features usando bbox de YOLO
        all_feat = {}
        
        # Pharmacokinetic y Kinetic
        pk_feat = extractor.pharmacokinetic.extract(phase_signals)
        all_feat.update(pk_feat)
        
        kinetic_feat = extractor.kinetic.extract(phase_signals)
        all_feat.update(kinetic_feat)
        
        # Cargar imagen real con YOLO bbox
        import cv2
        best_feat = patient_data['T0'].get(best_phase or 'Ph3', {})
        if best_feat.get('image_path'):
            try:
                img = cv2.imread(best_feat['image_path'], cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    x1 = int(yolo_bbox[0] * w)
                    y1 = int(yolo_bbox[1] * h)
                    x2 = int(yolo_bbox[2] * w)
                    y2 = int(yolo_bbox[3] * h)
                    
                    roi = img[y1:y2, x1:x2].astype(np.float32)
                    
                    if roi.size > 100:
                        # GLCM
                        glcm_feat = extractor.glcm.extract(roi)
                        all_feat.update(glcm_feat)
                        
                        # First-order
                        fo_feat = extractor.first_order.extract(roi)
                        all_feat.update(fo_feat)
                        
                        # Peritumoral
                        peri_feat = extractor.peritumoral.extract(img.astype(np.float32), (x1, y1, x2, y2))
                        all_feat.update(peri_feat)
                        
                        # Shape
                        shape_feat = extractor.shape.extract((x1, y1, x2, y2), (h, w))
                        all_feat.update(shape_feat)
                    else:
                        # Fallback a zeros
                        for name in extractor.glcm.feature_names:
                            all_feat[name] = 0.0
                        for name in extractor.first_order.feature_names:
                            all_feat[name] = 0.0
                        for name in extractor.peritumoral.feature_names:
                            all_feat[name] = 0.0
                        for name in extractor.shape.feature_names:
                            all_feat[name] = 0.0
            except Exception as e:
                for name in extractor.glcm.feature_names + extractor.first_order.feature_names + extractor.peritumoral.feature_names + extractor.shape.feature_names:
                    all_feat[name] = 0.0
        else:
            for name in extractor.glcm.feature_names + extractor.first_order.feature_names + extractor.peritumoral.feature_names + extractor.shape.feature_names:
                all_feat[name] = 0.0
        
        feature_values = list(all_feat.values())
        patient_features.append(feature_values)
        labels.append(pcr)
        patient_ids_valid.append(patient_id)
    
    if len(patient_features) < 20:
        print(f"❌ Muy pocos pacientes: {len(patient_features)}")
        return None
    
    X = np.array(patient_features)
    y = np.array(labels)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    feature_names = list(all_feat.keys())
    
    print(f"\n✓ {X.shape[0]} pacientes, {X.shape[1]} features")
    print(f"  pCR=0: {np.sum(y==0)}, pCR=1: {np.sum(y==1)}")
    
    # Evaluar
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import SelectKBest, f_classif
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
    
    for k in [5, 10, 15, 20]:
        if k > X.shape[1]:
            continue
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        auc_k = cross_val_score(lr, X_selected, y, cv=5, scoring='roc_auc')
        print(f"📊 AUC (top-{k}): {auc_k.mean():.4f} ± {auc_k.std():.4f}")
    
    return X, y, feature_names, patient_ids_valid


def evaluate_phase2_delta_features(phase1_data=None):
    """
    PHASE 2: Agregar delta features (cambios T0 → T1/T2/T3).
    """
    print("\n" + "="*70)
    print("PHASE 2: Evaluación con Delta Features (T0 → T1/T2)")
    print("="*70)
    
    with open(OLD_FEATURES_FILE, 'rb') as f:
        features_data = pickle.load(f)
    
    audit_df = pd.read_csv(AUDIT_CSV)
    extractor = ScientificFeatureExtractor()
    
    patient_features = []
    labels = []
    
    for patient_id in tqdm(features_data.keys(), desc="Phase 2 - Delta"):
        row = audit_df[audit_df['PatientID'] == patient_id]
        if len(row) == 0 or pd.isna(row.iloc[0]['pCR']):
            continue
        
        pcr = int(row.iloc[0]['pCR'])
        patient_data = features_data[patient_id]
        
        # Obtener señales para T0 y T1
        def get_phase_signals(tp):
            signals = []
            if tp not in patient_data:
                return np.zeros(6)
            for phase in PHASES:
                if phase in patient_data[tp]:
                    feat = patient_data[tp][phase]
                    if feat['image_path'] != '':
                        densenet = np.array(feat['densenet_features'])
                        signals.append(np.mean(densenet) * 1000)
                    else:
                        signals.append(0.0)
                else:
                    signals.append(0.0)
            return np.array(signals)
        
        t0_signals = get_phase_signals('T0')
        t1_signals = get_phase_signals('T1')
        t2_signals = get_phase_signals('T2')
        
        if np.all(t0_signals == 0):
            continue
        
        # Features base
        all_feat = {}
        
        pk_feat = extractor.pharmacokinetic.extract(t0_signals)
        all_feat.update(pk_feat)
        
        kinetic_feat = extractor.kinetic.extract(t0_signals)
        all_feat.update(kinetic_feat)
        
        # Delta features T0→T1
        if not np.all(t1_signals == 0):
            delta_feat = extractor.delta.extract(t0_signals, t1_signals)
            all_feat.update({f't0t1_{k}': v for k, v in delta_feat.items()})
        else:
            for name in extractor.delta.feature_names:
                all_feat[f't0t1_{name}'] = 0.0
        
        # Delta features T0→T2
        if not np.all(t2_signals == 0):
            delta_feat2 = extractor.delta.extract(t0_signals, t2_signals)
            all_feat.update({f't0t2_{k}': v for k, v in delta_feat2.items()})
        else:
            for name in extractor.delta.feature_names:
                all_feat[f't0t2_{name}'] = 0.0
        
        patient_features.append(list(all_feat.values()))
        labels.append(pcr)
    
    X = np.array(patient_features)
    y = np.array(labels)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✓ {X.shape[0]} pacientes, {X.shape[1]} features (con delta)")
    print(f"  pCR=0: {np.sum(y==0)}, pCR=1: {np.sum(y==1)}")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import SelectKBest, f_classif
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
    
    for k in [5, 10, 15, 20]:
        if k > X.shape[1]:
            continue
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        auc_k = cross_val_score(lr, X_selected, y, cv=5, scoring='roc_auc')
        print(f"📊 AUC (top-{k}): {auc_k.mean():.4f} ± {auc_k.std():.4f}")
    
    return X, y


def evaluate_phase3_with_densenet():
    """
    PHASE 3: Combinar scientific features + top DenseNet features.
    """
    print("\n" + "="*70)
    print("PHASE 3: Combinar Scientific + DenseNet Features")
    print("="*70)
    
    with open(OLD_FEATURES_FILE, 'rb') as f:
        features_data = pickle.load(f)
    
    audit_df = pd.read_csv(AUDIT_CSV)
    extractor = ScientificFeatureExtractor()
    
    patient_features_sci = []
    patient_features_dn = []
    labels = []
    
    for patient_id in tqdm(features_data.keys(), desc="Phase 3 - Combined"):
        row = audit_df[audit_df['PatientID'] == patient_id]
        if len(row) == 0 or pd.isna(row.iloc[0]['pCR']):
            continue
        
        pcr = int(row.iloc[0]['pCR'])
        patient_data = features_data[patient_id]
        
        if 'T0' not in patient_data:
            continue
        
        # Extraer DenseNet features promediadas
        dn_feats = []
        phase_signals = []
        
        for phase in PHASES:
            if phase in patient_data['T0']:
                feat = patient_data['T0'][phase]
                if feat['image_path'] != '':
                    dn = np.array(feat['densenet_features'])
                    dn_feats.append(dn)
                    phase_signals.append(np.mean(dn) * 1000)
                else:
                    phase_signals.append(0.0)
            else:
                phase_signals.append(0.0)
        
        if len(dn_feats) < 3 or np.all(np.array(phase_signals) == 0):
            continue
        
        # Promediar DenseNet features
        dn_mean = np.mean(dn_feats, axis=0)
        
        # Scientific features
        phase_signals = np.array(phase_signals)
        
        all_sci = {}
        pk_feat = extractor.pharmacokinetic.extract(phase_signals)
        all_sci.update(pk_feat)
        
        kinetic_feat = extractor.kinetic.extract(phase_signals)
        all_sci.update(kinetic_feat)
        
        patient_features_sci.append(list(all_sci.values()))
        patient_features_dn.append(dn_mean)
        labels.append(pcr)
    
    X_sci = np.array(patient_features_sci)
    X_dn = np.array(patient_features_dn)
    y = np.array(labels)
    
    X_sci = np.nan_to_num(X_sci, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✓ {X_sci.shape[0]} pacientes")
    print(f"  Scientific features: {X_sci.shape[1]}")
    print(f"  DenseNet features: {X_dn.shape[1]}")
    print(f"  pCR=0: {np.sum(y==0)}, pCR=1: {np.sum(y==1)}")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
    
    lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
    
    # 1. Solo Scientific
    scaler1 = StandardScaler()
    X_sci_scaled = scaler1.fit_transform(X_sci)
    auc_sci = cross_val_score(lr, X_sci_scaled, y, cv=5, scoring='roc_auc')
    print(f"\n📊 Scientific only ({X_sci.shape[1]} feat): {auc_sci.mean():.4f} ± {auc_sci.std():.4f}")
    
    # 2. Solo DenseNet (top-50)
    scaler2 = StandardScaler()
    X_dn_scaled = scaler2.fit_transform(X_dn)
    selector_dn = SelectKBest(f_classif, k=50)
    X_dn_selected = selector_dn.fit_transform(X_dn_scaled, y)
    auc_dn = cross_val_score(lr, X_dn_selected, y, cv=5, scoring='roc_auc')
    print(f"📊 DenseNet top-50: {auc_dn.mean():.4f} ± {auc_dn.std():.4f}")
    
    # 3. Combinado: Scientific + DenseNet top-50
    X_combined = np.hstack([X_sci_scaled, X_dn_selected])
    auc_comb = cross_val_score(lr, X_combined, y, cv=5, scoring='roc_auc')
    print(f"📊 Combined ({X_combined.shape[1]} feat): {auc_comb.mean():.4f} ± {auc_comb.std():.4f}")
    
    # 4. Combinado + selección
    for k in [20, 30, 40]:
        selector_comb = SelectKBest(f_classif, k=k)
        X_comb_selected = selector_comb.fit_transform(X_combined, y)
        auc_sel = cross_val_score(lr, X_comb_selected, y, cv=5, scoring='roc_auc')
        print(f"📊 Combined top-{k}: {auc_sel.mean():.4f} ± {auc_sel.std():.4f}")
    
    return X_combined, y


def run_all_phases():
    """Ejecutar las 3 fases de evaluación."""
    print("\n" + "#"*70)
    print("# EVALUACIÓN INCREMENTAL - 3 FASES")
    print("#"*70)
    
    # Baseline
    print("\n" + "="*70)
    print("BASELINE: Sin mejoras")
    print("="*70)
    print("(Resultados anteriores: Top-10 AUC = 0.635)")
    
    # Phase 1
    result1 = evaluate_phase1_yolo_bbox()
    
    # Phase 2
    result2 = evaluate_phase2_delta_features()
    
    # Phase 3
    result3 = evaluate_phase3_with_densenet()
    
    print("\n" + "#"*70)
    print("# RESUMEN FINAL")
    print("#"*70)


def main():
    parser = argparse.ArgumentParser(description='Scientific Feature Extraction')
    parser.add_argument('--test', action='store_true', help='Test extractores')
    parser.add_argument('--analyze', action='store_true', help='Analizar discriminación')
    parser.add_argument('--phase1', action='store_true', help='Eval Phase 1: YOLO bbox')
    parser.add_argument('--phase2', action='store_true', help='Eval Phase 2: Delta features')
    parser.add_argument('--phase3', action='store_true', help='Eval Phase 3: +DenseNet')
    parser.add_argument('--all', action='store_true', help='Ejecutar 3 fases')
    args = parser.parse_args()
    
    if args.test:
        test_extractors()
    elif args.analyze:
        analyze_discrimination()
    elif args.phase1:
        evaluate_phase1_yolo_bbox()
    elif args.phase2:
        evaluate_phase2_delta_features()
    elif args.phase3:
        evaluate_phase3_with_densenet()
    elif args.all:
        run_all_phases()
    else:
        print("Uso:")
        print("  --test     Test de extractores")
        print("  --analyze  Analizar poder discriminativo")
        print("  --phase1   Eval Phase 1: YOLO bbox")
        print("  --phase2   Eval Phase 2: Delta features")
        print("  --phase3   Eval Phase 3: +DenseNet")
        print("  --all      Ejecutar las 3 fases")


if __name__ == "__main__":
    main()
