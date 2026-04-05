"""
FASE 6: Duke DICOM to PNG Converter
=====================================
Convierte imágenes DICOM de Duke Breast Cancer MRI a PNG 16-bit.

Estructura de entrada:
    Duke-Breast-Cancer-MRI/
    └── Breast_MRI_XXX/
        └── date-study/
            ├── pre_phase/
            │   └── 1-001.dcm, 1-002.dcm, ...
            ├── 1st_pass/
            ├── 2nd_pass/
            └── etc.

Estructura de salida:
    Duke_PNG/
    └── Breast_MRI_XXX/
        ├── phase_0/    (pre-contrast)
        │   └── 001.png, 002.png, ...
        ├── phase_1/    (1st pass)
        ├── phase_2/    (2nd pass)
        └── etc.

Features:
- Conversión a PNG 16-bit preservando rango dinámico completo
- Detección automática de fases DCE por nombre de carpeta
- Procesamiento paralelo con multiprocessing
- Logging detallado

Uso:
    python fase6_duke_dicom_to_png.py [--num-workers 4] [--test]
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# Importar pydicom
try:
    import pydicom
except ImportError:
    print("Error: pydicom no instalado. Ejecutar: pip install pydicom")
    sys.exit(1)

# Para guardar PNG 16-bit
try:
    import cv2
except ImportError:
    print("Error: opencv-python no instalado. Ejecutar: pip install opencv-python")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DUKE_DICOM_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/manifest-1654812109500/Duke-Breast-Cancer-MRI")
OUTPUT_DIR = Path("/media/alexander/585e7fd5-328a-4c3f-af02-97e1ec64e8b8/proyecto-ispy2/data/processed/Duke_PNG")

# Phase detection patterns (case insensitive)
PHASE_PATTERNS = {
    'phase_0': [
        r'.*pre.*',           # pre-contrast, ax dyn pre
        r'.*ax\s*3d\s*dyn$',  # "ax 3d dyn" sin pass
        r'.*ax\s*dyn$',       # "ax dyn" sin pass
        r'.*600\.0+.*dyn.*',  # 600.000000-ax dyn
        r'.*500\.0+.*dyn\s*mp$',  # 500.000000-ax 3d dyn MP
        r'.*ax\s*dynamic$',   # ax dynamic
        r'.*vibrant.*multiphase$',  # Ax Vibrant MultiPhase (pre)
        r'.*400\.0+.*vibrant.*',  # 400.000000-Ax Vibrant MultiPhase
    ],
    'phase_1': [
        r'.*1st\s*pass.*',    # 1st pass
        r'.*ph1.*',           # Ph1ax dyn, Ph1ax 3d dyn
        r'.*601\.0+.*',       # 601.000000-Ph1ax 3d dyn
        r'.*501\.0+.*',       # 501.000000-Ph1ax 3d dyn
        r'.*401\.0+.*',       # 401.000000-Ph1Ax Vibrant MultiPhase
        r'.*801\.0+.*',       # 801.000000-Ph1ax 3d dyn
    ],
    'phase_2': [
        r'.*2nd\s*pass.*',    # 2nd pass
        r'.*ph2.*',           # Ph2ax dyn
        r'.*602\.0+.*',       # 602.000000-Ph2ax 3d dyn
        r'.*502\.0+.*',       # 502.000000-Ph2ax 3d dyn
        r'.*402\.0+.*',       # 402.000000-Ph2Ax Vibrant MultiPhase
        r'.*802\.0+.*',       # 802.000000-Ph2ax 3d dyn
    ],
    'phase_3': [
        r'.*3rd\s*pass.*',    # 3rd pass
        r'.*ph3.*',           # Ph3ax dyn
        r'.*603\.0+.*',       # 603.000000-Ph3ax 3d dyn
        r'.*503\.0+.*',       # 503.000000-Ph3ax 3d dyn
        r'.*403\.0+.*',       # 403.000000-Ph3Ax Vibrant MultiPhase
        r'.*803\.0+.*',       # 803.000000-Ph3ax 3d dyn
    ],
    'phase_4': [
        r'.*4th\s*pass.*',    # 4th pass
        r'.*ph4.*',           # Ph4ax dyn
        r'.*604\.0+.*',       # 604.000000-Ph4ax 3d dyn
        r'.*504\.0+.*',       # 504.000000-Ph4ax 3d dyn
        r'.*404\.0+.*',       # 404.000000-Ph4Ax Vibrant MultiPhase
        r'.*804\.0+.*',       # 804.000000-Ph4ax 3d dyn
    ],
}

# Patterns to SKIP (no son DCE phases relevantes)
SKIP_PATTERNS = [
    r'.*t1\s*tse.*',         # T1 TSE (no DCE)
    r'.*ax\s*t1$',           # ax t1 (sin contraste, no dinámico)
    r'.*ax\s*t1\s*c$',       # ax t1 c (post-contraste estático)
    r'.*segmentation.*',     # Segmentation masks
    r'.*t2.*',               # T2 sequences
    r'.*stir.*',             # STIR
    r'.*scout.*',            # Localizers
    r'.*loc.*',              # Localizers
    r'.*bilateral.*t1.*',    # 3D T1 bilateral (no DCE)
    r'.*3d\s*t1.*',          # 3D T1 (no DCE)
]


def detect_phase(folder_name: str) -> Optional[str]:
    """
    Detecta la fase DCE basándose en el nombre de la carpeta.
    
    Returns:
        'phase_0', 'phase_1', ..., 'phase_4' o None si no es DCE
    """
    folder_lower = folder_name.lower()
    
    # Primero verificar si es una carpeta a ignorar
    for skip_pattern in SKIP_PATTERNS:
        if re.match(skip_pattern, folder_lower):
            return None
    
    # Buscar coincidencia con phases
    for phase, patterns in PHASE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, folder_lower):
                return phase
    
    return None


def convert_dicom_to_png16(dicom_path: Path, output_path: Path) -> bool:
    """
    Convierte un archivo DICOM a PNG 16-bit.
    
    Args:
        dicom_path: Ruta al archivo DICOM
        output_path: Ruta de salida para el PNG
    
    Returns:
        True si la conversión fue exitosa
    """
    try:
        # Leer DICOM
        ds = pydicom.dcmread(str(dicom_path))
        
        # Obtener pixel array
        pixel_array = ds.pixel_array
        
        # Aplicar rescale si está presente
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Convertir a 16-bit unsigned
        # Normalizar al rango completo de uint16
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        
        if pixel_max > pixel_min:
            # Escalar a 0-65535
            normalized = (pixel_array - pixel_min) / (pixel_max - pixel_min)
            img_16bit = (normalized * 65535).astype(np.uint16)
        else:
            # Imagen constante
            img_16bit = np.zeros_like(pixel_array, dtype=np.uint16)
        
        # Crear directorio de salida si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar como PNG 16-bit
        cv2.imwrite(str(output_path), img_16bit)
        
        return True
        
    except Exception as e:
        logger.error(f"Error convirtiendo {dicom_path}: {e}")
        return False


def process_patient(patient_dir: Path) -> Dict:
    """
    Procesa todas las fases DCE de un paciente.
    
    Returns:
        Dict con estadísticas del procesamiento
    """
    patient_id = patient_dir.name
    stats = {
        'patient_id': patient_id,
        'phases_found': {},
        'total_slices': 0,
        'errors': 0
    }
    
    # Buscar la carpeta de estudio (hay una sola por paciente generalmente)
    study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
    
    for study_dir in study_dirs:
        # Buscar carpetas de series/fases
        series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
        
        for series_dir in series_dirs:
            # Detectar fase
            phase = detect_phase(series_dir.name)
            
            if phase is None:
                continue
            
            # Encontrar archivos DICOM
            dicom_files = sorted(series_dir.glob("*.dcm"))
            
            if not dicom_files:
                continue
            
            # Crear directorio de salida
            output_phase_dir = OUTPUT_DIR / patient_id / phase
            
            # Convertir cada slice
            for dicom_file in dicom_files:
                # Nombre de salida: slice_XXX.png
                slice_num = dicom_file.stem.split('-')[-1]  # "1-001" -> "001"
                output_file = output_phase_dir / f"{slice_num}.png"
                
                if convert_dicom_to_png16(dicom_file, output_file):
                    stats['total_slices'] += 1
                else:
                    stats['errors'] += 1
            
            # Registrar fase encontrada
            if phase not in stats['phases_found']:
                stats['phases_found'][phase] = 0
            stats['phases_found'][phase] += len(dicom_files)
    
    return stats


def process_patient_wrapper(args):
    """Wrapper para multiprocessing."""
    patient_dir, = args
    try:
        return process_patient(patient_dir)
    except Exception as e:
        return {
            'patient_id': patient_dir.name,
            'phases_found': {},
            'total_slices': 0,
            'errors': 1,
            'exception': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Convert Duke DICOM to PNG 16-bit')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--test', action='store_true', help='Test with first 10 patients')
    parser.add_argument('--start', type=int, default=0, help='Start from patient index')
    parser.add_argument('--end', type=int, default=None, help='End at patient index')
    args = parser.parse_args()
    
    print("="*60)
    print("FASE 6: Duke DICOM to PNG 16-bit Converter")
    print("="*60)
    
    # Verificar directorio de entrada
    if not DUKE_DICOM_DIR.exists():
        logger.error(f"No se encontró: {DUKE_DICOM_DIR}")
        return
    
    # Listar pacientes
    patient_dirs = sorted([
        d for d in DUKE_DICOM_DIR.iterdir() 
        if d.is_dir() and d.name.startswith('Breast_MRI')
    ])
    
    logger.info(f"Pacientes encontrados: {len(patient_dirs)}")
    
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
    
    # Crear directorio de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Procesar pacientes
    all_stats = []
    
    print(f"\nConvirtiendo con {args.num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_patient_wrapper, (patient_dir,)): patient_dir 
            for patient_dir in patient_dirs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pacientes"):
            stats = future.result()
            all_stats.append(stats)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    total_slices = sum(s['total_slices'] for s in all_stats)
    total_errors = sum(s['errors'] for s in all_stats)
    patients_with_dce = sum(1 for s in all_stats if s['total_slices'] > 0)
    
    print(f"Pacientes procesados: {len(all_stats)}")
    print(f"Pacientes con DCE válido: {patients_with_dce}")
    print(f"Total slices convertidos: {total_slices}")
    print(f"Total errores: {total_errors}")
    
    # Estadísticas por fase
    phase_totals = {}
    for stats in all_stats:
        for phase, count in stats['phases_found'].items():
            if phase not in phase_totals:
                phase_totals[phase] = {'patients': 0, 'slices': 0}
            phase_totals[phase]['patients'] += 1
            phase_totals[phase]['slices'] += count
    
    print("\n--- Por Fase ---")
    for phase in sorted(phase_totals.keys()):
        info = phase_totals[phase]
        print(f"  {phase}: {info['patients']} pacientes, {info['slices']} slices")
    
    print(f"\n✅ Salida guardada en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
