import kagglehub
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

file_dir = Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = file_dir
RAW_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" # Se mantiene para consistencia, aunque no se use inmediatamente.

def setup_paths():
    """Crea o verifica los directorios RAW y PROCESSED."""
    try:
        RAW_PATH.mkdir(parents=True, exist_ok=True)
        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directorios verificados: {RAW_PATH} y {PROCESSED_PATH}")
    except Exception as e:
        logging.critical(f"Error al crear directorios: {e}")
        raise

def download_and_process_data():
    """Descarga el dataset, imprime la estructura y mueve la carpeta EuroSAT a la ruta RAW."""
    
    download_root: Optional[Path] = None
    try:
        logging.info("Iniciando descarga del dataset 'apollo2506/eurosat-dataset'.")
        download_root = Path(kagglehub.dataset_download("apollo2506/eurosat-dataset"))
        logging.info(f"Descarga completa. Ruta temporal raíz: {download_root}")
    except Exception as e:
        logging.error(f"Error durante la descarga de Kaggle: {e}", exc_info=True)
        return

    logging.info("Contenido de la carpeta raíz de descarga:")
    for item in download_root.iterdir():
        logging.info(f" - {item.name} {'(DIR)' if item.is_dir() else '(FILE)'}")
        

    eurosat_source_path = None
    
    eurosat_list = [p for p in download_root.glob('**/EuroSAT') if p.is_dir()]
    if eurosat_list:
        eurosat_source_path = eurosat_list[0]
        logging.info(f"Encontrada la carpeta 'EuroSAT' en: {eurosat_source_path}")
    
    elif not eurosat_list:
        eurosat_allbands_list = [p for p in download_root.glob('**/EuroSATallBands') if p.is_dir()]
        if eurosat_allbands_list:
            eurosat_source_path = eurosat_allbands_list[0]
            logging.warning(f"No se encontró 'EuroSAT'. Usando 'EuroSATallBands' en: {eurosat_source_path}")


    if not eurosat_source_path:
        logging.error("¡ERROR FATAL! No se encontraron las carpetas 'EuroSAT' ni 'EuroSATallBands'.")
        return

    eurosat_target_path = RAW_PATH / eurosat_source_path.name

    try:
        logging.info(f"Moviendo la estructura completa de {eurosat_source_path.name} a {RAW_PATH.name}...")
        shutil.move(eurosat_source_path, eurosat_target_path)
        logging.info(f"Movimiento finalizado. Dataset guardado en: {eurosat_target_path}")

    except Exception as e:
        logging.error(f"Error al mover la carpeta {eurosat_source_path.name}: {e}", exc_info=True)


if __name__ == '__main__':
    try:
        setup_paths()
        download_and_process_data()
    except Exception as e:
        logging.critical(f"El proceso principal falló: {e}")