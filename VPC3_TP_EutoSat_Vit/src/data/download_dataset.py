import kagglehub
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
RAW_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

def setup_paths():
    try:
        RAW_PATH.mkdir(parents=True, exist_ok=True)
        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directorios creados/verificados: {RAW_PATH} y {PROCESSED_PATH}")
    except Exception as e:
        logging.critical(f"Error al crear directorios: {e}")
        raise

def download_and_process_data():
    try:
        logging.info("Iniciando descarga del dataset 'apollo2506/eurosat-dataset'.")
        dataset_download_path = Path(kagglehub.dataset_download("apollo2506/eurosat-dataset"))
        logging.info(f"Descarga completa. Ruta temporal: {dataset_download_path}")
    except Exception as e:
        logging.error(f"Error durante la descarga de Kaggle: {e}", exc_info=True)
        return
    eurosat_dir_list = list(dataset_download_path.glob("EuroSAT"))
    if not eurosat_dir_list:
            logging.error(f"No se encontró la carpeta 'EuroSAT' en {dataset_download_path}")
            return
    
    eurosat_dir = eurosat_dir_list[0]

    for file_path in eurosat_dir.iterdir():
        file_name = file_path.name
        
        if file_name.endswith(".csv"):
            destination_dir = PROCESSED_PATH
        else:
            destination_dir = RAW_PATH
        
        destination_path = destination_dir / file_name

        try:
            shutil.move(file_path, destination_path)
            logging.debug(f"Movido: {file_name} a {destination_dir.name}")
        except Exception as e:
            logging.error(f"Error al mover el archivo {file_name}: {e}", exc_info=True)
        
    logging.info("Proceso de descarga y movimiento finalizado.")


if __name__ == '__main__':
    try:
        setup_paths()
        download_and_process_data()
    except Exception as e:
        logging.critical(f"El proceso principal falló: {e}")