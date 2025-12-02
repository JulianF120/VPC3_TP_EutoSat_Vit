from pathlib import Path
import json
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datasets import load_dataset
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE RUTAS ---
file_dir = Path(__file__).resolve().parent

PROJECT_ROOT = file_dir
while PROJECT_ROOT.name != "VPC3_TP_EutoSat_Vit" and PROJECT_ROOT.parent != PROJECT_ROOT:
    if (PROJECT_ROOT / "src").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

if (PROJECT_ROOT.name == "models") or (PROJECT_ROOT.name == "src"):
    PROJECT_ROOT = PROJECT_ROOT.parent

logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

CSV_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT"
LABELS_MAP = IMAGE_DIR / "label_map.json"

# CAMBIO: Ruta apuntando al archivo .pt de YOLO
MODEL_DIR = PROJECT_ROOT / "models" / "YOLO-eurosat-model"/ "train" / "weights" / "best.pt" 
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "yolo-v8"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"CSV_DIR: {CSV_DIR}")
logger.info(f"IMAGE_DIR: {IMAGE_DIR}")
logger.info(f"MODEL_DIR: {MODEL_DIR}")
logger.info(f"FIGURES_DIR: {FIGURES_DIR}")

with open(LABELS_MAP, "r") as f:
    label_map = json.load(f)

id2label_gt = {i: label for i, label in enumerate(label_map)}
label2id_gt = {label: i for i, label in enumerate(label_map)}

logger.info(f"Clases esperadas (Ground Truth): {id2label_gt}")

try:
    model = YOLO(MODEL_DIR)
    logger.info("Modelo YOLO cargado exitosamente.")
except Exception as e:
    logger.error(f"Error cargando el modelo YOLO en {MODEL_DIR}. Asegúrate de que sea un archivo .pt válido.")
    raise e

model_names = model.names 
logger.info(f"Clases del modelo YOLO: {model_names}")


def predict_image(image_path: Path, top_k: int = 3):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    results = model.predict(source=str(image_path), verbose=False)[0]
    
    probs = results.probs.data.cpu().numpy()
    
    top_indices = probs.argsort()[::-1][:top_k]
    
    prediction_results = []
    for idx in top_indices:
        class_name = model.names[int(idx)]
        
        prediction_results.append({
            "label_id": int(idx),
            "label_name": class_name,
            "prob": float(probs[idx]),
        })
    return prediction_results

def show_and_save_prediction(image_path: Path, out_path: Path, top_k: int = 3):
    image_path = Path(image_path)
    out_path = Path(out_path)

    result = predict_image(image_path, top_k=top_k)
    img = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")

    title = "\n".join([f"{r['label_name']} ({r['prob']:.2f})" for r in result])
    plt.title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Guardada figura de predicción: {out_path}")


def pick_example_images():
    examples = []
    candidates = ["AnnualCrop", "Forest", "River"]

    for cls in candidates:
        cls_dir = IMAGE_DIR / cls
        if not cls_dir.exists():
            continue

        img_path = None
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs = list(cls_dir.glob(ext))
            if imgs:
                img_path = imgs[0]
                break

        if img_path:
            examples.append((cls, img_path))

    return examples


def evaluate_and_save_figures():
    logger.info("Iniciando evaluación en el set de test con YOLO...")

    test_csv = CSV_DIR / "test.csv"
    logger.info(f"test_csv: {test_csv}")

    df_test = pd.read_csv(test_csv)
    
    y_true_indices = []
    y_pred_probs_all = [] 
    

    image_paths = []
    valid_indices = [] 

    logger.info("Preparando lista de imágenes...")
    for idx, row in df_test.iterrows():
        fname = row["Filename"]
        classname = row["ClassName"] 
        
        full_path = IMAGE_DIR / fname
        if full_path.exists():
            image_paths.append(str(full_path))
            y_true_indices.append(label2id_gt[classname])
            valid_indices.append(idx)
        else:
            logger.warning(f"Imagen no encontrada: {full_path}")

    batch_size = 64
    total_images = len(image_paths)
    logger.info(f"Ejecutando inferencia en {total_images} imágenes...")

    num_classes = len(label_map)
    final_probs_matrix = np.zeros((total_images, num_classes))

    for i in tqdm(range(0, total_images, batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        
        results = model.predict(batch_paths, verbose=False)
        
        for j, res in enumerate(results):
            probs_tensor = res.probs.data.cpu().numpy()
            
            mapped_probs = np.zeros(num_classes)
            
            for model_id, prob in enumerate(probs_tensor):
                name = model.names[model_id]
                if name in label2id_gt:
                    gt_id = label2id_gt[name]
                    mapped_probs[gt_id] = prob
            
            final_probs_matrix[i + j] = mapped_probs

    
    y_true = np.array(y_true_indices)
    y_pred = final_probs_matrix.argmax(axis=-1)
    
    acc = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(y_true, final_probs_matrix, k=3)

    print(f"Accuracy en test (Top-1): {acc:.4f}")
    print(f"Top-3 accuracy en test: {top3:.4f}")

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=label_map,
        digits=3
    )

    print("\n=== Classification report por clase ===")
    print(report_str)

    
    metrics_txt_path = FIGURES_DIR / "metrics_report.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo: YOLOv8 Classification\n")
        f.write(f"Accuracy (Top-1): {acc:.4f}\n")
        f.write(f"Top-3 accuracy: {top3:.4f}\n\n")
        f.write(report_str)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    text = f"YOLOv8 Results\nAccuracy (Top-1): {acc:.4f}\nTop-3 accuracy: {top3:.4f}\n\n{report_str}"
    ax.text(0, 1, text, fontsize=8, va="top", family="monospace")
    fig.tight_layout()
    metrics_fig_path = FIGURES_DIR / "metrics_report.png"
    fig.savefig(metrics_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=label_map, yticklabels=label_map)
    plt.xlabel("Predicción (YOLO)")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - EuroSAT (YOLOv8)")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=label_map, yticklabels=label_map)
    plt.xlabel("Predicción (YOLO)")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión Normalizada (YOLOv8)")
    plt.tight_layout()
    cmn_path = FIGURES_DIR / "confusion_matrix_normalized.png"
    plt.savefig(cmn_path, dpi=200, bbox_inches="tight")
    plt.close()

    dist = df_test["ClassName"].value_counts().reindex(label_map)
    plt.figure(figsize=(7, 4))
    ax = dist.plot(kind="bar", color='green', alpha=0.7)
    plt.ylabel("Número de imágenes")
    plt.title("Distribución test por clase")
    plt.xticks(rotation=45, ha="right")
    for p_ in ax.patches:
        height = p_.get_height()
        ax.text(p_.get_x() + p_.get_width()/2, height + 5, f"{int(height)}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "test_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    x = np.arange(len(per_class_acc))
    plt.figure(figsize=(8, 4))
    bars = plt.bar(x, per_class_acc, color='green', alpha=0.7)
    plt.ylabel("Recall por clase")
    plt.ylim(0, 1.10)
    plt.xticks(x, label_map, rotation=45, ha="right")
    plt.title("Rendimiento por clase (YOLOv8)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_class_recall.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Figuras de evaluación YOLO guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    logger.info("Iniciando script de evaluación YOLOv8")

    evaluate_and_save_figures()

    examples = pick_example_images()
    if examples:
        for i, (cls, img_path) in enumerate(examples, start=1):
            out_file = FIGURES_DIR / f"prediction_example_{i}_{cls}.png"
            show_and_save_prediction(img_path, out_file, top_k=3)