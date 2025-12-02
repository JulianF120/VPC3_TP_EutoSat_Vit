from pathlib import Path
import json
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import ViTImageProcessor, ViTForImageClassification
import matplotlib.pyplot as plt

from datasets import load_dataset
from torch.utils.data import DataLoader
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


file_dir = Path(__file__).resolve().parent

PROJECT_ROOT = file_dir
while PROJECT_ROOT.name != "VPC3_TP_EutoSat_Vit" and PROJECT_ROOT.parent != PROJECT_ROOT:
    if (PROJECT_ROOT / "src").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

if (PROJECT_ROOT.name == "models") or (PROJECT_ROOT.name == "src"):
    PROJECT_ROOT = PROJECT_ROOT.parent

logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Rutas del proyecto
CSV_DIR = PROJECT_ROOT / "src" / "data" / "data" / "processed"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_MAP = IMAGE_DIR / "label_map.json"
MODEL_DIR = PROJECT_ROOT / "models" / "vit-eurosat-model" / "best"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"CSV_DIR: {CSV_DIR}")
logger.info(f"IMAGE_DIR: {IMAGE_DIR}")
logger.info(f"MODEL_DIR: {MODEL_DIR}")
logger.info(f"LABELS_MAP: {LABELS_MAP}")
logger.info(f"FIGURES_DIR: {FIGURES_DIR}")


# Carga de etiquetas, processor y modelo
with open(LABELS_MAP, "r") as f:
    label_map = json.load(f)

id2label = {i: label for i, label in enumerate(label_map)}
label2id = {label: i for i, label in enumerate(label_map)}

logger.info(f"Clases: {id2label}")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

model = ViTForImageClassification.from_pretrained(
    MODEL_DIR,
    id2label=id2label,
    label2id=label2id,
    local_files_only=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
logger.info(f"Usando dispositivo: {device}")


# Función de inferencia para una imagen
def predict_image(image_path: Path, top_k: int = 3):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top_indices = probs.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "label_id": int(idx),
            "label_name": id2label[idx],
            "prob": float(probs[idx]),
        })
    return results



# Mostrar y guardar predicción de una imagen
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


# Escoger imágenes de ejemplo (AnnualCrop, Forest, River)
def pick_example_images():
    examples = []
    candidates = [
        "AnnualCrop",
        "Forest",
        "River",
    ]

    for cls in candidates:
        cls_dir = IMAGE_DIR / cls
        if not cls_dir.exists():
            logger.warning(f"No existe el directorio de clase: {cls_dir}")
            continue

        img_path = None
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs = list(cls_dir.glob(ext))
            if imgs:
                img_path = imgs[0]
                break

        if img_path is None:
            logger.warning(f"No se encontraron imágenes para la clase {cls}")
            continue

        examples.append((cls, img_path))

    return examples


# Evaluación en el set de test + figuras
def evaluate_and_save_figures():
    logger.info("Iniciando evaluación en el set de test...")

    test_csv = CSV_DIR / "test.csv"
    logger.info(f"test_csv: {test_csv}")

    test_ds = load_dataset(
        "csv",
        data_files={"test": str(test_csv)}
    )["test"]

    logger.info(f"Ejemplo test_ds[0]: {test_ds[0]}")

    def preprocess_batch(batch):
        from PIL import Image as PILImage
        image_paths = [IMAGE_DIR / fname for fname in batch["Filename"]]
        images = [PILImage.open(p).convert("RGB") for p in image_paths]
        processed = processor(images, return_tensors="pt")
        batch["pixel_values"] = processed["pixel_values"]
        batch["labels"] = batch["Label"]
        return batch

    test_ds = test_ds.map(preprocess_batch, batched=True)

    for col in ["Filename", "ClassName", "Unnamed: 0"]:
        if col in test_ds.column_names:
            test_ds = test_ds.remove_columns(col)

    test_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    batch_size = 64
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    y_true = all_labels
    y_pred = all_logits.argmax(axis=-1)

    acc = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(y_true, all_logits, k=3)

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

    # Guardar el classification report en un txt
    metrics_txt_path = FIGURES_DIR / "metrics_report.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy (Top-1): {acc:.4f}\n")
        f.write(f"Top-3 accuracy: {top3:.4f}\n\n")
        f.write(report_str)
    logger.info(f"Reporte de métricas guardado en: {metrics_txt_path}")

    # Crear figura con texto de métricas
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    text = f"Accuracy (Top-1): {acc:.4f}\nTop-3 accuracy: {top3:.4f}\n\n{report_str}"
    ax.text(0, 1, text, fontsize=8, va="top", family="monospace")
    fig.tight_layout()
    metrics_fig_path = FIGURES_DIR / "metrics_report.png"
    fig.savefig(metrics_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura de métricas guardada en: {metrics_fig_path}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Heatmap absoluta
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_map,
        yticklabels=label_map
    )
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - EuroSAT (test)")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Matriz de confusión guardada en: {cm_path}")

    # Matriz de confusión normalizada
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_map,
        yticklabels=label_map
    )
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión EuroSAT (test) normalizada")
    plt.tight_layout()
    cmn_path = FIGURES_DIR / "confusion_matrix_normalized.png"
    plt.savefig(cmn_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Matriz de confusión normalizada guardada en: {cmn_path}")

    # Distribución del test por clase
    df_test = pd.read_csv(test_csv)
    dist = df_test["ClassName"].value_counts().reindex(label_map)

    plt.figure(figsize=(7, 4))
    ax = dist.plot(kind="bar")
    plt.ylabel("Número de imágenes")
    plt.title("Distribución test por clase")
    plt.xticks(rotation=45, ha="right")

    for p_ in ax.patches:
        height = p_.get_height()
        ax.text(
            p_.get_x() + p_.get_width() / 2,
            height + (max(dist) * 0.02),
            f"{int(height)}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    dist_path = FIGURES_DIR / "test_distribution.png"
    plt.savefig(dist_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Distribución del test guardada en: {dist_path}")

    # Recall por clase (igual a accuracy por clase en multiclase)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    class_names = [id2label[i] for i in range(len(per_class_acc))]
    x = np.arange(len(per_class_acc))

    plt.figure(figsize=(8, 4))
    bars = plt.bar(x, per_class_acc)

    plt.ylabel("Recall por clase")
    plt.ylim(0, 1.10)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.title("Rendimiento por clase en el set de test")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    recall_path = FIGURES_DIR / "per_class_recall.png"
    plt.savefig(recall_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Recall por clase guardado en: {recall_path}")

    print(f"\n✅ Figuras de evaluación guardadas en: {FIGURES_DIR}")


# Main
if __name__ == "__main__":
    logger.info("Iniciando script de predicción y evaluación (predict_model.py)")

    # 1) Evaluación y generación de figuras globales
    evaluate_and_save_figures()

    # 2) Ejemplos de inferencia de imágenes
    examples = pick_example_images()
    if not examples:
        logger.error("No se encontraron imágenes de ejemplo para inferencia.")
    else:
        saved_paths = []
        for i, (cls, img_path) in enumerate(examples, start=1):
            out_file = FIGURES_DIR / f"prediction_example_{i}_{cls}.png"
            show_and_save_prediction(img_path, out_file, top_k=3)
            saved_paths.append(out_file)

        print("\nSe generaron figuras de inferencia para las siguientes imágenes:")
        for p in saved_paths:
            print(f"  - {p}")

        print(f"\nTodas las imágenes (evaluación + predicción) se guardaron en: {FIGURES_DIR}")
