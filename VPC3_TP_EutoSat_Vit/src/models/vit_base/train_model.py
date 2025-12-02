from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
from datasets import load_dataset
import logging
import os
import json
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

IMAGE_DIR = "VPC3_TP_EutoSat_Vit/data/raw"
CSV_DIR = "VPC3_TP_EutoSat_Vit/data/processed"
DATA_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "validation": "validation.csv"
}

LABELS_MAP = "VPC3_TP_EutoSat_Vit/data/raw/label_map.json"

logging.info("Cargando archivos CSV...")
dataset = load_dataset(path = CSV_DIR, data_files = DATA_FILES)

with open(LABELS_MAP, "r") as f:
    label_map = json.load(f)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

def load_and_transform(batch):
    image_paths = [os.path.join(IMAGE_DIR, f) for f in batch['Filename']]

    try:
        pil_images = [Image.open(path).convert("RGB") for path in image_paths]
    except FileNotFoundError as e:
        logging.error(f"No se pudo encontrar la imagen: {e.filename}")

    processed_data = processor(pil_images, return_tensors="pt")
    batch['pixel_values'] = processed_data.pixel_values
    batch['labels'] = batch['Label']

    return batch

logging.info("\n--- Aplicando pre-procesamiento (Cargando y Tensorizando) ---")

train_ds = dataset['train'].map(load_and_transform, batched=True)
val_ds = dataset['validation'].map(load_and_transform, batched=True)
test_ds = dataset['test'].map(load_and_transform, batched=True)

cols_to_remove = ['Unnamed: 0', 'Filename', 'Label', 'ClassName']
train_ds = train_ds.remove_columns(cols_to_remove)
val_ds = val_ds.remove_columns(cols_to_remove)
test_ds = test_ds.remove_columns(cols_to_remove)

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(label_map),
    id2label={i: label for i, label in enumerate(label_map)},
    label2id={label: i for i, label in enumerate(label_map)},
    ignore_mismatched_sizes=True
    )

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir = "VPC3_TP_EutoSat_Vit/models",
    per_device_train_batch_size=16,
    num_train_epochs=50,                 
    eval_strategy="epoch",        
    save_strategy="epoch",              
    logging_steps=10,                 
    load_best_model_at_end=True,        
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)


logging.info("--- Iniciando entrenamiento ---")
trainer.train()

logging.info("--- Entrenamiento finalizado. Guardando el mejor modelo. ---")
trainer.save_model("VPC3_TP_EutoSat_Vit/models/vit-base-eurosat-model/best")
