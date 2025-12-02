from ultralytics import YOLO
import torch

model = YOLO("yolov8n-cls.pt")  # modelo pequeno

results = model.train(
    data="/home/timstark/GitHub/new/VPC3_TP_EutoSat_Vit/VPC3_TP_EutoSat_Vit/data/datasets/eurosat",
    epochs=5,
    imgsz=128,
    batch=64,
    lr0=1e-3,
    patience=5,
    device=0 if torch.cuda.is_available() else "cpu",
    project='/home/timstark/GitHub/new/VPC3_TP_EutoSat_Vit/VPC3_TP_EutoSat_Vit/models/YOLO-eurosat-model'
)