from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

RAW_ROOT = Path("data/raw/EuroSAT")
OUT_ROOT = Path("data/datasets/eurosat")
TRAIN_DIR = OUT_ROOT / "train"
VAL_DIR   = OUT_ROOT / "val"
TEST_DIR  = OUT_ROOT / "test"

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

classes = [d.name for d in RAW_ROOT.iterdir() if d.is_dir()]
print("Clases:", classes)

test_size = 0.2
val_size = 0.1
random_state = 42

for cls in classes:
    src_cls_dir = RAW_ROOT / cls
    images = list(src_cls_dir.glob("*.jpg")) + list(src_cls_dir.glob("*.png"))

    if len(images) == 0:
        print(f"⚠ No imágenes en {src_cls_dir}, se omite.")
        continue

    train_imgs, temp_imgs = train_test_split(
        images, test_size=(test_size + val_size), random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=(1 - val_ratio), random_state=random_state
    )

    for subset_name, subset_imgs in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]:
        out_cls_dir = OUT_ROOT / subset_name / cls
        out_cls_dir.mkdir(parents=True, exist_ok=True)
        for img_path in subset_imgs:
            shutil.copy2(img_path, out_cls_dir / img_path.name)

print("✅ Dataset preparado en formato YOLO (clasificación) en datasets/eurosat")
print("Ejemplo train:", list(TRAIN_DIR.iterdir())[:3])
