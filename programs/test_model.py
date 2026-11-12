from ultralytics import YOLO
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename_candidates):
    for candidate in filename_candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return filename_candidates[0]

# Recherche automatique de best4.pt
model_path = find_file([
    os.path.join(SCRIPT_DIR, "..", "best4", "best4.pt"),
    os.path.join(SCRIPT_DIR, "best4", "best4.pt"),
    "best4/best4.pt",
    "best4.pt"
])

# Recherche automatique de data.yaml
yaml_path = find_file([
    os.path.join(SCRIPT_DIR, "..", "..", "Dataset_Global", "data.yaml"),
    os.path.join(SCRIPT_DIR, "..", "Dataset_Global", "data.yaml"),
    "Dataset_Global/data.yaml",
    "data.yaml"
])

print(f"🔄 Modèle chargé : {model_path}")
print(f"📂 Dataset YAML  : {yaml_path}")

model = YOLO(model_path)

# Lancer l'évaluation sur le jeu de test
metrics = model.val(
    data=yaml_path,
    split="test",
    imgsz=640,
    device='cpu'
)

print("\n" + "=" * 40)
print(f"mAP50-95 global : {metrics.box.map:.4f}")
print(f"mAP50           : {metrics.box.map50:.4f}")
precision = metrics.box.p[0] if hasattr(metrics.box.p, "__len__") else metrics.box.p
recall = metrics.box.r[0] if hasattr(metrics.box.r, "__len__") else metrics.box.r
print(f"Précision       : {precision:.4f}")
print(f"Rappel          : {recall:.4f}")
print("=" * 40)
