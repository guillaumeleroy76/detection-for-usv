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

# Recherche automatique de la vidéo test
video_path = find_file([
    os.path.join(SCRIPT_DIR, "..", "images and video", "modelisation_3d_boat.mp4"),
    os.path.join(SCRIPT_DIR, "images and video", "modelisation_3d_boat.mp4"),
    "images and video/modelisation_3d_boat.mp4",
    "modelisation_3d_boat.mp4"
])

print(f"🔄 Modèle chargé : {model_path}")
print(f"🎬 Vidéo source  : {video_path}")

model = YOLO(model_path)

# Lancer le tracking
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    persist=True,
    imgsz=640,
    conf=0.35,
    iou=0.5,
    show=True
)
