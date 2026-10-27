from ultralytics import YOLO

# 1. Charger ton modèle best4.pt
model = YOLO("models/best4.pt")

# 2. Lancer le tracking avec des paramètres adaptés au milieu marin
results = model.track(
    source="images and video/modelisation_3d_boat.mp4", 
    tracker="bytetrack.yaml", 
    persist=True,               # Conserve l'historique des ID
    imgsz=640,
    conf=0.35,                  # Seuil de confiance un peu plus bas pour éviter de perdre le bateau
    iou=0.5,                    # Tolérance de chevauchement des boîtes
    show=True                   # Visualisation en direct sur ton PC
)