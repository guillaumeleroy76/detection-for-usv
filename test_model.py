from ultralytics import YOLO

# 1. Charger ton modèle fraîchement entraîné
model = YOLO("models/best3.pt")

# 2. Lancer l'évaluation sur le jeu de test
metrics = model.val(
    data="Dataset_Global/data.yaml",
    split="test",  # Utilise spécifiquement le dossier test
    imgsz=640,
    device='cpu'
)

# 3. Afficher les métriques principales (mAP, Précision, Rappel)
print(f"mAP50-95 global : {metrics.box.map}")
print(f"mAP50 : {metrics.box.map50}")
print(f"Précision : {metrics.box.p}")
print(f"Rappel : {metrics.box.r}")