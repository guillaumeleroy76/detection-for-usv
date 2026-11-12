from ultralytics import YOLO
import cv2
import time
import json

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename_candidates):
    for candidate in filename_candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return filename_candidates[0]

# 1. Charger ton modèle entraîné (résolution auto)
model_path = find_file([
    os.path.join(SCRIPT_DIR, "..", "best4", "best4.pt"),
    os.path.join(SCRIPT_DIR, "best4", "best4.pt"),
    "best4/best4.pt",
    "best4.pt"
])
print(f"🔄 Modèle chargé : {model_path}")
model = YOLO(model_path)

# 2. Ouvrir la source vidéo (ou remplacer par 0 pour la caméra en direct)
video_path = find_file([
    os.path.join(SCRIPT_DIR, "..", "images and video", "modelisation_3d_boat.mp4"),
    os.path.join(SCRIPT_DIR, "images and video", "modelisation_3d_boat.mp4"),
    "images and video/modelisation_3d_boat.mp4",
    "modelisation_3d_boat.mp4"
])
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Erreur : Impossible d'ouvrir la source vidéo : {video_path}")
    exit()

print("🚀 Démarrage de l'acquisition et du tracking...")
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Si la vidéo est terminée (cap.read() renvoie False), on sort de la boucle
        if not ret:
            print("🏁 Fin de la vidéo atteinte. Arrêt de l'acquisition.")
            break

        # Temps écoulé depuis le début (en secondes)
        t_actuel = round(time.time() - start_time, 2)

        # Inférence avec tracking pour maintenir les identifiants d'une image à l'autre
        results = model.track(
            frame, 
            tracker="bytetrack.yaml", 
            persist=True, 
            imgsz=640, 
            conf=0.4, 
            verbose=False
        )
        
        boxes = results[0].boxes
        pistes_list = []

        # Récupération des détections et des pistes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = xyxy
                
                # Format de la boîte : [x, y, largeur, hauteur]
                w = int(xmax - xmin)
                h = int(ymax - ymin)
                x = int(xmin)
                y = int(ymin)

                conf = float(box.conf[0])

                # Récupération de l'ID de piste du tracker (si disponible)
                if box.id is not None:
                    id_piste = int(box.id[0])
                else:
                    id_piste = 0  # ID par défaut si non assigné

                piste = {
                    "id_piste": id_piste,
                    "boite": [x, y, w, h],
                    "confiance": round(conf, 2)
                }
                pistes_list.append(piste)

        # Construction de la structure JSON demandée pour l'image courante
        observation_data = {
            "t": t_actuel,
            "pistes": pistes_list
        }

        # Affichage du JSON dans la console pour chaque image
        print(json.dumps(observation_data, ensure_ascii=False))

        # Affichage optionnel de la vidéo annotée en direct sur ton PC
        annotated_frame = results[0].plot()
        cv2.imshow("Suivi USV - PC Test", annotated_frame)

        # Touche 'q' pour quitter manuellement si besoin
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interruption manuelle par l'utilisateur.")
            break

except KeyboardInterrupt:
    print("\n🛑 Arrêt brusque du programme.")

finally:
    # Libération propre des ressources
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Ressources libérées.")