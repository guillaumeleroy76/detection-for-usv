import os
import sys
import glob
import time
import json
import argparse
import cv2
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Suivi vidéo USV avec export de flux JSON temps réel")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4",
        help="Modèle à utiliser : alias ('best4', 'best4_2'), nom ou chemin (défaut: best4)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="images and video/modelisation_3d_boat.mp4",
        help="Source vidéo ou index de caméra '0' (défaut: images and video/modelisation_3d_boat.mp4)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Algorithme de tracking (défaut: bytetrack.yaml)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Seuil de confiance minimal (défaut: 0.4)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Résolution d'inférence (défaut: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device : '0' (GPU CUDA) ou 'cpu' (défaut: auto-détection)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Désactiver la fenêtre OpenCV d'affichage en direct"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout le modèle YOLO (alias ou chemin)."""
    if not model_arg:
        model_arg = "best4"
        
    if os.path.isabs(model_arg) and os.path.exists(model_arg):
        return os.path.abspath(model_arg)
        
    cwd = os.getcwd()
    parent_dir = os.path.dirname(base_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    search_dirs = [cwd, base_dir, parent_dir, grandparent_dir]
    
    names = [model_arg]
    if not model_arg.lower().endswith(".pt"):
        names.append(model_arg + ".pt")
        
    candidates = []
    for name in names:
        candidates.append(name)
        for s_dir in search_dirs:
            candidates.append(os.path.join(s_dir, name))
            candidates.append(os.path.join(s_dir, "best4", name))
            candidates.append(os.path.join(s_dir, "models", name))
            
    bare_name = os.path.splitext(os.path.basename(model_arg))[0]
    for s_dir in search_dirs:
        candidates.append(os.path.join(s_dir, bare_name, f"{bare_name}.pt"))
        if bare_name == "best4_2":
            candidates.append(os.path.join(s_dir, "best4", "best4_2.pt"))
            
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
            
    target_pattern = bare_name + ".pt" if not model_arg.lower().endswith(".pt") else os.path.basename(model_arg)
    for root in [parent_dir, grandparent_dir]:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", target_pattern), recursive=True)
            if matches:
                return os.path.abspath(matches[0])
                
    return os.path.abspath(model_arg)


def resolve_source_path(source_arg: str, base_dir: str = SCRIPT_DIR):
    """Résout la source vidéo (entier ou chemin absolu)."""
    if source_arg.isdigit():
        return int(source_arg)
    if os.path.isabs(source_arg) and os.path.exists(source_arg):
        return os.path.abspath(source_arg)
        
    candidates = [
        source_arg,
        os.path.join(base_dir, source_arg),
        os.path.join(os.path.dirname(base_dir), source_arg),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), source_arg),
        os.path.join(os.path.dirname(base_dir), "images and video", os.path.basename(source_arg)),
        os.path.join(base_dir, "images and video", os.path.basename(source_arg)),
        os.path.join(os.path.dirname(base_dir), "images and video", "modelisation_3d_boat.mp4")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return source_arg


def main():
    args = parse_args()
    
    model_path = resolve_model_path(args.model)
    source_path = resolve_source_path(args.source)
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable : {args.model}")
        print(f"   Chemin testé : {model_path}")
        sys.exit(1)
        
    print("=" * 65)
    print("📡 ACQUISITION & FLUX JSON TEMPS RÉEL (USV TRACKING)")
    print("=" * 65)
    print(f"• Modèle chargé : {model_path}")
    print(f"• Source vidéo  : {source_path}")
    print(f"• Tracker       : {args.tracker}")
    print(f"• Confiance min : {args.conf}")
    print(f"• Device        : {args.device}")
    print(f"• Fenêtre GUI   : {'Non (--no-show)' if args.no_show else 'Oui'}")
    print("-" * 65)
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        print(f"❌ Erreur : Impossible d'ouvrir la source vidéo : {source_path}")
        sys.exit(1)
        
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("🏁 Fin de la vidéo atteinte.")
                break
                
            t_actuel = round(time.time() - start_time, 2)
            
            results = model.track(
                frame,
                tracker=args.tracker,
                persist=True,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            
            boxes = results[0].boxes
            pistes_list = []
            
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = xyxy
                    w = int(xmax - xmin)
                    h = int(ymax - ymin)
                    x = int(xmin)
                    y = int(ymin)
                    conf = float(box.conf[0])
                    id_piste = int(box.id[0]) if box.id is not None else 0
                    
                    pistes_list.append({
                        "id_piste": id_piste,
                        "boite": [x, y, w, h],
                        "confiance": round(conf, 2)
                    })
                    
            observation_data = {
                "t": t_actuel,
                "pistes": pistes_list
            }
            
            # Émission du JSON sur stdout
            print(json.dumps(observation_data, ensure_ascii=False))
            
            if not args.no_show:
                annotated_frame = results[0].plot()
                cv2.imshow("Suivi USV - JSON Stream", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Arrêt manuel par l'utilisateur.")
                    break
                    
    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel interrompu.")
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print("✅ Ressources libérées.")


if __name__ == "__main__":
    main()
