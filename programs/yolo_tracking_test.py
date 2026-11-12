import os
import sys
import glob
import argparse
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Suivi vidéo multi-objets (tracking) avec YOLO USV")
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
        help="Source vidéo : fichier vidéo, dossier ou index de webcam '0' (défaut: images and video/modelisation_3d_boat.mp4)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Algorithme de tracking : 'bytetrack.yaml' ou 'botsort.yaml' (défaut: bytetrack.yaml)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Seuil de confiance minimal (défaut: 0.35)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Seuil IoU de NMS (défaut: 0.5)"
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
        help="Désactiver l'affichage de la fenêtre graphique vidéo"
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


def resolve_source_path(source_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout le chemin de la source vidéo si ce n'est pas un index numérique."""
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
    print("🎬 SUIVI TEMPOREL MULTI-OBJETS (TRACKING) YOLO USV")
    print("=" * 65)
    print(f"• Modèle chargé : {model_path}")
    print(f"• Source vidéo  : {source_path}")
    print(f"• Tracker       : {args.tracker}")
    print(f"• Confiance min : {args.conf}")
    print(f"• Device        : {args.device} ({'GPU CUDA' if args.device == '0' else 'CPU'})")
    print(f"• Affichage     : {'Non (--no-show)' if args.no_show else 'Oui'}")
    print("-" * 65)
    
    model = YOLO(model_path)
    
    results = model.track(
        source=source_path,
        tracker=args.tracker,
        persist=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()
