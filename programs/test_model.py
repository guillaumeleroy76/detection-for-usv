import os
import sys
import glob
import argparse
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation des performances d'un modèle YOLO USV sur Dataset_Global")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4",
        help="Modèle à évaluer : alias ('best4', 'best4_2'), nom de fichier ou chemin (défaut: best4)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="Dataset_Global/data.yaml",
        help="Chemin vers data.yaml (défaut: Dataset_Global/data.yaml)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Split du dataset à évaluer : 'test' ou 'val' (défaut: test)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Résolution d'image pour l'inférence (défaut: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Périphérique d'exécution : '0' (GPU CUDA) ou 'cpu' (défaut: auto-détection GPU)"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """
    Résout de manière flexible et robuste le chemin d'un modèle YOLO.
    Supporte les alias courts ('best4', 'best4_2', 'best5'), noms de fichiers et chemins.
    """
    if not model_arg:
        model_arg = "best4"
        
    if os.path.isabs(model_arg) and os.path.exists(model_arg):
        return os.path.abspath(model_arg)
        
    cwd = os.getcwd()
    parent_dir = os.path.dirname(base_dir)          # detection-for-usv
    grandparent_dir = os.path.dirname(parent_dir)   # yolo models lab
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


def resolve_data_path(data_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout le chemin vers data.yaml."""
    if os.path.isabs(data_arg) and os.path.exists(data_arg):
        return os.path.abspath(data_arg)
        
    candidates = [
        data_arg,
        os.path.join(base_dir, data_arg),
        os.path.join(os.path.dirname(base_dir), data_arg),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), data_arg),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), "Dataset_Global", "data.yaml"),
        os.path.join(os.path.dirname(base_dir), "Dataset_Global", "data.yaml"),
        "Dataset_Global/data.yaml",
        "data.yaml"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return os.path.abspath(data_arg)


def main():
    args = parse_args()
    
    model_path = resolve_model_path(args.model)
    data_path = resolve_data_path(args.data)
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable : {args.model}")
        print(f"   Chemin testé : {model_path}")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"❌ Erreur : Dataset YAML introuvable : {args.data}")
        print(f"   Chemin testé : {data_path}")
        sys.exit(1)
        
    print("=" * 65)
    print("🚀 ÉVALUATION DU MODÈLE YOLO USV")
    print("=" * 65)
    print(f"• Modèle évalué    : {model_path}")
    print(f"• Dataset YAML     : {data_path}")
    print(f"• Split            : {args.split}")
    print(f"• Résolution image : {args.imgsz}")
    print(f"• Périphérique     : {args.device} ({'GPU CUDA' if args.device == '0' else 'CPU'})")
    print("-" * 65)
    
    model = YOLO(model_path)
    
    metrics = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device
    )
    
    precision = metrics.box.p[0] if hasattr(metrics.box.p, "__len__") else metrics.box.p
    recall = metrics.box.r[0] if hasattr(metrics.box.r, "__len__") else metrics.box.r
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print("\n" + "=" * 65)
    print(f"📊 RÉSULTATS D'ÉVALUATION [{os.path.basename(model_path)}]")
    print("=" * 65)
    print(f"• Précision (P)       : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"• Rappel (R)          : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"• mAP50               : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"• mAP50-95 global     : {map50_95:.4f}  ({map50_95*100:.2f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
