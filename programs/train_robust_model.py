

import os
import sys
import argparse
import torch
from ultralytics import YOLO


import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning robuste du modèle YOLO pour USV")
    parser.add_argument(
        "--weights", "--model", "-m",
        type=str,
        default="best4",
        help="Poids initiaux de départ : alias ('best4', 'best4_2'), nom ou chemin (défaut: best4)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="Dataset_Global/data.yaml",
        help="Chemin vers data.yaml (défaut: Dataset_Global/data.yaml)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Nombre d'époques de fine-tuning (défaut: 20)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Taille du lot / batch size (défaut: 16, baisser à 8 si mémoire GPU saturée)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Résolution d'image (défaut: 640)"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Taux d'apprentissage initial pour fine-tuning (défaut: 0.001)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="best4_robust_run",
        help="Nom du dossier de sortie de l'entraînement"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str) -> str:
    """Résout le chemin d'un modèle YOLO (alias 'best4', 'best4_2', nom de fichier ou chemin)."""
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


def resolve_file(candidates):
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0]

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Résolution robuste des chemins
    weights_path = resolve_model_path(args.weights, script_dir)
    
    data_path = resolve_file([
        args.data,
        os.path.join(script_dir, "..", "..", "Dataset_Global", "data.yaml"),
        os.path.join(script_dir, "..", "Dataset_Global", "data.yaml"),
        os.path.join(script_dir, "Dataset_Global", "data.yaml")
    ])

    # Détection automatique du device (GPU Nvidia RTX 3050 si CUDA disponible, sinon CPU)
    device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("🚀 LANCEMENT DU FINE-TUNING ROBUSTE YOLO (USV CEREMA)")
    print("=" * 70)
    print(f"• Modèle source       : {weights_path}")
    print(f"• Dataset             : {data_path}")
    print(f"• Device d'exécution  : {device} ({'GPU CUDA détecté' if device == '0' else 'CPU'})")
    print(f"• Nombre d'époques    : {args.epochs}")
    print(f"• Batch size          : {args.batch}")
    print(f"• Taux d'apprentissage: {args.lr0}")
    print(f"• Dossier de run      : runs/detect/{args.name}")
    print("-" * 70)

    # 1. Chargement des poids existants de best4
    print(f"🔄 Chargement des poids de base depuis {weights_path}...")
    model = YOLO(weights_path)

    # Vérification et mise à jour automatique du chemin absolu dans data.yaml
    try:
        dataset_root = os.path.dirname(os.path.abspath(data_path)).replace("\\", "/")
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(data_path, "w", encoding="utf-8") as f:
            for line in lines:
                if line.strip().startswith("path:"):
                    f.write(f"path: {dataset_root}\n")
                else:
                    f.write(line)
    except Exception as e:
        print(f"⚠️ Avertissement config yaml : {e}")

    # 2. Lancement du fine-tuning
    print("▶️  Démarrage de l'apprentissage...")
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        lr0=args.lr0,
        name=args.name,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("🎉 Entraînement terminé !")
    best_weights = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    print(f"✅ Meilleurs poids sauvegardés sous : {best_weights}")
    
    # 3. Évaluation automatique sur le jeu de test
    print("\n📊 Évaluation des performances sur le jeu de test propre...")
    eval_model = YOLO(best_weights)
    metrics = eval_model.val(data=args.data, split="test", imgsz=args.imgsz, device=device)
    
    map50_95 = metrics.box.map
    map50 = metrics.box.map50
    precision = metrics.box.p[0] if hasattr(metrics.box.p, "__len__") else metrics.box.p
    recall = metrics.box.r[0] if hasattr(metrics.box.r, "__len__") else metrics.box.r
    
    print("-" * 70)
    print(f"• Précision (P)   : {precision:.4f}")
    print(f"• Rappel (R)      : {recall:.4f}")
    print(f"• mAP50           : {map50:.4f}")
    print(f"• mAP50-95 global : {map50_95:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
