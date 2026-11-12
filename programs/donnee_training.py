import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualisation des courbes d'entraînement YOLO (losses, mAP, P/R)")
    parser.add_argument(
        "--run", "-r", "--model", "-m",
        type=str,
        default="after",
        help="Run à tracer : 'after' ('best4_2', 'robust') ou 'before' ('best4', 'initial') (défaut: after)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Chemin direct vers un fichier results.csv spécifique"
    )
    return parser.parse_args()


def resolve_csv_path(run_arg: str, csv_arg: str = None, base_dir: str = SCRIPT_DIR) -> str:
    """Résout le chemin du fichier results.csv selon le modèle/run sélectionné."""
    if csv_arg and os.path.exists(csv_arg):
        return os.path.abspath(csv_arg)
        
    parent_dir = os.path.dirname(base_dir) # detection-for-usv
    
    run_lower = (run_arg or "after").lower().strip()
    
    if run_lower in ["after", "best4_2", "robust", "transformations", "after_transformations"]:
        candidates = [
            os.path.join(parent_dir, "best4", "after_transformations", "results_training_after_transformations", "results.csv"),
            os.path.join(base_dir, "..", "best4", "after_transformations", "results_training_after_transformations", "results.csv"),
            os.path.join(base_dir, "best4", "after_transformations", "results_training_after_transformations", "results.csv"),
        ]
    elif run_lower in ["before", "best4", "initial", "nominal", "before_transformations"]:
        candidates = [
            os.path.join(parent_dir, "best4", "before_transformations", "results_training_before_transformations", "results.csv"),
            os.path.join(base_dir, "..", "best4", "before_transformations", "results_training_before_transformations", "results.csv"),
            os.path.join(base_dir, "best4", "before_transformations", "results_training_before_transformations", "results.csv"),
        ]
    else:
        # Chemin direct ou nom de dossier
        candidates = [
            run_arg,
            os.path.join(base_dir, run_arg),
            os.path.join(parent_dir, run_arg),
            os.path.join(parent_dir, "best4", run_arg, "results.csv")
        ]
        
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
            
    return candidates[0]


def main():
    args = parse_args()
    csv_path = resolve_csv_path(args.run, args.csv)
    
    if not os.path.exists(csv_path):
        print(f"❌ Erreur : Fichier results.csv introuvable pour le run '{args.run}'.")
        print(f"   Chemin recherché : {csv_path}")
        print("💡 Astuce : Utilisez --run after (pour best4_2) ou --run before (pour best4 initial).")
        sys.exit(1)
        
    run_label = "Fine-Tuning Robuste (best4_2 / after)" if "after" in csv_path.lower() else "Entraînement Initial (best4 / before)"
    print("=" * 65)
    print("📈 TRACÉ DES COURBES D'ENTRAÎNEMENT YOLO")
    print("=" * 65)
    print(f"• Modèle / Run    : {run_label}")
    print(f"• Fichier CSV     : {csv_path}")
    print("-" * 65)
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Création de la figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Suivi de l'entraînement YOLO - {run_label}", fontsize=15, fontweight='bold')
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    epochs = df['epoch']

    # --- Graphique 1 : Pertes (Losses) Entraînement ---
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns and 'train/dfl_loss' in df.columns:
        ax1.plot(epochs, df['train/box_loss'], label='Box Loss', color='#1f77b4', linewidth=2)
        ax1.plot(epochs, df['train/cls_loss'], label='Class Loss', color='#ff7f0e', linewidth=2)
        ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss', color='#2ca02c', linewidth=2)
        ax1.set_title('Pertes d\'entraînement (Train Losses)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Perte')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 2 : mAP ---
    ax2 = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
        ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', color='#d62728', linewidth=2)
        ax2.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', color='#9467bd', linewidth=2)
        ax2.set_title('Évolution du mAP', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Époques')
        ax2.set_ylabel('Score mAP')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 3 : Précision et Rappel ---
    ax3 = axes[1, 0]
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        ax3.plot(epochs, df['metrics/precision(B)'], label='Précision (Precision)', color='#8c564b', linewidth=2)
        ax3.plot(epochs, df['metrics/recall(B)'], label='Rappel (Recall)', color='#e377c2', linewidth=2)
        ax3.set_title('Précision et Rappel', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Époques')
        ax3.set_ylabel('Taux')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 4 : Pertes de Validation ---
    ax4 = axes[1, 1]
    val_loss_cols = [c for c in df.columns if 'val/' in c]
    if val_loss_cols:
        colors_val = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, col in enumerate(val_loss_cols):
            ax4.plot(epochs, df[col], label=col.replace('val/', 'Val '), linewidth=2, color=colors_val[idx % len(colors_val)])
        ax4.set_title('Pertes de Validation (Val Losses)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Époques')
        ax4.set_ylabel('Perte')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)
    else:
        ax4.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = os.path.join(os.path.dirname(csv_path), "courbes_entrainement.png")
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Courbes enregistrées sous : {output_filename}")
    plt.close()


if __name__ == "__main__":
    main()
