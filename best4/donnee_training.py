import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Chemin vers ton fichier results.csv (mets à jour le nom du dossier de ton run si besoin)
csv_path = "results.csv"

if not os.path.exists(csv_path):
    print(f"❌ Fichier introuvable : {csv_path}. Vérifie le chemin de ton dossier de run.")
else:
    # 2. Charger les données et nettoyer les espaces superflus dans les en-têtes
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 3. Création de la figure pour de belles courbes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Suivi de l'entraînement YOLO - Évolution des performances", fontsize=16, fontweight='bold')

    # Style global des courbes
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    epochs = df['epoch']

    # --- Graphique 1 : Les Pertes (Losses) Entraînement ---
    ax1 = axes[0, 0]
    if 'train/box_loss' in df.columns and 'train/cls_loss' in df.columns and 'train/dfl_loss' in df.columns:
        ax1.plot(epochs, df['train/box_loss'], label='Box Loss', color='#1f77b4', linewidth=2)
        ax1.plot(epochs, df['train/cls_loss'], label='Class Loss', color='#ff7f0e', linewidth=2)
        ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss', color='#2ca02c', linewidth=2)
        ax1.set_title('Pertes d\'entraînement (Losses)')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Perte')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 2 : mAP (Précision moyenne) ---
    ax2 = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns and 'metrics/mAP50-95(B)' in df.columns:
        ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP50', color='#d62728', linewidth=2)
        ax2.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95', color='#9467bd', linewidth=2)
        ax2.set_title('Évolution du mAP')
        ax2.set_xlabel('Époques')
        ax2.set_ylabel('Score mAP')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 3 : Précision et Rappel ---
    ax3 = axes[1, 0]
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        ax3.plot(epochs, df['metrics/precision(B)'], label='Précision (Precision)', color='#8c564b', linewidth=2)
        ax3.plot(epochs, df['metrics/recall(B)'], label='Rappel (Recall)', color='#e377c2', linewidth=2)
        ax3.set_title('Précision et Rappel')
        ax3.set_xlabel('Époques')
        ax3.set_ylabel('Taux')
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 4 : Perte de Validation (si disponible) ---
    ax4 = axes[1, 1]
    val_loss_cols = [c for c in df.columns if 'val/' in c]
    if val_loss_cols:
        for col in val_loss_cols:
            ax4.plot(epochs, df[col], label=col, linewidth=2)
        ax4.set_title('Pertes de Validation')
        ax4.set_xlabel('Époques')
        ax4.set_ylabel('Perte')
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.6)
    else:
        ax4.set_visible(False)

    # Ajustement de l'affichage et sauvegarde
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = "courbes_entrainement.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Belles courbes générées et sauvegardées sous : {output_filename}")
    
    # Affichage direct dans le notebook
    plt.show()