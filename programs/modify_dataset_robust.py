

import os
import sys
import glob
import json
import random
from typing import Dict, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import du moteur de dégradation
from image_degradation import load_image_safe, save_image_safe, degrade_image


def modify_dataset(
    dataset_dir: str = "Dataset_Global",
    ratio_per_transform: float = 0.10,
    seed: int = 42,
    generate_samples_sheet: bool = True
):
    """
    Sélectionne aléatoirement 10% des images d'entraînement pour chaque transformation
    et les modifie avec une sévérité réaliste pour la détection navale.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Résolution automatique du dossier dataset
    if not os.path.exists(dataset_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "..", dataset_dir),
            os.path.join(script_dir, "..", dataset_dir),
            os.path.join(script_dir, dataset_dir)
        ]
        for c in candidates:
            if os.path.exists(c):
                dataset_dir = os.path.abspath(c)
                break
                
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Dossier introuvable : {train_img_dir}")
        
    all_images = glob.glob(os.path.join(train_img_dir, "*.*"))
    all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    total_count = len(all_images)
    print("=" * 70)
    print("🛠️  MODIFICATION DU DATASET D'ENTRAÎNEMENT - DATA AUGMENTATION MARITIME")
    print("=" * 70)
    print(f"• Dossier images train : {train_img_dir}")
    print(f"• Total images train   : {total_count}")
    print(f"• Ratio par transform  : {ratio_per_transform * 100:.1f}% ({int(total_count * ratio_per_transform)} images chacune)")
    print(f"• Graine aléatoire     : {seed}")
    print("-" * 70)

    # 5 transformations adaptées au domaine maritime
    transformations = [
        ("gouttes", "Gouttes d'eau & Embruns", (0.20, 0.45)),
        ("pixels_noirs", "Pixels Noirs & Masquage", (0.12, 0.30)),
        ("flou", "Flou Optique & Houle", (0.18, 0.40)),
        ("resolution", "Baisse de Résolution", (0.20, 0.40)),
        ("brouillard", "Brume Marine & Contraste", (0.25, 0.50))
    ]

    # Mélange aléatoire de toutes les images pour obtenir des sous-ensembles disjoints
    shuffled_images = list(all_images)
    random.shuffle(shuffled_images)
    
    n_per_group = int(total_count * ratio_per_transform)
    
    manifest = {
        "dataset": dataset_dir,
        "total_images": total_count,
        "modified_per_transform": n_per_group,
        "seed": seed,
        "transformations": {}
    }

    sample_visuals = {} # {transform_name: [(clean_img, degraded_img, filename)]}

    for idx, (t_key, t_label, (sev_min, sev_max)) in enumerate(transformations):
        start_i = idx * n_per_group
        end_i = start_i + n_per_group
        selected_files = shuffled_images[start_i:end_i]
        
        print(f"\n🔄 Application de [{t_label}] sur {len(selected_files)} images ({start_i} à {end_i-1})...")
        
        modified_list = []
        sample_visuals[t_key] = []
        
        for file_idx, file_path in enumerate(selected_files):
            # Tirage d'une sévérité réaliste pour l'apprentissage
            sev = round(random.uniform(sev_min, sev_max), 3)
            
            try:
                # 1. Chargement de l'image
                original_img = load_image_safe(file_path)
                
                # Sauvegarde d'échantillons visuels pour la planche de contrôle
                if len(sample_visuals[t_key]) < 2:
                    clean_copy = original_img.copy()
                
                # 2. Application de la transformation
                degraded_img = degrade_image(original_img, t_key, severity=sev, seed=seed + file_idx)
                
                # 3. Écrasement / sauvegarde de l'image modifiée
                save_image_safe(file_path, degraded_img)
                
                if len(sample_visuals[t_key]) < 2:
                    sample_visuals[t_key].append((clean_copy, degraded_img, os.path.basename(file_path), sev))
                
                modified_list.append({
                    "file": os.path.basename(file_path),
                    "severity": sev
                })
                
            except Exception as e:
                print(f"   ⚠️ Erreur sur {file_path}: {e}")

            if (file_idx + 1) % 300 == 0 or (file_idx + 1) == len(selected_files):
                print(f"   Progression: {file_idx + 1}/{len(selected_files)} images traitées...")
                
        manifest["transformations"][t_key] = {
            "label": t_label,
            "count": len(modified_list),
            "severity_range": [sev_min, sev_max],
            "files": modified_list
        }

    # Sauvegarde du manifeste de traçabilité
    manifest_path = "dataset_modification_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Manifeste sauvegardé sous : {manifest_path}")

    # Génération de la planche d'échantillons de contrôle
    if generate_samples_sheet:
        print("🎨 Génération de la planche visuelle de vérification...")
        fig, axes = plt.subplots(len(transformations), 2, figsize=(8, 3.2 * len(transformations)))
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        for r_idx, (t_key, t_label, _) in enumerate(transformations):
            samples = sample_visuals.get(t_key, [])
            if samples:
                clean_img, deg_img, fname, sev_used = samples[0]
                
                # Image Clean
                clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
                axes[r_idx, 0].imshow(clean_rgb)
                axes[r_idx, 0].set_title(f"Originale ({t_label})", fontsize=10, fontweight="bold", color="green")
                axes[r_idx, 0].axis("off")
                
                # Image Modifiée
                deg_rgb = cv2.cvtColor(deg_img, cv2.COLOR_BGR2RGB)
                axes[r_idx, 1].imshow(deg_rgb)
                axes[r_idx, 1].set_title(f"Modifiée ({t_label} - Sévérité {int(sev_used*100)}%)", fontsize=10, fontweight="bold", color="blue")
                axes[r_idx, 1].axis("off")
                
        plt.suptitle("Échantillons des Images Modifiées dans le Dataset Train", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        samples_path = "echantillon_images_modifiees.png"
        plt.savefig(samples_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"🖼️  Planche de contrôle sauvegardée sous : {samples_path}")

    total_modified = n_per_group * len(transformations)
    pct_modified = (total_modified / total_count) * 100
    print("\n" + "=" * 70)
    print("✅ OPÉRATION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"• Total images train : {total_count}")
    print(f"• Images modifiées   : {total_modified} ({pct_modified:.1f} % du dataset)")
    print(f"• Images intactes    : {total_count - total_modified} ({100 - pct_modified:.1f} %)")
    print("=" * 70)


if __name__ == "__main__":
    modify_dataset()
