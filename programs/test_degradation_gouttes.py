"""
Script: test_degradation_gouttes.py
Description: Banc d'essai visuel spécialisé sur les gouttes d'eau et embruns marins
             avec montée en sévérité jusqu'à 100% (et comparaison best4 vs best4_2).
Projet: USV Cerema - IMT Atlantique
Auteur: Guillaume
"""

import os
import sys
import glob
import argparse
import tempfile
from typing import List, Tuple, Dict, Any, Optional

# Éviter l'avertissement de cache Matplotlib sous Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# Import local du moteur de dégradation
from image_degradation import (
    load_image_safe,
    save_image_safe,
    add_water_droplets
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test visuel spécifique aux gouttes d'eau et embruns (jusqu'à 100% de sévérité)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4_2",
        help="Modèle principal à tester : alias ('best4', 'best4_2'), nom ou chemin (défaut: best4_2)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Activer la comparaison côte-à-côte entre best4 (initial) et best4_2 (robuste)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="images and video/bateau_test.jpg",
        help="Chemin vers l'image test (défaut: images and video/bateau_test.jpg)"
    )
    parser.add_argument(
        "--severities", "-s",
        nargs="+",
        type=float,
        default=[0.20, 0.40, 0.60, 0.80, 1.00],
        help="Niveaux de sévérité à tester (défaut: 0.20 0.40 0.60 0.80 1.00)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results_degradation/visual_gouttes",
        help="Dossier de sortie (défaut: results_degradation/visual_gouttes)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Seuil de confiance minimal YOLO (défaut: 0.35)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Périphérique d'inférence : '0' (GPU CUDA) ou 'cpu' (défaut: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour reproductibilité des gouttes (défaut: 42)"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout de façon robuste le chemin d'un modèle YOLO."""
    if not model_arg:
        model_arg = "best4_2"
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


def resolve_output_dir(base_dir: str, output_arg: str) -> str:
    """Résout le dossier de sortie toujours relativement au dossier du script si relatif."""
    if os.path.isabs(output_arg):
        return output_arg
    return os.path.abspath(os.path.join(base_dir, output_arg))


def resolve_path(base_dir: str, rel_or_abs_path: str) -> str:
    """Résout un chemin de fichier image."""
    if os.path.isabs(rel_or_abs_path) and os.path.exists(rel_or_abs_path):
        return rel_or_abs_path
    
    candidates = [
        os.path.join(base_dir, rel_or_abs_path),
        os.path.join(os.path.dirname(base_dir), rel_or_abs_path),
        os.path.join(os.path.dirname(base_dir), "images and video", os.path.basename(rel_or_abs_path)),
        os.path.join(base_dir, "images and video", os.path.basename(rel_or_abs_path)),
        rel_or_abs_path
    ]
        
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
            
    return os.path.abspath(rel_or_abs_path)


def draw_detections(
    image: np.ndarray,
    detections: Any,
    class_names: Dict[int, str],
    color: Tuple[int, int, int] = (0, 220, 0)
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Dessine les boîtes de détection YOLO avec étiquettes et score de confiance."""
    annotated = image.copy()
    extracted = []
    
    for box in detections.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = class_names.get(cls_id, f"id_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        extracted.append({
            "class": cls_name,
            "conf": conf,
            "bbox": (x1, y1, x2, y2)
        })
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        label = f"{cls_name}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        y_text = max(y1, label_h + 5)
        cv2.rectangle(
            annotated,
            (x1, y_text - label_h - 4),
            (x1 + label_w + 4, y_text + baseline - 2),
            color,
            -1
        )
        text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y_text - 2),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA
        )
        
    return annotated, extracted


def run_droplet_tests(
    model_path: str,
    image_path: str,
    output_dir: str,
    severities: List[float],
    conf_thresh: float,
    device: str,
    seed: int,
    compare_mode: bool = False
):
    """Exécute les tests de dégradation progressive aux gouttes d'eau."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("💧 USV YOLO - BANC D'ESSAI VISUEL : GOUTTES D'EAU & EMBRUNS (100%)")
    print("=" * 70)
    print(f"• Image source       : {image_path}")
    print(f"• Modèle principal   : {os.path.basename(model_path)} ({model_path})")
    print(f"• Mode comparaison   : {'Oui (best4 vs best4_2)' if compare_mode else 'Non'}")
    print(f"• Sévérités testées  : {[f'{int(s*100)}%' for s in severities]}")
    print(f"• Seuil confiance    : {conf_thresh:.2f}")
    print(f"• Device d'inférence : {device} ({'GPU CUDA' if device == '0' else 'CPU'})")
    print(f"• Dossier de sortie  : {output_dir}")
    print("-" * 70)

    # 1. Chargement modèle(s) et image
    model_primary = YOLO(model_path)
    orig_img = load_image_safe(image_path)
    
    models_to_test = [("primary", os.path.splitext(os.path.basename(model_path))[0], model_primary)]
    
    if compare_mode:
        model_b4_path = resolve_model_path("best4")
        model_b42_path = resolve_model_path("best4_2")
        models_to_test = [
            ("initial", "best4 (Initial)", YOLO(model_b4_path)),
            ("robuste", "best4_2 (Robuste)", YOLO(model_b42_path))
        ]

    # Inférence sur image nominale (propre)
    clean_annotated = {}
    clean_top_confs = {}
    for key, name, mdl in models_to_test:
        res = mdl(orig_img, conf=conf_thresh, device=device, verbose=False)[0]
        annot, dets = draw_detections(orig_img, res, mdl.names, color=(0, 220, 0))
        clean_annotated[key] = annot
        clean_top_confs[key] = dets[0]["conf"] if dets else 0.0
        status_txt = f"✅ Détecté (Conf: {clean_top_confs[key]:.2f})" if dets else "❌ Non Détecté"
        print(f"• [{name}] Image Propre : {status_txt}")

    # Batterie de tests par sévérité
    matrix_results = {}  # (key, sev) -> {"annotated": img, "conf": float, "detected": bool, "drops": int}
    
    for sev in severities:
        sev_pct = int(round(sev * 100))
        # Génération dégradation gouttes
        deg_img = add_water_droplets(orig_img, severity=sev, seed=seed)
        approx_drops = int(10 + sev * 80)
        
        print(f"\n🌊 Sévérité {sev_pct}% (~{approx_drops} gouttes d'eau projetées) :")
        
        for key, name, mdl in models_to_test:
            res = mdl(deg_img, conf=conf_thresh, device=device, verbose=False)[0]
            if len(res.boxes) > 0:
                top_conf = float(res.boxes.conf.max())
                box_color = (0, 220, 0) if top_conf >= 0.70 else (0, 165, 255)
            else:
                top_conf = 0.0
                box_color = (0, 0, 255)
                
            deg_annot, deg_dets = draw_detections(deg_img, res, mdl.names, color=box_color)
            is_det = len(deg_dets) > 0
            matrix_results[(key, sev)] = {
                "annotated": deg_annot,
                "conf": top_conf,
                "detected": is_det,
                "drops": approx_drops
            }
            
            status = f"✅ Détecté ({top_conf:.2f})" if is_det else "❌ Non Détecté"
            print(f"   [{name:<18}] : {status}")

        # Génération d'une image comparative individuelle Clean vs Dégradé
        primary_key = models_to_test[0][0]
        prim_annot = matrix_results[(primary_key, sev)]["annotated"]
        prim_conf = matrix_results[(primary_key, sev)]["conf"]
        h, w = orig_img.shape[:2]
        header_h = 45
        
        pair_canvas = np.zeros((h + header_h, w * 2 + 10, 3), dtype=np.uint8)
        pair_canvas[:] = (35, 35, 35)
        pair_canvas[header_h:header_h + h, :w] = clean_annotated[primary_key]
        pair_canvas[header_h:header_h + h, w + 10:w * 2 + 10] = prim_annot
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            pair_canvas,
            f"Original (Conf: {clean_top_confs[primary_key]:.2f})",
            (15, 30),
            font, 0.7, (240, 240, 240), 2, cv2.LINE_AA
        )
        status_header = f"Gouttes {sev_pct}% - Conf: {prim_conf:.2f}" if prim_conf > 0 else f"Gouttes {sev_pct}% - NON DETECTE"
        cv2.putText(
            pair_canvas,
            status_header,
            (w + 25, 30),
            font, 0.65, (0, 220, 255) if prim_conf > 0 else (100, 100, 255), 2, cv2.LINE_AA
        )
        pair_filename = f"comparaison_gouttes_sev{sev_pct:03d}.jpg"
        save_image_safe(os.path.join(output_dir, pair_filename), pair_canvas)

    # 3. Génération de la planche récapitulative globale
    fig_rows = len(models_to_test)
    fig_cols = len(severities) + 1  # 1 col pour Clean + cols pour les sévérités
    
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3.8 * fig_cols, 2.7 * fig_rows))
    
    if fig_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    for r_idx, (key, name, _) in enumerate(models_to_test):
        # Colonne 0 : Clean
        clean_rgb = cv2.cvtColor(clean_annotated[key], cv2.COLOR_BGR2RGB)
        axes[r_idx, 0].imshow(clean_rgb)
        axes[r_idx, 0].set_ylabel(name, fontsize=11, fontweight="bold")
        if r_idx == 0:
            axes[r_idx, 0].set_title(
                f"Image Propre\nConf: {clean_top_confs[key]:.2f}",
                fontsize=10, fontweight="bold", color="darkgreen"
            )
        else:
            axes[r_idx, 0].set_title(f"Conf: {clean_top_confs[key]:.2f}", fontsize=10, color="darkgreen")
        axes[r_idx, 0].axis("off")
        
        # Colonnes Sévérités
        for c_idx, sev in enumerate(severities):
            col_ax = axes[r_idx, c_idx + 1]
            rec = matrix_results[(key, sev)]
            rgb_annot = cv2.cvtColor(rec["annotated"], cv2.COLOR_BGR2RGB)
            col_ax.imshow(rgb_annot)
            
            sev_pct = int(round(sev * 100))
            conf_val = rec["conf"]
            if rec["detected"]:
                color = "darkgreen" if conf_val >= 0.70 else "darkorange"
                title_txt = f"Gouttes {sev_pct}%\nConf: {conf_val:.2f}"
            else:
                color = "red"
                title_txt = f"Gouttes {sev_pct}%\nÉchec Détection"
                
            col_ax.set_title(title_txt, fontsize=10, color=color, fontweight="bold")
            col_ax.axis("off")
            
    title_main = (
        "Comparatif de Résistance aux Gouttes d'Eau et Embruns (Jusqu'à 100%)"
        if compare_mode
        else f"Résistance aux Gouttes d'Eau et Embruns - Modèle {os.path.basename(model_path)}"
    )
    plt.suptitle(title_main, fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    summary_plot_name = "comparatif_gouttes_best4_vs_best4_2.png" if compare_mode else "planche_gouttes_eau_100pct.png"
    summary_plot_path = os.path.join(output_dir, summary_plot_name)
    plt.savefig(summary_plot_path, dpi=250, bbox_inches="tight")
    plt.close()
    
    print("\n" + "=" * 70)
    print("📊 BILAN DES TESTS GOUTTES D'EAU (100%)")
    print("=" * 70)
    print(f"✅ Images comparatives par niveau enregistrées dans : {output_dir}")
    print(f"✅ Planche matricielle globale sauvegardée : {summary_plot_path}")
    
    if compare_mode:
        print("\n📈 TABLEAU COMPARATIF FINAL (best4 vs best4_2) :")
        print("-" * 70)
        print(f"{'Sévérité':<12} | {'best4 (Initial)':<20} | {'best4_2 (Robuste)':<20} | {'Gain':<10}")
        print("-" * 70)
        for sev in severities:
            sev_pct = f"{int(round(sev * 100))}%"
            c4 = matrix_results[("initial", sev)]["conf"]
            c42 = matrix_results[("robuste", sev)]["conf"]
            diff = c42 - c4
            diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            c4_txt = f"{c4:.2f}" if c4 > 0 else "NON DÉTECTÉ"
            c42_txt = f"{c42:.2f}" if c42 > 0 else "NON DÉTECTÉ"
            print(f"{sev_pct:<12} | {c4_txt:<20} | {c42_txt:<20} | {diff_str:<10}")
        print("=" * 70)


def main():
    args = parse_args()
    model_resolved = resolve_model_path(args.model, SCRIPT_DIR)
    image_resolved = resolve_path(SCRIPT_DIR, args.image)
    output_resolved = resolve_output_dir(SCRIPT_DIR, args.output_dir)
    
    run_droplet_tests(
        model_path=model_resolved,
        image_path=image_resolved,
        output_dir=output_resolved,
        severities=args.severities,
        conf_thresh=args.conf,
        device=args.device,
        seed=args.seed,
        compare_mode=args.compare
    )


if __name__ == "__main__":
    main()
