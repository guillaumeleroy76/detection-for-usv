

import os
import sys
import argparse
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import local du moteur de dégradation
from image_degradation import (
    load_image_safe,
    save_image_safe,
    degrade_image
)
from ultralytics import YOLO


import glob
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Test visuel de dégradation d'image pour YOLO USV")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4_2",
        help="Modèle YOLO à tester : alias ('best4', 'best4_2'), nom ou chemin (défaut: best4_2)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="images and video/bateau_test.jpg",
        help="Chemin vers l'image test (défaut: images and video/bateau_test.jpg)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results_degradation/visual",
        help="Dossier de sortie pour les comparaisons (défaut: results_degradation/visual)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Seuil de confiance minimal YOLO (défaut: 0.35)"
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="Niveaux de sévérité à tester (défaut: 0.25 0.50 0.75)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device : '0' (GPU CUDA) ou 'cpu' (défaut: auto-détection)"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str) -> str:
    """Résout le chemin d'un modèle YOLO (alias 'best4', 'best4_2', nom de fichier ou chemin)."""
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


def resolve_path(base_dir: str, rel_or_abs_path: str) -> str:
    """Résout un chemin d'image ou de dossier."""
    if os.path.isabs(rel_or_abs_path) and os.path.exists(rel_or_abs_path):
        return rel_or_abs_path
    
    candidates = [
        rel_or_abs_path,
        os.path.join(base_dir, rel_or_abs_path),
        os.path.join(os.path.dirname(base_dir), rel_or_abs_path),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), rel_or_abs_path),
        os.path.join(os.path.dirname(base_dir), "images and video", os.path.basename(rel_or_abs_path)),
        os.path.join(base_dir, "images and video", os.path.basename(rel_or_abs_path))
    ]
        
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
            
    return os.path.abspath(rel_or_abs_path)


def draw_detections(
    image: np.ndarray,
    detections: Any,
    class_names: Dict[int, str],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Dessine les boîtes de détection YOLO avec étiquettes et niveau de confiance.
    
    Returns:
        Tuple (image annotée, liste des détections extraites)
    """
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
        
        # Tracé rectangle boîte englobante
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label avec fond coloré pour lisibilité
        label = f"{cls_name}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Placer le label au-dessus ou en dedans si trop proche du bord
        y_text = max(y1, label_h + 5)
        cv2.rectangle(
            annotated,
            (x1, y_text - label_h - 4),
            (x1 + label_w + 4, y_text + baseline - 2),
            color,
            -1
        )
        # Texte en noir sur fond clair ou blanc sur fond foncé
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


def run_visual_tests(
    model_path: str,
    image_path: str,
    output_dir: str,
    severities: List[float],
    conf_thresh: float
):
    """Exécute la batterie de tests visuels et produit le rapport d'images."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print("🧪 USV YOLO - TESTS QUALITATIFS DE DÉGRADATION D'IMAGES")
    print("=" * 65)
    print(f"• Modèle YOLO       : {model_path}")
    print(f"• Image source      : {image_path}")
    print(f"• Seuil confiance   : {conf_thresh:.2f}")
    print(f"• Sévérités testées : {severities}")
    print(f"• Dossier de sortie : {output_dir}")
    print("-" * 65)

    # 1. Chargement modèle et image
    model = YOLO(model_path)
    orig_img = load_image_safe(image_path)
    
    # 2. Inférence de référence sur l'image originale
    clean_results = model(orig_img, conf=conf_thresh, verbose=False)[0]
    clean_annotated, clean_dets = draw_detections(orig_img, clean_results, model.names, color=(0, 220, 0))
    clean_conf_str = f"{clean_dets[0]['conf']:.2f}" if clean_dets else "0.00 (Non Détecté)"
    print(f"✅ Image Originale : {len(clean_dets)} détection(s) [Confiance max: {clean_conf_str}]")

    degradations = [
        ("gouttes", "Gouttes d'eau & Embruns"),
        ("pixels_noirs", "Pixels Noirs & Masquage"),
        ("flou", "Flou de Défocalisation"),
        ("resolution", "Baisse de Résolution"),
        ("brouillard", "Brume Marine & Perte de Contraste")
    ]

    summary_records = []
    matrix_images = {} # {(deg_key, sev): annotated_image}

    for deg_key, deg_label in degradations:
        print(f"\n🌊 Test de la dégradation : {deg_label} ({deg_key})")
        
        for sev in severities:
            # Application de la dégradation
            deg_img = degrade_image(orig_img, deg_key, severity=sev, seed=42)
            
            # Inférence YOLO
            deg_results = model(deg_img, conf=conf_thresh, verbose=False)[0]
            
            # Couleur: vert si détecté avec haute conf, orange si moyenne, rouge si non détecté
            if len(deg_results.boxes) > 0:
                top_conf = float(deg_results.boxes.conf.max())
                box_color = (0, 220, 0) if top_conf >= 0.70 else (0, 165, 255)
            else:
                top_conf = 0.0
                box_color = (0, 0, 255)
                
            deg_annotated, deg_dets = draw_detections(deg_img, deg_results, model.names, color=box_color)
            matrix_images[(deg_key, sev)] = deg_annotated
            
            status = f"✅ Détecté ({top_conf:.2f})" if len(deg_dets) > 0 else "❌ Non Détecté"
            print(f"   - Sévérité {sev*100:4.1f}% : {status}")
            
            summary_records.append({
                "degradation": deg_label,
                "key": deg_key,
                "severity": sev,
                "detected": len(deg_dets) > 0,
                "count": len(deg_dets),
                "top_conf": top_conf
            })

            # Génération d'une image comparative côte-à-côte (Original vs Dégradé)
            h, w = orig_img.shape[:2]
            header_h = 45
            comparison_canvas = np.zeros((h + header_h, w * 2 + 10, 3), dtype=np.uint8)
            comparison_canvas[:] = (35, 35, 35) # fond gris anthracite
            
            # Placement des images
            comparison_canvas[header_h:header_h + h, :w] = clean_annotated
            comparison_canvas[header_h:header_h + h, w + 10:w * 2 + 10] = deg_annotated
            
            # Titres
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                comparison_canvas,
                f"Original (Conf: {clean_conf_str})",
                (15, 30),
                font, 0.7, (240, 240, 240), 2, cv2.LINE_AA
            )
            
            status_text = f"Conf: {top_conf:.2f}" if len(deg_dets) > 0 else "NON DETECTE"
            header_deg_title = f"{deg_label} ({int(sev*100)}%) - {status_text}"
            cv2.putText(
                comparison_canvas,
                header_deg_title,
                (w + 25, 30),
                font, 0.65, (0, 220, 255) if len(deg_dets) > 0 else (100, 100, 255), 2, cv2.LINE_AA
            )

            pair_filename = f"comparaison_{deg_key}_sev{int(sev*100):02d}.jpg"
            pair_path = os.path.join(output_dir, pair_filename)
            save_image_safe(pair_path, comparison_canvas)

    # 3. Génération de la planche récapitulative globale (Figure Matplotlib)
    fig_rows = len(degradations)
    fig_cols = len(severities) + 1  # 1 col pour Clean + cols pour les sévérités
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 2.6 * fig_rows))
    
    if fig_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    for r_idx, (deg_key, deg_label) in enumerate(degradations):
        # Colonne 0 : Clean
        clean_rgb = cv2.cvtColor(clean_annotated, cv2.COLOR_BGR2RGB)
        axes[r_idx, 0].imshow(clean_rgb)
        axes[r_idx, 0].set_ylabel(deg_label, fontsize=10, fontweight="bold")
        if r_idx == 0:
            axes[r_idx, 0].set_title(f"Original\nConf: {clean_conf_str}", fontsize=11, fontweight="bold", color="green")
        axes[r_idx, 0].axis("off")
        
        # Colonnes sévérités
        for c_idx, sev in enumerate(severities):
            col_ax = axes[r_idx, c_idx + 1]
            img_annot = matrix_images[(deg_key, sev)]
            rgb_annot = cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB)
            col_ax.imshow(rgb_annot)
            
            rec = next(r for r in summary_records if r["key"] == deg_key and r["severity"] == sev)
            conf_val = rec["top_conf"]
            if rec["detected"]:
                title_color = "darkgreen" if conf_val >= 0.70 else "darkorange"
                title_txt = f"Sévérité {int(sev*100)}%\nConf: {conf_val:.2f}"
            else:
                title_color = "red"
                title_txt = f"Sévérité {int(sev*100)}%\nÉchec Détection"
                
            col_ax.set_title(title_txt, fontsize=10, color=title_color, fontweight="bold")
            col_ax.axis("off")
            
    plt.suptitle("Planche Comparative de Robustesse YOLO aux Dégradations Marines", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    summary_plot_path = os.path.join(output_dir, "planche_recapitulative_degradations.png")
    plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print("\n" + "=" * 65)
    print("📊 BILAN DES TESTS QUALITATIFS")
    print("=" * 65)
    print(f"✅ Images côte-à-côte générées dans : {output_dir}")
    print(f"✅ Planche matricielle globale sauvegardée : {summary_plot_path}")
    print("=" * 65)


if __name__ == "__main__":
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_resolved = resolve_model_path(args.model, script_dir)
    image_resolved = resolve_path(script_dir, args.image)
    output_resolved = resolve_path(script_dir, args.output_dir)
    
    run_visual_tests(
        model_path=model_resolved,
        image_path=image_resolved,
        output_dir=output_resolved,
        severities=args.severities,
        conf_thresh=args.conf
    )
