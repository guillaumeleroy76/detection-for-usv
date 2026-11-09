"""
Script: test_degradation_benchmark.py
Description: Benchmark quantitatif de la robustesse de YOLO face aux dégradations d'images (Dataset de Test)
Projet: USV Cerema - IMT Atlantique
Auteur: Guillaume

Ce script évalue de manière rigoureuse et reproductible l'impact de dégradations
progressives (gouttes d'eau, masquage pixels noirs, flou, résolution, brouillard)
sur les performances du modèle YOLO (Rappel, Précision, F1-Score, Confiance Moyenne).
Il exporte les métriques en CSV et trace les courbes d'affaiblissement de performances.
"""

import os
import sys
import glob
import argparse
import csv
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_degradation import (
    load_image_safe,
    degrade_image
)
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark quantitatif de dégradation YOLO pour USV")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best4.pt",
        help="Chemin vers le modèle YOLO (.pt) (défaut: models/best4.pt)"
    )
    parser.add_argument(
        "--dataset_test",
        type=str,
        default="../Dataset_Global/test",
        help="Dossier du split test contenant 'images' et 'labels' (défaut: ../Dataset_Global/test)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_degradation/benchmark",
        help="Dossier où sauvegarder les résultats CSV et graphiques (défaut: results_degradation/benchmark)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Nombre d'images à échantillonner pour le benchmark (défaut: 50, mettre 0 pour toutes)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Seuil de confiance minimal YOLO (défaut: 0.35)"
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.50,
        help="Seuil IoU pour valider un Vrai Positif (défaut: 0.50)"
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        default=[0.0, 0.20, 0.40, 0.60, 0.80],
        help="Niveaux de sévérité à tester (défaut: 0.0 0.20 0.40 0.60 0.80)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire d'échantillonnage et de dégradation (défaut: 42)"
    )
    return parser.parse_args()


def resolve_path(base_dir: str, path_str: str) -> str:
    """Résout un chemin de façon portable."""
    if os.path.isabs(path_str):
        return path_str
    candidate = os.path.join(base_dir, path_str)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    parent_candidate = os.path.join(os.path.dirname(base_dir), path_str)
    if os.path.exists(parent_candidate):
        return os.path.abspath(parent_candidate)
    return os.path.abspath(path_str)


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Calcule l'Intersection over Union (IoU) entre deux boîtes englobantes [x1, y1, x2, y2].
    
    Complexité : O(1) temps et espace.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_ground_truth(label_path: str, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    """
    Charge les boîtes vérités terrain au format YOLO (classe, xc, yc, w, h)
    et les convertit en coordonnées pixels absolues [x1, y1, x2, y2].
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Format: cls xc yc w h
                _, xc, yc, bw, bh = map(float, parts[:5])
                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                x2 = (xc + bw / 2.0) * img_w
                y2 = (yc + bh / 2.0) * img_h
                boxes.append((x1, y1, x2, y2))
    return boxes


def evaluate_detections(
    predictions: List[Tuple[Tuple[float, float, float, float], float]],
    ground_truths: List[Tuple[float, float, float, float]],
    iou_thresh: float = 0.50
) -> Tuple[int, int, int, List[float]]:
    """
    Associe les prédictions YOLO aux vérités terrain par IoU maximale.
    
    Returns:
        Tuple (True Positives, False Positives, False Negatives, Confidences des TP)
    """
    matched_gt = set()
    tp = 0
    fp = 0
    tp_confs = []
    
    # Tri des prédictions par confiance décroissante
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    for pred_box, conf in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            tp_confs.append(conf)
        else:
            fp += 1
            
    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn, tp_confs


def run_benchmark(
    model_path: str,
    dataset_test_dir: str,
    output_dir: str,
    sample_size: int,
    conf_thresh: float,
    iou_thresh: float,
    severities: List[float],
    seed: int
):
    """Exécute l'évaluation benchmark multi-dégradations."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    
    images_dir = os.path.join(dataset_test_dir, "images")
    labels_dir = os.path.join(dataset_test_dir, "labels")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Dossier test/images introuvable : {images_dir}")
        
    all_img_files = glob.glob(os.path.join(images_dir, "*.*"))
    all_img_files = [f for f in all_img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    total_available = len(all_img_files)
    if total_available == 0:
        raise ValueError(f"Aucune image trouvée dans {images_dir}")
        
    if 0 < sample_size < total_available:
        selected_files = list(np.random.choice(all_img_files, size=sample_size, replace=False))
    else:
        selected_files = all_img_files
        sample_size = total_available

    print("=" * 70)
    print("🚀 USV YOLO - BENCHMARK QUANTITATIF DE ROBUSTESSE")
    print("=" * 70)
    print(f"• Modèle YOLO           : {model_path}")
    print(f"• Dataset Test          : {dataset_test_dir} ({sample_size}/{total_available} images)")
    print(f"• Seuil Confiance       : {conf_thresh:.2f}")
    print(f"• Seuil IoU             : {iou_thresh:.2f}")
    print(f"• Sévérités à évaluer   : {severities}")
    print(f"• Dossier Résultats     : {output_dir}")
    print("-" * 70)

    # Chargement modèle
    model = YOLO(model_path)

    # Pré-chargement des images et vérités terrain en mémoire pour maximiser la vitesse
    print("📥 Pré-chargement du jeu de test...")
    cached_dataset = []
    total_gt_boxes = 0
    
    for img_path in selected_files:
        try:
            img = load_image_safe(img_path)
            h, w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            gt_boxes = load_ground_truth(label_path, w, h)
            total_gt_boxes += len(gt_boxes)
            cached_dataset.append((img, gt_boxes))
        except Exception as e:
            continue
            
    print(f"✅ {len(cached_dataset)} images chargées ({total_gt_boxes} bateaux au total).")

    degradations = [
        ("gouttes", "Gouttes d'eau & Embruns"),
        ("pixels_noirs", "Pixels Noirs & Masquage"),
        ("flou", "Flou de Défocalisation"),
        ("resolution", "Baisse de Résolution"),
        ("brouillard", "Brume Marine / Contraste")
    ]

    all_results = [] # Liste de dictionnaires pour le CSV

    for deg_key, deg_label in degradations:
        print(f"\n🔍 Évaluation : {deg_label} ({deg_key})...")
        
        for sev in severities:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            all_confs = []
            
            for img_idx, (img, gt_boxes) in enumerate(cached_dataset):
                # Si sev == 0.0, image intacte sans altération
                if sev == 0.0:
                    eval_img = img
                else:
                    eval_img = degrade_image(img, deg_key, severity=sev, seed=seed + img_idx)
                
                # Inférence YOLO
                results = model(eval_img, conf=conf_thresh, verbose=False)[0]
                
                # Extraction des prédictions
                preds = []
                for box in results.boxes:
                    coords = tuple(box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    preds.append((coords, conf))
                    
                tp, fp, fn, confs = evaluate_detections(preds, gt_boxes, iou_thresh=iou_thresh)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_confs.extend(confs)

            # Calcul des métriques globales
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
            
            record = {
                "Degradation": deg_label,
                "Cle": deg_key,
                "Severite": sev,
                "Images": len(cached_dataset),
                "Bateaux_GT": total_tp + total_fn,
                "TP": total_tp,
                "FP": total_fp,
                "FN": total_fn,
                "Rappel": round(recall, 4),
                "Precision": round(precision, 4),
                "F1_Score": round(f1, 4),
                "Confiance_Moyenne": round(avg_conf, 4)
            }
            all_results.append(record)
            
            print(f"   - Sévérité {sev*100:4.1f}% | Rappel: {recall*100:5.1f}% | Précision: {precision*100:5.1f}% | Conf Moy: {avg_conf:.3f} | F1: {f1:.3f}")

    # 1. Sauvegarde CSV des résultats
    csv_path = os.path.join(output_dir, "benchmark_metrics.csv")
    fieldnames = [
        "Degradation", "Cle", "Severite", "Images", "Bateaux_GT",
        "TP", "FP", "FN", "Rappel", "Precision", "F1_Score", "Confiance_Moyenne"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n💾 Résultats bruts sauvegardés sous : {csv_path}")

    # 2. Génération des courbes Matplotlib de référence
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.28, wspace=0.22)
    
    colors = {
        "gouttes": "#1f77b4",       # Bleu
        "pixels_noirs": "#2ca02c",  # Vert
        "flou": "#d62728",          # Rouge
        "resolution": "#ff7f0e",    # Orange
        "brouillard": "#9467bd"     # Violet
    }
    
    markers = {
        "gouttes": "o",
        "pixels_noirs": "s",
        "flou": "^",
        "resolution": "D",
        "brouillard": "v"
    }

    metrics_config = [
        ("Rappel", "Rappel (Taux de Détection)", axes[0, 0], (0, 1.05)),
        ("Confiance_Moyenne", "Score de Confiance Moyen", axes[0, 1], (0, 1.05)),
        ("Precision", "Précision des Détections", axes[1, 0], (0, 1.05)),
        ("F1_Score", "F1-Score Global", axes[1, 1], (0, 1.05))
    ]

    for metric_key, metric_title, ax, ylim in metrics_config:
        for deg_key, deg_label in degradations:
            deg_rows = [r for r in all_results if r["Cle"] == deg_key]
            x_vals = [r["Severite"] * 100 for r in deg_rows]
            y_vals = [r[metric_key] for r in deg_rows]
            
            ax.plot(
                x_vals,
                y_vals,
                marker=markers[deg_key],
                linewidth=2.2,
                markersize=6.5,
                label=deg_label,
                color=colors[deg_key]
            )
            
        ax.set_title(metric_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Sévérité de la dégradation (%)", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(ylim)
        ax.set_xlim(-2, max([s * 100 for s in severities]) + 2)
        ax.grid(True, linestyle="--", alpha=0.6)
        # Seuil critique à 50%
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Seuil critique (50%)" if metric_key == "Rappel" else "")
        ax.legend(fontsize=9, loc="best")

    plt.suptitle("Impact des Dégradations Marines sur les Performances de YOLO (best4)", fontsize=15, fontweight="bold", y=0.98)
    
    plot_path = os.path.join(output_dir, "courbes_performances_degradation.png")
    plt.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"📈 Graphiques des courbes sauvegardés sous : {plot_path}")

    # 3. Synthèse des seuils de rupture pour le rapport
    print("\n" + "=" * 70)
    print("📋 SYNTHÈSE DES SEUILS DE RUPTURE (Rappel < 50%)")
    print("=" * 70)
    baseline_recall = next((r["Rappel"] for r in all_results if r["Severite"] == 0.0), 0.0)
    print(f"• Rappel Nominal (Image Parfaite) : {baseline_recall * 100:.1f}%\n")
    
    for deg_key, deg_label in degradations:
        deg_rows = [r for r in all_results if r["Cle"] == deg_key]
        breakdown_sev = None
        for r in deg_rows:
            if r["Rappel"] < 0.50:
                breakdown_sev = r["Severite"]
                break
        if breakdown_sev is not None:
            print(f"⚠️  {deg_label:<32} : Rupture critique dès {breakdown_sev*100:.0f}% de dégradation")
        else:
            min_recall = min(r["Rappel"] for r in deg_rows)
            print(f"✅  {deg_label:<32} : Robuste ! Reste au-dessus de 50% (min: {min_recall*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_resolved = resolve_path(script_dir, args.model)
    dataset_resolved = resolve_path(script_dir, args.dataset_test)
    output_resolved = resolve_path(script_dir, args.output_dir)
    
    run_benchmark(
        model_path=model_resolved,
        dataset_test_dir=dataset_resolved,
        output_dir=output_resolved,
        sample_size=args.sample_size,
        conf_thresh=args.conf,
        iou_thresh=args.iou_thresh,
        severities=args.severities,
        seed=args.seed
    )
