"""
Script: test_confidence_benchmark.py
Description: Évaluation fine des performances (Précision, Rappel, F1-score) en faisant varier 
             le seuil de confiance de 40% à 90% sur le jeu de test de Dataset_Global.
             Génère des courbes comparatives haute résolution (best4 vs best4_2) et un tableau récapitulatif.
Auteur: Guillaume (Projet Cerema USV / IMT Atlantique)
"""

import os
import sys
import glob
import tempfile
import argparse

# Configuration des dossiers temporaires pour éviter les erreurs d'accès sous Windows
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark des performances YOLO selon le seuil de confiance (40% à 90%)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4_2",
        help="Modèle principal à tester : alias ('best4', 'best4_2') ou chemin (défaut: best4_2)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        default=True,
        help="Comparer automatiquement best4 (initial) et best4_2 (robuste) (défaut: True)"
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="Évaluer uniquement le modèle principal sans comparaison"
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
        choices=["test", "val"],
        help="Split à évaluer (défaut: test)"
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.40,
        help="Seuil de confiance minimal (défaut: 0.40)"
    )
    parser.add_argument(
        "--max-conf",
        type=float,
        default=0.90,
        help="Seuil de confiance maximal (défaut: 0.90)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Pas d'incrément de confiance (défaut: 0.05)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results_confidence",
        help="Dossier d'export des graphiques et CSV (défaut: results_confidence)"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout de manière robuste le chemin d'un modèle YOLO."""
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
            candidates.append(os.path.join(s_dir, "best_final", name))
            
    bare_name = os.path.splitext(os.path.basename(model_arg))[0]
    for s_dir in search_dirs:
        candidates.append(os.path.join(s_dir, bare_name, f"{bare_name}.pt"))
        if bare_name == "best4_2":
            candidates.append(os.path.join(s_dir, "best4", "best4_2.pt"))
            candidates.append(os.path.join(s_dir, "best_final", "best4_2.pt"))
            
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
    """Résout le chemin vers data.yaml de manière robuste."""
    if os.path.isabs(data_arg) and os.path.exists(data_arg):
        return os.path.abspath(data_arg)
        
    cwd = os.getcwd()
    parent_dir = os.path.dirname(base_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    root_proj = os.path.dirname(os.path.dirname(grandparent_dir)) if "cerema" in base_dir else grandparent_dir

    candidates = [
        data_arg,
        os.path.join(base_dir, data_arg),
        os.path.join(parent_dir, data_arg),
        os.path.join(grandparent_dir, data_arg),
        os.path.join(grandparent_dir, "Dataset_Global", "data.yaml"),
        os.path.join(parent_dir, "Dataset_Global", "data.yaml"),
        os.path.join(cwd, data_arg),
        os.path.join(cwd, "modèle pi cam", "yolo models lab", "Dataset_Global", "data.yaml"),
        os.path.join(cwd, "modèle pi cam", "best_final", "dataset", "Dataset_Global", "data.yaml"),
        "Dataset_Global/data.yaml",
        "data.yaml"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)

    for search_root in [cwd, grandparent_dir, root_proj]:
        if os.path.exists(search_root):
            matches = glob.glob(os.path.join(search_root, "**", "Dataset_Global", "data.yaml"), recursive=True)
            if matches:
                return os.path.abspath(matches[0])
                
    return os.path.abspath(data_arg)


def extract_curves_from_model(model_path: str, data_path: str, split: str = "test", device: str = "0"):
    """
    Exécute l'évaluation globale Ultralytics et extrait les courbes complètes P(c), R(c), F1(c).
    """
    print(f"🔄 Évaluation du modèle {os.path.basename(model_path)} sur split '{split}'...")
    model = YOLO(model_path)
    metrics = model.val(data=data_path, split=split, device=device, verbose=False)
    
    # Extraire les courbes depuis metrics.box.curves_results
    # 0: Recall vs Precision, 1: Conf vs F1, 2: Conf vs Precision, 3: Conf vs Recall
    curves = metrics.box.curves_results
    f1_curve = curves[1]
    p_curve = curves[2]
    r_curve = curves[3]
    pr_curve = curves[0]
    
    conf_x = f1_curve[0]
    f1_y = f1_curve[1][0]
    p_y = p_curve[1][0]
    r_y = r_curve[1][0]
    
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    
    return {
        "model_path": model_path,
        "name": os.path.splitext(os.path.basename(model_path))[0],
        "conf_x": conf_x,
        "f1_y": f1_y,
        "p_y": p_y,
        "r_y": r_y,
        "recall_pr": pr_curve[0],
        "precision_pr": pr_curve[1][0],
        "map50": map50,
        "map50_95": map50_95
    }


def sample_metrics_at_thresholds(eval_dict: dict, thresholds: list) -> pd.DataFrame:
    """Échantillonne les métriques aux seuils de confiance demandés."""
    conf_x = eval_dict["conf_x"]
    p_y = eval_dict["p_y"]
    r_y = eval_dict["r_y"]
    f1_y = eval_dict["f1_y"]
    
    rows = []
    for c in thresholds:
        idx = int(np.argmin(np.abs(conf_x - c)))
        p_val = float(p_y[idx])
        r_val = float(r_y[idx])
        f1_val = float(f1_y[idx])
        rows.append({
            "Modele": eval_dict["name"],
            "Confiance": round(c, 2),
            "Precision": p_val,
            "Rappel": r_val,
            "F1_Score": f1_val
        })
    return pd.DataFrame(rows)


def plot_single_model_curves(eval_data: dict, out_path: str):
    """Génère le graphique complet Précision / Rappel / F1 pour un modèle."""
    conf_x = eval_data["conf_x"]
    p_y = eval_data["p_y"]
    r_y = eval_data["r_y"]
    f1_y = eval_data["f1_y"]
    
    best_idx = int(np.argmax(f1_y))
    opt_conf = float(conf_x[best_idx])
    opt_f1 = float(f1_y[best_idx])
    opt_p = float(p_y[best_idx])
    opt_r = float(r_y[best_idx])
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Masque pour la zone de confiance d'intérêt [0.30, 0.95]
    mask = (conf_x >= 0.20) & (conf_x <= 0.95)
    
    plt.plot(conf_x[mask], p_y[mask] * 100, label="Précision (P) : Moins de faux positifs", color="#10b981", lw=2.5)
    plt.plot(conf_x[mask], r_y[mask] * 100, label="Rappel (R) : Moins de bateaux manqués", color="#3b82f6", lw=2.5)
    plt.plot(conf_x[mask], f1_y[mask] * 100, label="F1-Score (Compromis harmonique)", color="#f59e0b", lw=3.0)
    
    # Ligne verticale seuil optimal
    plt.axvline(x=opt_conf, color="#ef4444", linestyle="--", lw=1.8, alpha=0.8)
    plt.plot(opt_conf, opt_f1 * 100, marker="*", color="#ef4444", markersize=14, label=f"Seuil optimal F1 ({opt_conf*100:.1f}%)")
    
    # Boîte d'annotation du seuil optimal
    text_annot = (
        f"Point Optimal :\n"
        f"• Seuil Confiance : {opt_conf*100:.1f} %\n"
        f"• F1-Score max   : {opt_f1*100:.1f} %\n"
        f"• Précision      : {opt_p*100:.1f} %\n"
        f"• Rappel         : {opt_r*100:.1f} %"
    )
    plt.annotate(
        text_annot,
        xy=(opt_conf, opt_f1 * 100),
        xytext=(opt_conf + 0.08, opt_f1 * 100 - 15),
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#ef4444", lw=1.5, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5),
        fontsize=10,
        fontweight="bold"
    )
    
    # Zone recommandée pour l'USV Cerema (0.55 à 0.65)
    plt.axvspan(0.55, 0.65, color="#10b981", alpha=0.10, label="Zone recommandée USV (anti-faux positifs)")
    
    plt.title(f"Évolution des Performances selon le Seuil de Confiance — Modèle {eval_data['name']}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Seuil de Confiance (Confidence Threshold)", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.xlim(0.30, 0.92)
    plt.ylim(10, 105)
    plt.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Graphique généré : {out_path}")


def plot_comparative_curves(eval1: dict, eval2: dict, out_path: str):
    """Génère un grand panneau comparatif 1x3 entre best4 et best4_2."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    mask1 = (eval1["conf_x"] >= 0.35) & (eval1["conf_x"] <= 0.90)
    mask2 = (eval2["conf_x"] >= 0.35) & (eval2["conf_x"] <= 0.90)
    
    x1 = eval1["conf_x"][mask1] * 100
    x2 = eval2["conf_x"][mask2] * 100
    
    # 1. Précision
    ax = axes[0]
    ax.plot(x1, eval1["p_y"][mask1] * 100, label=f"{eval1['name']} (Initial)", color="#64748b", lw=2, linestyle="--")
    ax.plot(x2, eval2["p_y"][mask2] * 100, label=f"{eval2['name']} (Robuste)", color="#10b981", lw=2.8)
    ax.set_title("Précision (%) — Capacité à éviter les faux positifs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Seuil de Confiance (%)", fontsize=10)
    ax.set_ylabel("Précision (%)", fontsize=10)
    ax.set_ylim(70, 102)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Rappel
    ax = axes[1]
    ax.plot(x1, eval1["r_y"][mask1] * 100, label=f"{eval1['name']} (Initial)", color="#64748b", lw=2, linestyle="--")
    ax.plot(x2, eval2["r_y"][mask2] * 100, label=f"{eval2['name']} (Robuste)", color="#3b82f6", lw=2.8)
    ax.set_title("Rappel (%) — Capacité à détecter tous les bateaux", fontsize=11, fontweight="bold")
    ax.set_xlabel("Seuil de Confiance (%)", fontsize=10)
    ax.set_ylabel("Rappel (%)", fontsize=10)
    ax.set_ylim(0, 85)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # 3. F1-Score
    ax = axes[2]
    ax.plot(x1, eval1["f1_y"][mask1] * 100, label=f"{eval1['name']} (Initial)", color="#64748b", lw=2, linestyle="--")
    ax.plot(x2, eval2["f1_y"][mask2] * 100, label=f"{eval2['name']} (Robuste)", color="#f59e0b", lw=3.0)
    
    opt_idx2 = int(np.argmax(eval2["f1_y"]))
    opt_conf2 = float(eval2["conf_x"][opt_idx2]) * 100
    opt_f1_2 = float(eval2["f1_y"][opt_idx2]) * 100
    ax.plot(opt_conf2, opt_f1_2, marker="*", color="#ef4444", markersize=13, label=f"Max F1 {eval2['name']} ({opt_conf2:.1f}%)")
    
    ax.set_title("F1-Score (%) — Compromis global d'efficacité", fontsize=11, fontweight="bold")
    ax.set_xlabel("Seuil de Confiance (%)", fontsize=10)
    ax.set_ylabel("F1-Score (%)", fontsize=10)
    ax.set_ylim(0, 85)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle(
        f"Comparatif des Performances selon la Confiance : {eval1['name']} vs {eval2['name']} (Jeu de test)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Graphique comparatif généré : {out_path}")


def main():
    args = parse_args()
    
    # Résolution des chemins
    model_path = resolve_model_path(args.model)
    data_path = resolve_data_path(args.data)
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable : {args.model}")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"❌ Erreur : Dataset data.yaml introuvable : {args.data}")
        sys.exit(1)
        
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(SCRIPT_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    print("=" * 75)
    print("📈 BENCHMARK DES PERFORMANCES SELON LE SEUIL DE CONFIANCE (40% - 90%)")
    print("=" * 75)
    print(f"• Modèle cible      : {model_path}")
    print(f"• Dataset YAML      : {data_path}")
    print(f"• Split             : {args.split}")
    print(f"• Plage de seuils   : {args.min_conf*100:.0f}% à {args.max_conf*100:.0f}% (pas: {args.step*100:.0f}%)")
    print(f"• Mode comparaison  : {'Oui (best4 vs best4_2)' if args.compare else 'Non'}")
    print(f"• Périphérique      : {device} ({'GPU CUDA' if device == '0' else 'CPU'})")
    print(f"• Dossier de sortie : {out_dir}")
    print("-" * 75)
    
    # 1. Évaluation du modèle principal
    eval_main = extract_curves_from_model(model_path, data_path, split=args.split, device=device)
    
    # 2. Liste des seuils cibles
    thresholds = [round(x, 2) for x in np.arange(args.min_conf, args.max_conf + 1e-4, args.step)]
    df_main = sample_metrics_at_thresholds(eval_main, thresholds)
    
    # 3. Évaluation comparative si demandée
    eval_base = None
    if args.compare:
        base_name = "best4" if eval_main["name"] != "best4" else "best4_2"
        base_path = resolve_model_path(base_name)
        if os.path.exists(base_path):
            eval_base = extract_curves_from_model(base_path, data_path, split=args.split, device=device)
            df_base = sample_metrics_at_thresholds(eval_base, thresholds)
        else:
            print(f"⚠️ Avertissement : Modèle comparatif {base_name} introuvable.")
            
    # 4. Affichage du tableau récapitulatif
    print("\n" + "=" * 75)
    print("📋 TABLEAU COMPARATIF DES MÉTRIQUES (40% À 90%)")
    print("=" * 75)
    
    if eval_base is not None:
        header = f"| Seuil Conf | Précision ({eval_base['name']}) | Précision ({eval_main['name']}) | Rappel ({eval_base['name']}) | Rappel ({eval_main['name']}) | F1-Score ({eval_main['name']}) |"
        sep = "|:---:|:---:|:---:|:---:|:---:|:---:|"
        print(header)
        print(sep)
        
        comparison_rows = []
        for c in thresholds:
            r_main = df_main[df_main["Confiance"] == c].iloc[0]
            r_base = df_base[df_base["Confiance"] == c].iloc[0]
            
            p_gain = (r_main["Precision"] - r_base["Precision"]) * 100
            r_gain = (r_main["Rappel"] - r_base["Rappel"]) * 100
            
            row_str = (
                f"| **{c*100:2.0f} %** | "
                f"{r_base['Precision']*100:5.2f} % | "
                f"**{r_main['Precision']*100:5.2f} %** ({'+' if p_gain>=0 else ''}{p_gain:.2f}%) | "
                f"{r_base['Rappel']*100:5.2f} % | "
                f"**{r_main['Rappel']*100:5.2f} %** ({'+' if r_gain>=0 else ''}{r_gain:.2f}%) | "
                f"**{r_main['F1_Score']*100:5.2f} %** |"
            )
            print(row_str)
            comparison_rows.append({
                "Confiance": c,
                f"Precision_{eval_base['name']}": r_base["Precision"],
                f"Precision_{eval_main['name']}": r_main["Precision"],
                f"Rappel_{eval_base['name']}": r_base["Rappel"],
                f"Rappel_{eval_main['name']}": r_main["Rappel"],
                f"F1_{eval_base['name']}": r_base["F1_Score"],
                f"F1_{eval_main['name']}": r_main["F1_Score"]
            })
            
        csv_path = os.path.join(out_dir, f"comparatif_confiance_{eval_base['name']}_vs_{eval_main['name']}.csv")
        pd.DataFrame(comparison_rows).to_csv(csv_path, index=False, sep=";")
        print(f"\n💾 Tableau exporté en CSV : {csv_path}")
        
    else:
        print(df_main.to_markdown(index=False))
        csv_path = os.path.join(out_dir, f"metrics_confiance_{eval_main['name']}.csv")
        df_main.to_csv(csv_path, index=False, sep=";")
        print(f"\n💾 Données exportées en CSV : {csv_path}")

    # 5. Seuil optimal F1
    best_idx = int(np.argmax(eval_main["f1_y"]))
    opt_conf = float(eval_main["conf_x"][best_idx])
    print("-" * 75)
    print(f"🎯 RECOMMANDATION OPÉRATIONNELLE POUR L'USV :")
    print(f"• Seuil optimal théorique (F1 max) : {opt_conf*100:.1f} % (F1 = {eval_main['f1_y'][best_idx]*100:.2f} %)")
    print(f"• Seuil recommandé anti-faux positifs : 60 % (Précision = {df_main[df_main['Confiance']==0.60]['Precision'].values[0]*100:.2f} %, Rappel = {df_main[df_main['Confiance']==0.60]['Rappel'].values[0]*100:.2f} %)")
    print("=" * 75)

    # 6. Tracé des graphiques haute résolution
    out_graph_single = os.path.join(out_dir, f"courbes_confiance_{eval_main['name']}.png")
    plot_single_model_curves(eval_main, out_graph_single)
    
    if eval_base is not None:
        out_graph_comp = os.path.join(out_dir, f"comparatif_confiance_{eval_base['name']}_vs_{eval_main['name']}.png")
        plot_comparative_curves(eval_base, eval_main, out_graph_comp)

    print("\n✅ Benchmark terminé avec succès.")


if __name__ == "__main__":
    main()
