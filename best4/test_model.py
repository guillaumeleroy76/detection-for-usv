from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# 1. Charger ton meilleur modèle (mets à jour le chemin vers ton run si nécessaire)
model_path = "best4.pt"
print(f"🔄 Chargement du modèle : {model_path}")
model = YOLO(model_path)


yaml_path = "Dataset_Global/data.yaml"

print("🚀 Lancement de l'évaluation sur le jeu de test...")
metrics = model.val(
    data=yaml_path,
    split="test",
    imgsz=640,
    device="cpu"
)
# 3. Récupération et affichage textuel des métriques principales
map50_95 = metrics.box.map
map50 = metrics.box.map50
precision = metrics.box.p[0] if hasattr(metrics.box.p, "__len__") else metrics.box.p
recall = metrics.box.r[0] if hasattr(metrics.box.r, "__len__") else metrics.box.r

print("\n" + "="*40)
print("📊 RÉSULTATS DE L'ÉVALUATION DU TEST")
print("="*40)
print(f"• Précision (P)   : {precision:.4f}")
print(f"• Rappel (R)      : {recall:.4f}")
print(f"• mAP50           : {map50:.4f}")
print(f"• mAP50-95 global : {map50_95:.4f}")
print("="*40 + "\n")

# 4. Génération du graphique récapitulatif des performances
categories = ['Précision (P)', 'Rappel (R)', 'mAP50', 'mAP50-95']
values = [precision, recall, map50, map50_95]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(9, 6))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

bars = plt.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)

# Ajout des valeurs exactes au-dessus de chaque barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylim(0, 1.1)  # Les métriques s'échelonnent de 0 à 1
plt.title("Bilan des Performances sur le Jeu de Test Global", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Score", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Sauvegarde de la figure
output_img = "bilan_test_metriques.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ Graphique des tests sauvegardé sous : {output_img}")

# Affichage du graphique dans le notebook
plt.show()