# AI Vision for USV Platform (`detection-for-usv`)

Système de vision par ordinateur embarqué pour drone de surface (USV - Unmanned Surface Vehicle) développé dans le cadre du projet Cerema / IMT Atlantique.

Le système assure la veille visuelle autonome basse consommation via caméra et modèle YOLO optimisé, permettant la détection et le suivi (*tracking*) de navires avant d'activer les modules experts d'analyse approfondie du flux vidéo.

---

## 📁 Arborescence réorganisée du répertoire

```text
detection-for-usv/
├── best4/                                  # Modèle de référence et comparaisons avant/après
│   ├── after_transformations/              # Échantillons et résultats post-transformations
│   │   └── echantillon_images_modifiees.png# Planche visuelle des altérations maritimes injectées (10% par type)
│   ├── before_transformations/             # Données et benchmarks initiaux sur image nominale
│   │   ├── results_training_before_transformations/ # Données du run d'origine (results.csv, BoxF1, loss)
│   │   ├── benchmark_metrics.csv           # Données brutes du benchmark initial de dégradation
│   │   ├── bilan_test_metriques.png        # Barres récapitulatives initiales (P, R, mAP50, mAP50-95)
│   │   ├── courbes_performances_degradation.png # Graphique 4-en-1 de chute des scores
│   │   └── planche_recapitulative_degradations.png # Matrice visuelle originale des dégradations
│   ├── best4.pt                            # Poids du modèle YOLOv11 fine-tuné (référence)
│   └── dataset_modification_manifest.json  # Traçabilité complète des 4 545 images modifiées
├── images and video/                       # Jeux de médias de test (images fixes et flux vidéo)
│   ├── bateau_test.jpg                     # Photo réelle de navire pour inférence unitaire
│   ├── baleine.png                         # Cas négatif / hors-domaine
│   ├── modelisation_3d_boat.mp4            # Vidéo de test de tracking (animation 3D)
│   └── scene-blender-01.mp4                # Simulation réaliste Blender (houle et clapot)
└── programs/                               # Ensemble des scripts exécutables
    ├── donnee_training.py                  # Trace les courbes d'apprentissage à partir de results.csv
    ├── image_degradation.py                # Moteur physique de dégradation maritime (gouttes, sel, flou...)
    ├── modify_dataset_robust.py            # Modifie 10% du dataset d'entraînement par transformation
    ├── test_degradation_benchmark.py       # Benchmark quantitatif sur Dataset_Global/test
    ├── test_degradation_visual.py          # Test qualitatif produisant des planches côte-à-côte
    ├── test_model.py                       # Évaluation standard du modèle sur Dataset_Global
    ├── tracking_test_json.py               # Tracking temps réel avec export JSON structuré
    ├── train_robust_model.py               # Fine-tuning à partir de best4.pt sur dataset augmenté
    ├── yolo_detect.py                      # Détection unitaire (image, vidéo, caméra USB, Picamera)
    ├── yolo_tracking_test.py               # Tracking temps réel avec affichage vidéo
    ├── test.md                             # Guide d'exécution détaillé
    └── results_degradation/                # Sorties des bancs de test
        ├── benchmark/                      # benchmark_metrics.csv & courbes_performances_degradation.png
        └── visual/                         # Planches et comparaisons visuelles unitaires
```

---

## 📂 Détail des dossiers

### 1. `best4/` — Modèle de référence et analyses comparatives
Ce dossier regroupe tous les artefacts du modèle principal :
* **`best4.pt`** : Les poids de référence actuels (5.5 Mo).
* **`before_transformations/`** : L'état initial du modèle avant tout ajout de données dégradées :
  * Les courbes d'entraînement de base (`results_training_before_transformations/`).
  * Les métriques et courbes d'affaiblissement face aux dégradations (`courbes_performances_degradation.png`).
* **`after_transformations/`** : L'état du jeu d'entraînement augmenté avec 10 % de perturbations marines (gouttes, sel, flou, résolution, brume) visualisable dans `echantillon_images_modifiees.png`.
* **`dataset_modification_manifest.json`** : Registre horodaté listant précisément chaque image modifiée et son niveau de sévérité.

---

### 2. `images and video/` — Médias de test
* **`bateau_test.jpg`** : Photographie de test nominale.
* **`baleine.png`** : Test hors-distribution.
* **`modelisation_3d_boat.mp4`** & **`scene-blender-01.mp4`** : Vidéos pour tester le maintien des identifiants temporels (*track IDs*).

---

### 3. `programs/` — Tous les programmes et bancs d'essai
Tous les scripts sont désormais centralisés ici et disposent d'une **résolution automatique de chemins** (ils fonctionnent quel que soit le dossier depuis lequel ils sont lancés).

---

## 🚀 Guide des commandes (depuis `programs/`)

```bash
# 1. Évaluer les métriques du modèle sur le jeu de test propre
python test_model.py

# 2. Tracer les courbes d'entraînement de best4 (pertes, mAP, précision/rappel)
python donnee_training.py

# 3. Tester la détection sur une image ou une vidéo
python yolo_detect.py --model ../best4/best4.pt --source "../images and video/bateau_test.jpg" --thresh 0.5

# 4. Lancer le tracking vidéo en direct
python yolo_tracking_test.py

# 5. Lancer le tracking avec génération du flux JSON temps réel
python tracking_test_json.py

# 6. Exécuter le test visuel de dégradation (génère les planches côte-à-côte)
python test_degradation_visual.py

# 7. Exécuter le benchmark quantitatif (ex: 50 images du test set)
python test_degradation_benchmark.py --sample_size 50

# 8. Lancer le fine-tuning sur le dataset augmenté pour créer le nouveau modèle
python train_robust_model.py --epochs 20 --batch 16
```
