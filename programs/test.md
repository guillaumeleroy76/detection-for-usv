# Guide d'exécution des programmes - `detection-for-usv`

Ce document répertorie l'ensemble des scripts disponibles dans `programs/` et détaille comment les lancer sans erreur de chemin.

---

## 🛠️ Description des 10 programmes

| Script | Rôle |
|---|---|
| **`test_model.py`** | Évalue les performances globales du modèle (`best4.pt`) sur le split test de `Dataset_Global` (mAP50, mAP50-95, P, R). |
| **`donnee_training.py`** | Trace et exporte les 4 courbes d'entraînement à partir de `results.csv` (losses, mAP, précision/rappel). |
| **`yolo_detect.py`** | Détection d'objets sur image fixe, vidéo, caméra USB ou Raspberry Pi Camera. |
| **`yolo_tracking_test.py`** | Suivi temporel d'embarcations (*multi-object tracking*) avec affichage en direct. |
| **`tracking_test_json.py`** | Suivi temporel avec sérialisation de chaque observation en objet JSON horodaté. |
| **`image_degradation.py`** | Moteur physique de simulation des dégradations marines (gouttes d'eau, masquage/sel, flou, résolution, brume). |
| **`test_degradation_visual.py`** | Test qualitatif générant des planches comparatives côte-à-côte (Clean vs Dégradé) à différentes sévérités. |
| **`test_degradation_benchmark.py`** | Benchmark quantitatif mesurant la dégradation du Rappel, de la Précision et de la Confiance sur le dataset. |
| **`modify_dataset_robust.py`** | Applique les 5 transformations maritimes aléatoires sur 10 % du dataset d'entraînement chacune. |
| **`train_robust_model.py`** | Lance le fine-tuning à partir de `best4.pt` sur le jeu de données enrichi. |

---

## 💻 Commandes à lancer dans le terminal

Tous les scripts intègrent une **résolution automatique de chemins** : ils peuvent être lancés aussi bien depuis `detection-for-usv/programs/` que depuis la racine de `detection-for-usv/`.

### 1. Visualisation des courbes d'entraînement
```bash
python donnee_training.py
```

### 2. Évaluation des métriques nominales
```bash
python test_model.py
```

### 3. Détection simple sur image ou vidéo
```bash
python yolo_detect.py --model ../best4/best4.pt --source "../images and video/bateau_test.jpg" --thresh 0.5
python yolo_detect.py --model ../best4/best4.pt --source "../images and video/modelisation_3d_boat.mp4" --thresh 0.5
```

### 4. Tracking temps réel
```bash
# Avec affichage vidéo live :
python yolo_tracking_test.py

# Avec génération du flux JSON temps réel :
python tracking_test_json.py
```

### 5. Tests de dégradation d'images
```bash
# Test visuel unitaire :
python test_degradation_visual.py

# Benchmark quantitatif (échantillon de 50 images de test) :
python test_degradation_benchmark.py --sample_size 50

# Benchmark complet sur les 644 images du test set :
python test_degradation_benchmark.py --sample_size 0
```

### 6. Fine-tuning du modèle robuste
```bash
python train_robust_model.py --epochs 20 --batch 16
```
