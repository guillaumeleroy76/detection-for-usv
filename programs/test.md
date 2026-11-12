# Guide d'exécution des programmes - `detection-for-usv`

Ce document répertorie l'ensemble des scripts disponibles dans `programs/` et détaille comment choisir dynamiquement le modèle YOLO (`--model` / `-m`) et lancer les tests sans aucune erreur de chemin.

---

## 🎯 Sélection du modèle : Alias et Chemins

Tous les scripts qui utilisent un modèle YOLO supportent l'argument `--model` (ou son raccourci `-m`).

| Valeur passée à `--model` | Modèle chargé | Description |
|---|---|---|
| `best4` ou `best4.pt` | `best4/best4.pt` | Modèle initial de référence (images nominales non dégradées). |
| `best4_2` ou `best4_2.pt` | `best4/best4_2.pt` | Modèle robuste fine-tuné (avec 10 % de perturbations maritimes injectées). |
| `best5` (futur modèle) | `best5/best5.pt` ou `best5.pt` | Détection automatique si vous créez un dossier `best5/`. |
| Chemin relatif ou absolu | Ex: `../best4/best4.pt` | Résolution directe de tout fichier `.pt`. |
| *(Omission du paramètre)* | Modèle par défaut | Chaque script utilise une valeur par défaut fonctionnelle (`best4` ou `best4_2`). |

> **Note :** La détection du périphérique d'exécution est **automatique** : si votre carte graphique NVIDIA (ex: RTX 3050 Laptop GPU) avec CUDA est disponible, elle est utilisée (`device='0'`) ; sinon, le calcul bascule automatiquement sur le processeur (`device='cpu'`).

---

## 🛠️ Tableau récapitulatif des programmes

| Script | Rôle | Paramètre modèle |
|---|---|---|
| **`test_model.py`** | Évalue les métriques globales (mAP50, mAP50-95, P, R) sur le jeu de test de `Dataset_Global`. | `--model` (défaut: `best4`) |
| **`donnee_training.py`** | Trace et exporte les 4 courbes d'entraînement (pertes, mAP, P/R, val loss). | `--run` / `--model` (défaut: `after`) |
| **`yolo_detect.py`** | Détection d'objets sur image fixe, vidéo, caméra USB ou Raspberry Pi Camera. | `--model` (défaut: `best4`) |
| **`yolo_tracking_test.py`** | Suivi temporel d'embarcations (*multi-object tracking*) avec affichage vidéo live. | `--model` (défaut: `best4`) |
| **`tracking_test_json.py`** | Suivi temporel avec sérialisation de chaque observation en objet JSON horodaté. | `--model` (défaut: `best4`) |
| **`test_degradation_visual.py`** | Test qualitatif produisant des planches comparatives côte-à-côte (Clean vs Dégradé). | `--model` (défaut: `best4_2`) |
| **`test_degradation_benchmark.py`** | Benchmark quantitatif mesurant la dégradation du Rappel, Précision et Confiance. | `--model` (défaut: `best4`) |
| **`train_robust_model.py`** | Lance un fine-tuning à partir d'un modèle de départ sur le jeu enrichi. | `--weights` / `--model` (défaut: `best4`) |
| **`modify_dataset_robust.py`** | Applique les 5 transformations maritimes aléatoires sur 10 % du dataset d'entraînement. | *(Dataset)* |
| **`image_degradation.py`** | Moteur physique de simulation des dégradations maritimes (gouttes, sel, flou, etc.). | *(Moteur)* |

---

## 💻 Exemples détaillés de commandes (depuis `programs/`)

### 1. Évaluation des performances globales (`test_model.py`)
```bash
# Tester le modèle initial best4 sur le jeu de test :
python test_model.py --model best4

# Tester le modèle robuste best4_2 sur le jeu de test :
python test_model.py --model best4_2

# Spécifier explicitement le split ou le périphérique :
python test_model.py --model best4_2 --split test --device 0
```

---

### 2. Visualisation des courbes d'apprentissage (`donnee_training.py`)
```bash
# Tracer les courbes du fine-tuning robuste (best4_2 / after_transformations) :
python donnee_training.py --run after

# Tracer les courbes de l'entraînement initial (best4 / before_transformations) :
python donnee_training.py --run before

# Spécifier un fichier CSV personnalisé :
python donnee_training.py --csv "../best4/after_transformations/results_training_after_transformations/results.csv"
```

---

### 3. Détection d'objets (`yolo_detect.py`)
```bash
# Détection avec le modèle robuste best4_2 sur une photo de navire :
python yolo_detect.py --model best4_2 --source "../images and video/bateau_test.jpg" --thresh 0.5

# Détection avec le modèle initial best4 sur une vidéo :
python yolo_detect.py --model best4 --source "../images and video/modelisation_3d_boat.mp4" --thresh 0.5

# Détection sur caméra USB en direct (ex: index 0) :
python yolo_detect.py --model best4_2 --source usb0 --thresh 0.4
```

---

### 4. Tracking temps réel (`yolo_tracking_test.py` et `tracking_test_json.py`)
```bash
# Tracking vidéo avec affichage graphique :
python yolo_tracking_test.py --model best4_2
python yolo_tracking_test.py --model best4 --source "../images and video/scene-blender-01.mp4"

# Tracking avec émission du flux JSON horodaté (terminal) :
python tracking_test_json.py --model best4_2

# Mode sans interface graphique (idéal pour exécution embarquée basse consommation) :
python tracking_test_json.py --model best4_2 --no-show
```

---

### 5. Tests de dégradation d'images
```bash
# Test qualitatif visuel (planche côte-à-côte à 25%, 50%, 75% de sévérité) :
python test_degradation_visual.py --model best4_2
python test_degradation_visual.py --model best4

# Benchmark quantitatif sur un échantillon de 50 images :
python test_degradation_benchmark.py --model best4_2 --sample_size 50
python test_degradation_benchmark.py --model best4 --sample_size 50

# Benchmark complet sur l'intégralité du jeu de test (644 images) :
python test_degradation_benchmark.py --model best4_2 --sample_size 0
```

---

### 6. Fine-tuning d'un nouveau modèle (`train_robust_model.py`)
```bash
# Lancer un fine-tuning de 20 époques à partir des poids de best4 :
python train_robust_model.py --weights best4 --epochs 20 --batch 16 --name "best4_robust_run"

# Repartir d'un autre modèle existant :
python train_robust_model.py --weights best4_2 --epochs 10 --batch 16 --name "best4_v3_run"
```
