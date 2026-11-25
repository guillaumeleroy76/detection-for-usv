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
| **`live_pi_camera.py`** | Streaming direct de la caméra Raspberry Pi sur le PC (OpenCV, FFplay, détection YOLO temps réel). | `--model` / `--yolo` / `--ffplay` |
| **`test_model.py`** | Évalue les métriques globales (mAP50, mAP50-95, P, R) sur le jeu de test de `Dataset_Global`. | `--model` (défaut: `best4`) |
| **`test_confidence_benchmark.py`** | Benchmark et courbes des performances (Précision, Rappel, F1) selon le seuil de confiance (40% à 90%) avec comparatif `best4` vs `best4_2`. | `--model` / `--compare` / `--min-conf` / `--max-conf` |
| **`donnee_training.py`** | Trace et exporte les 4 courbes d'entraînement (pertes, mAP, P/R, val loss). | `--run` / `--model` (défaut: `after`) |
| **`yolo_detect.py`** | Détection d'objets sur image fixe, vidéo, caméra USB ou Raspberry Pi Camera. | `--model` (défaut: `best4`) |
| **`yolo_tracking_test.py`** | Suivi temporel d'embarcations (*multi-object tracking*) avec affichage vidéo live. | `--model` (défaut: `best4`) |
| **`tracking_test_json.py`** | Suivi temporel avec validation spatio-temporelle (images consécutives, IoU, distance centroïdes) et export JSON filtré. | `--model` / `--min-consecutive` / `--emit-mode` |
| **`test_expert_trigger.py`** | Teste le déclencheur d'experts sur vidéo ou dossier d'images séquentielles (alerte console & HUD dès 5 images d'affilée, rejet des vagues). | `--source` / `--min-consecutive` / `--save` |
| **`test_degradation_visual.py`** | Test qualitatif produisant des planches comparatives côte-à-côte (Clean vs Dégradé). | `--model` (défaut: `best4_2`) |
| **`test_degradation_gouttes.py`** | Test visuel spécialisé sur les gouttes d'eau (sévérité jusqu'à 100%) avec mode comparatif. | `--model` / `--compare` |
| **`test_degradation_benchmark.py`** | Benchmark quantitatif mesurant la dégradation du Rappel, Précision et Confiance. | `--model` (défaut: `best4`) |
| **`train_robust_model.py`** | Lance un fine-tuning à partir d'un modèle de départ sur le jeu enrichi. | `--weights` / `--model` (défaut: `best4`) |
| **`modify_dataset_robust.py`** | Applique les 5 transformations maritimes aléatoires sur 10 % du dataset d'entraînement. | *(Dataset)* |
| **`image_degradation.py`** | Moteur physique de simulation des dégradations maritimes (gouttes, sel, flou, etc.). | *(Moteur)* |

---

## 💻 Exemples détaillés de commandes (depuis `programs/`)

### 1. Visualisation directe de la caméra Raspberry Pi sur le PC (`live_pi_camera.py`)
```bash
# Lancement direct du flux caméra (fenêtre interactive OpenCV) :
python live_pi_camera.py

# Si connecté par câble Ethernet direct sans Wi-Fi :
python live_pi_camera.py --host 192.168.137.200

# Lancer avec détection d'embarcations YOLO (best4_2) exécutée en direct sur le GPU du PC :
python live_pi_camera.py --yolo --model best4_2

# Lancer en mode FFplay (ultra basse latence) :
python live_pi_camera.py --ffplay

# Raccourcis clavier dans la fenêtre vidéo OpenCV :
# [Q] ou [Échap] : Quitter et libérer le capteur sur le Pi
# [S]            : Prendre une photo instantanée (.jpg)
# [R]            : Démarrer / Arrêter l'enregistrement vidéo (.avi)
# [D]            : Activer / Désactiver la détection YOLO à la volée
```

---

### 2. Évaluation des performances globales (`test_model.py`)
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

### 4. Tracking temps réel & validation spatio-temporelle (`yolo_tracking_test.py` et `tracking_test_json.py`)
```bash
# Tracking vidéo standard avec affichage graphique :
python yolo_tracking_test.py --model best4_2
python yolo_tracking_test.py --model best4 --source "../images and video/scene-blender-01.mp4"

# Tracking avec validation spatio-temporelle (filtre anti faux positifs) :
# Requiert 3 images consécutives avec continuité spatiale (même objet) avant d'émettre le JSON :
python tracking_test_json.py --model best4_2 --min-consecutive 3

# Émission d'un événement JSON unique à chaque nouveau bateau confirmé :
python tracking_test_json.py --model best4_2 --min-consecutive 5 --emit-mode event

# Mode embarqué sans affichage (headless) sur GPU ou CPU :
python tracking_test_json.py --model best4_2 --min-consecutive 3 --emit-mode confirmed --no-show

# Ajustement de la tolérance spatiale (pixels max entre 2 centroïdes consécutifs) et du recouvrement IoU :
python tracking_test_json.py --min-consecutive 3 --max-dist 150 --min-iou 0.2

# Test du déclencheur d'experts sur vidéo avec alertes console claires et bandeau visuel HUD (défaut N=5) :
python test_expert_trigger.py --source "images and video/modelisation_3d_boat.mp4" --min-consecutive 5

# Test sur un dossier d'images séquentielles (ex: frames extraites) :
python test_expert_trigger.py --source "mon_dossier_images/" --min-consecutive 5

# Exporter une vidéo de démonstration MP4 annotée avec les alertes visuelles :
python test_expert_trigger.py --source "images and video/modelisation_3d_boat.mp4" --save "demo_experts_declenches.mp4"
```

---

### 5. Tests de dégradation d'images
```bash
# Test visuel unitaire généraliste (5 dégradations) :
python test_degradation_visual.py --model best4_2
python test_degradation_visual.py --model best4

# Test visuel spécifique aux gouttes d'eau et embruns (jusqu'à 100% de sévérité) :
python test_degradation_gouttes.py --model best4_2

# Test comparatif côte-à-côte best4 vs best4_2 face aux gouttes (100%) :
python test_degradation_gouttes.py --compare

# Personnaliser les sévérités de gouttes à tester (ex: 25%, 50%, 75%, 100%) :
python test_degradation_gouttes.py --severities 0.25 0.50 0.75 1.00

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

---

### 7. Benchmark de sensibilité au seuil de confiance (`test_confidence_benchmark.py`)
```bash
# Lancer le benchmark comparatif automatique (best4 vs best4_2) de 40% à 90% (pas de 5%) :
python test_confidence_benchmark.py --model best4_2

# Évaluer uniquement un modèle spécifique sans comparaison :
python test_confidence_benchmark.py --model best4_2 --no-compare

# Personnaliser la plage de confiance (ex: de 30% à 95% par pas de 5%) :
python test_confidence_benchmark.py --model best4_2 --min-conf 0.30 --max-conf 0.95 --step 0.05

# Spécifier un autre dossier de sortie pour les graphiques et le tableau CSV :
python test_confidence_benchmark.py --output-dir mon_dossier_resultats
```
