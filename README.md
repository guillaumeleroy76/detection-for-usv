# AI Vision for USV Platform (`detection-for-usv`)

Système de vision par ordinateur embarqué pour drone de surface (USV - Unmanned Surface Vehicle) développé dans le cadre du projet Cerema / IMT Atlantique.

Le système assure la veille visuelle autonome basse consommation via caméra et modèle YOLO optimisé, permettant la détection et le suivi (*tracking*) de navires avant d'activer les modules experts d'analyse approfondie du flux vidéo.

---

## 📁 Arborescence du répertoire

```text
detection-for-usv/
├── best4/                  # Résultats d'entraînement et métriques du modèle de référence best4
│   ├── results_training/   # Courbes d'entraînement (BoxF1, BoxPR, BoxP, BoxR, loss, CSV)
│   ├── runs/               # Sorties de validation Ultralytics (matrices de confusion, prédictions)
│   ├── best4.pt            # Poids du modèle YOLOv11 fine-tuné
│   ├── bilan_test_metriques.png
│   ├── donnee_training.py  # Script d'analyse des métriques d'entraînement
│   └── test_model.py       # Script d'évaluation sur le jeu de test
├── images and video/       # Jeux de médias de test (images fixes et flux vidéo)
│   ├── bateau_test.jpg     # Image réelle de navire pour inférence unitaire
│   ├── baleine.png         # Image de test hors-domaine
│   ├── modelisation_3d_boat.mp4 # Vidéo de test de tracking (animation 3D)
│   └── scene-blender-01.mp4     # Vidéo de test réaliste issue de simulation Blender
├── models/                 # Historique des poids des modèles YOLO entraînés
│   ├── best2.pt            # Modèle YOLO v2 (5.5 Mo)
│   ├── best3.pt            # Modèle YOLO v3 grande capacité (26.6 Mo)
│   └── best4.pt            # Meilleur modèle actuel (5.5 Mo, classe unique 'boat')
└── programs/               # Scripts d'inférence, tracking et bancs d'essai de dégradation
    ├── yolo_detect.py      # Détection simple image/vidéo/caméra USB/Picamera
    ├── yolo_tracking_test.py # Tracking multi-objets en direct avec identifiants stables
    ├── tracking_test_json.py # Tracking avec export JSON temps réel (t, id_piste, bbox, conf)
    ├── test_model.py       # Validation rapide des métriques (mAP50, mAP50-95, P, R)
    ├── image_degradation.py # Moteur physique de dégradation (gouttes, sel, flou, résolution, brume)
    ├── test_degradation_visual.py # Test qualitatif générant des planches comparatives côte-à-côte
    ├── test_degradation_benchmark.py # Benchmark quantitatif (Rappel, Précision, seuils de rupture)
    ├── test.md             # Guide d'exécution détaillé des programmes
    └── results_degradation/# Résultats générés par les tests de dégradation
        ├── visual/         # Comparaisons visuelles et planche matricielle globale
        └── benchmark/      # benchmark_metrics.csv et courbes_performances_degradation.png
```

---

## 📂 Contenu détaillé de chaque sous-répertoire

### 1. `best4/` — Modèle de référence et historique de validation
Contient les artefacts de validation et d'analyse du meilleur modèle actuel :
* **`best4.pt`** : Poids entraînés optimisés pour la détection navale.
* **`results_training/`** : Graphiques générés lors de l'entraînement (`BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `courbes_entrainement.png`) et le fichier de suivi `results.csv`.
* **`runs/detect/val-3/`** : Résultats bruts du test YOLO : matrices de confusion (`confusion_matrix.png`, `confusion_matrix_normalized.png`) et lots d'images prédites (`val_batch*_pred.jpg`).
* **`bilan_test_metriques.png`** : Diagramme en barres récapitulatif des 4 métriques clés (Précision, Rappel, mAP50, mAP50-95).
* **`test_model.py` & `donnee_training.py`** : Scripts utilisés pour évaluer le modèle et tracer les graphiques d'entraînement.

---

### 2. `images and video/` — Médias de test et de démonstration
Ressources multimédias utilisées pour tester les algorithmes en local sans caméra connectée :
* **`bateau_test.jpg`** : Photographie de navire en mer, utilisée comme cas de test nominal pour la détection et les tests de dégradation.
* **`baleine.png`** : Cas négatif / hors-domaine pour tester l'absence de fausse alerte.
* **`modelisation_3d_boat.mp4`** : Vidéo de synthèse permettant de valider le maintien des identifiants de suivi (*track IDs*) dans le temps.
* **`scene-blender-01.mp4`** : Séquence vidéo haute définition simulant un environnement maritime avec houle et déplacement réaliste.

---

### 3. `models/` — Archive des poids YOLO
Centralise les différentes itérations de modèles entraînés :
* **`best2.pt`** : Modèle léger entraîné sur le premier jeu de données Roboflow Boat (5.5 Mo).
* **`best3.pt`** : Modèle de plus grande taille (26.6 Mo).
* **`best4.pt`** : Modèle léger de référence (5.5 Mo), offrant le meilleur compromis rapidité/précision sur le jeu global `Dataset_Global`.

---

### 4. `programs/` — Scripts exécutables et bancs de test

#### A. Détection & Tracking
* **`yolo_detect.py`** : Effectue la détection d'objets sur image unique, dossier d'images, flux vidéo, caméra USB ou Raspberry Pi Camera (`Picamera2`).
* **`yolo_tracking_test.py`** : Exécute le tracking temporel d'embarcations et affiche le flux annoté en direct.
* **`tracking_test_json.py`** : Réalise le tracking et structure chaque détection en temps réel sous forme d'objet JSON (horodatage, ID de piste, coordonnées de boîte, score de confiance) prêt à être transmis au décideur USV.

#### B. Évaluation & Dégradations d'images
* **`test_model.py`** : Lance l'évaluation standard Ultralytics (`model.val`) sur le split `test` et affiche mAP, Précision et Rappel.
* **`image_degradation.py`** : Module simulant 5 dégradations optiques et environnementales marines (gouttes d'eau/embruns, pixels noirs/masquage de sel, flou de défocalisation/houle, sous-échantillonnage de résolution, brume marine).
* **`test_degradation_visual.py`** : Applique les dégradations à sévérité croissante sur une image et génère des comparatifs côte-à-côte avec bounding boxes et confiance dans `results_degradation/visual/`.
* **`test_degradation_benchmark.py`** : Banc de test quantitatif évaluant l'évolution du Rappel, de la Précision, du F1-Score et de la Confiance sur le dataset de test. Génère `benchmark_metrics.csv` et le graphique à 4 panneaux `courbes_performances_degradation.png` dans `results_degradation/benchmark/`.
* **`test.md`** : Guide pas-à-pas regroupant les commandes terminal prêtes à l'emploi.

---

## 🚀 Commandes rapides (depuis `programs/`)

```bash
# 1. Détection simple sur image
python yolo_detect.py --model ../models/best4.pt --source "../images and video/bateau_test.jpg" --thresh 0.5

# 2. Tracking sur vidéo
python yolo_tracking_test.py --model ../models/best4.pt --source "../images and video/modelisation_3d_boat.mp4"

# 3. Test visuel qualitatif de dégradation
python test_degradation_visual.py --model ../models/best4.pt --image "../images and video/bateau_test.jpg"

# 4. Benchmark quantitatif de dégradation (ex: 50 images de test)
python test_degradation_benchmark.py --model ../models/best4.pt --sample_size 50
```
