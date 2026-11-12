# AI Vision for USV Platform (`detection-for-usv`)

Système de vision par ordinateur embarqué pour drone de surface (USV - Unmanned Surface Vehicle) développé dans le cadre du projet Cerema / IMT Atlantique.

Le système assure la veille visuelle autonome basse consommation via caméra et modèle YOLO optimisé, permettant la détection et le suivi (*tracking*) de navires avant d'activer les modules experts d'analyse approfondie du flux vidéo.

---

## 🎯 Sélection universelle du modèle (`--model` / `-m`)

Tous les programmes acceptent désormais l'argument `--model` (ou `-m`), permettant de sélectionner dynamiquement le modèle à exécuter sans modifier une seule ligne de code :

* **Alias supportés automatiquement :**
  * `best4` ou `best4.pt` : Modèle initial de référence (entraîné sur images nominales propres).
  * `best4_2` ou `best4_2.pt` : Modèle robuste fine-tuné (entraîné avec 10 % d'altérations maritimes injectées : embruns, sel, flou, brume, baisse de résolution).
  * Tout autre modèle (`best5`, `yolo11n.pt`, ou chemin relatif / absolu `path/to/model.pt`).
* **Résolution automatique :** Vous pouvez lancer les commandes depuis n'importe quel dossier (`programs/`, `detection-for-usv/`, etc.), les chemins sont automatiquement détectés.

---

## 📁 Arborescence réorganisée du répertoire

```text
detection-for-usv/
├── best4/                                  # Modèles de détection et résultats comparatifs
│   ├── best4.pt                            # Modèle YOLO initial de référence (5.5 Mo)
│   ├── best4_2.pt                          # Modèle YOLO fine-tuné robuste aux dégradations (5.5 Mo)
│   ├── before_transformations/             # Données et benchmarks du modèle initial (best4)
│   │   ├── results_training_before_transformations/ # Données du run d'origine (results.csv, pertes, mAP)
│   │   ├── courbes_entrainement.png        # 4 courbes d'entraînement initial (50 époques)
│   │   ├── benchmark_metrics.csv           # Données brutes du benchmark initial de dégradation
│   │   ├── bilan_test_metriques.png        # Barres récapitulatives initiales (P, R, mAP50, mAP50-95)
│   │   ├── courbes_performances_degradation.png # Chute des performances face aux dégradations
│   │   └── planche_recapitulative_degradations.png # Matrice visuelle des dégradations initiales
│   └── after_transformations/              # Données et benchmarks du modèle robuste (best4_2)
│       ├── results_training_after_transformations/  # Données du fine-tuning (results.csv, BoxF1, args.yaml)
│       ├── courbes_entrainement.png        # 4 courbes du fine-tuning robuste (20 époques)
│       ├── courbes_performances_degradation.png # Robustesse validée sur images dégradées
│       ├── echantillon_images_modifiees.png# Planche visuelle des altérations maritimes injectées
│       └── planche_recapitulative_degradations.png # Matrice visuelle comparative post-entraînement
├── images and video/                       # Médias de test (images fixes et flux vidéo)
│   ├── bateau_test.jpg                     # Photo réelle de navire pour inférence unitaire
│   ├── baleine.png                         # Cas négatif / hors-domaine
│   ├── modelisation_3d_boat.mp4            # Vidéo de test de tracking (animation 3D)
│   └── scene-blender-01.mp4                # Simulation réaliste Blender (houle et clapot)
├── programs/                               # Ensemble des scripts exécutables
│   ├── donnee_training.py                  # Visualisation des courbes d'apprentissage (--run before / after)
│   ├── image_degradation.py                # Moteur physique de dégradation maritime (gouttes, sel, flou...)
│   ├── modify_dataset_robust.py            # Modifie 10% du dataset d'entraînement par transformation
│   ├── test_degradation_benchmark.py       # Benchmark quantitatif (--model best4 / best4_2)
│   ├── test_degradation_visual.py          # Test qualitatif côte-à-côte (--model best4 / best4_2)
│   ├── test_model.py                       # Évaluation standard sur Dataset_Global (--model best4 / best4_2)
│   ├── tracking_test_json.py               # Tracking temps réel avec export JSON (--model best4 / best4_2)
│   ├── train_robust_model.py               # Fine-tuning à partir d'un modèle source (--weights best4)
│   ├── yolo_detect.py                      # Détection unitaire image/vidéo/caméra (--model best4 / best4_2)
│   ├── yolo_tracking_test.py               # Tracking temps réel avec affichage (--model best4 / best4_2)
│   ├── test.md                             # Guide d'exécution détaillé
│   └── results_degradation/                # Sorties des bancs de test
│       ├── benchmark/                      # benchmark_metrics.csv & courbes_performances_degradation.png
│       └── visual/                         # Planches et comparaisons visuelles unitaires
└── dataset_modification_manifest.json      # Registre complet des 4 545 images modifiées
```

---

## 🚀 Guide rapide des commandes (depuis `programs/`)

### 1. Évaluer et comparer les performances nominales
```bash
# Évaluer le modèle initial best4
python test_model.py --model best4

# Évaluer le modèle robuste best4_2
python test_model.py --model best4_2
```

### 2. Visualiser les courbes d'apprentissage
```bash
# Courbes du fine-tuning robuste (best4_2, 20 époques)
python donnee_training.py --run after

# Courbes de l'entraînement initial (best4, 50 époques)
python donnee_training.py --run before
```

### 3. Détection simple sur image ou vidéo
```bash
# Détection avec le modèle robuste best4_2 sur une photo
python yolo_detect.py --model best4_2 --source "../images and video/bateau_test.jpg" --thresh 0.5

# Détection avec le modèle initial best4 sur une vidéo
python yolo_detect.py --model best4 --source "../images and video/modelisation_3d_boat.mp4" --thresh 0.5
```

### 4. Tracking temps réel
```bash
# Suivi multi-objets avec affichage vidéo
python yolo_tracking_test.py --model best4_2

# Suivi multi-objets avec export JSON temps réel
python tracking_test_json.py --model best4_2
```

### 5. Tests de robustesse aux dégradations marines
```bash
# Test visuel unitaire (génère les comparaisons Clean vs Dégradé)
python test_degradation_visual.py --model best4_2

# Benchmark quantitatif sur 50 images de test
python test_degradation_benchmark.py --model best4_2 --sample_size 50
```

### 6. Fine-tuning d'un nouveau modèle
```bash
# Lancer un fine-tuning à partir de best4 pour générer un nouveau run
python train_robust_model.py --weights best4 --epochs 20 --batch 16
```
