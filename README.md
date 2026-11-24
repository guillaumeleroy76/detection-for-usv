# Module de Détection & Veille Visuelle USV (`ia/detection`)

Ce module assure la **veille visuelle autonome et la détection d'embarcations** pour le drone de surface (USV) Cerema / IMT Atlantique.

Le système opère en amont de la chaîne de décision IA : il détecte et suit (*tracking*) les navires en temps réel sur flux caméra basse consommation (Raspberry Pi Camera / IMX500), émet les flux d'observations structurés en JSON et active les modules experts (classification fine, approche, OCR).

---

## 🎯 Statut des objectifs du volet Détection

- [x] **Reprendre et optimiser le modèle YOLO (YOLO11n) :** Modèle de base `best4.pt` (5.5 Mo, inférence rapide compatible Raspberry Pi 4).
- [x] **Confirmer et améliorer la détection en conditions difficiles :**
  - Modélisation physique des perturbations maritimes : embruns, gouttes d'eau, masquage/sel, brume, flou de clapot, baisse de résolution.
  - Entraînement robuste par injection de 10 % d'altérations marines dans le jeu d'entraînement.
  - Création du modèle robuste `best4_2.pt` validant un gain de résistance sans régression nominale.
- [x] **Suivi temporel & flux de communication :** Module de tracking (*ByteTrack*) générant des flux JSON horodatés pour la station sol et le superviseur.

---

## 📁 Arborescence du répertoire

```text
ia/detection/
├── best4/                                  # Modèles entraînés et résultats comparatifs
│   ├── best4.pt                            # Modèle YOLO initial de référence (5.5 Mo)
│   ├── best4_2.pt                          # Modèle YOLO robuste fine-tuné (5.5 Mo)
│   ├── before_transformations/             # Données et courbes de l'entraînement initial (50 époques)
│   │   ├── results_training_before_transformations/ # Données brutes (results.csv, courbes de pertes)
│   │   ├── courbes_entrainement.png        # 4 courbes d'entraînement initial
│   │   ├── benchmark_metrics.csv           # Benchmark initial de dégradation
│   │   ├── bilan_test_metriques.png        # Synthèse des métriques nominales
│   │   └── courbes_performances_degradation.png # Chute des scores face aux altérations
│   ├── after_transformations/              # Données du fine-tuning robuste (20 époques)
│   │   ├── results_training_after_transformations/  # Données brutes (results.csv, BoxF1, args.yaml)
│   │   ├── courbes_entrainement.png        # 4 courbes du fine-tuning robuste
│   │   ├── courbes_performances_degradation.png # Robustesse validée
│   │   └── echantillon_images_modifiees.png# Planche visuelle des altérations injectées
│   └── comparatif_gouttes_best4_vs_best4_2.png # Matrice visuelle comparative sous embruns
├── images and video/                       # Médias de test (images fixes et séquences vidéo)
│   ├── bateau_test.jpg                     # Photographie réelle de navire pour inférence unitaire
│   ├── baleine.png                         # Cas négatif / hors-domaine
│   ├── modelisation_3d_boat.mp4            # Animation 3D de navire pour validation de tracking
│   ├── scene-blender-01.mp4                # Simulation réaliste Blender (forte houle et clapot)
│   └── scene-blender-02.mp4                # Simulation réaliste Blender (vue dégagée)
├── programs/                               # Ensemble des bancs d'essai et scripts exécutables
│   ├── live_pi_camera.py                   # Streaming direct de la caméra Pi sur PC (OpenCV / FFplay / YOLO)
│   ├── test_model.py                       # Évaluation standard sur Dataset_Global (--model best4 / best4_2)
│   ├── test_degradation_benchmark.py       # Benchmark quantitatif de robustesse multi-dégradations
│   ├── test_degradation_gouttes.py         # Test visuel spécialisé gouttes d'eau jusqu'à 100% (--compare)
│   ├── test_degradation_visual.py          # Test qualitatif côte-à-côte (5 dégradations)
│   ├── donnee_training.py                  # Visualisation des courbes d'apprentissage (--run before / after)
│   ├── yolo_detect.py                      # Détection unitaire (image, vidéo, caméra USB, Picamera)
│   ├── yolo_tracking_test.py               # Tracking multi-objets avec affichage vidéo
│   ├── tracking_test_json.py               # Tracking avec validation spatio-temporelle (images consécutives) et export JSON
│   ├── train_robust_model.py               # Fine-tuning à partir d'un modèle source (--weights best4)
│   ├── modify_dataset_robust.py            # Injection de 10% de perturbations maritimes dans le dataset
│   ├── image_degradation.py                # Moteur physique de simulation des dégradations maritimes
│   └── test.md                             # Guide d'exécution détaillé des programmes
└── README.md                               # Présentation générale du module
```

---

## 📊 Bilan comparatif des performances : `best4` vs `best4_2`

### 1. Évaluation nominale (Jeu de test propre, 644 images, 799 navires)

Évaluation menée avec IoU = 0.50 :

| Métrique | `best4.pt` *(Initial)* | `best4_2.pt` *(Robuste)* | Évolution | Interprétation opérationnelle pour l'USV |
|---|:---:|:---:|:---:|---|
| **Précision (P)** | **77.90 %** | **79.34 %** | **+1.44 %** 🟢 | Réduction des fausses alertes générées par l'écume et le clapot. |
| **Rappel (R)** | **75.44 %** | **76.22 %** | **+0.78 %** 🟢 | Moins de navires manqués dans la zone de veille. |
| **mAP50** | **80.13 %** | **82.40 %** | **+2.27 %** 🟢 | Gain de plus de 2 points sur la détection standard. |
| **mAP50-95** | **56.32 %** | **56.86 %** | **+0.54 %** 🟢 | Boîtes englobantes plus rigoureuses et fidèles aux contours. |

> [!TIP]
> **Conclusion clé :** L'ajout de 10 % de données dégradées n'a provoqué **aucun oubli catastrophique** (*Catastrophic Forgetting*). Au contraire, les altérations ont agi comme un puissant régularisateur, renforçant la généralisation du modèle même sur images nettes.

### 2. Résistance sous forte projection de gouttes d'eau (jusqu'à 100 %)

Test de résistance optique mené avec `test_degradation_gouttes.py` sur `bateau_test.jpg` :

| Sévérité des gouttes | `best4` *(Initial)* | `best4_2` *(Robuste)* | Écart de confiance |
|:---:|:---:|:---:|:---:|
| **0 % (Image nette)** | 0.89 | 0.89 | Identique |
| **50 % (Embruns modérés)** | 0.89 | 0.89 | Identique |
| **80 % (Forte pluie/projections)** | 0.87 | 0.87 | Identique |
| **100 % (~90 gouttes, objectif saturé)** | **0.63** ⚠️ | **0.83** 🟢 | **+0.20 (+20 points)** |

---

## 🎯 Sélection universelle du modèle (`--model` / `-m`)

Tous les programmes acceptent l'argument `--model` (ou `-m`), permettant de basculer facilement entre les modèles :
* `best4` ou `best4.pt` : Modèle initial de référence.
* `best4_2` ou `best4_2.pt` : Modèle robuste fine-tuné.
* Tout autre fichier ou chemin (`best5`, `path/to/model.pt`).

La détection du GPU (NVIDIA CUDA si disponible, sinon CPU) et la résolution des chemins relatifs sont automatiques.

---

## 🚀 Commandes d'exécution rapide (depuis `programs/`)

```bash
# 1. Lancer la caméra du Raspberry Pi en direct sur le PC :
python live_pi_camera.py

# 2. Lancer la caméra du Pi avec détection YOLO en direct sur le PC :
python live_pi_camera.py --yolo --model best4_2

# 3. Évaluer les métriques sur le test set :
python test_model.py --model best4_2

# 2. Visualiser les courbes d'apprentissage (pertes, mAP, P/R) :
python donnee_training.py --run after

# 3. Détection simple sur image ou vidéo :
python yolo_detect.py --model best4_2 --source "../images and video/bateau_test.jpg" --thresh 0.5

# 4. Suivi vidéo temps réel (tracking live) :
python yolo_tracking_test.py --model best4_2 --source "../images and video/scene-blender-01.mp4"

# 5. Suivi avec émission du flux JSON horodaté :
python tracking_test_json.py --model best4_2

# 6. Test visuel spécialisé gouttes d'eau (comparatif best4 vs best4_2 jusqu'à 100%) :
python test_degradation_gouttes.py --compare

# 7. Benchmark quantitatif de robustesse face aux 5 dégradations :
python test_degradation_benchmark.py --model best4_2 --sample_size 50
```

Pour le détail complet de chaque argument et option avancée, consultez **[programs/test.md](programs/test.md)**.
