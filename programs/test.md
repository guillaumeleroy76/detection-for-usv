# USV Project - Embedded Intelligence System

This project develops an embedded computer vision system for an USV (Unmanned Surface Vehicle). The approach relies on lightweight and autonomous monitoring via a camera and an optimized YOLO model, enabling the detection and tracking of boats while minimizing energy consumption before triggering expert modules for an in-depth acquisition of the video stream.

---

## What is done in this directory

This working directory brings together the local development and testing environment on a PC:
- Storage and structuring of training and validation datasets (`Dataset_Global`, `Fishing Boat.v1i.yolov11`).
- Archiving of the various trained YOLO models (`best2.pt`, `best3.pt`, `best4.pt`).
- Utility scripts for evaluating performance, manipulating data, testing video tracking, and testing robustness against environmental degradations.

---

## What each program does

- **`yolo_detect.py`**: Performs simple object detection (without temporal tracking) on a source image or video (allows visualization).
- **`yolo_tracking_test.py`**: Runs multi-object tracking on a test video while maintaining stable track IDs and displaying the result live.
- **`tracking_test_json.py`**: Performs tracking and formats real-time observations as a structured JSON object (containing time `t`, track ID `id_piste`, bounding box coordinates, and confidence).
- **`test_model.py`**: Evaluates and tests the overall performance of the model on a dedicated test dataset.
- **`image_degradation.py`**: Module simulant les dégradations optiques et environnementales marines (gouttes d'eau/embruns, pixels noirs/masquage, flou de défocalisation/houle, sous-échantillonnage de résolution, brume marine).
- **`test_degradation_visual.py`**: Évalue qualitativement la robustesse de détection de YOLO face aux dégradations et génère des planches comparatives côte-à-côte avec bounding boxes et confiance.
- **`test_degradation_benchmark.py`**: Benchmark quantitatif systématique sur le dataset de test (`Dataset_Global/test`), générant les métriques (Rappel, Précision, F1, Confiance) au format CSV et traçant les courbes d'affaiblissement de performances.
- **`get_pourcentage.py`**: Utility script to provide the training/validation/test split of the dataset.
- **`new_merge.py`**: Allows merging, cleaning, or reorganizing training datasets with each other.

---

## How to run the programs

Here are the commands to run in your terminal (within the project directory):

### 1. Run simple detection on an image or a video
```bash
python programs/yolo_detect.py --model models/best4.pt --source "images and video/bateau_test.jpg" --thresh 0.5
```
```bash
python programs/yolo_detect.py --model models/best4.pt --source "images and video/modelisation_3d_boat.mp4" --thresh 0.5
```

### 2. Run visual degradation tests (qualitative evaluation)
```bash
python programs/test_degradation_visual.py --model models/best4.pt --image "images and video/bateau_test.jpg" --severities 0.25 0.50 0.75
```
Output files are saved in `results_degradation/visual/`.

### 3. Run quantitative benchmark on the test dataset
```bash
# Fast benchmark on 50 sampled test images:
python programs/test_degradation_benchmark.py --model models/best4.pt --sample_size 50

# Full benchmark on all 644 test images:
python programs/test_degradation_benchmark.py --model models/best4.pt --sample_size 0
```
Output metrics (`benchmark_metrics.csv`) and performance degradation plots (`courbes_performances_degradation.png`) are saved in `results_degradation/benchmark/`.
