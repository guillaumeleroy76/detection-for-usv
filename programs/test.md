# USV Project - Embedded Intelligence System

This project develops an embedded computer vision system for an USV (Unmanned Surface Vehicle). The approach relies on lightweight and autonomous monitoring via a camera and an optimized YOLO model, enabling the detection and tracking of boats while minimizing energy consumption before triggering expert modules for an in-depth acquisition of the video stream.

---

## What is done in this directory

This working directory brings together the local development and testing environment on a PC:
- Storage and structuring of training and validation datasets (`Dataset_Global`, `Fishing Boat.v1i.yolov11`).
- Archiving of the various trained YOLO models (`best2.pt`, `best3.pt`, `best4.pt`).
- Utility scripts for evaluating performance, manipulating data, and testing video tracking.

---

## What each program does

- **`yolo_detect.py`**: Performs simple object detection (without temporal tracking) on a source image or video (allows visualization).
- **`yolo_tracking_test.py`**: Runs multi-object tracking on a test video while maintaining stable track IDs and displaying the result live.
- **`tracking_test_json.py`**: Performs tracking and formats real-time observations as a structured JSON object (containing time `t`, track ID `id_piste`, bounding box coordinates, and confidence).
- **`test_model.py`**: Evaluates and tests the overall performance of the model on a dedicated test dataset.
- **`get_pourcentage.py`**: Utility script to provide the training/validation/test split of the dataset.
- **`new_merge.py`**: Allows merging, cleaning, or reorganizing training datasets with each other.

---

## How to run the programs

Here are the commands to run in your terminal (within the project directory):

### 1. Run simple detection on an image or a video
```bash
python yolo_detect.py --model models/best4.pt --source "images and video/bateau_test.jpg" --thresh 0.5
```
```bash
python yolo_detect.py --model models/best4.pt --source "images and video/modelisation_3d_boat.mp4" --thresh 0.5
```

