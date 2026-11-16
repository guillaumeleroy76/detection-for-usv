# Détection avec le modèle robuste best4_2 sur une photo de navire :
python yolo_detect.py --model best4_2 --source "../images and video/bateau_test.jpg" --thresh 0.5

# Détection avec le modèle initial best4 sur une vidéo :
python yolo_detect.py --model best4 --source "../images and video/modelisation_3d_boat.mp4" --thresh 0.5