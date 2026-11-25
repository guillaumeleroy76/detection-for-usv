# Détection avec le modèle robuste best4_2 sur une photo de navire :

python programs/yolo_detect.py --model best4_2 --source "../images and video/bateau_test.jpg" --thresh 0.5

python programs/tracking_test_json.py --source "images and video/scene-blender-02.mp4" --model best4_2 --tracker botsort.yaml --min-consecutive 4 --conf 0.55    

# Détection avec le modèle initial best4 sur une vidéo :
python programs/yolo_detect.py --model best4 --source "../images and video/scene-blender-01.mp4" --thresh 0.5
