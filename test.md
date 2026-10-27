# Projet USV - Système d'Intelligence Embarquée

Ce projet développe un système de vision par ordinateur embarqué pour un USV (Unmanned Surface Vehicle). L'approche repose sur une veille légère et autonome via une caméra et un modèle YOLO optimisé, permettant de détecter et suivre des bateaux en minimisant la consommation énergétique avant de déclencher des modules experts pour une acquisition approfondie du flux vidéo.

---

## Ce qui est fait dans ce répertoire

Ce répertoire de travail rassemble l'environnement de développement et de test local sur PC :
- Le stockage et la structuration des jeux de données d'entraînement et de validation (`Dataset_Global`, `Fishing Boat.v1i.yolov11`).
- L'archivage des différents modèles YOLO entraînés (`best2.pt`, `best3.pt`, `best4.pt`).
- Les scripts utilitaires pour évaluer les performances, manipuler les données et tester le tracking vidéo.

---

## Ce que fait chaque programme

- **`yolo_detect.py`** : Effectue une détection d'objets simple (sans suivi temporel) sur une image ou une vidéo source (permet de visualiser).
- **`yolo_tracking_test.py`** : Lance le suivi (tracking) multi-objets sur une vidéo de test en conservant des identifiants de piste stables et en affichant le résultat en direct.
- **`tracking_test_json.py`** : Réalise le tracking et formate les observations en temps réel sous forme d'un objet JSON structuré (contenant le temps `t`, l'identifiant `id_piste`, les coordonnées de la boîte et la confiance).
- **`test_model.py`** : Évalue et teste les performances globales du modèle sur un jeu de données dédiéau test.
- **`get_pourcentage.py`** : Script utilitaire permettant de donner la répartition training/validation/test du dataset.
- **`new_merge.py`** : Permet de fusionner, nettoyer ou réorganiser des ensembles de données d'entraînement entre eux.

---

## Comment lancer les programmes

Voici les commandes à exécuter dans votre terminal (dans le répertoire du projet) :

### 1. Lancer une détection simple sur une image
```bash
python yolo_detect.py --model models/best4.pt --source "images and video/bateau_test.jpg" --thresh 0.5
```

