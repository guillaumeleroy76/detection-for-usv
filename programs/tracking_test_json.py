import os
import sys
import glob
import time
import json
import argparse
import cv2
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Suivi vidéo USV avec filtrage spatial et temporel (images consécutives) et export JSON"
    )
    # Paramètres du modèle et de la source
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4",
        help="Modèle à utiliser : alias ('best4', 'best4_2'), nom ou chemin (défaut: best4)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="images and video/modelisation_3d_boat.mp4",
        help="Source vidéo ou index de caméra '0' (défaut: images and video/modelisation_3d_boat.mp4)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Algorithme de tracking : 'bytetrack.yaml' ou 'botsort.yaml' (défaut: bytetrack.yaml)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Seuil de confiance minimal (défaut: 0.4)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Résolution d'inférence (défaut: 640)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device : '0' (GPU CUDA) ou 'cpu' (défaut: auto-détection)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Désactiver la fenêtre OpenCV d'affichage en direct"
    )

    # Filtrage spatial et temporel (anti faux positifs)
    parser.add_argument(
        "--min-consecutive", "--min-frames",
        dest="min_consecutive",
        type=int,
        default=3,
        help="Nombre minimal d'images consécutives pour confirmer la détection d'un bateau (défaut: 3)"
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=120.0,
        help="Distance spatiale maximale en pixels entre deux centroïdes consécutifs pour valider le même objet (défaut: 120.0 px)"
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.15,
        help="Seuil d'IoU minimal pour valider immédiatement le recouvrement spatial (défaut: 0.15)"
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=0,
        help="Tolérance aux images sautées sans détection avant rupture de la continuité (défaut: 0 = consécutif strict)"
    )
    parser.add_argument(
        "--emit-mode",
        type=str,
        default="confirmed",
        choices=["confirmed", "event", "all"],
        help="Mode d'émission JSON sur la console : 'confirmed' (défaut: uniquement si un bateau confirmé est présent), 'event' (uniquement lors d'un événement de confirmation), 'all' (à chaque image)"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout le modèle YOLO (alias ou chemin)."""
    if not model_arg:
        model_arg = "best4"
        
    if os.path.isabs(model_arg) and os.path.exists(model_arg):
        return os.path.abspath(model_arg)
        
    cwd = os.getcwd()
    parent_dir = os.path.dirname(base_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    search_dirs = [cwd, base_dir, parent_dir, grandparent_dir]
    
    names = [model_arg]
    if not model_arg.lower().endswith(".pt"):
        names.append(model_arg + ".pt")
        
    candidates = []
    for name in names:
        candidates.append(name)
        for s_dir in search_dirs:
            candidates.append(os.path.join(s_dir, name))
            candidates.append(os.path.join(s_dir, "best4", name))
            candidates.append(os.path.join(s_dir, "models", name))
            
    bare_name = os.path.splitext(os.path.basename(model_arg))[0]
    for s_dir in search_dirs:
        candidates.append(os.path.join(s_dir, bare_name, f"{bare_name}.pt"))
        if bare_name == "best4_2":
            candidates.append(os.path.join(s_dir, "best4", "best4_2.pt"))
            
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
            
    target_pattern = bare_name + ".pt" if not model_arg.lower().endswith(".pt") else os.path.basename(model_arg)
    for root in [parent_dir, grandparent_dir]:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", target_pattern), recursive=True)
            if matches:
                return os.path.abspath(matches[0])
                
    return os.path.abspath(model_arg)


def resolve_source_path(source_arg: str, base_dir: str = SCRIPT_DIR):
    """Résout la source vidéo (entier ou chemin absolu)."""
    if source_arg.isdigit():
        return int(source_arg)
    if os.path.isabs(source_arg) and os.path.exists(source_arg):
        return os.path.abspath(source_arg)
        
    candidates = [
        source_arg,
        os.path.join(base_dir, source_arg),
        os.path.join(os.path.dirname(base_dir), source_arg),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), source_arg),
        os.path.join(os.path.dirname(base_dir), "images and video", os.path.basename(source_arg)),
        os.path.join(base_dir, "images and video", os.path.basename(source_arg)),
        os.path.join(os.path.dirname(base_dir), "images and video", "modelisation_3d_boat.mp4")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return source_arg


# =============================================================================
# MODULE DE VALIDATION SPATIALE ET TEMPORELLE (ANTI-FAUX POSITIFS)
# =============================================================================

def compute_iou(boxA, boxB):
    """Calcule l'Intersection-over-Union (IoU) entre deux boîtes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union_area = areaA + areaB - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def compute_centroid_distance(boxA, boxB):
    """Calcule la distance euclidienne entre les centres de deux boîtes [x, y, w, h]."""
    cAx = boxA[0] + boxA[2] / 2.0
    cAy = boxA[1] + boxA[3] / 2.0
    cBx = boxB[0] + boxB[2] / 2.0
    cBy = boxB[1] + boxB[3] / 2.0
    return ((cAx - cBx) ** 2 + (cAy - cBy) ** 2) ** 0.5


def is_spatially_consistent(old_box, new_box, max_dist=120.0, min_iou=0.15):
    """
    Vérifie si deux détections consécutives correspondent physiquement au MÊME objet :
    1. Recouvrement spatial IoU suffisant, OU
    2. Proximité spatiale du centre (distance <= max_dist adaptable à la taille), ET
    3. Stabilité d'échelle (l'objet ne grossit/rétrécit pas de façon absurde en 40ms).
    """
    iou = compute_iou(old_box, new_box)
    if iou >= min_iou:
        return True, iou, 0.0

    dist = compute_centroid_distance(old_box, new_box)
    old_diag = (old_box[2] ** 2 + old_box[3] ** 2) ** 0.5
    # Seuil dynamique : tolère un déplacement proportionnel à l'objet (vagues/tangage)
    dist_thresh = max(max_dist, old_diag * 1.5)

    area_old = max(1.0, float(old_box[2] * old_box[3]))
    area_new = max(1.0, float(new_box[2] * new_box[3]))
    ratio = area_new / area_old
    scale_ok = (0.2 <= ratio <= 5.0)

    if dist <= dist_thresh and scale_ok:
        return True, iou, dist

    return False, iou, dist


class SpatialTrack:
    """État temporel et spatial d'une piste de détection."""
    def __init__(self, track_id: int, bbox: list, conf: float, frame_idx: int, t: float):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.conf = conf
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_t = t
        self.last_t = t
        self.consecutive_frames = 1
        self.total_frames = 1
        self.confirmed = False
        self.already_notified = False  # pour mode event

    @property
    def centroid(self):
        return [int(self.bbox[0] + self.bbox[2] / 2.0), int(self.bbox[1] + self.bbox[3] / 2.0)]


class SpatialTrackValidator:
    """
    Gestionnaire assurant que les détections consécutives correspondent
    au même objet dans l'espace physique avant de valider la présence d'un bateau.
    """
    def __init__(self, min_consecutive: int = 3, max_dist: float = 120.0,
                 min_iou: float = 0.15, max_missed: int = 0, prune_timeout: int = 30):
        self.min_consecutive = min_consecutive
        self.max_dist = max_dist
        self.min_iou = min_iou
        self.max_missed = max_missed
        self.prune_timeout = prune_timeout
        self.tracks = {}  # track_id -> SpatialTrack
        self.next_fallback_id = 1000

    def update(self, raw_detections: list, frame_idx: int, t: float):
        """
        Met à jour les pistes avec les détections brutes de la trame.
        raw_detections: list of dict {'id_piste': int, 'boite': [x,y,w,h], 'confiance': float}
        Retourne :
        - active_tracks: toutes les pistes vues à cette trame
        - confirmed_tracks: les pistes ayant validé le critère d'images consécutives
        - new_confirmed_events: les pistes devenues confirmées EXACTEMENT à cette trame
        """
        active_tracks = []
        confirmed_tracks = []
        new_confirmed_events = []

        seen_track_ids = set()

        for det in raw_detections:
            id_piste = det.get("id_piste", 0)
            bbox = det["boite"]
            conf = det["confiance"]

            # Si le tracker n'a pas assigné d'ID (0 ou None), trouver la piste spatiale la plus proche
            if id_piste <= 0:
                best_id = None
                best_dist = float("inf")
                for tid, track in self.tracks.items():
                    if tid in seen_track_ids:
                        continue
                    ok, _, dist = is_spatially_consistent(track.bbox, bbox, self.max_dist, self.min_iou)
                    if ok and dist < best_dist:
                        best_dist = dist
                        best_id = tid
                if best_id is not None:
                    id_piste = best_id
                else:
                    id_piste = self.next_fallback_id
                    self.next_fallback_id += 1

            seen_track_ids.add(id_piste)

            if id_piste in self.tracks:
                track = self.tracks[id_piste]
                frames_gap = frame_idx - track.last_frame

                # 1. Vérification de la continuité temporelle (images consécutives)
                is_temporally_continuous = (frames_gap <= (self.max_missed + 1))

                # 2. Vérification de la continuité spatiale (même objet physique)
                is_spatially_ok, _, _ = is_spatially_consistent(
                    track.bbox, bbox, max_dist=self.max_dist, min_iou=self.min_iou
                )

                if is_temporally_continuous and is_spatially_ok:
                    track.consecutive_frames += 1
                    track.total_frames += 1
                else:
                    # Rupture temporelle ou saut spatial (téléportation / faux positif distinct) :
                    # On réinitialise le compteur consécutif
                    track.consecutive_frames = 1

                track.bbox = bbox
                track.conf = conf
                track.last_frame = frame_idx
                track.last_t = t

            else:
                # Nouvelle piste
                track = SpatialTrack(track_id=id_piste, bbox=bbox, conf=conf, frame_idx=frame_idx, t=t)
                self.tracks[id_piste] = track

            # Validation du seuil de confirmation
            was_confirmed = track.confirmed
            if track.consecutive_frames >= self.min_consecutive:
                track.confirmed = True
            else:
                track.confirmed = False

            if track.confirmed and not was_confirmed:
                new_confirmed_events.append(track)

            active_tracks.append(track)
            if track.confirmed:
                confirmed_tracks.append(track)

        # Nettoyage des anciennes pistes inactives
        to_prune = [
            tid for tid, track in self.tracks.items()
            if (frame_idx - track.last_frame) > self.prune_timeout
        ]
        for tid in to_prune:
            del self.tracks[tid]

        return active_tracks, confirmed_tracks, new_confirmed_events


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    args = parse_args()
    
    model_path = resolve_model_path(args.model)
    source_path = resolve_source_path(args.source)
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable : {args.model}")
        print(f"   Chemin testé : {model_path}")
        sys.exit(1)
        
    print("=" * 70)
    print("📡 ACQUISITION & SUIVI USV AVEC VALIDATION SPATIALE & TEMPORELLE")
    print("=" * 70)
    print(f"• Modèle chargé       : {model_path}")
    print(f"• Source vidéo        : {source_path}")
    print(f"• Tracker             : {args.tracker}")
    print(f"• Confiance minimale  : {args.conf}")
    print(f"• Images consécutives : {args.min_consecutive} frames (seuil de confirmation)")
    print(f"• Distance max spatiale: {args.max_dist} px (tolérance de déplacement)")
    print(f"• Seuil IoU spatial   : {args.min_iou}")
    print(f"• Tolérance saut fr   : {args.max_missed} frame(s)")
    print(f"• Mode émission JSON  : {args.emit_mode}")
    print(f"• Device d'inférence  : {args.device} ({'GPU CUDA' if args.device == '0' else 'CPU'})")
    print(f"• Fenêtre graphique   : {'Désactivée (--no-show)' if args.no_show else 'Active'}")
    print("=" * 70)
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        print(f"❌ Erreur : Impossible d'ouvrir la source vidéo : {source_path}")
        sys.exit(1)
        
    validator = SpatialTrackValidator(
        min_consecutive=args.min_consecutive,
        max_dist=args.max_dist,
        min_iou=args.min_iou,
        max_missed=args.max_missed
    )

    start_time = time.time()
    frame_idx = 0
    total_confirmed_events = 0
    
    window_name = "Suivi USV - Validation Spatio-Temporelle & JSON"
    target_w = 960
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1000
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 600
    target_h = int(target_w * (vid_h / float(vid_w)))
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, target_w, target_h)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("🏁 Fin de la vidéo atteinte.")
                break
                
            frame_idx += 1
            t_actuel = round(time.time() - start_time, 2)
            
            # Tracking YOLO avec ByteTrack
            results = model.track(
                frame,
                tracker=args.tracker,
                persist=True,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            
            # Extraction des détections brutes
            boxes = results[0].boxes
            raw_detections = []
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = xyxy
                    w = int(xmax - xmin)
                    h = int(ymax - ymin)
                    x = int(xmin)
                    y = int(ymin)
                    conf = float(box.conf[0])
                    id_piste = int(box.id[0]) if box.id is not None else 0
                    
                    raw_detections.append({
                        "id_piste": id_piste,
                        "boite": [x, y, w, h],
                        "confiance": round(conf, 2)
                    })
                    
            # Mise à jour et validation spatio-temporelle
            active_tracks, confirmed_tracks, new_confirmed_events = validator.update(
                raw_detections, frame_idx=frame_idx, t=t_actuel
            )
            
            total_confirmed_events += len(new_confirmed_events)

            # Préparation des données JSON
            pistes_export = []
            for trk in (confirmed_tracks if args.emit_mode != "all" else active_tracks):
                pistes_export.append({
                    "id_piste": trk.track_id,
                    "boite": trk.bbox,
                    "centre": trk.centroid,
                    "confiance": round(trk.conf, 2),
                    "images_consecutives": trk.consecutive_frames,
                    "statut": "confirme" if trk.confirmed else "en_cours_validation"
                })

            # Gestion de l'émission JSON sur la console stdout
            if args.emit_mode == "confirmed":
                # Émet uniquement s'il y a au moins un bateau confirmé dans la trame
                if len(confirmed_tracks) > 0:
                    json_msg = {
                        "t": t_actuel,
                        "frame": frame_idx,
                        "nb_confirmes": len(confirmed_tracks),
                        "pistes": pistes_export
                    }
                    print(json.dumps(json_msg, ensure_ascii=False), flush=True)

            elif args.emit_mode == "event":
                # Émet un JSON d'événement uniquement lors de la transition vers "confirmé"
                for trk in new_confirmed_events:
                    event_msg = {
                        "event": "BATEAU_CONFIRME",
                        "t": t_actuel,
                        "frame": frame_idx,
                        "id_piste": trk.track_id,
                        "images_consecutives": trk.consecutive_frames,
                        "boite": trk.bbox,
                        "centre": trk.centroid,
                        "confiance": round(trk.conf, 2)
                    }
                    print(json.dumps(event_msg, ensure_ascii=False), flush=True)

            elif args.emit_mode == "all":
                # Émet à chaque trame (compatibilité totale avec l'ancien comportement)
                json_msg = {
                    "t": t_actuel,
                    "frame": frame_idx,
                    "nb_confirmes": len(confirmed_tracks),
                    "pistes": pistes_export
                }
                print(json.dumps(json_msg, ensure_ascii=False), flush=True)

            # Affichage graphique OpenCV
            if not args.no_show:
                display_frame = frame.copy()
                
                # Tracé des boîtes selon leur statut de validation spatiale
                for trk in active_tracks:
                    x, y, w, h = trk.bbox
                    
                    if trk.confirmed:
                        # Bateau validé spatialement et temporellement : Vert
                        color = (0, 220, 0)
                        label = f"BATEAU #{trk.track_id} [CONFIRME {trk.consecutive_frames}fr] {trk.conf:.2f}"
                        thickness = 3
                    else:
                        # Détection en cours de validation spatiale : Orange
                        color = (0, 165, 255)
                        label = f"Piste #{trk.track_id} [{trk.consecutive_frames}/{args.min_consecutive}fr] {trk.conf:.2f}"
                        thickness = 1
                        
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                    cv2.circle(display_frame, (trk.centroid[0], trk.centroid[1]), 4, color, -1)
                    
                    # Fond du label texte
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        display_frame,
                        (x, max(0, y - label_size[1] - 8)),
                        (x + label_size[0] + 6, max(label_size[1] + 8, y)),
                        color, -1
                    )
                    cv2.putText(
                        display_frame, label,
                        (x + 3, max(label_size[1] + 2, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )

                # Bandeau d'information supérieur (HUD)
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 36), (30, 30, 30), -1)
                hud_txt = (
                    f"Frame: {frame_idx:04d} | Pistes: {len(active_tracks)} | "
                    f"Confirmes: {len(confirmed_tracks)} (seuil: {args.min_consecutive} fr) | "
                    f"Mode JSON: {args.emit_mode}"
                )
                cv2.putText(display_frame, hud_txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

                if not args.no_show:
                    if display_frame.shape[1] != target_w:
                        show_frame = cv2.resize(display_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    else:
                        show_frame = display_frame
                    cv2.imshow(window_name, show_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("🛑 Arrêt manuel par l'utilisateur.")
                        break
                    
    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel interrompu.")
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print("-" * 70)
        print(f"📊 RÉSUMÉ D'EXÉCUTION :")
        print(f"• Trames analysées           : {frame_idx}")
        print(f"• Événements bateaux validés : {total_confirmed_events}")
        print("✅ Ressources libérées.")


if __name__ == "__main__":
    main()
