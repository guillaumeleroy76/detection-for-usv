"""
Script: test_expert_trigger.py
Description: Validation du filtrage temporel (ex: 5 images d'affilée) sur flux vidéo ou dossier
             d'images séquentielles. Affiche clairement dans la console et sur la vidéo
             le déclenchement des experts (uniquement si 5 détections consécutives du même bateau)
             et le rejet des artefacts fugitifs (vagues, reflets < 5 images).
Auteur: Guillaume (Projet Cerema USV / IMT Atlantique)
"""

import os
import re
import sys
import glob
import time
import tempfile
import argparse
import cv2
import torch

# Configuration des dossiers temporaires pour éviter les verrous Windows
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test du déclencheur d'experts par persistance temporelle (N images d'affilée)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="images and video/modelisation_3d_boat.mp4",
        help="Chemin de la vidéo (.mp4/.avi) ou d'un dossier d'images séquentielles (défaut: modelisation_3d_boat.mp4)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4_2",
        help="Modèle YOLO : alias ('best4', 'best4_2') ou chemin (défaut: best4_2)"
    )
    parser.add_argument(
        "--min-consecutive", "-n",
        type=int,
        default=5,
        help="Nombre d'images d'affilée nécessaires pour lancer les experts (défaut: 5)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.50,
        help="Seuil de confiance minimal YOLO (défaut: 0.50)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Algorithme de tracking : 'botsort.yaml' (recommandé vagues) ou 'bytetrack.yaml'"
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=150.0,
        help="Distance max en pixels entre 2 centroïdes consécutifs pour valider le même objet (défaut: 150 px)"
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=1,
        help="Tolérance de trames perdues sans rompre la piste (défaut: 1)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Désactiver la fenêtre OpenCV (mode console pure)"
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Largeur de la fenêtre d'affichage en pixels (défaut: 960 pour s'adapter à l'écran sans déborder)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Chemin d'export d'une vidéo MP4 annotée avec les alertes visuelles (optionnel)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device : '0' (GPU CUDA) ou 'cpu'"
    )
    return parser.parse_args()


def natural_sort_key(s):
    """Tri naturel pour les suites d'images (frame_1, frame_2, ... frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def resolve_source(source_arg: str, base_dir: str = SCRIPT_DIR):
    """Résout le chemin d'une vidéo ou d'un dossier d'images."""
    candidates = [
        source_arg,
        os.path.join(base_dir, source_arg),
        os.path.join(os.path.dirname(base_dir), source_arg),
        os.path.join(os.path.dirname(os.path.dirname(base_dir)), source_arg),
        os.path.join(os.path.dirname(base_dir), "images and video", os.path.basename(source_arg)),
        os.path.join(base_dir, "images and video", os.path.basename(source_arg)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return source_arg


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout le modèle YOLO."""
    if not model_arg:
        model_arg = "best4_2"
    if os.path.isabs(model_arg) and os.path.exists(model_arg):
        return os.path.abspath(model_arg)
        
    cwd = os.getcwd()
    parent_dir = os.path.dirname(base_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    search_dirs = [cwd, base_dir, parent_dir, grandparent_dir]
    
    names = [model_arg]
    if not model_arg.lower().endswith(".pt"):
        names.append(model_arg + ".pt")
        
    for name in names:
        for s_dir in search_dirs:
            p1 = os.path.join(s_dir, name)
            p2 = os.path.join(s_dir, "best4", name)
            if os.path.exists(p1):
                return os.path.abspath(p1)
            if os.path.exists(p2):
                return os.path.abspath(p2)
    return model_arg


class FrameReader:
    """Lecteur unifié capable de lire soit une vidéo, soit un dossier d'images séquentielles."""
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.is_image_folder = os.path.isdir(source_path)
        self.image_files = []
        self.cap = None
        self.idx = 0
        
        if self.is_image_folder:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
            for ext in exts:
                self.image_files.extend(glob.glob(os.path.join(source_path, ext)))
            self.image_files.sort(key=natural_sort_key)
            self.total_frames = len(self.image_files)
            if self.total_frames == 0:
                raise ValueError(f"Aucune image trouvée dans le dossier : {source_path}")
            # Lire la 1ère image pour récupérer la taille
            sample = cv2.imread(self.image_files[0])
            self.height, self.width = sample.shape[:2]
            self.fps = 25.0
        else:
            self.cap = cv2.VideoCapture(source_path)
            if not self.cap.isOpened():
                raise ValueError(f"Impossible d'ouvrir la vidéo : {source_path}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def read(self):
        if self.is_image_folder:
            if self.idx >= self.total_frames:
                return False, None
            img = cv2.imread(self.image_files[self.idx])
            self.idx += 1
            return (img is not None), img
        else:
            ret, frame = self.cap.read()
            self.idx += 1
            return ret, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class PersistentTrackerValidator:
    """Gestionnaire de persistance temporelle assurant le déclenchement strict au bout de N trames."""
    def __init__(self, min_consecutive: int = 5, max_dist: float = 150.0, max_missed: int = 1):
        self.min_consecutive = min_consecutive
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.tracks = {}  # track_id -> dict
        
    def update(self, raw_detections, frame_idx):
        """
        raw_detections: list of dict {'id_piste': int, 'boite': [x,y,w,h], 'confiance': float}
        Retourne :
        - active_tracks: liste des pistes vues
        - new_expert_launches: liste des pistes franchissant le seuil N exactement à cette trame
        - dropped_transient_tracks: liste des pistes disparues AVANT d'atteindre N (fausses alertes filtrées)
        """
        seen_ids = set()
        new_expert_launches = []
        active_tracks = []
        
        for det in raw_detections:
            tid = det["id_piste"]
            bbox = det["boite"]
            conf = det["confiance"]
            seen_ids.add(tid)
            
            if tid not in self.tracks:
                # Nouvelle piste candidate
                self.tracks[tid] = {
                    "id": tid,
                    "bbox": bbox,
                    "conf": conf,
                    "consecutive": 1,
                    "confirmed": False,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "missed": 0
                }
            else:
                trk = self.tracks[tid]
                trk["bbox"] = bbox
                trk["conf"] = conf
                trk["last_frame"] = frame_idx
                trk["missed"] = 0
                trk["consecutive"] += 1
                
                # Déclenchement exact au seuil min_consecutive
                if trk["consecutive"] >= self.min_consecutive and not trk["confirmed"]:
                    trk["confirmed"] = True
                    new_expert_launches.append(trk)
                    
            active_tracks.append(self.tracks[tid])
            
        # Détection des pistes perdues ou disparues
        dropped_transient_tracks = []
        all_ids = list(self.tracks.keys())
        for tid in all_ids:
            if tid not in seen_ids:
                trk = self.tracks[tid]
                trk["missed"] += 1
                if trk["missed"] > self.max_missed:
                    # Si la piste disparaît avant d'avoir atteint N : c'était une vague/reflet !
                    if not trk["confirmed"]:
                        dropped_transient_tracks.append(trk)
                    del self.tracks[tid]
                    
        return active_tracks, new_expert_launches, dropped_transient_tracks


def main():
    args = parse_args()
    
    model_path = resolve_model_path(args.model)
    source_path = resolve_source(args.source)
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Modèle introuvable : {model_path}")
        sys.exit(1)
        
    if not os.path.exists(source_path):
        print(f"❌ Erreur : Source introuvable : {source_path}")
        sys.exit(1)
        
    try:
        reader = FrameReader(source_path)
    except Exception as e:
        print(f"❌ Erreur d'initialisation de la source : {e}")
        sys.exit(1)
        
    print("=" * 80)
    print("🚢 TEST DU DÉCLENCHEUR D'EXPERTS PAR PERSISTANCE TEMPORELLE")
    print("=" * 80)
    print(f"• Source de test        : {source_path}")
    print(f"• Type de source        : {'Dossier d images' if reader.is_image_folder else 'Fichier vidéo'}")
    print(f"• Total trames          : {reader.total_frames} frames ({reader.width}x{reader.height} @ {reader.fps:.1f} FPS)")
    print(f"• Modèle YOLO           : {model_path}")
    print(f"• Tracker               : {args.tracker} (Global Motion Compensation)")
    print(f"• Seuil de confirmation : {args.min_consecutive} images d'affilée pour lancer les experts")
    print(f"• Seuil de confiance    : {args.conf * 100:.0f}%")
    print(f"• Tolérance saut        : {args.max_missed} image(s)")
    print(f"• Device                : {args.device} ({'GPU CUDA' if args.device == '0' else 'CPU'})")
    print("=" * 80)
    print("💡 RÈGLE OPÉRATIONNELLE :")
    print(f"   -> Les détections < {args.min_consecutive} frames (vagues, écume) sont REJETÉES sans réveiller les experts.")
    print(f"   -> Dès qu'un bateau est vu {args.min_consecutive} fois d'affilée : ALERTE IMMÉDIATE 'LANCEMENT EXPERTS'.")
    print("-" * 80)
    print("Démarrage de l'analyse...\n")
    
    model = YOLO(model_path)
    validator = PersistentTrackerValidator(
        min_consecutive=args.min_consecutive,
        max_dist=args.max_dist,
        max_missed=args.max_missed
    )
    
    # Enregistreur vidéo optionnel
    writer = None
    if args.save:
        save_path = args.save if os.path.isabs(args.save) else os.path.join(SCRIPT_DIR, args.save)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, reader.fps, (reader.width, reader.height))
        print(f"💾 Vidéo annotée exportée en direct vers : {save_path}\n")

    # Configuration de la fenêtre graphique adaptée à l'écran
    window_name = f"Test Declenchement Experts USV (Persistance N={args.min_consecutive})"
    target_w = args.display_width
    target_h = int(target_w * (reader.height / float(reader.width)))
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, target_w, target_h)

    frame_idx = 0
    total_expert_triggers = 0
    total_waves_rejected = 0
    
    try:
        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                break
                
            frame_idx += 1
            
            # 1. Tracking avec YOLO
            results = model.track(
                frame,
                tracker=args.tracker,
                persist=True,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            
            # 2. Extraction des détections
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
                    tid = int(box.id[0]) if (box.id is not None) else 0
                    if tid > 0:
                        raw_detections.append({
                            "id_piste": tid,
                            "boite": [x, y, w, h],
                            "confiance": conf
                        })
                        
            # 3. Mise à jour de la persistance temporelle
            active, launches, dropped = validator.update(raw_detections, frame_idx)
            
            # 4. Affichage console clair et explicite
            # A. Artefacts / vagues rejetés
            for d in dropped:
                total_waves_rejected += 1
                print(f"[Trame {frame_idx:04d}] 🛡️  ARTEFACT/VAGUE REJETÉ : Piste #{d['id']} disparue après {d['consecutive']} frame(s) (< {args.min_consecutive}) -> EXPERTS NON LANCÉS")
                
            # B. Nouvel événement de déclenchement des experts
            for l in launches:
                total_expert_triggers += 1
                bx, by, bw, bh = l["bbox"]
                print("\n" + "=" * 80)
                print(f"🚨🚨🚨 [LANCEMENT DES EXPERTS !] 🚨🚨🚨")
                print(f"   • Trame            : {frame_idx}")
                print(f"   • Condition        : Bateau confirmé sur {args.min_consecutive} images consécutives !")
                print(f"   • Identifiant      : Bateau ID #{l['id']}")
                print(f"   • Confiance        : {l['conf'] * 100:.1f} %")
                print(f"   • Coordonnées boîte: [x={bx}, y={by}, largeur={bw}, hauteur={bh}]")
                print(f"   • Action USV       : Transmission immédiate aux modules de décision et d'évitement.")
                print("=" * 80 + "\n")
                
            # C. État courant si candidat en cours d'accumulation
            if len(active) > 0 and len(launches) == 0:
                for trk in active:
                    if not trk["confirmed"]:
                        print(f"[Trame {frame_idx:04d}] ⏳ Candidat en cours : Piste #{trk['id']} vue {trk['consecutive']}/{args.min_consecutive} fois (Conf: {trk['conf']*100:.0f}%) -> En attente de confirmation...")
                    else:
                        if frame_idx % 10 == 0:  # n'affiche que toutes les 10 trames pour ne pas polluer
                            print(f"[Trame {frame_idx:04d}] 🟢 [EXPERTS ACTIFS] Bateau ID #{trk['id']} suivi en continu (Persistance: {trk['consecutive']} frames)")
            elif len(active) == 0 and len(dropped) == 0:
                if frame_idx % 25 == 0:
                    print(f"[Trame {frame_idx:04d}] 🌊 Eau libre / Vagues (Aucun obstacle détecté)")

            # 5. Rendu Visuel OpenCV (HUD et boîtes colorées)
            if not args.no_show or writer is not None:
                display_frame = frame.copy()
                has_confirmed = any(trk["confirmed"] for trk in active)
                has_candidate = any(not trk["confirmed"] for trk in active)
                
                # Tracé des boîtes
                for trk in active:
                    x, y, w, h = trk["bbox"]
                    if trk["confirmed"]:
                        # Vert éclatant : Bateau confirmé -> Experts actifs
                        color = (0, 230, 0)
                        label = f"BATEAU #{trk['id']} [CONFIRME {trk['consecutive']}fr] -> EXPERTS ACTIFS"
                        thickness = 3
                    else:
                        # Orange / Jaune : Candidat en attente
                        color = (0, 165, 255)
                        label = f"Candidat #{trk['id']} [{trk['consecutive']}/{args.min_consecutive}fr] (En attente)"
                        thickness = 2
                        
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # Fond texte
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(display_frame, (x, max(0, y - th - 8)), (x + tw + 6, max(th + 8, y)), color, -1)
                    cv2.putText(display_frame, label, (x + 3, max(th + 2, y - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                                
                # Bandeau d'état supérieur (HUD)
                hud_h = 42
                if has_confirmed:
                    # Vert foncé si expert lancé
                    banner_color = (0, 140, 0)
                    banner_txt = f"🚨 EXPERTS DECLENCHES : BATEAU DETECTE ({args.min_consecutive}+ IMAGES D'AFFILEE)"
                elif has_candidate:
                    # Orange si un candidat est en cours de validation
                    banner_color = (0, 100, 200)
                    banner_txt = f"⏳ EN COURS DE VALIDATION : OBJET OBSERVE (< {args.min_consecutive} IMAGES - EXPERTS EN VEILLE)"
                else:
                    # Gris si eau libre
                    banner_color = (40, 40, 40)
                    banner_txt = f"🌊 EAU LIBRE : AUCUN OBSTACLE CONFIRME (EXPERTS EN VEILLE)"
                    
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], hud_h), banner_color, -1)
                cv2.putText(display_frame, banner_txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Info trame en bas à gauche
                info_txt = f"Frame: {frame_idx:04d}/{reader.total_frames} | Seuil persistance: N={args.min_consecutive} | Vagues rejetees: {total_waves_rejected}"
                cv2.putText(display_frame, info_txt, (14, display_frame.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                if writer is not None:
                    writer.write(display_frame)
                    
                if not args.no_show:
                    # Redimensionnement doux pour que 100% de la vidéo soit visible sans débordement
                    if display_frame.shape[1] != target_w:
                        show_frame = cv2.resize(display_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    else:
                        show_frame = display_frame
                    cv2.imshow(window_name, show_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("🛑 Arrêt manuel demandé par l'utilisateur.")
                        break
                        
    except KeyboardInterrupt:
        print("\n🛑 Interruption clavier.")
    finally:
        reader.release()
        if writer is not None:
            writer.release()
            print(f"🎬 Vidéo de démonstration finalisée : {save_path}")
        if not args.no_show:
            cv2.destroyAllWindows()
            
        print("\n" + "=" * 80)
        print("📊 BILAN DU TEST DE PERSISTANCE TEMPORELLE")
        print("=" * 80)
        print(f"• Trames totales analysées          : {frame_idx}")
        print(f"• Déclenchements d'experts validés  : {total_expert_triggers} fois (seuil {args.min_consecutive} images atteint)")
        print(f"• Faux positifs / vagues rejetés    : {total_waves_rejected} artefact(s) filtré(s) avant d'atteindre {args.min_consecutive} images")
        print("=" * 80)


if __name__ == "__main__":
    main()
