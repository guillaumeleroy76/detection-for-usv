"""
Script: live_pi_camera.py
Description: Affichage en direct sur PC du flux caméra du Raspberry Pi 4 (USV Cerema / IMT Atlantique).
             Permet la visualisation directe à faible latence (OpenCV / FFplay) et l'inférence YOLO temps réel.
Auteur: Guillaume
"""

import os
import sys
import time
import glob
import signal
import argparse
import subprocess
from datetime import datetime

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Affichage en direct sur PC du flux caméra Raspberry Pi (USV Cerema)"
    )
    # Paramètres de connexion réseau
    parser.add_argument(
        "--host",
        type=str,
        default="cerema-pi.local",
        help="Nom d'hôte ou adresse IP du Raspberry Pi (défaut: cerema-pi.local ou 192.168.137.200)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="pi",
        help="Nom d'utilisateur SSH sur le Raspberry Pi (défaut: pi)"
    )
    
    # Paramètres vidéo de la caméra
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Largeur de la résolution vidéo (défaut: 640 pour faible latence, ou 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Hauteur de la résolution vidéo (défaut: 480 ou 720)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Fréquence d'images par seconde (défaut: 25 fps)"
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=180,
        choices=[0, 90, 180, 270],
        help="Rotation du capteur (défaut: 180 car monté inversé sur l'USV)"
    )
    parser.add_argument(
        "--imx500",
        action="store_true",
        help="Activer le modèle interne embarqué sur capteur Sony IMX500"
    )
    
    # Modes d'affichage
    parser.add_argument(
        "--ffplay",
        action="store_true",
        help="Utiliser le lecteur ultra basse latence FFplay au lieu de la fenêtre OpenCV"
    )
    
    # Options IA / YOLO en direct sur le PC
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Activer la détection d'objets YOLO en direct sur le flux vidéo"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best4_2",
        help="Modèle YOLO à utiliser : alias ('best4', 'best4_2') ou chemin .pt (défaut: best4_2)"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.45,
        help="Seuil de confiance minimal pour afficher les détections (défaut: 0.45)"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Démarrer l'enregistrement vidéo local sur le PC dès l'ouverture"
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str, base_dir: str = SCRIPT_DIR) -> str:
    """Résout de manière flexible et robuste le chemin du modèle YOLO."""
    if not model_arg:
        model_arg = "best4_2"
    if os.path.isabs(model_arg) and os.path.exists(model_arg):
        return os.path.abspath(model_arg)
        
    cwd = os.getcwd()
    parent_dir = os.path.dirname(base_dir)          # detection
    grandparent_dir = os.path.dirname(parent_dir)   # ia
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


def test_ssh_connection(host: str, user: str) -> bool:
    """Teste si la connexion SSH au Raspberry Pi fonctionne."""
    try:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", f"{user}@{host}", "echo OK"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False


def run_ffplay_mode(args):
    """Lance le streaming vidéo direct via pipeline FFplay."""
    print("🎬 Démarrage du flux en mode FFplay (ultra basse latence)...")
    
    extra_flags = ""
    if args.imx500:
        extra_flags += " --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
        
    remote_cmd = (
        f"rpicam-vid -t 0 -n --inline --rotation {args.rotation} "
        f"--width {args.width} --height {args.height} --framerate {args.fps}{extra_flags} -o -"
    )
    
    pipeline_cmd = (
        f'ssh {args.user}@{args.host} "{remote_cmd}" | '
        f'ffplay -fflags nobuffer -flags low_delay -framedrop -probesize 32 -sync ext -'
    )
    
    print(f"Commande exécutée : {pipeline_cmd}")
    try:
        subprocess.run(pipeline_cmd, shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel du flux FFplay.")
    finally:
        # Nettoyage processus résiduels sur le Pi
        subprocess.run(["ssh", f"{args.user}@{args.host}", "killall -q rpicam-vid"], stderr=subprocess.DEVNULL)


def run_opencv_mode(args):
    """Récupère le flux MJPEG de la caméra Pi via SSH et l'affiche dans une fenêtre interactive OpenCV."""
    # 1. Chargement conditionnel du modèle YOLO si demandé
    yolo_enabled = args.yolo
    model = None
    labels = {}
    
    if yolo_enabled:
        try:
            from ultralytics import YOLO
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            model_path = resolve_model_path(args.model)
            print(f"🤖 Initialisation du modèle YOLO : {model_path} sur device {device}")
            model = YOLO(model_path)
            labels = model.names
            print("✅ Modèle YOLO prêt pour l'inférence temps réel sur PC.")
        except Exception as e:
            print(f"⚠️ Avertissement : Impossible d'activer YOLO ({e}). Affichage en mode brut.")
            yolo_enabled = False

    # 2. Construction de la commande distante rpicam-vid
    remote_cam_cmd = (
        f"rpicam-vid -t 0 -n --inline --codec mjpeg --rotation {args.rotation} "
        f"--width {args.width} --height {args.height} --framerate {args.fps} -o -"
    )
    ssh_cmd = ["ssh", f"{args.user}@{args.host}", remote_cam_cmd]

    print("=" * 70)
    print("📡 LANCEMENT DU FLUX CAMÉRA RASPBERRY PI EN DIRECT")
    print("=" * 70)
    print(f"• Hôte Pi            : {args.user}@{args.host}")
    print(f"• Résolution         : {args.width}x{args.height} @ {args.fps} FPS")
    print(f"• Rotation           : {args.rotation}°")
    print(f"• Détection YOLO     : {'Activee (' + args.model + ')' if yolo_enabled else 'Desactivee (appuyer sur [D] pour activer)'}")
    print(f"• Enregistrement     : {'Actif des le depart' if args.record else 'Inactif (appuyer sur [R] pour enregistrer)'}")
    print("-" * 70)
    print("🎮 CONTRÔLES CLAVIER DANS LA FENÊTRE VIDÉO :")
    print("   [Q] ou [Échap] : Quitter et libérer la caméra sur le Pi")
    print("   [S]            : Prendre une capture d'écran photo")
    print("   [R]            : Démarrer / Arrêter l'enregistrement vidéo")
    print("   [D]            : Activer / Désactiver la détection YOLO en direct")
    print("=" * 70)

    # Démarrage du sous-processus SSH avec capture du flux binaire stdout
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)

    # Palette de couleurs pour les boîtes de détection
    bbox_colors = [(0, 220, 0), (255, 140, 0), (0, 165, 255), (180, 105, 255), (255, 0, 128)]

    # Variables d'enregistrement
    recording = args.record
    video_writer = None
    if recording:
        rec_name = f"stream_pi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(rec_name, fourcc, args.fps, (args.width, args.height))
        print(f"🔴 Enregistrement vidéo en cours vers : {rec_name}")

    buffer = b""
    fps_history = []
    t_prev = time.time()

    try:
        while True:
            # Lecture par blocs binaires du flux MJPEG
            chunk = proc.stdout.read(4096)
            if not chunk:
                # Vérifier si le processus s'est arrêté avec une erreur
                if proc.poll() is not None:
                    err_out = proc.stderr.read().decode('utf-8', errors='ignore')
                    print(f"\n❌ Le flux distant s'est arrêté. Message du Pi : {err_out.strip()}")
                break
                
            buffer += chunk

            # Recherche des délimiteurs de trames JPEG (SOI: 0xFFD8, EOI: 0xFFD9)
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                # Extraction de la trame complète
                jpg_data = buffer[a:b+2]
                buffer = buffer[b+2:]

                # Décodage de l'image
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                t_now = time.time()
                instant_fps = 1.0 / max(1e-5, (t_now - t_prev))
                t_prev = t_now
                fps_history.append(instant_fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                mean_fps = np.mean(fps_history)

                display_frame = frame.copy()

                # --- 1. Inférence YOLO si activée ---
                if yolo_enabled and model is not None:
                    results = model(frame, conf=args.thresh, verbose=False)[0]
                    for idx, box in enumerate(results.boxes):
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = labels.get(cls_id, f"obj_{cls_id}")
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        color = bbox_colors[cls_id % len(bbox_colors)]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label_txt = f"{cls_name}: {conf:.2f}"
                        cv2.putText(
                            display_frame, label_txt, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
                        )

                # --- 2. Incrustation Informations Écran (OSD) ---
                # Bandeau d'état supérieur
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 35), (25, 25, 25), -1)
                
                status_txt = f"RPi Camera ({args.host}) | FPS: {mean_fps:4.1f} | YOLO: {'ON (' + args.model + ')' if yolo_enabled else 'OFF'}"
                cv2.putText(display_frame, status_txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

                # Indicateur d'enregistrement
                if recording:
                    cv2.circle(display_frame, (display_frame.shape[1] - 20, 18), 8, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (display_frame.shape[1] - 65, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

                # Écriture dans le fichier vidéo si enregistrement actif
                if recording and video_writer is not None:
                    video_writer.write(display_frame)

                # Affichage de la fenêtre OpenCV
                cv2.imshow("Flux Caméra Raspberry Pi (USV Cerema)", display_frame)

                # Gestion des touches clavier
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]: # 'q' ou Échap
                    print("\n🛑 Fermeture demandée par l'utilisateur.")
                    break
                elif key in [ord('s'), ord('S')]: # Capture d'écran
                    snapshot_name = f"capture_pi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snapshot_name, frame)
                    print(f"📸 Capture d'écran enregistrée : {snapshot_name}")
                elif key in [ord('r'), ord('R')]: # Basculer Enregistrement
                    recording = not recording
                    if recording:
                        rec_name = f"stream_pi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        video_writer = cv2.VideoWriter(rec_name, fourcc, args.fps, (args.width, args.height))
                        print(f"🔴 Enregistrement DÉMARRÉ : {rec_name}")
                    else:
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        print("⏹️  Enregistrement ARRÊTÉ.")
                elif key in [ord('d'), ord('D')]: # Basculer YOLO
                    if model is None:
                        try:
                            from ultralytics import YOLO
                            model_path = resolve_model_path(args.model)
                            model = YOLO(model_path)
                            labels = model.names
                            yolo_enabled = True
                            print(f"🤖 YOLO ACTIVÉ : {model_path}")
                        except Exception as e:
                            print(f"❌ Impossible d'activer YOLO : {e}")
                    else:
                        yolo_enabled = not yolo_enabled
                        print(f"🤖 YOLO {'ACTIVÉ' if yolo_enabled else 'DÉSACTIVÉ'}.")

            # Protection contre l'accumulation infinie du buffer si trames corrompues
            if len(buffer) > 5 * 1024 * 1024:
                buffer = buffer[-1024*1024:]

    except KeyboardInterrupt:
        print("\n🛑 Arrêt manuel par Ctrl+C.")
    finally:
        # Libération propre de toutes les ressources
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Envoi d'un signal d'arrêt propre au sous-processus SSH et arrêt de rpicam-vid sur le Pi
        print("🧹 Libération du capteur caméra sur le Raspberry Pi...")
        try:
            proc.terminate()
            proc.kill()
        except Exception:
            pass
            
        subprocess.run(
            ["ssh", f"{args.user}@{args.host}", "killall -q rpicam-vid"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Caméra libérée avec succès.")


def main():
    args = parse_args()
    
    print(f"🔍 Test de la connexion SSH vers {args.user}@{args.host}...")
    if not test_ssh_connection(args.host, args.user):
        print(f"⚠️  Attention : Impossible de contacter directement '{args.host}'.")
        print("💡 Vérifications recommandées :")
        print("   1. Le Raspberry Pi est-il bien allumé ?")
        print("   2. Êtes-vous connecté au même réseau Wi-Fi ou le câble Ethernet est-il branché ?")
        print("   3. Si vous êtes branché en câble direct sans Wi-Fi, tentez :")
        print("      python live_pi_camera.py --host 192.168.137.200")
        choice = input("\nVoulez-vous tenter de vous connecter quand même ? (O/n) : ").strip().lower()
        if choice == 'n':
            sys.exit(1)
            
    if args.ffplay:
        run_ffplay_mode(args)
    else:
        run_opencv_mode(args)


if __name__ == "__main__":
    main()
