"""
Module: image_degradation.py
Description: Moteur de simulation de dégradations d'images pour caméras USV
Projet: USV Cerema - IMT Atlantique
Auteur: Guillaume

Ce module implémente des dégradations réalistes subies par une caméra embarquée
sur un drone de surface (USV) en milieu marin :
1. Gouttes d'eau et embruns marins (refraction, flou local, halo de bordure, reflet spéculaire).
2. Pixels noirs aléatoires et masquages (dropout de capteur, occlusion, saleté séchée/sel).
3. Flou de défocalisation et de mouvement (houle, caméra mal réglée).
4. Baisse de résolution / sous-échantillonnage (mode basse consommation / flux dégradé).
5. Brouillard maritime / brume (perte de contraste et diffusion atmosphérique).

Complexité algorithmique générale:
- Temps: O(H * W) par image traitée (linéaire en nombre de pixels).
- Espace mémoire: O(H * W) copie temporaire du tableau d'image.
"""

from typing import Tuple, Optional, Union, List
import os
import cv2
import numpy as np


def load_image_safe(filepath: str) -> np.ndarray:
    """
    Charge une image de façon robuste sous Windows, même avec des accents dans le chemin.
    
    Args:
        filepath: Chemin absolu ou relatif vers le fichier image.
        
    Returns:
        Image au format BGR (np.ndarray uint8).
        
    Raises:
        FileNotFoundError: Si le fichier est introuvable ou illisible.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier image introuvable : {filepath}")
    
    with open(filepath, "rb") as f:
        bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        
    if img is None:
        raise ValueError(f"Impossible de décoder l'image : {filepath}")
    return img


def save_image_safe(filepath: str, image: np.ndarray) -> bool:
    """
    Sauvegarde une image sous Windows même avec des accents dans le chemin.
    
    Args:
        filepath: Chemin de destination.
        image: Image BGR à sauvegarder.
        
    Returns:
        True si succès, False sinon.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    ext = os.path.splitext(filepath)[1]
    if not ext:
        ext = ".jpg"
        filepath += ext
    is_success, buffer = cv2.imencode(ext, image)
    if is_success:
        with open(filepath, "wb") as f:
            f.write(buffer)
        return True
    return False


def add_water_droplets(
    image: np.ndarray,
    severity: float = 0.5,
    num_drops: Optional[int] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simule la présence de gouttes d'eau et d'embruns sur l'objectif de la caméra USV.
    
    Chaque goutte est modélisée optiquement par :
    - Un flou local et une distorsion de réfraction.
    - Un liseré sombre de bordure (réflexion interne du ménisque).
    - Un reflet spéculaire brillant (soleil / ciel marin).

    Args:
        image: Image d'entrée (H, W, 3) BGR.
        severity: Degré de dégradation normalisé dans [0.0, 1.0].
        num_drops: Nombre explicite de gouttes (si None, déduit de severity).
        seed: Graine aléatoire pour reproductibilité des tests.

    Returns:
        Image dégradée avec gouttes d'eau.
    """
    if severity <= 0.0:
        return image.copy()
        
    if seed is not None:
        np.random.seed(seed)
        
    h, w = image.shape[:2]
    result = image.copy()
    
    # Nombre de gouttes proportionnel à la sévérité et à la taille de l'image
    base_drops = int(10 + severity * 80)
    count = num_drops if num_drops is not None else base_drops
    
    # Rayon des gouttes (proportionnel à la résolution min)
    min_dim = min(h, w)
    min_radius = max(4, int(min_dim * 0.015))
    max_radius = max(min_radius + 4, int(min_dim * (0.04 + 0.05 * severity)))
    
    for _ in range(count):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        r = np.random.randint(min_radius, max_radius + 1)
        
        # Découpe de la boîte englobante de la goutte
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r + 1)
        y2 = min(h, cy + r + 1)
        
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
            
        patch = result[y1:y2, x1:x2].copy()
        ph, pw = patch.shape[:2]
        
        # Grille relative au centre de la boîte
        yc, xc = np.ogrid[:ph, :pw]
        dist_sq = (xc - (cx - x1)) ** 2 + (yc - (cy - y1)) ** 2
        drop_mask = dist_sq <= (r ** 2)
        
        if not np.any(drop_mask):
            continue
            
        # 1. Réfraction / flou local à l'intérieur de la goutte
        # Kernel impair pour le flou
        ksize = max(3, (int(r * 0.7) // 2) * 2 + 1)
        blurred_patch = cv2.GaussianBlur(patch, (ksize, ksize), 0)
        
        # Légère amplification de contraste / luminosité simulant l'effet loupe
        magnified = cv2.convertScaleAbs(blurred_patch, alpha=1.08, beta=5)
        
        # Application du patch réfracté
        patch[drop_mask] = magnified[drop_mask]
        
        # 2. Liseré sombre du ménisque (périphérie de la goutte : r*0.82 <= dist <= r)
        rim_mask = (dist_sq >= ((r * 0.82) ** 2)) & drop_mask
        patch[rim_mask] = (patch[rim_mask].astype(np.float32) * 0.55).astype(np.uint8)
        
        # 3. Reflet spéculaire blanc (soleil ou ciel)
        # Positionné vers le haut/gauche de la goutte (source lumineuse zénithale)
        glare_cx = int(cx - x1 - 0.30 * r)
        glare_cy = int(cy - y1 - 0.30 * r)
        glare_r = max(1, int(0.18 * r))
        
        glare_dist_sq = (xc - glare_cx) ** 2 + (yc - glare_cy) ** 2
        glare_mask = glare_dist_sq <= (glare_r ** 2)
        glare_mask = glare_mask & drop_mask
        
        # Reflet spéculaire blanc doux avec fondu
        if np.any(glare_mask):
            patch[glare_mask] = np.clip(
                0.25 * patch[glare_mask].astype(np.float32) + 0.75 * 255.0,
                0, 255
            ).astype(np.uint8)
        
        result[y1:y2, x1:x2] = patch
        
    return result


def add_black_pixels(
    image: np.ndarray,
    severity: float = 0.5,
    mode: str = "combined",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simule la présence de pixels noirs aléatoires et d'occlusions partielles.
    
    Modes supportés :
    - 'salt_pepper' (ou 'dropout'): pixels isolés noirs aléatoires (erreurs capteur/transmission).
    - 'cutout' (ou 'patches'): blocs noirs opaques (saleté, dépôt de sel, algue séchée).
    - 'combined': combinaison réaliste de pixels morts et de petites taches d'occlusion.

    Args:
        image: Image d'entrée (H, W, 3) BGR.
        severity: Degré de dégradation normalisé dans [0.0, 1.0].
        mode: Type de dégradation ('salt_pepper', 'cutout', 'combined').
        seed: Graine aléatoire pour reproductibilité.

    Returns:
        Image dégradée avec pixels noirs / masquages.
    """
    if severity <= 0.0:
        return image.copy()
        
    if seed is not None:
        np.random.seed(seed)
        
    h, w = image.shape[:2]
    result = image.copy()
    
    # Mode pixels noirs isolés (dropout / salt & pepper noir)
    if mode in ["salt_pepper", "combined"]:
        # Ratio de pixels noirs : de 0.5% (severity=0.1) à 25% (severity=1.0)
        pixel_ratio = 0.005 + severity * 0.22
        num_pixels = int(h * w * pixel_ratio)
        
        y_coords = np.random.randint(0, h, num_pixels)
        x_coords = np.random.randint(0, w, num_pixels)
        result[y_coords, x_coords] = 0
        
    # Mode occlusion par blocs (cutout / taches opaques de sel ou saleté)
    if mode in ["cutout", "combined"]:
        num_patches = int(2 + severity * 14)
        max_patch_size = max(8, int(min(h, w) * (0.04 + 0.10 * severity)))
        
        for _ in range(num_patches):
            pw = np.random.randint(5, max_patch_size)
            ph = np.random.randint(5, max_patch_size)
            px = np.random.randint(0, max(1, w - pw))
            py = np.random.randint(0, max(1, h - ph))
            
            # Forme rectangulaire ou circulaire aléatoire
            if np.random.rand() > 0.5:
                result[py:py + ph, px:px + pw] = 0
            else:
                center = (px + pw // 2, py + ph // 2)
                radius = min(pw, ph) // 2
                cv2.circle(result, center, radius, (0, 0, 0), -1)
                
    return result


def add_blur(
    image: np.ndarray,
    severity: float = 0.5,
    blur_type: str = "gaussian"
) -> np.ndarray:
    """
    Simule une défocalisation (optique mal réglée) ou un flou de bougé (vague / USV).
    
    Args:
        image: Image d'entrée BGR.
        severity: Degré de dégradation normalisé dans [0.0, 1.0].
        blur_type: 'gaussian' (défocalisation) ou 'motion' (houle/mouvement).

    Returns:
        Image floutée.
    """
    if severity <= 0.0:
        return image.copy()
        
    if blur_type == "gaussian":
        # Kernel size impair variant de 3 (légère perte de netteté) à 45 (fort flou)
        k = int(3 + severity * 38)
        if k % 2 == 0:
            k += 1
        sigma = 1.0 + severity * 9.0
        return cv2.GaussianBlur(image, (k, k), sigma)
        
    elif blur_type == "motion":
        # Noyau linéaire incliné simulant le roulis du drone marin
        k = int(5 + severity * 30)
        kernel = np.zeros((k, k), dtype=np.float32)
        # Ligne diagonale simulant le mouvement
        np.fill_diagonal(kernel, 1.0 / k)
        return cv2.filter2D(image, -1, kernel)
        
    return image.copy()


def add_downsampling(
    image: np.ndarray,
    severity: float = 0.5
) -> np.ndarray:
    """
    Simule une baisse drastique de résolution (mode caméra basse consommation / flux éco).
    
    L'image est sous-échantillonnée puis ré-agrandie à sa résolution nominale.
    
    Args:
        image: Image d'entrée BGR.
        severity: Degré de dégradation normalisé dans [0.0, 1.0].

    Returns:
        Image pixelisée / basse résolution.
    """
    if severity <= 0.0:
        return image.copy()
        
    h, w = image.shape[:2]
    # Facteur de réduction de x1.5 (severity=0.1) à x16 (severity=1.0)
    scale = 1.0 / (1.0 + severity * 15.0)
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    
    low_res = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_NEAREST)
    return restored


def add_fog_haze(
    image: np.ndarray,
    severity: float = 0.5
) -> np.ndarray:
    """
    Simule la brume marine ou le brouillard côtier (diffusion lumineuse, perte de contraste).
    
    Args:
        image: Image d'entrée BGR.
        severity: Degré de dégradation normalisé dans [0.0, 1.0].

    Returns:
        Image avec brume/brouillard.
    """
    if severity <= 0.0:
        return image.copy()
        
    alpha = min(0.85, severity * 0.85)
    fog_layer = np.full_like(image, 215, dtype=np.uint8)
    return cv2.addWeighted(image, 1.0 - alpha, fog_layer, alpha, 0)


def degrade_image(
    image: np.ndarray,
    degradation_type: str,
    severity: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Fonction chapeau unifiée appliquant une dégradation choisie selon une sévérité [0, 1].
    
    Types supportés:
    - 'gouttes' ou 'water_droplets'
    - 'pixels_noirs' ou 'black_pixels'
    - 'flou' ou 'blur'
    - 'resolution' ou 'downsampling'
    - 'brouillard' ou 'fog'

    Args:
        image: Image BGR uint8.
        degradation_type: Nom de la dégradation.
        severity: Flottant entre 0.0 et 1.0.
        seed: Graine optionnelle pour reproductibilité.

    Returns:
        Image modifiée.
    """
    dtype = degradation_type.lower().strip()
    
    if dtype in ["gouttes", "gouttes_eau", "water", "water_droplets"]:
        return add_water_droplets(image, severity=severity, seed=seed)
    elif dtype in ["pixels_noirs", "black_pixels", "dropout", "pixels_noirs_aleatoires"]:
        return add_black_pixels(image, severity=severity, mode="combined", seed=seed)
    elif dtype in ["flou", "blur", "defocus"]:
        return add_blur(image, severity=severity, blur_type="gaussian")
    elif dtype in ["resolution", "downsampling", "basse_resolution"]:
        return add_downsampling(image, severity=severity)
    elif dtype in ["brouillard", "fog", "brume", "haze"]:
        return add_fog_haze(image, severity=severity)
    else:
        raise ValueError(
            f"Type de dégradation inconnu : '{degradation_type}'. "
            f"Choisir parmi : ['gouttes', 'pixels_noirs', 'flou', 'resolution', 'brouillard']"
        )
