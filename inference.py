import cv2
import torch
import matplotlib.pyplot as plt
from modules.xfeat import XFeat

DATA_PATH = "assets/"

reference = "ref.png"
target = "tgt.png"

# 1. Inicializar modelo
xfeat = XFeat()

# 2. Cargar imágenes
img0 = cv2.imread(DATA_PATH + reference)
img1 = cv2.imread(DATA_PATH + target)

# CASO 1: Ambas imágenes se cargaron -> Emparejamiento
if img0 is not None and img1 is not None:
    mkpts0, mkpts1 = xfeat.match_xfeat(img0, img1)
    print(f"Puntos coincidentes encontrados: {len(mkpts0)}")

    matched_img = cv2.drawMatches(
        img0, [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts0],
        img1, [cv2.KeyPoint(p[0], p[1], 1) for p in mkpts1],
        [cv2.DMatch(i, i, 0) for i in range(len(mkpts0))],
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Mostrar con Matplotlib en lugar de cv2.imshow
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("XFeat Matches")
    plt.show()

# CASO 2: Solo una imagen disponible -> Keypoints individuales
else:
    img = img0 if img0 is not None else img1
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar ninguna imagen desde {DATA_PATH}")

    outputs = xfeat.detectAndCompute(img, top_k=4096)[0]
    keypoints = outputs['keypoints']

    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    print(f"Keypoints extraídos: {keypoints.shape[0]}")

    cv_kpts = [cv2.KeyPoint(pt[0], pt[1], 3) for pt in keypoints]
    img_kpts = cv2.drawKeypoints(
        img, cv_kpts, None, 
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Mostrar con Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_kpts, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("XFeat Single Image Keypoints")
    plt.show()