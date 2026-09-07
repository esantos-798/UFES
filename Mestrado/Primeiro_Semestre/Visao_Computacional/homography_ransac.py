import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pillow_heif
from PIL import Image, ImageOps
import os
import glob

def normalize_points(pts):
    centroid = np.mean(pts, axis=0)
    shifted  = pts - centroid

    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale     = np.sqrt(2) / (mean_dist + 1e-10)

    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,         0,                    1]
    ])

    pts_h    = np.column_stack([pts, np.ones(len(pts))])
    pts_norm = (T @ pts_h.T).T[:, :2]

    return pts_norm, T


def compute_homography_dlt(pts1, pts2):
    assert len(pts1) >= 4 and len(pts2) >= 4, "Need at least 4 point correspondences"

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    N = len(pts1_norm)
    A = np.zeros((2 * N, 9))

    for i in range(N):
        x, y   = pts1_norm[i]
        xp, yp = pts2_norm[i]

        A[2 * i]     = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]
        A[2 * i + 1] = [x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp]

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    H = np.linalg.inv(T2) @ H_norm @ T1

    if np.abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]

    return H


def compute_homography_ransac(pts1, pts2,
                              num_iterations=2000,
                              inlier_threshold=5.0,
                              min_inliers=10):
    N = len(pts1)
    assert N >= 4, "Need at least 4 correspondences"

    best_H       = None
    best_mask    = np.zeros(N, dtype=bool)
    best_inliers = 0

    for _ in range(num_iterations):
        idx    = np.random.choice(N, 4, replace=False)
        s_pts1 = pts1[idx]
        s_pts2 = pts2[idx]

        try:
            H_cand = compute_homography_dlt(s_pts1, s_pts2)
        except np.linalg.LinAlgError:
            continue

        pts1_h = np.column_stack([pts1, np.ones(N)]).T
        pts2_proj_h = H_cand @ pts1_h

        w = pts2_proj_h[2, :]
        w[np.abs(w) < 1e-10] = 1e-10
        
        pts2_proj = (pts2_proj_h[:2, :] / w).T
        errors = np.sqrt(np.sum((pts2 - pts2_proj) ** 2, axis=1))

        mask     = errors < inlier_threshold
        n_inliers = np.sum(mask)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_mask    = mask
            best_H       = H_cand

    if best_inliers >= min_inliers:
        pts1_in = pts1[best_mask]
        pts2_in = pts2[best_mask]
        H = compute_homography_dlt(pts1_in, pts2_in)
    else:
        print(f"RANSAC: apenas {best_inliers} inliers encontrados. Retornando melhor candidato.")
        pts1_in = pts1[best_mask]
        pts2_in = pts2[best_mask]
        H = best_H

    return H, pts1_in, pts2_in


def buscar_e_carregar_imagem(nome_base):
    """Busca a imagem e corrige orientações EXIF indesejadas (comum em fotos de celular)."""
    nome_sem_ext = os.path.splitext(nome_base)[0]
    arquivos_encontrados = glob.glob(f"{nome_sem_ext}.*")
    
    if not arquivos_encontrados:
        raise FileNotFoundError(f"Não foi possível encontrar nenhum arquivo correspondente a '{nome_sem_ext}'")
    
    caminho_arquivo = arquivos_encontrados[0]
    print(f"Carregando e tratando orientação: {caminho_arquivo}")
    
    try:
        img_pil = Image.open(caminho_arquivo)
        img_pil = ImageOps.exif_transpose(img_pil)  # Corrige a rotação física dos pixels
        img_pil = img_pil.convert('RGB')
        
        img_np = np.array(img_pil)
        return cv.cvtColor(img_np, cv.COLOR_RGB2GRAY)
    except Exception as e:
        raise ValueError(f"Erro ao processar a imagem {caminho_arquivo}: {e}")


# --- Configurações Iniciais ---
MIN_MATCH_COUNT = 10
pillow_heif.register_heif_opener()

nome_img1 = 'imag_1' 
nome_img2 = 'imag_2'

img1 = buscar_e_carregar_imagem(nome_img1)
img2 = buscar_e_carregar_imagem(nome_img2)

# --- Detecção de Características (SIFT) ---
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --- Casamento de Características (FLANN) ---
FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann   = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

img4 = np.zeros_like(img2)

# --- Estimação Geométrica ---
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    M, _, _ = compute_homography_ransac(src_pts, dst_pts)
    
    if M is not None:
        h_dest, w_dest = img2.shape[:2]
        img4 = cv.warpPerspective(img1, M, (w_dest, h_dest))
    else:
        print("RANSAC falhou em estimar a matriz homográfica.")
else:
    print(f"Correspondências insuficientes encontradas - {len(good)}/{MIN_MATCH_COUNT}")

# --- Geração Gráfica ---
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(20, 12))

axs[0, 0].imshow(img3, 'gray')
axs[0, 0].set_title('Correspondências (Matches)')
axs[0, 0].axis('off')

axs[0, 1].imshow(img1, 'gray')
axs[0, 1].set_title('First image')

axs[1, 0].imshow(img2, 'gray')
axs[1, 0].set_title('Second image')

axs[1, 1].imshow(img4, 'gray')
axs[1, 1].set_title('First image after warping')

plt.tight_layout()
plt.show()