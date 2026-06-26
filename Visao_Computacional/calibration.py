import numpy as np
import cv2
import glob
import zipfile
import matplotlib.pyplot as plt

# ── parâmetros do enunciado ──────────────────────────────────────────────────
ZipFile     = True
ZipFileName = 'canon_charuco.zip'
img_path    = './canon_charuco/'

SHOW_CV     = True

square_size = 0.045  # 4.5 cm -> 0.045 m
aruco_size  = 0.033  # 3.3 cm -> 0.033 m

sq_x = 9
sq_y = 7

# ── descompactar ──────────────────────────────────────────────────────────────
if ZipFile:
    with zipfile.ZipFile(ZipFileName, 'r') as z:
        z.extractall('./')

# ── Configuração do ChArUco Board ─────────────────────────────────────────────
aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board    = cv2.aruco.CharucoBoard(
    (sq_x, sq_y), square_size, aruco_size, aruco_dict
)
charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

# Acumuladores para ChArUco
all_charuco_corners = []
all_charuco_ids     = []

# ══════════════════════════════════════════════════════════════════════════════
# LOOP DE DETECÇÃO
# ══════════════════════════════════════════════════════════════════════════════
images = sorted(glob.glob(img_path + '*.jpg'))
print(f'Imagens encontradas: {len(images)}')

img_size = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
        
    # Ajuste de orientação (garante que todas fiquem em landscape se necessário)
    h0, w0 = img.shape[:2]
    if w0 < h0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1] # Salva o tamanho da imagem (W, H)

    vis = img.copy()

    # Detecção exclusiva ChArUco
    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
    
    # São necessários ao menos 4 cantos para estimar a pose
    if ch_ids is not None and len(ch_ids) >= 4:
        all_charuco_corners.append(ch_corners)
        all_charuco_ids.append(ch_ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

    if SHOW_CV:
        scale = min(1.0, 900 / max(vis.shape[:2]))
        cv2.imshow('Deteccoes ChArUco', cv2.resize(vis, None, fx=scale, fy=scale))
        cv2.waitKey(50)

if SHOW_CV:
    cv2.destroyAllWindows()

print(f'ChArUco — Imagens válidas para calibração: {len(all_charuco_corners)}')
assert len(all_charuco_corners) >= 4, 'Imagens válidas insuficientes para calibração.'

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRAÇÃO UNIVERSAL (Compatível com qualquer versão do OpenCV)
# ══════════════════════════════════════════════════════════════════════════════
obj_pts_all = []
img_pts_all = []

for ch_corners, ch_ids in zip(all_charuco_corners, all_charuco_ids):
    # Associa os cantos 2D detectados aos seus respectivos pontos 3D no objeto
    obj_pts, img_pts = charuco_board.matchImagePoints(ch_corners, ch_ids)
    obj_pts_all.append(obj_pts)
    img_pts_all.append(img_pts)

# Calibração padrão do OpenCV utilizando os pontos emparelhados
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts_all, img_pts_all, img_size, None, None
)

print('\n=== RESULTADO DA CALIBRAÇÃO ===')
print(f'Erro de Reprojeção Total: {ret:.4f} pixels')
print('\nMatriz Intrínseca (K):\n', mtx)
print('\nCoeficientes de Distorção (D):\n', dist.ravel())

# ══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO DE DISTORÇÃO (Exemplo com a primeira imagem)
# ══════════════════════════════════════════════════════════════════════════════
img_sample = cv2.imread(images[0])
h, w = img_sample.shape[:2]
if w < h:
    img_sample = cv2.rotate(img_sample, cv2.ROTATE_90_CLOCKWISE)
    h, w = img_sample.shape[:2]

new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img_sample, mtx, dist, None, new_mtx)

# Plot comparativo final
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original com Distorção')
axes[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
axes[1].set_title('Corrigida (Undistorted)')
plt.show()