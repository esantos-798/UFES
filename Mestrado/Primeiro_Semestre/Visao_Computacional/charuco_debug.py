import numpy as np
import cv2
import glob
import zipfile

# ── configuração ──────────────────────────────────────────────────────────────
ZipFile     = True
ZipFileName = 'canon_charuco.zip'
img_path    = './canon_charuco/'

if ZipFile:
    with zipfile.ZipFile(ZipFileName, 'r') as z:
        z.extractall('./')

# ── parâmetros — AJUSTE ESTES se o diagnóstico mostrar 0 detecções ────────────
square_size = 45
aruco_size  = 33
l, c        = 9, 7   # linhas x colunas de CANTOS INTERNOS

# ── monta o board ChArUco ─────────────────────────────────────────────────────
aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board = cv2.aruco.CharucoBoard(
    (c + 1, l + 1), square_size, aruco_size, aruco_dict
)
charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

# ══════════════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO — mostra o que está sendo detectado em cada imagem
# ══════════════════════════════════════════════════════════════════════════════
images = sorted(glob.glob(img_path + '*.jpg'))

for fname in images[:5]:   # testa nas primeiras 5 imagens
    img  = cv2.imread(fname)
    h0, w0 = img.shape[:2]
    if w0 < h0:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f'\n── {fname}  ({img.shape[1]}×{img.shape[0]}) ──')

    # --- chessboard ---
    ret, corners = cv2.findChessboardCorners(
        gray, (c, l),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS
    )
    print(f'  Chessboard ({c}×{l} cantos internos): {"OK" if ret else "FALHOU"}')

    # testa tamanhos alternativos para ajudar a descobrir o certo
    for tc, tl in [(6,8),(8,6),(6,9),(9,6),(7,10),(10,7)]:
        r2, _ = cv2.findChessboardCorners(gray, (tc, tl), None)
        if r2:
            print(f'    → detectou com ({tc}×{tl}) — tente l={tl}, c={tc}')

    # --- ArUco sozinho (sem ChArUco) ---
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector  = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners_a, ids_a, _ = aruco_detector.detectMarkers(gray)
    n_aruco = len(ids_a) if ids_a is not None else 0
    print(f'  Marcadores ArUco detectados: {n_aruco}')

    # --- ChArUco completo ---
    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
    n_ch = len(ch_ids) if ch_ids is not None else 0
    print(f'  Cantos ChArUco interpolados: {n_ch}  (mínimo necessário: 4)')

    # mostra visualmente
    vis = img.copy()
    if ids_a is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners_a, ids_a)
    if ch_ids is not None and len(ch_ids) >= 4:
        cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

    scale = min(1.0, 900 / max(vis.shape[:2]))
    vis_s = cv2.resize(vis, None, fx=scale, fy=scale)
    cv2.imshow(f'diagnóstico — {fname.split("/")[-1]}', vis_s)
    cv2.waitKey(0)

cv2.destroyAllWindows()