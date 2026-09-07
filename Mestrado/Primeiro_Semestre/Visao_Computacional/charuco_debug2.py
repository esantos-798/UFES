import numpy as np
import cv2
import glob
import zipfile

ZipFileName = 'canon_charuco.zip'
img_path    = './canon_charuco/'

with zipfile.ZipFile(ZipFileName, 'r') as z:
    z.extractall('./')

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# detecta marcadores ArUco na primeira imagem
fname = sorted(glob.glob(img_path + '*.jpg'))[0]
img   = cv2.imread(fname)
h0, w0 = img.shape[:2]
if w0 < h0:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector_params = cv2.aruco.DetectorParameters()
aruco_detector  = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
corners_a, ids_a, _ = aruco_detector.detectMarkers(gray)

if ids_a is not None:
    ids_sorted = sorted(ids_a.flatten().tolist())
    print(f'IDs detectados ({len(ids_sorted)} marcadores): {ids_sorted}')
    print(f'ID mínimo: {min(ids_sorted)},  ID máximo: {max(ids_sorted)}')

# testa combinações de tamanho de board até achar a que interpola cantos
square_size = 45
aruco_size  = 33

print('\nTestando tamanhos de board...')
for sq_x in range(4, 16):
    for sq_y in range(4, 16):
        board = cv2.aruco.CharucoBoard(
            (sq_x, sq_y), square_size, aruco_size, aruco_dict
        )
        detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        n = len(ch_ids) if ch_ids is not None else 0
        if n >= 4:
            print(f'  squaresX={sq_x}, squaresY={sq_y} → {n} cantos interpolados  ← CANDIDATO')