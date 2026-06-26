# -*- coding: utf-8 -*-
"""
charuco_calibration.py

Camera Calibration using ChArUco Board
"""

import numpy as np
import cv2
import glob
import zipfile
import time
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use("TkAgg")

########################################################
# Complementary functions for ploting points and vectors with Y-axis swapped with Z-axis
def set_plot(ax=None, figure=None, figsize=(9, 8),
             limx=[-2, 2], limy=[-2, 2], limz=[-2, 2]):
    if figure is None:
        figure = plt.figure(figsize=figsize)
    if ax is None:
        ax = plt.axes(projection='3d')

    ax.set_xlim(limx);  ax.set_xlabel("x axis")
    ax.set_ylim(limy);  ax.set_ylabel("y axis")
    ax.set_zlim(limz);  ax.set_zlabel("z axis")
    return ax

def draw_arrows(point, base, axis, length=1.5):
    axis.quiver(point[0], point[1], point[2],
                base[0, 0], base[1, 0], base[2, 0],
                color='red',   pivot='tail', length=length)
    axis.quiver(point[0], point[1], point[2],
                base[0, 1], base[1, 1], base[2, 1],
                color='green', pivot='tail', length=length)
    axis.quiver(point[0], point[1], point[2],
                base[0, 2], base[1, 2], base[2, 2],
                color='blue',  pivot='tail', length=length)
    return axis

### Opções de impressão numpy
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3, suppress=True)
#########################################################

"""#Read images and detect features"""

# Flag to read from zip file
ZipFile     = True
ZipFileName = 'canon_charuco.zip'
img_path    = './canon_charuco/'

SHOW_MATPLOTLIB = False
SHOW_CV         = True

# ChArUco Board parameters
square_size = 0.045   # cm → m
aruco_size  = 0.033   # cm → m
sq_x = 9              # squares direction X
sq_y = 7              # squares direction Y

# Read zip file and unzip images
if ZipFile:
    with zipfile.ZipFile(ZipFileName, 'r') as z:
        z.extractall('./')

# Configuração do ChArUco Board e detector
aruco_dict       = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board    = cv2.aruco.CharucoBoard(
    (sq_x, sq_y), square_size, aruco_size, aruco_dict
)
charuco_detector = cv2.aruco.CharucoDetector(charuco_board)

# Arrays to store object points and image points from all the images.
all_charuco_corners = []   
all_charuco_ids     = []   

# Read images for calibration
images   = sorted(glob.glob(img_path + '*.jpg'))
img_size = None
print('Número de imagens lidas: ', len(images))

# Matplotlib view
if SHOW_MATPLOTLIB:
    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.show()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    imsize = img.shape
    print('Tamanho da imagem: ', imsize[0], ' ', imsize[1])
    if imsize[1] < imsize[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]   # (W, H)

    if SHOW_MATPLOTLIB:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    vis = img.copy()

    # ChArUco detection
    ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)

    if ch_ids is not None and len(ch_ids) >= 4:
        all_charuco_corners.append(ch_corners)
        all_charuco_ids.append(ch_ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)

        if SHOW_MATPLOTLIB:
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            plt.imshow(vis_rgb)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

    if SHOW_CV:
        scale = min(1.0, 900 / max(vis.shape[:2]))
        cv2.imshow('Detecções ChArUco', cv2.resize(vis, None, fx=scale, fy=scale))
        cv2.waitKey(100)

if SHOW_CV:
    cv2.destroyAllWindows()
if SHOW_MATPLOTLIB:
    plt.ioff()

print('Imagens válidas para calibração:', len(all_charuco_corners))
assert len(all_charuco_corners) >= 4, 'Imagens válidas insuficientes para calibração.'

"""# Executar Calibração da Câmera
## Imprimir Matriz Intrínseca
## Imprimir coeficientes de distorção radial
"""

obj_pts_all = []
img_pts_all = []

for ch_corners, ch_ids in zip(all_charuco_corners, all_charuco_ids):
    obj_pts, img_pts = charuco_board.matchImagePoints(ch_corners, ch_ids)
    # matchImagePoints return (N,1,3) e (N,1,2) — reshape to (N,3) e (N,1,2)

    obj_pts_all.append(obj_pts.reshape(-1, 1, 3).astype(np.float32))
    img_pts_all.append(img_pts.reshape(-1, 1, 2).astype(np.float32))

# # Run camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts_all, img_pts_all, img_size, None, None
)

# Print calibration results
print('Number of images where the corners were detected', len(all_charuco_corners))
print('Calibration Matrix')
print(mtx)
print('Radil distortion coeficients')
print(dist)
print('Ret')
print(ret)

# Organize the extrinsic parameters
transl = np.hstack(tvecs)
rot    = np.hstack(rvecs)
print(transl.shape)
print(rot.shape)

"""## Corrigir distorção em uma imagem de exemplo
## Calcular erro de reprojeção
"""

img_sample = cv2.imread(images[0])
h0, w0 = img_sample.shape[:2]
if w0 < h0:
    img_sample = cv2.rotate(img_sample, cv2.ROTATE_90_CLOCKWISE)

img_rgb = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
h, w    = img_rgb.shape[:2]
print('Tamanho da imagem de exemplo: ', h, ' ', w)

# Undistort one image as example
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst          = cv2.undistort(img_rgb, mtx, dist, None, new_mtx)
x, y, wc, hc = roi
dst_crop     = dst[y:y+hc, x:x+wc]

# Show original image and corrected image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].set_title("Imagem Original")
axes[0].imshow(img_rgb)
axes[1].set_title("Imagem Corrigida (Undistorted)")
axes[1].imshow(dst_crop)
plt.show()

# Calculate the reprojection error
mean_error = 0
for i in range(len(obj_pts_all)):
    imgpoints2, _ = cv2.projectPoints(obj_pts_all[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_pts_all[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Erro de reprojeção total: ", mean_error / len(obj_pts_all))

"""# Visualizar Parâmetros Extrínsecos
## Padrão fixo na origem / câmera se movendo
"""

# Create base vector values
e1 = np.array([[1], [0], [0], [0]])
e2 = np.array([[0], [1], [0], [0]])
e3 = np.array([[0], [0], [1], [0]])
base   = np.hstack((e1, e2, e3))
# origin point
origin = np.array([[0], [0], [0], [1]])
# Create camera frame and world frame
cam    = np.hstack([base, origin])
world  = np.hstack([base, origin])

axis0 = set_plot(limx=[-0.5, 0.5], limy=[-0.5, 0.5], limz=[-1.0, 0.0])
axis0.set_title('Calibração ChArUco – Padrão Fixo / Câmera Movendo')

arrow_len = 0.06   # in meters

# Plot camera frames considering that the calibration patern is on the XY-plane and Z=0
for i in range(rot.shape[1]):
    R, _ = cv2.Rodrigues(rot[:, i])
    t    = transl[:, i]
    Rt   = np.eye(4)
    Rt[0:3, 0:3] = R
    Rt[0:3, -1]  = t
    M       = np.linalg.inv(Rt)
    new_cam = M @ cam
    axis0   = draw_arrows(new_cam[:, -1], new_cam[:, 0:3], axis0, arrow_len)

# Plot the calibration pattern
board_pts = charuco_board.getChessboardCorners()   
axis0.scatter(board_pts[:, 0], board_pts[:, 1], board_pts[:, 2],
              s=5, c='black', zorder=5)

axis0.view_init(elev=-60, azim=-111, roll=23)
plt.show()

"""##Considering the camera fixed at the origin and moving the calibration pattern"""

axis1 = set_plot(limx=[-0.5, 0.5], limy=[-0.5, 0.5], limz=[-0.1, 1.0])
axis1.set_title('Calibração ChArUco – Câmera Fixa / Padrão Movendo')
axis1 = draw_arrows(cam[:, -1], cam[:, 0:3], axis1, arrow_len)

# 3D point coordenates
board_pts_h = board_pts.T                                         
board_pts_h = np.vstack([board_pts_h, np.ones(board_pts_h.shape[1])])  

for i in range(rot.shape[1]):
    R, _ = cv2.Rodrigues(rot[:, i])
    t    = transl[:, i]
    Rt   = np.eye(4)
    Rt[0:3, 0:3] = R
    Rt[0:3, -1]  = t

    pts_cam = Rt @ board_pts_h
    axis1.scatter(pts_cam[0, :], pts_cam[1, :], pts_cam[2, :], s=3)

axis1.view_init(elev=-60, azim=-142, roll=53)
plt.show()
