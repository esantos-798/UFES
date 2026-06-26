from __future__ import annotations
import argparse, json, sys, colorsys
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import matplotlib
matplotlib.use("TkAgg")





# ── Skeleton metadata ────────────────────────────────────────
HKP_NAMES = {
    "NOSE": 0, "LEFT_EYE": 1, "RIGHT_EYE": 2, "LEFT_EAR": 3, "RIGHT_EAR": 4,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6, "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10, "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14, "LEFT_ANKLE": 15, "RIGHT_ANKLE": 16,
}
SHORT = {
    0:"NOS",1:"LEy",2:"REy",3:"LEa",4:"REa",5:"LSh",6:"RSh",7:"LEl",
    8:"REl",9:"LWr",10:"RWr",11:"LHi",12:"RHi",13:"LKn",14:"RKn",
    15:"LAn",16:"RAn",
}
LINKS = [
    (5,6),(5,11),(6,12),(11,12),(5,3),(6,4),(5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),(0,1),(1,3),(0,2),(2,4),
]

# ── Load calibration ────────────────────────────────────────
def load_calibration(json_path) -> dict:
    
    data = json.load(open(json_path))
    K = np.array(data["intrinsic"]["doubles"], dtype=np.float64).reshape(3,3)
    E = np.array(data["extrinsic"][0]["tf"]["doubles"], dtype=np.float64).reshape(4,4)
    w, h = int(data["resolution"]["width"]), int(data["resolution"]["height"])
    return {"K": K, "E": E, "w": w, "h": h}

# ── Load 2D skeleton points ─────────────────────────────────
    
def load_2d_points(json_path, cam_id: int) -> np.ndarray:
    """
    Return an array containing:

        (n_skeletons, n_keypoints, 2)
    """

    data = json.load(open(json_path))
    cam_key = f"camera_{cam_id}"

    if cam_key not in data:
        return np.empty((0, len(HKP_NAMES), 2), dtype=np.float64)

    skeletons = []

    # Sort skeleton_0, skeleton_1, ...
    for skel_name in sorted(
        data[cam_key].keys(),
        key=lambda s: int(s.split("_")[1])
    ):
        skel_data = data[cam_key][skel_name]
        pts = np.full((len(HKP_NAMES), 2),
                      np.nan,
                      dtype=np.float64)
        for kp_name, uv in skel_data.items():
            if kp_name in HKP_NAMES:
                kp_idx = HKP_NAMES[kp_name]
                pts[kp_idx] = uv                
        skeletons.append(pts)

    return np.stack(skeletons, axis=0)  


# ── Epipolar line helpers ────────────────────────────────────
def epipolar_line(F, pt):
    return F @ np.array([pt[0], pt[1], 1.0])

# ── Colors ───────────────────────────────────────────────────
def make_colors(n):
    return [colorsys.hsv_to_rgb((i*0.618)%1, 0.85, 0.95) for i in range(n)]

# ── Drawing ──────────────────────────────────────────────────

def draw_skeleton(ax, pts, colors):
    
    for (i,j) in LINKS:
        ax.plot([pts[i,0],pts[j,0]], [pts[i,1],pts[j,1]],
                    color=(0.9,0.9,0.9), lw=1.5, alpha=0.5, zorder=1)


        if i in pts and j in pts:
            ax.plot([pts[i][0],pts[j][0]], [pts[i][1],pts[j][1]],
                    color=(0.9,0.9,0.9), lw=1.5, alpha=0.5, zorder=1)
    for i in range(pts.shape[0]):
        c = colors[i % len(colors)]
        ax.scatter(pts[i,0], pts[i,1], c=[c], s=60, edgecolors="white", lw=1, zorder=3)
        ax.annotate(SHORT.get(i,str(i)), (pts[i,0], pts[i,1]),
                    textcoords="offset points", xytext=(6,6), fontsize=6,
                    fontweight="bold", color=c,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, ec="none"),
                    zorder=4)
    
def draw_epilines(ax,F, pts_src, w, h, colors):

  for i in range(pts_src.shape[0]):
    c = colors[i % len(colors)]
    line = epipolar_line(F, pts_src[i])

    # calculating points on the epipolar line 
    t = np.linspace(0,w,100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    
    # take only line points inside the image 
    ndx = (lt>=0) & (lt<h)
    ax.plot(t[ndx],lt[ndx],color=c, lw=1.5, alpha=0.8, ls="--", zorder=2)

def draw_skeleton3D(ax, pts, color="blue"):
    for (i,j) in LINKS:
        ax.plot([pts[0,i],pts[0,j]], [pts[1,i],pts[1,j]], [pts[2,i],pts[2,j]], color)
        ax.plot([pts[0,i],pts[0,j]], [pts[1,i],pts[1,j]], [pts[2,i],pts[2,j]], "*r")
    return
    

# ── Example of using the functions ─────────

cam_ref = 2

cam_other = [0,1,3]

# Load skeletons of the reference camera 
skel_2d_path = "./captures/skeletons_2d.json"
    
skel_src = load_2d_points(skel_2d_path, cam_ref)

nskeletons = skel_src.shape[0]
njoints = skel_src.shape[1]

skel_dst = []
# Load skeletons of the other cameras
skel_dst.append(load_2d_points(skel_2d_path, cam_other[0]))
skel_dst.append(load_2d_points(skel_2d_path, cam_other[1]))
skel_dst.append(load_2d_points(skel_2d_path, cam_other[2]))




colors = make_colors(njoints)

    

# Load images
img1 = cv2.cvtColor(cv2.imread("./captures/" + str(f"camera_{cam_ref}_clean.jpg")), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread("./captures/" + str(f"camera_{cam_other[0]}_clean.jpg")), cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(cv2.imread("./captures/" + str(f"camera_{cam_other[1]}_clean.jpg")), cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(cv2.imread("./captures/" + str(f"camera_{cam_other[2]}_clean.jpg")), cv2.COLOR_BGR2RGB)

# Create figure with images of the cameras
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle(f"Epipolar Geometry and Skeletons",
            fontsize=16, fontweight="bold", color="white", y=0.98)

ax1.imshow(img1); 
ax1.set_title(f"Camera {cam_ref} — 2D Skeletons", fontsize=13, fontweight="bold", color="white", pad=10)
ax2.imshow(img2); 
ax2.set_title(f"Camera {cam_other[0]} — 2D Skeletons", fontsize=13, fontweight="bold", color="white", pad=10)
ax3.imshow(img3); 
ax3.set_title(f"Camera {cam_other[1]} — 2D Skeletons", fontsize=13, fontweight="bold", color="white", pad=10)
ax4.imshow(img4); 
ax4.set_title(f"Camera {cam_other[2]} — 2D Skeletons", fontsize=13, fontweight="bold", color="white", pad=10)

for skel_id in range(nskeletons):

    draw_skeleton(ax1, skel_src[skel_id], colors)
    ax1.text(skel_src[skel_id,0,0],skel_src[skel_id,0,1]-40,f"Skel {skel_id}",
            color="#00FF00",fontweight="bold")
    draw_skeleton(ax2, skel_dst[0][skel_id], colors)
    ax2.text(skel_dst[0][skel_id,0,0],skel_dst[0][skel_id,0,1]-40, f"Skel {skel_id}",
            color="#00FF00",fontweight="bold")
    draw_skeleton(ax3, skel_dst[1][skel_id], colors)
    ax3.text(skel_dst[1][skel_id,0,0],skel_dst[1][skel_id,0,1]-40, f"Skel {skel_id}",
            color="#00FF00",fontweight="bold")
    draw_skeleton(ax4, skel_dst[2][skel_id], colors)
    ax4.text(skel_dst[2][skel_id,0,0],skel_dst[2][skel_id,0,1]-40, f"Skel {skel_id}",
            color="#00FF00",fontweight="bold")


ax1.set_xlim(0, img1.shape[1]); ax1.set_ylim(img1.shape[0], 0)
ax2.set_xlim(0, img2.shape[1]); ax2.set_ylim(img2.shape[0], 0)
ax3.set_xlim(0, img3.shape[1]); ax3.set_ylim(img3.shape[0], 0)
ax4.set_xlim(0, img4.shape[1]); ax4.set_ylim(img4.shape[0], 0)
    
plt.show()


        

        

