import numpy as np
import cv2
import matplotlib.pyplot as plt

from auxiliary_functions import (
    load_calibration,
    load_2d_points,
    draw_skeleton,
    draw_epilines,
    draw_skeleton3D,
    make_colors,
)
from epipolar_geometry import (
    fundamental_matrix,
    projection_matrix,
    best_matches,
)
from triangulation import (
    triangulate_skeleton,
    reprojection_error,
)

# ── 1. Load calibrations and 2D detections ──────────────────────────
calibrations = [load_calibration(f"./calibrations/{cam}.json") for cam in range(4)]

skel_2d_path = "./captures/skeletons_2d.json"
all_skeletons = [load_2d_points(skel_2d_path, cam) for cam in range(4)]

for cam in range(4):
    print(f"Camera {cam}: {all_skeletons[cam].shape}")

# ── 2. Choice ref camera ────────────────────────────────
cam_ref = 2
cam_others = [0, 1, 3]

njoints = all_skeletons[cam_ref].shape[1]
colors = make_colors(njoints)

# ── 3. Skeleton matches via epipolar geoemetry ─────────
correspondence = {cam_ref: {p: p for p in range(all_skeletons[cam_ref].shape[0])}}
F_matrices = {}

for cam in cam_others:
    F = fundamental_matrix(calibrations[cam_ref], calibrations[cam])
    F_matrices[cam] = F

    matches, cost = best_matches(F, all_skeletons[cam_ref], all_skeletons[cam])

    print(f"\nCusto epipolar (Camera {cam_ref} -> Camera {cam}):")
    print(cost)

    correspondence[cam] = {src: dst for src, dst, err in matches}

    for src, dst, err in matches:
        print(f"  Pessoa {src} (cam {cam_ref}) -> Pessoa {dst} (cam {cam})  erro = {err:.4f} px")

# ── 4. View: 2D skeletons + epipolar lines ──────────────
imgs = {
    cam: cv2.cvtColor(cv2.imread(f"./captures/camera_{cam}_clean.jpg"), cv2.COLOR_BGR2RGB)
    for cam in range(4)
}

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor("#1a1a2e")
fig.suptitle("Epipolar Geometry and Skeletons", fontsize=16, fontweight="bold", color="white", y=0.98)

ax_map = {cam_ref: axes[0, 0], cam_others[0]: axes[0, 1],
          cam_others[1]: axes[1, 0], cam_others[2]: axes[1, 1]}

for cam, ax in ax_map.items():
    ax.imshow(imgs[cam])
    ax.set_title(f"Camera {cam}", fontsize=13, fontweight="bold", color="white", pad=10)
    ax.set_xlim(0, imgs[cam].shape[1])
    ax.set_ylim(imgs[cam].shape[0], 0)

for person in range(all_skeletons[cam_ref].shape[0]):
    draw_skeleton(ax_map[cam_ref], all_skeletons[cam_ref][person], colors)
    ax_map[cam_ref].text(
        all_skeletons[cam_ref][person, 0, 0],
        all_skeletons[cam_ref][person, 0, 1] - 40,
        f"Pessoa {person}", color="#00FF00", fontweight="bold",
    )

for cam in cam_others:
    ax = ax_map[cam]
    for person in range(all_skeletons[cam_ref].shape[0]):
        dst_idx = correspondence[cam][person]
        draw_skeleton(ax, all_skeletons[cam][dst_idx], colors)
        ax.text(
            all_skeletons[cam][dst_idx, 0, 0],
            all_skeletons[cam][dst_idx, 0, 1] - 40,
            f"Pessoa {person}", color="#00FF00", fontweight="bold",
        )

        nose_ref = all_skeletons[cam_ref][person, 0].reshape(1, 2)
        if not np.any(np.isnan(nose_ref)):
            draw_epilines(ax, F_matrices[cam], nose_ref,
                           imgs[cam].shape[1], imgs[cam].shape[0], [colors[0]])

plt.tight_layout()

# ── 5. 3D triangulation using established correspondence ────────
skeletons3d = {}

for person in range(all_skeletons[cam_ref].shape[0]):
    indices_per_cam = {cam_ref: correspondence[cam_ref][person]}
    for cam in cam_others:
        indices_per_cam[cam] = correspondence[cam][person]

    skel3d = triangulate_skeleton(calibrations, all_skeletons, indices_per_cam)
    skeletons3d[person] = skel3d

    print(f"\nEsqueleto 3D - Pessoa {person}:")
    print(skel3d)

# ── 6. Validatin: reprojection error ─────────────────────────────────
Ps = [projection_matrix(cal) for cal in calibrations]

for person, skel3d in skeletons3d.items():
    print(f"\nErros de reprojeção - Pessoa {person}:")
    for cam in range(4):
        idx = correspondence[cam][person]
        errs = []
        for joint in range(njoints):
            obs = all_skeletons[cam][idx][joint]
            if np.any(np.isnan(obs)):
                continue
            err = reprojection_error(Ps[cam], skel3d[joint], obs)
            errs.append(err)
        if errs:
            print(f"  Camera {cam}: media = {np.mean(errs):.3f} px  (n={len(errs)})")
        else:
            print(f"  Camera {cam}: sem juntas validas")

# ── 7. 3D skeletons ──────────────────────────────────
fig3d = plt.figure(figsize=(8, 8))
ax3d = fig3d.add_subplot(111, projection="3d")

person_colors = ["blue", "green", "red", "orange"]
for person, skel3d in skeletons3d.items():
    draw_skeleton3D(ax3d, skel3d.T, person_colors[person % len(person_colors)])

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("Reconstrucao 3D dos esqueletos")

plt.show()
