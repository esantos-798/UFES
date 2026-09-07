from auxiliary_functions import *
from epipolar_geometry import *
from triangulation import *
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

path = "./calibrations/0.json"

print(path)

with open(path) as f:
    data = json.load(f)

print(data.keys())


calibrations = []

for cam in range(4):
    cal = load_calibration(f"./calibrations/{cam}.json")
    calibrations.append(cal)

print("Número de câmeras:", len(calibrations))


for calibration in calibrations:
    P = projection_matrix(calibration)
    print("P :" ,  P)

F20 = fundamental_matrix(calibrations[2], calibrations[0])

print(F20)
print("Shape:", F20.shape)

print("Rank:", np.linalg.matrix_rank(F20))



cam_ref = 2

cam_other = [0,1,3]

# Load skeletons of the reference camera 
skel_2d_path = "./captures/skeletons_2d.json"
    
all_skeletons = []

for cam in range(4):

    skel = load_2d_points(
        skel_2d_path,
        cam
    )

    all_skeletons.append(skel)

print(np.array(all_skeletons).shape)

nskeletons = skel.shape[0]
njoints = skel.shape[1]

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


nose = skel[0][0].reshape(1,2)

draw_epilines(
    ax2,
    F20,
    nose,
    img2.shape[1],
    img2.shape[0],
    [colors[0]]
)

print("Nariz (camera 2):")
print(nose)

#plt.show()


# Nariz da pessoa 0 na câmera de referência
nose = skel[0][0]

ax1.scatter(
    nose[0],
    nose[1],
    s=250,
    c="red",
    marker="x",
    linewidths=3,
    label="Nariz origem"
)

# Nariz correspondente da pessoa 0 na câmera 0
nose0 = skel_dst[0][0][0]

ax2.scatter(
    nose0[0],
    nose0[1],
    s=180,
    c="lime",
    marker="o",
    linewidths=3,
    label="Nariz destino"
)


nose = np.array([[779.52, 36.82]], dtype=np.float32)

lines = cv2.computeCorrespondEpilines(
    nose.reshape(-1,1,2),
    1,      # imagem 1 (câmera 2)
    F20
)

print(lines)
print("Pessoa 0 - câmera 0:", skel_dst[0][0][0])
print("Pessoa 1 - câmera 0:", skel_dst[0][1][0])
print("Pessoa 0 - câmera 2:", skel[0][0])
print("Pessoa 1 - câmera 2:", skel[1][0])
nose0 = skel_dst[0][0][0]
nose1 = skel_dst[0][1][0]

ax2.scatter(nose0[0], nose0[1], s=180, c="lime", marker="o")
ax2.scatter(nose1[0], nose1[1], s=180, c="yellow", marker="o")
#plt.show()

for i, cal in enumerate(calibrations):

    R = cal["E"][:3,:3]

    print(f"\nCamera {i}")

    print("det(R) =", np.linalg.det(R))

    print("R @ R.T =")

    print(R @ R.T)

def camera_center(cal):

    R = cal["E"][:3,:3]
    t = cal["E"][:3,3]

    C = -R.T @ t

    return C   


for i, cal in enumerate(calibrations):

    print(camera_center(cal))

print(type(skel))
print(type(skel[0]))
print(type(skel[0][0]))

print(skel[0][0])

print(type(skel_dst))
print(type(skel_dst[0]))
print(type(skel_dst[0][0]))

print(skel_dst[0][0])

def point_line_distance(line, point):
    """
    Distância entre um ponto e uma linha.

    line = (a,b,c)
    point = (x,y)
    """

    a, b, c = line
    x, y = point

    return abs(a*x + b*y + c) / np.sqrt(a*a + b*b)    

def skeleton_epipolar_error(F, skeleton_ref, skeleton_target):

    error = 0.0

    valid = 0

    for p_ref, p_dst in zip(skeleton_ref, skeleton_target):

        if np.any(np.isnan(p_ref)):
            continue

        if np.any(np.isnan(p_dst)):
            continue

        line = epipolar_line(F, p_ref)

        error += point_line_distance(line, p_dst)

        valid += 1

    if valid == 0:
        return np.inf

    return error / valid

for i, cal in enumerate(calibrations):

    P = projection_matrix(cal)

    print(f"\nCamera {i}")

    print(P)   



P_list = []

points = []


print(np.array(all_skeletons).shape)


for cam in range(4):
    print(f"Camera {cam}:")
    print(all_skeletons[cam].shape)



P_list = []
points = []

for cam in range(4):

    P_list.append(
        projection_matrix(calibrations[cam])
    )

    pt = all_skeletons[cam][0][0]      # Pessoa 0, nariz

    print(f"Cam {cam}: {pt}")

    points.append(pt)

X = triangulate_point(P_list, points)

print("\nNariz 3D:")
print(X)   




skel3d = triangulate_skeleton(
    calibrations,
    all_skeletons,
    person=0
)

print(skel3d.shape)

print(skel3d)

skel3d_p0 = triangulate_skeleton(
    calibrations,
    all_skeletons,
    person=0
)

print(skel3d_p0.shape)

print(skel3d_p0)

skel3d_p1 = triangulate_skeleton(
    calibrations,
    all_skeletons,
    person=1
)

print(skel3d_p1.shape)

print(skel3d_p1)

fig = plt.figure(figsize=(8,8))
ax3d = fig.add_subplot(111, projection="3d")

draw_skeleton3D(ax3d, skel3d_p0.T, "blue")
draw_skeleton3D(ax3d, skel3d_p1.T, "green")

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")

plt.show()

Ps = [projection_matrix(cal) for cal in calibrations]
nose = []

for cam in range(4):
    nose.append(all_skeletons[cam][0][0])

observations = np.array(nose)    

nose3d = triangulate_point(Ps, observations)
proj = project_point(Ps[0], nose3d)

print("Observado :", observations[0])
print("Projetado :", proj)
calibrations = []

for cam in range(4):
    calibrations.append(
        load_calibration(f"./calibrations/{cam}.json")
    )

Ps = []

for cal in calibrations:
    Ps.append(projection_matrix(cal))

for cam in range(4):

    err = reprojection_error(
        Ps[cam],
        nose3d,
        observations[cam]
    )

    print(f"Camera {cam}: {err:.3f} px")


for cam in range(4):

    R = calibrations[cam]["E"][:3, :3]
    t = calibrations[cam]["E"][:3, 3]

    C = -R.T @ t

    print(f"Camera {cam}: {C}")   

