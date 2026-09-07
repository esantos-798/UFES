import cv2
import numpy as np
import trimesh
import pyrender
from PIL import Image

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width = frame.shape[:2]

# ==========================================
# ARUCO
# ==========================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# ==========================================
# CAMERA MATRIX
# ==========================================
fx = width
fy = width

cx = width / 2
cy = height / 2

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))

# ==========================================
# ARUCO MARKER SIZE
# ==========================================
marker_size = 0.05

obj_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# ==========================================
# SCENE
# ==========================================
scene = pyrender.Scene(bg_color=[0, 0, 0, 0])

# ==========================================
# CAMERA 3D
# ==========================================
camera = pyrender.IntrinsicsCamera(fx=fx,fy=fy,cx=cx,cy=cy)
cam_node = scene.add(camera)

# ==========================================
# LIGHT
# ==========================================
light = pyrender.DirectionalLight(color=np.ones(3),intensity=20.0)
scene.add(light)

# ==========================================
# RENDERER
# ==========================================
renderer = pyrender.OffscreenRenderer(width,height)

# ==========================================
# PREPARING THE OBJECTS
# ==========================================
def load_obj(path,scale=0.01,rot_x=0,rot_y=0,rot_z=0,height=0.02,max_texture=2048):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    try:
        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if hasattr(material, 'image'):
                img = material.image
                if img is not None:
                    img_np = np.array(img)
                    h, w = img_np.shape[:2]
                    print(f"Original Texture: {w}x{h}")
                    # if the texture is too big
                    if w > max_texture or h > max_texture:
                        print("Redimensioning texture...")
                        pil_img = Image.fromarray(img_np)
                        pil_img.thumbnail((max_texture, max_texture))

                        material.image = np.array(pil_img)
                        nh, nw = material.image.shape[:2]
                        print(f"New texture: {nw}x{nh}")
    except Exception as e:
        print("Error processing texture:")
        print(e)

    # centraliza
    mesh.apply_translation(-mesh.centroid)

    # ==========================
    # ROTATIONS
    # ==========================
    rx = trimesh.transformations.rotation_matrix(np.radians(rot_x),[1, 0, 0])
    ry = trimesh.transformations.rotation_matrix(np.radians(rot_y),[0, 1, 0])
    rz = trimesh.transformations.rotation_matrix(np.radians(rot_z),[0, 0, 1])

    mesh.apply_transform(rx)
    mesh.apply_transform(ry)
    mesh.apply_transform(rz)
    
    # ==========================
    # LIFTING
    # ==========================
    mesh.apply_translation([0, 0, height])

    # ==========================
    # SCALE
    # ==========================
    mesh.apply_scale(scale)

    return pyrender.Mesh.from_trimesh(mesh)

# ==========================================
# LOAD OBJECTS
# ==========================================
meshes = {}

# ArUco ID 0
meshes[0] = load_obj(
    "./objects/fox/low-poly-fox-by-pixelmannen.obj",
    scale=0.0005,
    rot_x=90,
    height=40
)

# ArUco ID 1
meshes[1] = load_obj(
    "./objects/rio/low_poly_river.obj",
    scale=0.02,
    rot_x=90,
    height=0
)

# ArUco ID 2
meshes[2] = load_obj(
    "./objects/praia/the_beach.obj",
    scale=0.02,
    rot_x=90,
    height=0
)

# ArUco ID 3
meshes[3] = load_obj(
    "./objects/birds/birds.obj",
    scale=0.02,
    rot_x=180,
    height=5
)

# ArUco ID 4
meshes[4] = load_obj(
    "./objects/chuva/rain_1.obj",
    scale=0.0002,
    rot_x=90,
    height=1000
)

# ArUco ID 5
meshes[5] = load_obj(
    "./objects/fogueira/printable_fire_pit.obj",
    scale=0.01,
    rot_x=90,
    height=0
)

# ArUco ID 6
meshes[6] = load_obj(
    "./objects/vila/village_low_poly.obj",
    scale=0.1,
    rot_x=90,
    height=0,
    max_texture=4096
)

# ArUco ID 7
meshes[7] = load_obj(
    "./objects/barco/rowing_boat.obj",
    scale=0.00002,
    rot_x=90,
    height=1000
)

# ==========================================
# CRIATE NODES 
# ==========================================
nodes = {}
for marker_id, mesh in meshes.items():
    node = scene.add(mesh,pose=np.eye(4))
    nodes[marker_id] = node

# ==========================================
# LOOP
# ==========================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    # ======================================
    # HIDE OBJECTS
    # ======================================
    for node in nodes.values():
        scene.set_pose(node,
            np.array([
                [1,0,0,1000],
                [0,1,0,1000],
                [0,0,1,1000],
                [0,0,0,1]
            ])
        )

    # ======================================
    # DETECTION
    # ======================================
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame,corners,ids)
        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            img_points = corners[i][0]
            success, rvec, tvec = cv2.solvePnP(obj_points,img_points,camera_matrix,dist_coeffs)
            if success:
                rot_mat, _ = cv2.Rodrigues(rvec)
                pose = np.eye(4)
                pose[:3, :3] = rot_mat
                pose[:3, 3] = tvec.flatten()

                # ==================================
                # CORRECT OPENGL AXIS
                # ==================================
                pose[1, :] *= -1
                pose[2, :] *= -1

                # ==================================
                # UPDATE NODE POSE 
                # ==================================
                if marker_id in nodes:
                    scene.set_pose(nodes[marker_id],pose)

    # ======================================
    # RENDER ONCE
    # ======================================
    color, depth = renderer.render(scene)

    # RGB -> BGR
    color = color[:, :, ::-1]

    # mask
    mask = depth > 0

    # mix
    frame[mask] = color[:, :, :3][mask]

    cv2.imshow("Augmented Reality with ArUcos", cv2.flip(frame,1))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# ==========================================
# END
# ==========================================
cap.release()
cv2.destroyAllWindows()