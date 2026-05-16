import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import pi, cos, sin

# =============================================================================
# Funções auxiliares de plot (mantidas do código original)
# =============================================================================

def set_plot(ax=None, figure=None, lim=[-2, 2]):
    if figure is None:
        figure = plt.figure(figsize=(8, 8))
    if ax is None:
        ax = plt.axes(projection='3d')
    ax.set_xlim(lim); ax.set_xlabel("x axis")
    ax.set_ylim(lim); ax.set_ylabel("y axis")
    ax.set_zlim(lim); ax.set_zlabel("z axis")
    return ax

def draw_arrows(point, base, axis, length=1.5):
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',  pivot='tail',length=length)
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',length=length)
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue', pivot='tail',length=length)
    return axis

np.set_printoptions(precision=3, suppress=True)

# =============================================================================
# Transformações (mantidas do código original)
# =============================================================================

def move (dx,dy,dz):
    T = np.eye(4)
    T[0,-1] = dx
    T[1,-1] = dy
    T[2,-1] = dz
    return T

### Rotation
def z_rotation(angle):
    angle = angle*pi/180
    rotation_matrix=np.array([[cos(angle),-sin(angle),0,0],[sin(angle),cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
    return rotation_matrix

def x_rotation(angle):
    angle = angle*pi/180
    rotation_matrix=np.array([[1,0,0,0],[0, cos(angle),-sin(angle),0],[0, sin(angle), cos(angle),0],[0,0,0,1]])
    return rotation_matrix

def y_rotation(angle):
    angle = angle*pi/180
    rotation_matrix=np.array([[cos(angle),0, sin(angle),0],[0,1,0,0],[-sin(angle), 0, cos(angle),0],[0,0,0,1]])
    return rotation_matrix

def own_z_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[cos(angle),-sin(angle),0,0],[sin(angle),cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
    cam = cam@rotation_matrix
    return rotation_matrix, cam

def own_x_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[1,0,0,0],[0, cos(angle),-sin(angle),0],[0, sin(angle), cos(angle),0],[0,0,0,1]])
    cam = cam@rotation_matrix
    return rotation_matrix, cam

def own_y_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[cos(angle),0, sin(angle),0],[0,1,0,0],[-sin(angle), 0, cos(angle),0],[0,0,0,1]])
    cam = cam@rotation_matrix
    return rotation_matrix, cam

# =============================================================================
# Objeto: Robô (formato polilinha, compatível com plot3D)
# =============================================================================

def make_box_lines(cx, cy, cz, dx, dy, dz):
    x0,x1 = cx-dx,cx+dx; y0,y1 = cy-dy,cy+dy; z0,z1 = cz-dz,cz+dz
    nan = [float('nan')]*3
    pts = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z0], nan,
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[x0,y0,z1], nan,
        [x0,y0,z0],[x0,y0,z1], nan,
        [x1,y0,z0],[x1,y0,z1], nan,
        [x1,y1,z0],[x1,y1,z1], nan,
        [x0,y1,z0],[x0,y1,z1], nan,
    ]
    return np.array(pts).T

def make_cylinder_lines(cx, cy, cz_bot, cz_top, radius, segments=8):
    angles = np.linspace(0, 2*np.pi, segments+1)
    nan = [float('nan')]*3
    pts = []
    for a in angles: pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_bot])
    pts.append(nan)
    for a in angles: pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_top])
    pts.append(nan)
    for a in angles[::2]:
        pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_bot])
        pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_top])
        pts.append(nan)
    return np.array(pts).T

def concatenate_parts(*parts):
    nan_sep = np.full((3,1), np.nan)
    return np.hstack([p for part in parts for p in (part, nan_sep)])

def build_robot(scale=3.5):
    B  = make_box_lines
    CZ = make_cylinder_lines
    xyz = concatenate_parts(
        CZ( 0,    0,    4.10,4.50,0.04,6),  # antena haste
        CZ( 0,    0,    4.48,4.65,0.10,8),  # antena bola
        B( 0,    0,    3.70,0.38,0.35,0.40),# cabeça
        B(-0.18,-0.34, 3.82,0.10,0.03,0.09),# olho esq
        B( 0.18,-0.34, 3.82,0.10,0.03,0.09),# olho dir
        B( 0,   -0.34, 3.48,0.20,0.03,0.05),# boca
        B(-0.42, 0,    3.70,0.05,0.12,0.12),# orelha esq
        B( 0.42, 0,    3.70,0.05,0.12,0.12),# orelha dir
        CZ( 0,   0,    3.20,3.30,0.12,8),   # pescoço
        B( 0,    0,    2.55,0.55,0.38,0.65),# tronco
        B( 0,   -0.37, 2.70,0.30,0.03,0.35),# painel
        B(-0.15,-0.38, 2.85,0.06,0.03,0.06),# botão 1
        B( 0.15,-0.38, 2.85,0.06,0.03,0.06),# botão 2
        B( 0,   -0.38, 2.60,0.06,0.03,0.06),# botão 3
        CZ(-0.70,0,    2.95,3.05,0.14,8),   # ombro esq
        CZ( 0.70,0,    2.95,3.05,0.14,8),   # ombro dir
        B(-0.82, 0,    2.55,0.14,0.13,0.38),# braço esq
        B(-0.85, 0,    2.02,0.12,0.11,0.24),# antebraço esq
        B(-0.86,-0.02, 1.74,0.13,0.12,0.13),# mão esq
        B( 0.82, 0,    2.70,0.14,0.13,0.28),# braço dir
        B( 0.85,-0.22, 2.35,0.12,0.24,0.12),# antebraço dir
        B( 0.86,-0.48, 2.33,0.14,0.14,0.14),# mão dir
        B( 0,    0,    1.88,0.50,0.35,0.12),# quadril
        B(-0.27, 0,    1.52,0.20,0.18,0.34),# coxa esq
        B(-0.27, 0,    1.02,0.17,0.16,0.28),# canela esq
        B(-0.27,-0.18, 0.70,0.17,0.30,0.09),# pé esq
        B( 0.27, 0,    1.52,0.20,0.18,0.34),# coxa dir
        B( 0.27, 0,    1.02,0.17,0.16,0.28),# canela dir
        B( 0.27,-0.18, 0.70,0.17,0.30,0.09),# pé dir
        CZ(-0.27,0,    1.27,1.33,0.13,8),   # joelho esq
        CZ( 0.27,0,    1.27,1.33,0.13,8),   # joelho dir
    ) * scale
    hom = np.where(np.isnan(xyz[0,:]), np.nan, 1.0)
    return np.vstack([xyz, hom])

# =============================================================================
# Projeção perspectiva
# =============================================================================

def perspective_projection(cam, obj, K):
    """Projeta obj (4xN) com a câmera cam e matriz intrínseca K."""
    Proj  = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    G     = np.linalg.inv(cam)
    Mproj = K @ Proj @ G

    valid = ~np.isnan(obj[0,:])
    ping  = np.full((3, obj.shape[1]), np.nan)
    ping[:,valid] = Mproj @ obj[:,valid]

    p_img = np.full_like(ping, np.nan)
    nz = valid & (np.abs(ping[2,:]) > 1e-6)
    p_img[:,nz] = ping[:,nz] / ping[2,nz]
    return p_img

# =============================================================================
# Plot completo (3D + imagem)
# =============================================================================

def plot_scene(cam, cam_initial, world, obj, K, w_pixels, h_pixels, title=""):
    fig = plt.figure(figsize=(15,7))
    fig.suptitle(title)

    # -- Imagem projetada --
    p_img = perspective_projection(cam, obj, K)
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Image")
    ax1.set_xlim([0, w_pixels])
    ax1.set_ylim([h_pixels, 0])   # origem no canto superior esquerdo
    ax1.plot(p_img[0,:], p_img[1,:], 'red')
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_xlabel("u (pixels)"); ax1.set_ylabel("v (pixels)")

    # -- Cena 3D --
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    set_plot(ax=ax2, figure=fig, lim=[-15,20])
    ax2.plot3D(obj[0,:], obj[1,:], obj[2,:], 'red')
    ax2.set_title("3D Scene")
    draw_arrows(world[:,-1],  world[:,0:3],       ax2, 3.0)   # frame mundo
    draw_arrows(cam_initial[:,-1], cam_initial[:,0:3], ax2, 1.5) # frame inicial
    draw_arrows(cam[:,-1],    cam[:,0:3],          ax2, 1.5)  # frame atual

    plt.tight_layout()

# =============================================================================
# Leitura de parâmetros do terminal
# =============================================================================

def read_float(prompt, default):
    """Lê um float do terminal; usa default se o usuário pressionar Enter."""
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return float(val) if val != "" else default
    except ValueError:
        print(f"  Valor inválido, usando {default}.")
        return default

def read_int(prompt, default):
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return int(val) if val != "" else default
    except ValueError:
        print(f"  Valor inválido, usando {default}.")
        return default

def read_intrinsics(K_current, w_pixels, h_pixels):
    """Permite ao usuário alterar os parâmetros intrínsecos."""
    print("\n--- Parâmetros Intrínsecos ---")
    print("  (pressione Enter para manter o valor atual)")

    f        = read_float("Distância focal f (mm)", 20)
    w_cco    = read_float("Largura do sensor (mm)", 36)
    h_cco    = read_float("Altura do sensor  (mm)", 24)
    w_pixels = read_int  ("Largura da imagem (px)", w_pixels)
    h_pixels = read_int  ("Altura da imagem  (px)", h_pixels)
    skew     = read_float("Fator de skew", 0)

    sx = w_pixels / w_cco
    sy = h_pixels / h_cco
    ox = w_pixels / 2       # ponto principal: centro da imagem
    oy = h_pixels / 2

    K = np.array([[f*sx, skew, ox],
                  [0,    f*sy, oy],
                  [0,    0,    1 ]])

    print(f"\n  Matriz K:\n{K}")
    print(f"  Ponto principal: ({ox:.1f}, {oy:.1f})")
    return K, w_pixels, h_pixels

def read_extrinsics(cam):
    """Menu de movimentação da câmera por terminal."""
    print("\n--- Mover Câmera ---")
    print("  Referencial: (1) Mundo  (2) Câmera")
    ref = input("  Escolha [1]: ").strip()
    use_own = (ref == "2")

    print("\n  Tipo de transformação:")
    print("  (1) Translação   (2) Rotação X   (3) Rotação Y   (4) Rotação Z")
    tipo = input("  Escolha [1]: ").strip()

    if tipo == "2":
        angle = read_float("Ângulo de rotação em X (graus)", 0)
        if use_own:
            _, cam = own_x_rotation(angle, cam)
        else:
            cam = x_rotation(angle) @ cam
    elif tipo == "3":
        angle = read_float("Ângulo de rotação em Y (graus)", 0)
        if use_own:
            _, cam = own_y_rotation(angle, cam)
        else:
            cam = y_rotation(angle) @ cam
    elif tipo == "4":
        angle = read_float("Ângulo de rotação em Z (graus)", 0)
        if use_own:
            _, cam = own_z_rotation(angle, cam)
        else:
            cam = z_rotation(angle) @ cam
    else:
        dx = read_float("Translação dx", 0)
        dy = read_float("Translação dy", 0)
        dz = read_float("Translação dz", 0)
        if use_own:
            _, cam = own_move(dx, dy, dz, cam)
        else:
            cam = move(dx, dy, dz) @ cam

    return cam

# =============================================================================
# Loop principal
# =============================================================================

if __name__ == "__main__":

    # --- Frames iniciais ---
    e1 = np.array([[1],[0],[0],[0]])
    e2 = np.array([[0],[1],[0],[0]])
    e3 = np.array([[0],[0],[1],[0]])
    base   = np.hstack((e1,e2,e3))
    origin = np.array([[0],[0],[0],[1]])

    world = np.hstack([base, origin])
    cam   = np.hstack([base, origin])   # câmera começa na origem
    cam_initial = cam.copy()

    # --- Objeto ---
    obj = build_robot(scale=3.5)

    # --- Posição inicial da câmera (igual ao código original) ---
    print("=== Posicionamento inicial da câmera ===")
    _, cam = own_move(15, -5, 6, cam)
    _, cam = own_z_rotation(90, cam)
    _, cam = own_x_rotation(-90, cam)
    cam_initial = cam.copy()

    # --- Parâmetros intrínsecos iniciais ---
    w_pixels, h_pixels = 1280, 720
    f=20; w_cco=36; h_cco=24; skew=0
    sx=w_pixels/w_cco; sy=h_pixels/h_cco
    ox=w_pixels/2;     oy=h_pixels/2
    K = np.array([[f*sx,skew,ox],[0,f*sy,oy],[0,0,1]])

    print("\n=== Camera Visualizer – Robô ===")
    print("Parâmetros intrínsecos iniciais:")
    print(f"  f={f}mm, sensor={w_cco}x{h_cco}mm, imagem={w_pixels}x{h_pixels}px")
    print(f"  K =\n{K}")

    # --- Loop interativo ---
    plt.ion()   # modo interativo: plots não bloqueiam

    while True:
        # Atualiza o plot a cada iteração
        plt.close('all')
        plot_scene(cam, cam_initial, world, obj, K, w_pixels, h_pixels,
                   title="Perspective Projection – Robot")
        plt.draw()
        plt.pause(0.1)

        # Menu principal
        print("\n========================================")
        print("  (1) Mover câmera")
        print("  (2) Alterar parâmetros intrínsecos")
        print("  (3) Resetar câmera para posição inicial")
        print("  (4) Mostrar estado atual da câmera")
        print("  (0) Sair")
        print("========================================")
        op = input("Opção: ").strip()

        if op == "0":
            print("Encerrando.")
            break
        elif op == "1":
            cam = read_extrinsics(cam)
        elif op == "2":
            K, w_pixels, h_pixels = read_intrinsics(K, w_pixels, h_pixels)
        elif op == "3":
            cam = cam_initial.copy()
            print("  Câmera resetada para posição inicial.")
        elif op == "4":
            print(f"\n  Matriz da câmera (cam):\n{cam}")
            print(f"\n  Matriz extrínseca G = inv(cam):\n{np.linalg.inv(cam)}")
            print(f"\n  Matriz intrínseca K:\n{K}")
        else:
            print("  Opção inválida.")

    plt.ioff()
    plt.show()
