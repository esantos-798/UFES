"""
objects_3d.py
=============
Define três objetos 3D compatíveis com o camera_visualizer.py:

    - build_rocket()  : Foguete
    - build_mug()     : Xícara com alça
    - build_robot()   : Robô

Uso no visualizador:
    from objects_3d import build_rocket, build_mug, build_robot
    OBJECT_VERTS, OBJECT_FACES, OBJECT_COLORS = build_rocket()   # ou mug / robot

Teste independente:
    python objects_3d.py
"""

import numpy as np


# ===========================================================================
# Primitivas compartilhadas
# ===========================================================================

def _add_box(verts, faces, colors, cx, cy, cz, dx, dy, dz, color):
    """Paralelepípedo centrado em (cx,cy,cz) com semi-dims (dx,dy,dz)."""
    base = len(verts)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append([cx + sx*dx, cy + sy*dy, cz + sz*dz])
    for f in [
        [base+0, base+1, base+5, base+4],  # frente  (sy=-1)
        [base+2, base+3, base+7, base+6],  # atrás   (sy=+1)
        [base+0, base+2, base+6, base+4],  # esq     (sx=-1)
        [base+1, base+3, base+7, base+5],  # dir     (sx=+1)
        [base+0, base+1, base+3, base+2],  # baixo   (sz=-1)
        [base+4, base+5, base+7, base+6],  # cima    (sz=+1)
    ]:
        faces.append(f)
        colors.append(color)


def _add_pyramid(verts, faces, colors, cx, cy, cz, dx, dy, dz, color):
    """
    Pirâmide de base retangular com pico no topo (+Z).
        base centrada em (cx, cy, cz), semi-dims base (dx, dy)
        dz = altura até o pico
    """
    base = len(verts)
    verts.append([cx,      cy,      cz + dz])   # 0 pico
    verts.append([cx - dx, cy - dy, cz      ])   # 1 base frt-esq
    verts.append([cx + dx, cy - dy, cz      ])   # 2 base frt-dir
    verts.append([cx + dx, cy + dy, cz      ])   # 3 base trás-dir
    verts.append([cx - dx, cy + dy, cz      ])   # 4 base trás-esq
    faces.append([base+1, base+2, base+3, base+4])  # base
    faces.append([base+0, base+1, base+2])           # frente
    faces.append([base+0, base+2, base+3])           # dir
    faces.append([base+0, base+3, base+4])           # trás
    faces.append([base+0, base+4, base+1])           # esq
    for _ in range(5):
        colors.append(color)


def _add_cone_z(verts, faces, colors, cx, cy, cz_base, cz_tip,
                radius, segments, color_side, color_base):
    """
    Cone poligonal em torno do eixo Z.
        segments : número de lados (ex: 8 para octagonal)
    """
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    base_idx = len(verts)
    # vértices da base
    for a in angles:
        verts.append([cx + radius*np.cos(a), cy + radius*np.sin(a), cz_base])
    tip_idx = len(verts)
    verts.append([cx, cy, cz_tip])   # pico

    # faces laterais (triângulos)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([base_idx+i, base_idx+j, tip_idx])
        colors.append(color_side)

    # tampa da base (polígono)
    faces.append(list(range(base_idx, base_idx + segments)))
    colors.append(color_base)


def _add_cylinder_z(verts, faces, colors, cx, cy, cz_bot, cz_top,
                    radius, segments, color_side, color_top, color_bot):
    """Cilindro poligonal ao longo do eixo Z."""
    angles = np.linspace(0, 2*np.pi, segments, endpoint=False)
    b_bot = len(verts)
    for a in angles:
        verts.append([cx + radius*np.cos(a), cy + radius*np.sin(a), cz_bot])
    b_top = len(verts)
    for a in angles:
        verts.append([cx + radius*np.cos(a), cy + radius*np.sin(a), cz_top])

    # faces laterais (quads)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([b_bot+i, b_bot+j, b_top+j, b_top+i])
        colors.append(color_side)

    # tampas
    faces.append(list(range(b_bot, b_bot + segments)))
    colors.append(color_bot)
    faces.append(list(range(b_top, b_top + segments)))
    colors.append(color_top)


def _add_torus_arc(verts, faces, colors,
                   cx, cy, cz,          # centro do toro
                   R, r,                # raio maior, raio do tubo
                   theta_start, theta_end,  # arco em Y-Z plane (radianos)
                   n_theta, n_phi,      # segmentos
                   color):
    """
    Arco de toro no plano X=cx, varrendo theta de theta_start a theta_end.
    Usado para a alça da xícara.
    """
    thetas = np.linspace(theta_start, theta_end, n_theta)
    phis   = np.linspace(0, 2*np.pi, n_phi, endpoint=False)

    base = len(verts)
    for t in thetas:
        for p in phis:
            x = cx + r * np.cos(p)
            y = cy + (R + r * np.sin(p)) * np.sin(t)   # arco no plano YZ
            z = cz + (R + r * np.sin(p)) * np.cos(t)
            verts.append([x, y, z])

    nt = len(thetas)
    np_ = len(phis)
    for i in range(nt - 1):
        for j in range(np_):
            a = base + i  * np_ + j
            b = base + i  * np_ + (j+1) % np_
            c = base + (i+1)*np_ + (j+1) % np_
            d = base + (i+1)*np_ + j
            faces.append([a, b, c, d])
            colors.append(color)


def _finalize(verts, faces, colors):
    """Converte para numpy e coloca a base em Z=0."""
    v = np.array(verts, dtype=float)
    v[:, 2] -= v[:, 2].min()
    return v, faces, colors


# ===========================================================================
# 1. FOGUETE 🚀
# ===========================================================================

def build_rocket():
    """
    Foguete com:
        - Corpo cilíndrico branco
        - Nariz cônico vermelho (pico = CIMA)
        - Janela circular azul (FRENTE)
        - 4 aletas triangulares embaixo (BASE)
        - Bocal/motor preto (BAIXO)
        - Chamas alaranjadas saindo do bocal
    Orientação inequívoca: nariz=cima, chamas=baixo, janela=frente.
    """
    SEG = 12   # segmentos do cilindro

    verts  = []
    faces  = []
    colors = []

    B = lambda *a, **k: _add_box     (verts, faces, colors, *a, **k)
    P = lambda *a, **k: _add_pyramid (verts, faces, colors, *a, **k)
    CZ= lambda *a, **k: _add_cylinder_z(verts, faces, colors, *a, **k)
    CN= lambda *a, **k: _add_cone_z  (verts, faces, colors, *a, **k)

    # ---- Corpo principal ----
    CZ(0, 0,  0.0, 3.0,  0.40, SEG, "#EEEEEE", "#DDDDDD", "#DDDDDD")

    # ---- Nariz cônico (vermelho → CIMA) ----
    CN(0, 0,  3.0, 4.2,  0.40, SEG, "#CC2222", "#CC2222")

    # ---- Janela frontal (azul → FRENTE) ----
    # simulada com disco fino na frente do corpo
    CZ(0, -0.38, 1.6, 1.7,  0.06, 8, "#2266CC", "#2266CC", "#2266CC")

    # ---- Logo / faixa vermelha ----
    CZ(0, 0,  2.0, 2.2,  0.42, SEG, "#CC2222", "#CC2222", "#CC2222")

    # ---- Aletas (4) – base do foguete ----
    for angle_deg in [0, 90, 180, 270]:
        a = np.radians(angle_deg)
        ax_ = np.cos(a) * 0.38
        ay_ = np.sin(a) * 0.38
        # base da aleta: paralelepípedo fino
        B(ax_*1.8, ay_*1.8, 0.4,
          abs(np.sin(a))*0.06 + abs(np.cos(a))*0.35,
          abs(np.cos(a))*0.06 + abs(np.sin(a))*0.35,
          0.40, "#CC2222")

    # ---- Bocal / motor (preto → BAIXO) ----
    CN(0, 0,  0.0, -0.25, 0.28, SEG, "#222222", "#111111")

    # ---- Chamas (laranja/amarelo → sinal de BAIXO) ----
    CN(0, 0, -0.25, -0.9,  0.18, SEG, "#FF8800", "#FFCC00")
    CN(0, 0, -0.25, -0.7,  0.10, SEG, "#FFCC00", "#FFFFFF")

    return _finalize(verts, faces, colors)


# ===========================================================================
# 2. XÍCARA COM ALÇA ☕
# ===========================================================================

def build_mug():
    """
    Xícara com:
        - Corpo cilíndrico (ligeiramente mais largo no topo)
        - Interior oco (cilindro interno mais escuro)
        - Alça em arco apenas no lado DIREITO (+X) → marca a frente/lateral
        - Base plana mais larga (BAIXO)
        - Líquido (café) visível no topo (CIMA)
    Orientação: alça=direita, café=cima, base=baixo.
    """
    SEG = 16

    verts  = []
    faces  = []
    colors = []

    CZ = lambda *a, **k: _add_cylinder_z(verts, faces, colors, *a, **k)

    # ---- Base ----
    CZ(0, 0,  0.00, 0.15,  0.55, SEG, "#8B4513", "#8B4513", "#7A3B10")

    # ---- Corpo externo ----
    CZ(0, 0,  0.15, 2.20,  0.50, SEG, "#CC3300", "#BB2200", "#BB2200")

    # ---- Corpo interno (oco – mais escuro) ----
    CZ(0, 0,  0.30, 2.20,  0.40, SEG, "#441100", "#441100", "#441100")

    # ---- Café (superfície do líquido → CIMA) ----
    CZ(0, 0,  2.05, 2.15,  0.39, SEG, "#3B1F0A", "#3B1F0A", "#3B1F0A")

    # ---- Borda superior ----
    CZ(0, 0,  2.15, 2.25,  0.51, SEG, "#DD4411", "#CC3300", "#CC3300")

    # ---- Alça (arco de toro no lado +X → marca a direita/frente) ----
    _add_torus_arc(verts, faces, colors,
                   cx=0.50, cy=0, cz=1.15,   # centro do arco
                   R=0.65, r=0.09,            # raio maior e do tubo
                   theta_start=-np.pi*0.65,
                   theta_end  = np.pi*0.65,
                   n_theta=14, n_phi=8,
                   color="#AA2200")

    return _finalize(verts, faces, colors)


# ===========================================================================
# 3. ROBÔ 🤖
# ===========================================================================

def build_robot():
    """
    Robô humanóide com:
        - Cabeça quadrada com olhos (CIMA / FRENTE)
        - Antena no topo da cabeça (CIMA)
        - Tronco trapezoidal com painel de controle (FRENTE)
        - Braços com mãos (LADOS)
        - Pernas e pés pontudos para frente (BAIXO / FRENTE)
    Orientação inequívoca em qualquer ângulo.
    """
    verts  = []
    faces  = []
    colors = []

    B = lambda *a, **k: _add_box    (verts, faces, colors, *a, **k)
    P = lambda *a, **k: _add_pyramid(verts, faces, colors, *a, **k)
    CZ= lambda *a, **k: _add_cylinder_z(verts, faces, colors, *a, **k)

    # Paleta
    METAL  = "#88AACC"
    METAL2 = "#6688AA"
    DARK   = "#334455"
    OLHO   = "#00FFCC"
    BOCA   = "#CC4400"
    PAINEL = "#FFCC00"
    JOINT  = "#445566"
    PE     = "#223344"

    # ---- Antena (CIMA – inequívoco) ----
    CZ(0, 0, 4.10, 4.50, 0.04, 6, DARK,  DARK,  DARK)
    CZ(0, 0, 4.48, 4.65, 0.10, 8, "#FF3300", "#FF3300", "#FF3300")

    # ---- Cabeça ----
    B( 0,    0,    3.70, 0.38, 0.35, 0.40, METAL)

    # Olho esquerdo (FRENTE, lado -X)
    B(-0.18, -0.34, 3.82, 0.10, 0.03, 0.09, OLHO)
    # Olho direito (FRENTE, lado +X)
    B( 0.18, -0.34, 3.82, 0.10, 0.03, 0.09, OLHO)
    # Boca (FRENTE)
    B( 0,    -0.34, 3.48, 0.20, 0.03, 0.05, BOCA)

    # Orelha esq (lateral)
    B(-0.42,  0,    3.70, 0.05, 0.12, 0.12, METAL2)
    # Orelha dir (lateral)
    B( 0.42,  0,    3.70, 0.05, 0.12, 0.12, METAL2)

    # ---- Pescoço ----
    CZ(0, 0, 3.20, 3.30, 0.12, 8, JOINT, JOINT, JOINT)

    # ---- Tronco ----
    B( 0,    0,    2.55, 0.55, 0.38, 0.65, METAL)

    # Painel frontal (FRENTE – inequívoco)
    B( 0,   -0.37, 2.70, 0.30, 0.03, 0.35, PAINEL)
    # Botões no painel
    B(-0.15, -0.38, 2.85, 0.06, 0.03, 0.06, "#FF0000")
    B( 0.15, -0.38, 2.85, 0.06, 0.03, 0.06, "#00FF00")
    B( 0,    -0.38, 2.60, 0.06, 0.03, 0.06, "#0088FF")

    # ---- Ombro esquerdo ----
    CZ(-0.70, 0, 2.95, 3.05, 0.14, 8, JOINT, JOINT, JOINT)
    # Braço esquerdo
    B(-0.82,  0,    2.55, 0.14, 0.13, 0.38, METAL)
    # Antebraço esq
    B(-0.85,  0,    2.02, 0.12, 0.11, 0.24, METAL2)
    # Mão esq (punho)
    B(-0.86, -0.02, 1.74, 0.13, 0.12, 0.13, METAL)

    # ---- Ombro direito ----
    CZ( 0.70, 0, 2.95, 3.05, 0.14, 8, JOINT, JOINT, JOINT)
    # Braço direito (levemente levantado – posição diferente → assimetria)
    B( 0.82,  0,    2.70, 0.14, 0.13, 0.28, METAL)
    # Antebraço dir (estendido para frente)
    B( 0.85, -0.22, 2.35, 0.12, 0.24, 0.12, METAL2)
    # Mão dir
    B( 0.86, -0.48, 2.33, 0.14, 0.14, 0.14, METAL)

    # ---- Quadril ----
    B( 0,    0,    1.88, 0.50, 0.35, 0.12, METAL2)

    # ---- Perna esquerda ----
    B(-0.27,  0,    1.52, 0.20, 0.18, 0.34, METAL)
    B(-0.27,  0,    1.02, 0.17, 0.16, 0.28, METAL2)
    # Pé esq (pontudo para frente → marca FRENTE)
    B(-0.27, -0.18, 0.70, 0.17, 0.30, 0.09, PE)

    # ---- Perna direita ----
    B( 0.27,  0,    1.52, 0.20, 0.18, 0.34, METAL)
    B( 0.27,  0,    1.02, 0.17, 0.16, 0.28, METAL2)
    # Pé dir
    B( 0.27, -0.18, 0.70, 0.17, 0.30, 0.09, PE)

    # ---- Juntas dos joelhos ----
    CZ(-0.27, 0, 1.27, 1.33, 0.13, 8, JOINT, JOINT, JOINT)
    CZ( 0.27, 0, 1.27, 1.33, 0.13, 8, JOINT, JOINT, JOINT)

    return _finalize(verts, faces, colors)


# ===========================================================================
# Exporta os três objetos prontos para uso
# ===========================================================================
ROCKET_VERTS, ROCKET_FACES, ROCKET_COLORS = build_rocket()
MUG_VERTS,    MUG_FACES,    MUG_COLORS    = build_mug()
ROBOT_VERTS,  ROBOT_FACES,  ROBOT_COLORS  = build_robot()

# Aliases padrão (troque qual quiser usar no visualizador)
OBJECT_VERTS  = ROBOT_VERTS
OBJECT_FACES  = ROBOT_FACES
OBJECT_COLORS = ROBOT_COLORS


# ===========================================================================
# Teste independente: mostra os três lado a lado
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    objects = [
        ("Foguete 🚀",  ROCKET_VERTS, ROCKET_FACES, ROCKET_COLORS, (-55, -40)),
        ("Xícara ☕",   MUG_VERTS,    MUG_FACES,    MUG_COLORS,    (-30, -60)),
        ("Robô 🤖",     ROBOT_VERTS,  ROBOT_FACES,  ROBOT_COLORS,  (-20, -50)),
    ]

    fig = plt.figure(figsize=(15, 6), facecolor="#0d1117")
    fig.suptitle("Objetos 3D – camera_visualizer.py",
                 color="#AADDFF", fontsize=14, fontweight="bold")

    for i, (title, verts, faces, cols, view) in enumerate(objects):
        ax = fig.add_subplot(1, 3, i+1, projection="3d")
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color="#AADDFF", fontsize=12, pad=8)

        polys = [[verts[idx] for idx in face] for face in faces]
        pc = Poly3DCollection(polys, alpha=0.90,
                              facecolor=cols,
                              edgecolor="#1a2a3a", linewidth=0.2)
        ax.add_collection3d(pc)

        p = verts
        pad = 0.3
        ax.set_xlim(p[:,0].min()-pad, p[:,0].max()+pad)
        ax.set_ylim(p[:,1].min()-pad, p[:,1].max()+pad)
        ax.set_zlim(0, p[:,2].max()+pad)

        # Eixos do mundo
        for vec, c, lbl in [([1,0,0],"#ff6666","X"),
                              ([0,1,0],"#66ff66","Y"),
                              ([0,0,1],"#6699ff","Z")]:
            scale = p[:,2].max() * 0.3
            ax.quiver(0,0,0,*vec, color=c, length=scale,
                      normalize=False, linewidth=1.2)

        ax.set_xlabel("X", color="#557799", labelpad=4, fontsize=8)
        ax.set_ylabel("Y", color="#557799", labelpad=4, fontsize=8)
        ax.set_zlabel("Z", color="#557799", labelpad=4, fontsize=8)
        ax.tick_params(colors="#334455", labelsize=6)
        ax.view_init(elev=view[0], azim=view[1])

    plt.tight_layout()
    plt.show()
