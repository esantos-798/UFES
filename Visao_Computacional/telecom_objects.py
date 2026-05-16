"""
telecom_objects.py
==================
Objetos 3D de Telecomunicações em formato polilinha (plot3D),
compatíveis com o código do visualizador de câmera.

Objetos disponíveis:
    build_satellite()  → Satélite com painéis solares e antena
    build_dish()       → Antena parabólica com suporte e mastro

Uso:
    from telecom_objects import build_satellite, build_dish
    obj = build_satellite()   # ou build_dish()

Formato de saída (4 x N):
    linha 0: x
    linha 1: y
    linha 2: z
    linha 3: 1 (homogênea) ou NaN onde há separadores
"""

import numpy as np
from math import cos, sin, pi


# =============================================================================
# Primitivas compartilhadas
# =============================================================================

NAN3 = [float('nan')] * 3   # separador de polilinha


def _box(cx, cy, cz, dx, dy, dz):
    """Paralelepípedo como polilinha (12 arestas, NaN separando)."""
    x0, x1 = cx-dx, cx+dx
    y0, y1 = cy-dy, cy+dy
    z0, z1 = cz-dz, cz+dz
    pts = [
        # base inferior
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z0], NAN3,
        # base superior
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[x0,y0,z1], NAN3,
        # pilares
        [x0,y0,z0],[x0,y0,z1], NAN3,
        [x1,y0,z0],[x1,y0,z1], NAN3,
        [x1,y1,z0],[x1,y1,z1], NAN3,
        [x0,y1,z0],[x0,y1,z1], NAN3,
    ]
    return np.array(pts).T   # (3, N)


def _cylinder(cx, cy, cz_bot, cz_top, radius, seg=10):
    """Cilindro poligonal como polilinha."""
    angles = np.linspace(0, 2*pi, seg+1)
    pts = []
    for a in angles: pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_bot])
    pts.append(NAN3)
    for a in angles: pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_top])
    pts.append(NAN3)
    for a in angles[::2]:
        pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_bot])
        pts.append([cx+radius*cos(a), cy+radius*sin(a), cz_top])
        pts.append(NAN3)
    return np.array(pts).T


def _circle(cx, cy, cz, rx, ry, seg=32, normal='z'):
    """
    Elipse/círculo como polilinha num plano.
    normal='z' → plano XY, 'x' → plano YZ, 'y' → plano XZ
    """
    angles = np.linspace(0, 2*pi, seg+1)
    pts = []
    for a in angles:
        if normal == 'z':
            pts.append([cx + rx*cos(a), cy + ry*sin(a), cz])
        elif normal == 'x':
            pts.append([cx, cy + rx*cos(a), cz + ry*sin(a)])
        elif normal == 'y':
            pts.append([cx + rx*cos(a), cy, cz + ry*sin(a)])
    pts.append(NAN3)
    return np.array(pts).T


def _line(*points):
    """Segmento(s) de linha entre pontos dados."""
    pts = list(points) + [NAN3]
    return np.array(pts).T


def _join(*parts):
    """Concatena polilinhas com separador NaN."""
    nan_sep = np.full((3,1), np.nan)
    result = []
    for p in parts:
        result.append(p)
        result.append(nan_sep)
    return np.hstack(result)


def _finalize(xyz, scale=1.0):
    """Escala, centraliza base em Z=0 e adiciona linha homogênea."""
    xyz = xyz * scale
    # ignora NaN para encontrar mínimo Z real
    z_min = np.nanmin(xyz[2,:])
    xyz[2,:] -= z_min
    hom = np.where(np.isnan(xyz[0,:]), np.nan, 1.0)
    return np.vstack([xyz, hom])   # (4, N)


# =============================================================================
# 1. SATÉLITE 🛰️
# =============================================================================

def build_satellite(scale=2.5):
    """
    Satélite de comunicações com:
        - Corpo cúbico central (bus)
        - 2 painéis solares simétricos (eixo X) → marcam os LADOS
        - Antena parabólica apontando para -Y (FRENTE / nadir)
        - Antena de telemetria no topo (+Z) → marca o CIMA
        - Bocal do propulsor em +Y (TRÁS)

    Orientação inequívoca:
        CIMA  = antena de telemetria (haste fina no topo)
        BAIXO = propulsor (círculo no fundo)
        FRENTE= antena parabólica
        TRÁS  = bocal do propulsor
        LADOS = painéis solares
    """

    parts = []

    # ── Corpo principal (bus) ──
    parts.append(_box(0, 0, 0,  0.5, 0.5, 0.6))

    # ── Antena de telemetria no topo (CIMA) ──
    # haste
    parts.append(_cylinder(0, 0,  0.6, 1.1,  0.04, 6))
    # disco no topo
    parts.append(_circle(0, 0, 1.1,  0.18, 0.18, 16))

    # ── Painéis solares (LADOS, eixo X) ──
    for side in (+1, -1):
        px = side * 1.6
        # painel principal
        parts.append(_box(px, 0, 0,   0.9, 0.02, 0.45))
        # divisórias internas das células (3 faixas)
        for frac in [-0.3, 0, 0.3]:
            parts.append(_line(
                [px + frac*0.9, -0.02, -0.45],
                [px + frac*0.9, -0.02,  0.45],
            ))
        # haste de conexão ao corpo
        parts.append(_line(
            [side*0.5, 0, 0],
            [side*0.7, 0, 0],
        ))

    # ── Antena parabólica (FRENTE, -Y) ──
    # suporte do alimentador
    parts.append(_cylinder(0, -0.5, 0,   0, 0.06, 6))   # haste vertical
    parts.append(_line([0,-0.5,0],[0,-1.0,0]))            # haste horizontal
    # prato parabólico: arcos em diferentes ângulos
    r_dish = 0.55
    depth  = 0.20
    seg    = 24
    # perfil da parábola: z = depth*(r/r_dish)^2 - depth, centrado em y=-1.0
    for angle_deg in range(0, 180, 20):
        a = np.radians(angle_deg)
        xs, zs, ys = [], [], []
        for t in np.linspace(0, 2*pi, seg+1):
            r = r_dish * abs(cos(t/2))   # raio varia com t
            x = r_dish * cos(t)
            z = r_dish * sin(t)
            # deslocamento em Y simula profundidade do prato
            y = -1.0 - depth * (1 - (x**2 + z**2)/r_dish**2)
            xs.append(x); ys.append(y); zs.append(z)
        xs.append(float('nan')); ys.append(float('nan')); zs.append(float('nan'))
        parts.append(np.array([xs, ys, zs]))

    # borda do prato (círculo frontal)
    parts.append(_circle(0, -1.0, 0,  r_dish, r_dish, 32, normal='x'))
    # raios internos do prato (4 raios)
    for a in [0, pi/2, pi, 3*pi/2]:
        parts.append(_line(
            [0, -1.0, 0],
            [r_dish*cos(a), -1.0 - depth*0.3, r_dish*sin(a)],
        ))

    # ── Propulsor (TRÁS, +Y) ──
    parts.append(_cylinder(0, 0.5, 0,   0, 0.12, 8))   # câmara
    parts.append(_circle(0, 0.5, 0,  0.12, 0.12, 16, normal='y'))  # bocal
    # chamas (linhas irradiando)
    for a in np.linspace(0, 2*pi, 8, endpoint=False):
        parts.append(_line(
            [0.10*cos(a), 0.5, 0.10*sin(a)],
            [0.22*cos(a), 0.75, 0.22*sin(a)],
        ))

    xyz = _join(*parts)
    return _finalize(xyz, scale=scale)


# =============================================================================
# 2. ANTENA PARABÓLICA 📡
# =============================================================================

def build_dish(scale=2.0):
    """
    Antena parabólica com:
        - Prato parabólico apontando para +Y (FRENTE)
        - Alimentador (feed horn) no foco do prato
        - Suporte em forquilha (azimute + elevação)
        - Mastro/pedestal vertical
        - Base/ancoragem no chão

    Orientação inequívoca:
        CIMA   = ponto mais alto do prato
        BAIXO  = base no chão
        FRENTE = abertura do prato
        TRÁS   = face traseira do prato (fechada)
    """

    parts = []

    R      = 1.8    # raio do prato
    DEPTH  = 0.55   # profundidade do prato
    SEG    = 36     # segmentos angulares

    # ── Prato parabólico ──
    # A parábola: y = DEPTH * (r/R)^2,  r = sqrt(x^2+z^2)
    # Geramos perfis radiais (meridianos) e anéis (paralelos)

    # Meridianos (perfis radiais a cada 20°)
    for phi_deg in range(0, 360, 20):
        phi = np.radians(phi_deg)
        pts = []
        for r in np.linspace(0, R, 20):
            x = r * cos(phi)
            z = r * sin(phi)
            y = DEPTH * (r/R)**2   # parábola
            pts.append([x, y, z])
        pts.append(NAN3)
        parts.append(np.array(pts).T)

    # Anéis (paralelos a cada 0.3 de raio)
    for r in np.linspace(0, R, 7):
        pts = []
        y = DEPTH * (r/R)**2
        for phi in np.linspace(0, 2*pi, SEG+1):
            pts.append([r*cos(phi), y, r*sin(phi)])
        pts.append(NAN3)
        parts.append(np.array(pts).T)

    # Borda do prato (aro externo)
    parts.append(_circle(0, DEPTH, 0,  R, R, SEG, normal='y'))

    # Reforços estruturais (4 vigas radiais na traseira)
    for phi_deg in [0, 90, 180, 270]:
        phi = np.radians(phi_deg)
        parts.append(_line(
            [0, -0.05, 0],
            [R*cos(phi)*0.9, DEPTH*0.8, R*sin(phi)*0.9],
        ))

    # Struts (suporte do alimentador): 3 hastes do aro ao foco
    focal_y = -(R**2) / (4*DEPTH)  # foco da parábola
    for phi_deg in [60, 180, 300]:
        phi = np.radians(phi_deg)
        r_att = R * 0.7  # ponto de fixação no prato
        y_att = DEPTH * (r_att/R)**2
        parts.append(_line(
            [r_att*cos(phi), y_att, r_att*sin(phi)],
            [0, focal_y, 0],
        ))

    # Feed horn (alimentador no foco)
    parts.append(_cylinder(0, focal_y, 0,  focal_y-0.18, 0.12, 8))
    parts.append(_circle(0, focal_y,    0, 0.12, 0.12, 16, normal='y'))
    parts.append(_circle(0, focal_y-0.18, 0, 0.08, 0.08, 16, normal='y'))

    # ── Suporte em forquilha (elevation over azimuth) ──

    # Eixo de elevação (haste horizontal que inclina o prato)
    parts.append(_line([-0.35, -0.05, 0], [0.35, -0.05, 0]))

    # Braço esquerdo da forquilha
    parts.append(_box(-0.30, -0.30, -0.30,  0.06, 0.28, 0.06))
    # Braço direito da forquilha
    parts.append(_box( 0.30, -0.30, -0.30,  0.06, 0.28, 0.06))

    # Cabeça de azimute (bloco central)
    parts.append(_box(0, -0.30, -0.45,  0.20, 0.18, 0.12))

    # ── Mastro ──
    parts.append(_cylinder(0, -0.45, -0.60,  0, 0.10, 8))

    # Reforços do mastro (3 nervuras)
    for phi_deg in [0, 120, 240]:
        phi = np.radians(phi_deg)
        parts.append(_line(
            [0.10*cos(phi), -0.45, 0.10*sin(phi)],
            [0.22*cos(phi), -0.60, 0.22*sin(phi)],
        ))

    # ── Base / pedestal ──
    parts.append(_box(0, -0.62, -0.70,  0.28, 0.05, 0.08))
    # 4 pés de ancoragem
    for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        parts.append(_line(
            [sx*0.20, -0.57, sy*0.12],
            [sx*0.40, -0.57, sy*0.30],
        ))

    xyz = _join(*parts)
    return _finalize(xyz, scale=scale)


# =============================================================================
# Exporta aliases padrão
# =============================================================================
# Troque qual quiser usar no visualizador:
#   OBJECT = build_satellite()
#   OBJECT = build_dish()

def get_object(name='satellite'):
    if name == 'dish':
        return build_dish()
    return build_satellite()


# =============================================================================
# Teste independente: plota os dois lado a lado
# =============================================================================
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    objects = [
        ('Satelite de Comunicacoes', build_satellite(), (-20, -60)),
        ('Antena Parabolica',        build_dish(),      (-15, -50)),
    ]

    fig = plt.figure(figsize=(14, 7), facecolor='#f5f5f5')
    fig.suptitle('Objetos 3D – Telecomunicacoes', fontsize=13, fontweight='bold')

    for i, (title, obj, view) in enumerate(objects):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.set_facecolor('#ffffff')
        ax.set_title(title, fontsize=11, pad=8)

        # plot — ignora linha homogênea (linha 3)
        ax.plot3D(obj[0,:], obj[1,:], obj[2,:],
                  color='#c0392b', linewidth=0.8)

        # eixos do mundo
        scale = float(np.nanmax(obj[2,:])) * 0.25
        for vec, c, lbl in [([1,0,0],'red','X'),
                              ([0,1,0],'green','Y'),
                              ([0,0,1],'blue','Z')]:
            ax.quiver(0,0,0,*vec, color=c, length=scale, normalize=False)
            ax.text(*(np.array(vec)*scale*1.15), lbl, color=c, fontsize=8)

        p = obj[:3,:]
        pad = 0.5
        ax.set_xlim(np.nanmin(p[0,:])-pad, np.nanmax(p[0,:])+pad)
        ax.set_ylim(np.nanmin(p[1,:])-pad, np.nanmax(p[1,:])+pad)
        ax.set_zlim(0, np.nanmax(p[2,:])+pad)
        ax.set_xlabel('X', fontsize=8); ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=view[0], azim=view[1])

    plt.tight_layout()
    plt.show()
