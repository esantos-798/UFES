"""
ryu_object.py
=============
Define a geometria 3D estilizada do Ryu (Street Fighter) usando
vértices e faces poligonais, compatível com o camera_visualizer.py.

Para usar no visualizador, basta substituir:
    from ryu_object import OBJECT_VERTS, OBJECT_FACES, OBJECT_COLORS

no lugar das linhas equivalentes do camera_visualizer.py.
"""

import numpy as np

def build_ryu():
    """
    Constrói o Ryu com primitivas geométricas:
    - Cabeça      : cubo levemente achatado
    - Faixa       : fita fina ao redor da cabeça
    - Tronco      : paralelepípedo largo
    - Faixa preta : retângulo na cintura
    - Braço esq.  : paralelepípedo + mão
    - Braço dir.  : paralelepípedo + mão (em posição de soco)
    - Perna esq.  : dois segmentos (coxa + canela)
    - Perna dir.  : dois segmentos
    - Pé esq/dir  : caixas achatadas

    Sistema de coordenadas:
        X → largura (direita)
        Y → profundidade (frente)
        Z → altura (cima)

    O modelo fica de pé centrado na origem, base em Z=0.
    Altura total ≈ 4 unidades.
    """

    verts = []   # lista de np.array([x,y,z])
    faces = []   # lista de listas de índices
    colors = []  # cor por face

    def add_box(cx, cy, cz,  # centro
                dx, dy, dz,  # metades das dimensões
                color):
        """
        Adiciona um paralelepípedo (8 vértices, 6 faces quad).
        Retorna o índice base dos vértices adicionados.
        """
        base = len(verts)
        # 8 cantos: ±dx, ±dy, ±dz
        corners = [
            [cx-dx, cy-dy, cz-dz],  # 0 frente-esq-baixo
            [cx+dx, cy-dy, cz-dz],  # 1 frente-dir-baixo
            [cx+dx, cy+dy, cz-dz],  # 2 atrás-dir-baixo
            [cx-dx, cy+dy, cz-dz],  # 3 atrás-esq-baixo
            [cx-dx, cy-dy, cz+dz],  # 4 frente-esq-cima
            [cx+dx, cy-dy, cz+dz],  # 5 frente-dir-cima
            [cx+dx, cy+dy, cz+dz],  # 6 atrás-dir-cima
            [cx-dx, cy+dy, cz+dz],  # 7 atrás-esq-cima
        ]
        for c in corners:
            verts.append(c)
        # 6 faces (quads)
        quad_faces = [
            [base+0, base+1, base+5, base+4],  # frente
            [base+2, base+3, base+7, base+6],  # atrás
            [base+0, base+3, base+7, base+4],  # esquerda
            [base+1, base+2, base+6, base+5],  # direita
            [base+0, base+1, base+2, base+3],  # baixo
            [base+4, base+5, base+6, base+7],  # cima
        ]
        for f in quad_faces:
            faces.append(f)
            colors.append(color)
        return base

    # ------------------------------------------------------------------
    # Paleta de cores (estilo Street Fighter arcade)
    # ------------------------------------------------------------------
    SKIN   = "#F4A460"   # pele
    GI     = "#FFFFFF"   # kimono branco
    FAIXA  = "#000000"   # faixa preta
    FAIXA2 = "#CC0000"   # faixa vermelha da cabeça
    CABELO = "#1a1a1a"   # cabelo escuro
    SAPATO = "#222222"   # sapato escuro

    # ------------------------------------------------------------------
    # Cabeça  (centro em z=3.45, altura 0.7)
    # ------------------------------------------------------------------
    add_box( 0,  0, 3.45,   0.38, 0.32, 0.35,  SKIN)

    # Cabelo (parte superior da cabeça, mais escuro)
    add_box( 0,  0, 3.75,   0.38, 0.32, 0.07,  CABELO)

    # Faixa vermelha da cabeça
    add_box( 0,  0, 3.52,   0.40, 0.34, 0.06,  FAIXA2)

    # Ponta da faixa (à esquerda, saindo da cabeça)
    # triângulo aproximado por trapézio fino
    add_box(-0.55, -0.1, 3.48,  0.18, 0.04, 0.04, FAIXA2)

    # ------------------------------------------------------------------
    # Pescoço
    # ------------------------------------------------------------------
    add_box( 0,  0, 3.07,   0.14, 0.12, 0.06,  SKIN)

    # ------------------------------------------------------------------
    # Tronco (kimono)
    # ------------------------------------------------------------------
    add_box( 0,  0, 2.55,   0.55, 0.30, 0.48,  GI)

    # Lapela esquerda do kimono (faixa diagonal simulada)
    add_box(-0.15, -0.28, 2.55,  0.12, 0.03, 0.45,  "#dddddd")
    # Lapela direita
    add_box( 0.15, -0.28, 2.55,  0.12, 0.03, 0.45,  "#dddddd")

    # ------------------------------------------------------------------
    # Faixa preta na cintura
    # ------------------------------------------------------------------
    add_box( 0,  0, 2.08,   0.58, 0.33, 0.09,  FAIXA)

    # Nó da faixa (frente)
    add_box( 0, -0.33, 2.08,  0.18, 0.05, 0.10,  FAIXA)

    # ------------------------------------------------------------------
    # Braço esquerdo (posição neutra, levemente aberto)
    # ------------------------------------------------------------------
    # Parte superior do braço
    add_box(-0.80, 0.0, 2.60,  0.14, 0.13, 0.32,  GI)
    # Antebraço
    add_box(-0.90, 0.0, 2.10,  0.12, 0.11, 0.25,  SKIN)
    # Mão esquerda (punho fechado)
    add_box(-0.92, -0.02, 1.80,  0.13, 0.10, 0.12,  SKIN)

    # ------------------------------------------------------------------
    # Braço direito (em posição de soco estendido)
    # ------------------------------------------------------------------
    # Parte superior
    add_box( 0.80, 0.0, 2.60,  0.14, 0.13, 0.32,  GI)
    # Antebraço estendido para frente/baixo
    add_box( 0.90, -0.20, 2.20,  0.12, 0.22, 0.12,  SKIN)
    # Punho de soco
    add_box( 0.91, -0.48, 2.18,  0.14, 0.14, 0.14,  SKIN)

    # ------------------------------------------------------------------
    # Quadril
    # ------------------------------------------------------------------
    add_box( 0,  0, 1.92,   0.52, 0.28, 0.10,  GI)

    # ------------------------------------------------------------------
    # Perna esquerda
    # ------------------------------------------------------------------
    # Coxa
    add_box(-0.28,  0, 1.55,  0.20, 0.18, 0.35,  GI)
    # Canela
    add_box(-0.28,  0, 1.05,  0.17, 0.15, 0.30,  GI)
    # Pé esquerdo
    add_box(-0.28, -0.12, 0.72,  0.18, 0.28, 0.10,  SAPATO)

    # ------------------------------------------------------------------
    # Perna direita (ligeiramente afastada — postura de luta)
    # ------------------------------------------------------------------
    # Coxa
    add_box( 0.28,  0.05, 1.50,  0.20, 0.18, 0.32,  GI)
    # Canela
    add_box( 0.30,  0.05, 1.02,  0.17, 0.15, 0.28,  GI)
    # Pé direito
    add_box( 0.30, -0.10, 0.70,  0.18, 0.26, 0.10,  SAPATO)

    # ------------------------------------------------------------------
    # Converter para numpy
    # ------------------------------------------------------------------
    verts_np = np.array(verts, dtype=float)

    # Centralizar base em Z=0
    verts_np[:, 2] -= verts_np[:, 2].min()

    return verts_np, faces, colors


# Exporta os símbolos que o camera_visualizer.py espera
OBJECT_VERTS, OBJECT_FACES, OBJECT_COLORS = build_ryu()


# ------------------------------------------------------------------
# Teste independente: visualiza o Ryu sem o visualizador de câmera
# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 10), facecolor="#0a0a0a")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a0a")
    ax.set_title("Ryu – Street Fighter 3D", color="#cc0000",
                 fontsize=16, fontweight="bold", pad=12)

    # Monta polígonos
    polys = []
    for face in OBJECT_FACES:
        polys.append([OBJECT_VERTS[i] for i in face])

    col = Poly3DCollection(polys, alpha=0.90,
                           facecolor=OBJECT_COLORS,
                           edgecolor="#333333", linewidth=0.3)
    ax.add_collection3d(col)

    # Ajusta limites
    all_pts = OBJECT_VERTS
    pad = 0.3
    ax.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
    ax.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
    ax.set_zlim(0, all_pts[:,2].max()+pad)

    ax.set_xlabel("X", color="#666"); ax.set_ylabel("Y", color="#666")
    ax.set_zlabel("Z", color="#666")
    ax.tick_params(colors="#444")
    ax.view_init(elev=15, azim=-60)

    plt.tight_layout()
    plt.show()
