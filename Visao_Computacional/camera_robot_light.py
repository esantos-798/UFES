import matplotlib.pyplot as plt

from matplotlib.widgets import TextBox, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import pi, cos, sin

# =============================================================================
# Transformation functions
# =============================================================================

# ---------- translation ----------
def move (dx,dy,dz):
    T = np.eye(4)
    T[0,-1] = dx
    T[1,-1] = dy
    T[2,-1] = dz
    return T

# ---------- rotation ----------
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

# ---------- translation in its own frame ----------
def own_move (dx,dy,dz,cam):
    c0 = np.eye(4)
    T = np.eye(4)
    T[0,-1]=dx
    T[1,-1]=dy
    T[2,-1]=dz
    return cam@T@c0

# ---------- rotation in its own frame ----------
def own_z_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[cos(angle),-sin(angle),0,0],[sin(angle),cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
    return cam@rotation_matrix@c0

def own_x_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[1,0,0,0],[0, cos(angle),-sin(angle),0],[0, sin(angle), cos(angle),0],[0,0,0,1]])
    return cam@rotation_matrix@c0

def own_y_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[cos(angle),0, sin(angle),0],[0,1,0,0],[-sin(angle), 0, cos(angle),0],[0,0,0,1]])
    return cam@rotation_matrix@c0

# =============================================================================
# Object: Robot
# =============================================================================

def make_box_lines(cx, cy, cz, dx, dy, dz):
    x0,x1 = cx-dx,cx+dx; y0,y1 = cy-dy,cy+dy; z0,z1 = cz-dz,cz+dz
    nan = [float('nan')]*3
    pts = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z0], nan,
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[x0,y0,z1], nan,
        [x0,y0,z0],[x0,y0,z1], nan, [x1,y0,z0],[x1,y0,z1], nan,
        [x1,y1,z0],[x1,y1,z1], nan, [x0,y1,z0],[x0,y1,z1], nan,
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
        CZ( 0,    0,    4.10,4.50,0.04,6),
        CZ( 0,    0,    4.48,4.65,0.10,8),
        B( 0,    0,    3.70,0.38,0.35,0.40),
        B(-0.18,-0.34, 3.82,0.10,0.03,0.09),
        B( 0.18,-0.34, 3.82,0.10,0.03,0.09),
        B( 0,   -0.34, 3.48,0.20,0.03,0.05),
        B(-0.42, 0,    3.70,0.05,0.12,0.12),
        B( 0.42, 0,    3.70,0.05,0.12,0.12),
        CZ( 0,   0,    3.20,3.30,0.12,8),    
        B( 0,    0,    2.55,0.55,0.38,0.65), 
        B( 0,   -0.37, 2.70,0.30,0.03,0.35), 
        B(-0.15,-0.38, 2.85,0.06,0.03,0.06), 
        B( 0.15,-0.38, 2.85,0.06,0.03,0.06), 
        B( 0,   -0.38, 2.60,0.06,0.03,0.06), 
        CZ(-0.70,0,    2.95,3.05,0.14,8),    
        CZ( 0.70,0,    2.95,3.05,0.14,8),    
        B(-0.82, 0,    2.55,0.14,0.13,0.38), 
        B(-0.85, 0,    2.02,0.12,0.11,0.24), 
        B(-0.86,-0.02, 1.74,0.13,0.12,0.13), 
        B( 0.82, 0,    2.70,0.14,0.13,0.28), 
        B( 0.85,-0.22, 2.35,0.12,0.24,0.12), 
        B( 0.86,-0.48, 2.33,0.14,0.14,0.14), 
        B( 0,    0,    1.88,0.50,0.35,0.12), 
        B(-0.27, 0,    1.52,0.20,0.18,0.34), 
        B(-0.27, 0,    1.02,0.17,0.16,0.28), 
        B(-0.27,-0.18, 0.70,0.17,0.30,0.09), 
        B( 0.27, 0,    1.52,0.20,0.18,0.34), 
        B( 0.27, 0,    1.02,0.17,0.16,0.28), 
        B( 0.27,-0.18, 0.70,0.17,0.30,0.09), 
        CZ(-0.27,0,    1.27,1.33,0.13,8),    
        CZ( 0.27,0,    1.27,1.33,0.13,8),    
    ) * scale
    hom = np.where(np.isnan(xyz[0,:]), np.nan, 1.0)
    return np.vstack([xyz, hom])

# =============================================================================
# Perspective Projection
# =============================================================================

def perspective_projection(cam, obj, K):
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

def draw_arrows(point, base, axis, length=1.5):
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',  pivot='tail',length=length)
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',length=length)
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue', pivot='tail',length=length)

# =============================================================================
# General Application
# =============================================================================

class CameraApp:

    # ---------- estado inicial ----------
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)

        # frames — np.eye(4): cols = [X, Y, Z, origem]
        self.world = np.eye(4)

        # posição inicial da câmera (mesma do código original)
        cam = np.eye(4)
        cam = own_move(15, -5, 6, cam)
        cam = own_z_rotation(90, cam)
        cam = own_x_rotation(-90, cam)
        self.cam_initial = cam.copy()
        self.cam = cam.copy()

        # objeto
        self.obj = build_robot(scale=3.5)

        # intrínsecos iniciais
        self.f       = 20.0
        self.w_cco   = 36.0
        self.h_cco   = 24.0
        self.w_px    = 1280
        self.h_px    = 720
        self.skew    = 0.0
        self.K       = self._make_K()

        # referencial ativo
        self.ref = 'world'   # 'world' ou 'camera'

        self._build_ui()
        self._refresh()
        plt.show()

    # ---------- matriz K ----------
    def _make_K(self):
        sx = self.w_px / self.w_cco
        sy = self.h_px / self.h_cco
        ox = self.w_px / 2.0   # ponto principal: centro da imagem
        oy = self.h_px / 2.0
        return np.array([[self.f*sx, self.skew, ox],
                         [0,         self.f*sy, oy],
                         [0,         0,         1 ]])

    # ---------- construção da janela ----------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(17, 9))
        self.fig.patch.set_facecolor('#f5f5f5')

        # ── Plots: 3D (cima-esquerda) e Imagem (baixo-esquerda) ──
        # [left, bottom, width, height] em fração da figura
        self.ax3d  = self.fig.add_axes([0.02, 0.50, 0.62, 0.46], projection='3d')
        self.ax_img = self.fig.add_axes([0.02, 0.04, 0.62, 0.42])
        self.ax3d.set_facecolor('#ffffff')
        self.ax_img.set_facecolor('#ffffff')

        # ── Painel direito: posicionamento absoluto ──
        # Coluna esquerda do painel: x=0.67, largura=0.10
        # Coluna direita (campos): x=0.78, largura=0.20
        # Cada linha tem altura h=0.040, espaçamento gap=0.008

        fc  = '#e8e8e8'   # fundo campos
        tc  = '#222222'   # texto geral
        lc  = '#1a5276'   # azul labels seção
        btn = '#d0d0d0'   # fundo botões
        hover = '#aaaaaa'

        PX = 0.665   # início painel (x)
        PW = 0.320   # largura total do painel
        LW = 0.105   # largura coluna label
        FW = 0.200   # largura coluna campo/textbox
        BH = 0.038   # altura de cada linha
        GAP= 0.010   # espaço vertical entre linhas

        # y começa no topo e vai descendo
        def row_y(i):
            return 0.95 - i * (BH + GAP)

        def add_label(i, text, color=tc, fontsize=9, bold=False, fullwidth=False):
            w = PW if fullwidth else LW
            ax = self.fig.add_axes([PX, row_y(i), w, BH])
            ax.set_facecolor('#f5f5f5'); ax.axis('off')
            fw = 'bold' if bold else 'normal'
            ax.text(0.5, 0.5, text, ha='center', va='center',
                    color=color, fontsize=fontsize, fontweight=fw,
                    transform=ax.transAxes)

        def add_field_label(i, text, color=tc):
            ax = self.fig.add_axes([PX, row_y(i), LW, BH])
            ax.set_facecolor('#f5f5f5'); ax.axis('off')
            ax.text(0.9, 0.5, text, ha='right', va='center',
                    color=color, fontsize=9, fontweight='bold',
                    transform=ax.transAxes)

        def add_textbox(i, initial='0'):
            ax = self.fig.add_axes([PX + LW + 0.008, row_y(i), FW, BH])
            tb = TextBox(ax, '', initial=initial,
                         color=fc, hovercolor='#d0d0d0', label_pad=0.0)
            tb.label.set_color(tc)
            tb.text_disp.set_color(tc)
            return tb

        def add_button(i, text, x=PX, w=PW):
            ax = self.fig.add_axes([x, row_y(i), w, BH])
            b  = Button(ax, text, color=btn, hovercolor=hover)
            b.label.set_color(tc); b.label.set_fontsize(8)
            return b

        def add_radio(i, labels):
            ax = self.fig.add_axes([PX + LW + 0.008, row_y(i), FW, BH * 2.2])
            ax.set_facecolor('#f5f5f5')
            rb = RadioButtons(ax, labels, activecolor='#2e86c1')
            for l in rb.labels:
                l.set_color(tc); l.set_fontsize(8)
            return rb

        # ── Linha 0: Título ──
        add_label(0, '  Controles da Camera', color='#1a3a5c',
                  fontsize=10, bold=True, fullwidth=True)

        # ── Linha 1-2: Referencial (radio ocupa 2 linhas de altura) ──
        add_label(1, 'Referencial:', color=lc, bold=True, fullwidth=False)
        self.radio_ref = add_radio(1, ('Mundo', 'Camera'))

        # ── Linha 3: separador Translação ──
        add_label(3, '--- Translacao ---', color=lc, bold=True, fullwidth=True)

        # ── Linhas 4-6: dx dy dz ──
        add_field_label(4, 'dx', color='#c0392b')
        self.tb_dx = add_textbox(4, '0')
        add_field_label(5, 'dy', color='#1a7a40')
        self.tb_dy = add_textbox(5, '0')
        add_field_label(6, 'dz', color='#154360')
        self.tb_dz = add_textbox(6, '0')

        # ── Linha 7: botão translação ──
        self.btn_translate = add_button(7, 'Aplicar Translacao')

        # ── Linha 8: separador Rotação ──
        add_label(8, '--- Rotacao ---', color=lc, bold=True, fullwidth=True)

        # ── Linhas 9-11: Rx Ry Rz ──
        add_field_label(9,  'Rx (°)', color='#c0392b')
        self.tb_rx = add_textbox(9,  '0')
        add_field_label(10, 'Ry (°)', color='#1a7a40')
        self.tb_ry = add_textbox(10, '0')
        add_field_label(11, 'Rz (°)', color='#154360')
        self.tb_rz = add_textbox(11, '0')

        # ── Linha 12: botão rotação ──
        self.btn_rotate = add_button(12, 'Aplicar Rotacao')

        # ── Linha 13: separador Intrínsecos ──
        add_label(13, '--- Intrinsecos ---', color=lc, bold=True, fullwidth=True)

        # ── Linhas 14-16: f, w_px, h_px ──
        add_field_label(14, 'f (mm)', color=tc)
        self.tb_f   = add_textbox(14, str(self.f))
        add_field_label(15, 'w (px)', color=tc)
        self.tb_wpx = add_textbox(15, str(self.w_px))
        add_field_label(16, 'h (px)', color=tc)
        self.tb_hpx = add_textbox(16, str(self.h_px))

        # ── Linha 17: info ponto principal (só texto) ──
        ox, oy = self.w_px/2, self.h_px/2
        ax_pp = self.fig.add_axes([PX, row_y(17), PW, BH])
        ax_pp.set_facecolor('#f5f5f5'); ax_pp.axis('off')
        self.txt_pp = ax_pp.text(0.5, 0.5,
            f'Ponto principal: ({ox:.0f}, {oy:.0f})',
            ha='center', va='center', color='#2e86c1',
            fontsize=8, transform=ax_pp.transAxes)

        # ── Linha 18: Reset e Info lado a lado ──
        self.btn_reset = add_button(18, 'Resetar Camera',
                                    x=PX,          w=PW/2 - 0.005)
        self.btn_info  = add_button(18, 'Info (terminal)',
                                    x=PX+PW/2+0.005, w=PW/2 - 0.005)

        # ── Linha 19: status ──
        ax_st = self.fig.add_axes([PX, row_y(19), PW, BH])
        ax_st.set_facecolor('#eaf4fb'); ax_st.axis('off')
        self.txt_status = ax_st.text(
            0.5, 0.5, 'Pronto.', ha='center', va='center',
            color='#196f3d', fontsize=8, transform=ax_st.transAxes)

        # ── Conectar callbacks ──
        self.btn_translate.on_clicked(self._apply_translation)
        self.btn_rotate.on_clicked(self._apply_rotation)
        self.btn_reset.on_clicked(self._reset)
        self.btn_info.on_clicked(self._show_info)
        self.radio_ref.on_clicked(self._set_ref)
        self.tb_f.on_submit(self._apply_intrinsics)
        self.tb_wpx.on_submit(self._apply_intrinsics)
        self.tb_hpx.on_submit(self._apply_intrinsics)

    # ---------- callbacks ----------

    def _set_ref(self, label):
        self.ref = 'world' if label == 'Mundo' else 'camera'
        self._status(f'Referencial: {label}')

    def _read(self, tb, default=0.0):
        try:    return float(tb.text)
        except: return default

    def _apply_translation(self, _=None):
        dx = self._read(self.tb_dx)
        dy = self._read(self.tb_dy)
        dz = self._read(self.tb_dz)
        if self.ref == 'camera':
            self.cam = own_move(dx, dy, dz, self.cam)
        else:
            self.cam = move(dx, dy, dz) @ self.cam
        self._status(f'Translação ({dx:.2f}, {dy:.2f}, {dz:.2f}) [{self.ref}]')
        self._refresh()

    def _apply_rotation(self, _=None):
        rx = self._read(self.tb_rx)
        ry = self._read(self.tb_ry)
        rz = self._read(self.tb_rz)
        if self.ref == 'camera':
            if rx: self.cam = own_x_rotation(rx, self.cam)
            if ry: self.cam = own_y_rotation(ry, self.cam)
            if rz: self.cam = own_z_rotation(rz, self.cam)
        else:
            if rx: self.cam = x_rotation(rx) @ self.cam
            if ry: self.cam = y_rotation(ry) @ self.cam
            if rz: self.cam = z_rotation(rz) @ self.cam
        self._status(f'Rotação Rx={rx:.1f}° Ry={ry:.1f}° Rz={rz:.1f}° [{self.ref}]')
        self._refresh()

    def _apply_intrinsics(self, _=None):
        try:
            self.f    = float(self.tb_f.text)
            self.w_px = int(float(self.tb_wpx.text))
            self.h_px = int(float(self.tb_hpx.text))
            self.K    = self._make_K()
            ox, oy    = self.w_px/2, self.h_px/2
            self.txt_pp.set_text(f'Ponto principal: ({ox:.0f}, {oy:.0f})')
            self._status(f'K atualizado. pp=({ox:.0f},{oy:.0f})')
            self._refresh()
        except Exception as e:
            self._status(f'Erro: {e}')

    def _reset(self, _=None):
        self.cam = self.cam_initial.copy()
        self._status('Câmera resetada.')
        self._refresh()

    def _show_info(self, _=None):
        G = np.linalg.inv(self.cam)
        print('\n=== Estado atual ===')
        print(f'Referencial ativo : {self.ref}')
        print(f'Câmera (cam):\n{self.cam}')
        print(f'Extrínseca G=inv(cam):\n{G}')
        print(f'Intrínsecos: f={self.f}mm, img={self.w_px}x{self.h_px}px')
        print(f'K:\n{self.K}')
        self._status('Info impressa no terminal.')

    def _status(self, msg):
        self.txt_status.set_text(msg)
        self.fig.canvas.draw_idle()

    # ---------- redesenho ----------

    def _refresh(self):
        self._draw_3d()
        self._draw_image()
        self.fig.canvas.draw_idle()

    def _draw_3d(self):
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor('#ffffff')
        lim = [-15, 20]
        ax.set_xlim(lim); ax.set_xlabel('x', color='#333333', fontsize=8)
        ax.set_ylim(lim); ax.set_ylabel('y', color='#333333', fontsize=8)
        ax.set_zlim(lim); ax.set_zlabel('z', color='#333333', fontsize=8)
        ax.set_title('3D Scene', color='#1a1a1a', fontsize=10, pad=4)
        ax.tick_params(colors='#444444', labelsize=6)

        # objeto
        ax.plot3D(self.obj[0,:], self.obj[1,:], self.obj[2,:], '#c0392b', linewidth=0.8)

        # frames
        draw_arrows(self.world[:,-1],       self.world[:,0:3],       ax, 3.0)
        draw_arrows(self.cam_initial[:,-1], self.cam_initial[:,0:3], ax, 1.5)
        draw_arrows(self.cam[:,-1],         self.cam[:,0:3],         ax, 1.5)

    def _draw_image(self):
        ax = self.ax_img
        ax.cla()
        ax.set_facecolor('#ffffff')
        ax.set_title('Camera Image', color='#1a1a1a', fontsize=10, pad=4)
        ax.set_xlim([0, self.w_px])
        ax.set_ylim([self.h_px, 0])   # origem no canto superior esquerdo
        ax.set_xlabel('u (px)', color='#333333', fontsize=8)
        ax.set_ylabel('v (px)', color='#333333', fontsize=8)
        ax.tick_params(colors='#444444', labelsize=6)
        ax.grid(True, color='#cccccc', linewidth=0.5)

        p_img = perspective_projection(self.cam, self.obj, self.K)
        ax.plot(p_img[0,:], p_img[1,:], color='#c0392b', linewidth=0.8)

        # ponto principal
        ox, oy = self.w_px/2, self.h_px/2
        ax.plot(ox, oy, '+', color='#2e86c1', markersize=10, markeredgewidth=1.5)
        ax.text(ox+10, oy-15, 'pp', color='#2e86c1', fontsize=7)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    app = CameraApp()
