import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import pi, cos, sin

# =============================================================================
# Transformações
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



# Functions to move the camera in its own reference frame

# Translation in its own frame
def own_move (dx,dy,dz,cam):
    c0 = np.eye(4)

    T = np.eye(4)
    T[0,-1]=dx
    T[1,-1]=dy
    T[2,-1]=dz

    cam = cam@T@c0
    return T, cam

# Rotation in its own frame
def own_z_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[cos(angle),-sin(angle),0,0],[sin(angle),cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
    cam = cam@rotation_matrix@c0
    return rotation_matrix, cam

def own_x_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[1,0,0,0],[0, cos(angle),-sin(angle),0],[0, sin(angle), cos(angle),0],[0,0,0,1]])
    cam = cam@rotation_matrix@c0
    return rotation_matrix, cam

def own_y_rotation(angle, cam):
    angle = angle*pi/180
    c0 = np.eye(4)
    rotation_matrix=np.array([[cos(angle),0, sin(angle),0],[0,1,0,0],[-sin(angle), 0, cos(angle),0],[0,0,0,1]])
    cam = cam@rotation_matrix@c0
    return rotation_matrix, cam

# =============================================================================
# Objeto: Robô
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
        CZ( 0,    0,    4.10,4.50,0.04,6),   # antena haste
        CZ( 0,    0,    4.48,4.65,0.10,8),   # antena bola
        B( 0,    0,    3.70,0.38,0.35,0.40), # cabeça
        B(-0.18,-0.34, 3.82,0.10,0.03,0.09), # olho esq
        B( 0.18,-0.34, 3.82,0.10,0.03,0.09), # olho dir
        B( 0,   -0.34, 3.48,0.20,0.03,0.05), # boca
        B(-0.42, 0,    3.70,0.05,0.12,0.12), # orelha esq
        B( 0.42, 0,    3.70,0.05,0.12,0.12), # orelha dir
        CZ( 0,   0,    3.20,3.30,0.12,8),    # pescoço
        B( 0,    0,    2.55,0.55,0.38,0.65), # tronco
        B( 0,   -0.37, 2.70,0.30,0.03,0.35), # painel
        B(-0.15,-0.38, 2.85,0.06,0.03,0.06), # botão 1
        B( 0.15,-0.38, 2.85,0.06,0.03,0.06), # botão 2
        B( 0,   -0.38, 2.60,0.06,0.03,0.06), # botão 3
        CZ(-0.70,0,    2.95,3.05,0.14,8),    # ombro esq
        CZ( 0.70,0,    2.95,3.05,0.14,8),    # ombro dir
        B(-0.82, 0,    2.55,0.14,0.13,0.38), # braço esq
        B(-0.85, 0,    2.02,0.12,0.11,0.24), # antebraço esq
        B(-0.86,-0.02, 1.74,0.13,0.12,0.13), # mão esq
        B( 0.82, 0,    2.70,0.14,0.13,0.28), # braço dir
        B( 0.85,-0.22, 2.35,0.12,0.24,0.12), # antebraço dir
        B( 0.86,-0.48, 2.33,0.14,0.14,0.14), # mão dir
        B( 0,    0,    1.88,0.50,0.35,0.12), # quadril
        B(-0.27, 0,    1.52,0.20,0.18,0.34), # coxa esq
        B(-0.27, 0,    1.02,0.17,0.16,0.28), # canela esq
        B(-0.27,-0.18, 0.70,0.17,0.30,0.09), # pé esq
        B( 0.27, 0,    1.52,0.20,0.18,0.34), # coxa dir
        B( 0.27, 0,    1.02,0.17,0.16,0.28), # canela dir
        B( 0.27,-0.18, 0.70,0.17,0.30,0.09), # pé dir
        CZ(-0.27,0,    1.27,1.33,0.13,8),    # joelho esq
        CZ( 0.27,0,    1.27,1.33,0.13,8),    # joelho dir
    ) * scale
    hom = np.where(np.isnan(xyz[0,:]), np.nan, 1.0)
    return np.vstack([xyz, hom])

# =============================================================================
# Projeção perspectiva
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
# Aplicação principal
# =============================================================================

class CameraApp:

    # ---------- estado inicial ----------
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)

        # frames
        e1=np.array([[1],[0],[0],[0]]); e2=np.array([[0],[1],[0],[0]]); e3=np.array([[0],[0],[1],[0]])
        base   = np.hstack((e1,e2,e3))
        origin = np.array([[0],[0],[0],[1]])
        self.world = np.hstack([base, origin])

        # posição inicial da câmera (igual ao código original)
        cam = np.hstack([base, origin])
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
        self.fig.patch.set_facecolor('#1e1e2e')

        # Layout: plots à esquerda (70%), painel à direita (30%)
        gs = gridspec.GridSpec(
            1, 2,
            width_ratios=[7, 3],
            left=0.03, right=0.98,
            top=0.95, bottom=0.04,
            wspace=0.05
        )

        # --- sub-grid dos plots (3D em cima, imagem embaixo) ---
        gs_plots = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[0],
            hspace=0.35
        )
        self.ax3d  = self.fig.add_subplot(gs_plots[0], projection='3d')
        self.ax_img = self.fig.add_subplot(gs_plots[1])

        for ax in [self.ax3d, self.ax_img]:
            ax.set_facecolor('#11111b')

        # --- painel de controle (direita) ---
        gs_ctrl = gridspec.GridSpecFromSubplotSpec(
            18, 2, subplot_spec=gs[1],
            hspace=0.55, wspace=0.3
        )

        fc  = '#313244'   # fundo dos campos
        tc  = '#cdd6f4'   # cor do texto
        lc  = '#89b4fa'   # cor dos labels
        btn = '#45475a'   # fundo botões

        def lbl(row, col, text, colspan=2, color=lc):
            ax = self.fig.add_subplot(gs_ctrl[row, col:col+colspan])
            ax.set_facecolor('#1e1e2e'); ax.axis('off')
            ax.text(0.5, 0.5, text, ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold',
                    transform=ax.transAxes)
            return ax

        def textbox(row, col, initial='0'):
            ax = self.fig.add_subplot(gs_ctrl[row, col])
            ax.set_facecolor(fc)
            tb = TextBox(ax, '', initial=initial,
                         color=fc, hovercolor='#45475a',
                         label_pad=0.0)
            tb.label.set_color(tc)
            tb.text_disp.set_color(tc)
            return tb

        def button(row, col, text, colspan=1):
            ax = self.fig.add_subplot(gs_ctrl[row, col:col+colspan])
            b  = Button(ax, text, color=btn, hovercolor='#585b70')
            b.label.set_color(tc); b.label.set_fontsize(8)
            return b

        # ── Título painel ──
        lbl(0, 0, '🎥  Controles da Câmera', color='#cba6f7')

        # ── Referencial ──
        lbl(1, 0, 'Referencial:', colspan=1, color=lc)
        ax_radio = self.fig.add_subplot(gs_ctrl[1, 1])
        ax_radio.set_facecolor('#1e1e2e')
        self.radio_ref = RadioButtons(ax_radio, ('Mundo', 'Câmera'),
                                      activecolor='#cba6f7')
        for lbl_w in self.radio_ref.labels:
            lbl_w.set_color(tc); lbl_w.set_fontsize(8)

        # ── Translação ──
        lbl(2, 0, '── Translação ──')
        lbl(3, 0, 'Tx', colspan=1, color=tc)
        lbl(3, 1, 'Valor', colspan=1, color=tc)
        lbl(4, 0, 'dx', colspan=1, color='#f38ba8')
        self.tb_dx = textbox(4, 1, '0')
        lbl(5, 0, 'dy', colspan=1, color='#a6e3a1')
        self.tb_dy = textbox(5, 1, '0')
        lbl(6, 0, 'dz', colspan=1, color='#89b4fa')
        self.tb_dz = textbox(6, 1, '0')
        self.btn_translate = button(7, 0, 'Aplicar Translação', colspan=2)

        # ── Rotação ──
        lbl(8, 0, '── Rotação ──')
        lbl(9, 0, 'Eixo', colspan=1, color=tc)
        lbl(9, 1, 'Ângulo (°)', colspan=1, color=tc)
        lbl(10, 0, 'Rx', colspan=1, color='#f38ba8')
        self.tb_rx = textbox(10, 1, '0')
        lbl(11, 0, 'Ry', colspan=1, color='#a6e3a1')
        self.tb_ry = textbox(11, 1, '0')
        lbl(12, 0, 'Rz', colspan=1, color='#89b4fa')
        self.tb_rz = textbox(12, 1, '0')
        self.btn_rotate = button(13, 0, 'Aplicar Rotação', colspan=2)

        # ── Intrínsecos ──
        lbl(14, 0, '── Intrínsecos ──')
        lbl(15, 0, 'f (mm)', colspan=1, color=tc);  self.tb_f    = textbox(15, 1, str(self.f))
        lbl(16, 0, 'fx (px)', colspan=1, color=tc); self.tb_wpx  = textbox(16, 1, str(self.w_px))
        lbl(17, 0, 'fy (px)', colspan=1, color=tc); self.tb_hpx  = textbox(17, 1, str(self.h_px))

        # ── Botões de ação ──
        # Linha extra abaixo do grid para Reset e Info
        ax_reset = self.fig.add_axes([0.718, 0.09, 0.12, 0.04])
        ax_info  = self.fig.add_axes([0.848, 0.09, 0.12, 0.04])
        self.btn_reset = Button(ax_reset, '↺ Reset', color='#313244', hovercolor='#585b70')
        self.btn_info  = Button(ax_info,  'ℹ Info',  color='#313244', hovercolor='#585b70')
        for b in [self.btn_reset, self.btn_info]:
            b.label.set_color(tc); b.label.set_fontsize(9)

        # ── Label de status ──
        self.ax_status = self.fig.add_axes([0.715, 0.03, 0.27, 0.055])
        self.ax_status.set_facecolor('#11111b'); self.ax_status.axis('off')
        self.txt_status = self.ax_status.text(
            0.5, 0.5, 'Pronto.', ha='center', va='center',
            color='#a6e3a1', fontsize=8, transform=self.ax_status.transAxes,
            wrap=True)

        # ── Conectar callbacks ──
        self.btn_translate.on_clicked(self._apply_translation)
        self.btn_rotate.on_clicked(self._apply_rotation)
        self.btn_reset.on_clicked(self._reset)
        self.btn_info.on_clicked(self._show_info)
        self.radio_ref.on_clicked(self._set_ref)

        # Intrínsecos: aplica ao sair do campo (Enter)
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
        ax.set_facecolor('#11111b')
        lim = [-15, 20]
        ax.set_xlim(lim); ax.set_xlabel('x', color='#585b70', fontsize=8)
        ax.set_ylim(lim); ax.set_ylabel('y', color='#585b70', fontsize=8)
        ax.set_zlim(lim); ax.set_zlabel('z', color='#585b70', fontsize=8)
        ax.set_title('3D Scene', color='#cdd6f4', fontsize=10, pad=4)
        ax.tick_params(colors='#585b70', labelsize=6)

        # objeto
        ax.plot3D(self.obj[0,:], self.obj[1,:], self.obj[2,:], 'tomato', linewidth=0.8)

        # frames
        draw_arrows(self.world[:,-1],       self.world[:,0:3],       ax, 3.0)
        draw_arrows(self.cam_initial[:,-1], self.cam_initial[:,0:3], ax, 1.5)
        draw_arrows(self.cam[:,-1],         self.cam[:,0:3],         ax, 1.5)

    def _draw_image(self):
        ax = self.ax_img
        ax.cla()
        ax.set_facecolor('#11111b')
        ax.set_title('Camera Image', color='#cdd6f4', fontsize=10, pad=4)
        ax.set_xlim([0, self.w_px])
        ax.set_ylim([self.h_px, 0])   # origem no canto superior esquerdo
        ax.set_xlabel('u (px)', color='#585b70', fontsize=8)
        ax.set_ylabel('v (px)', color='#585b70', fontsize=8)
        ax.tick_params(colors='#585b70', labelsize=6)
        ax.grid(True, color='#313244', linewidth=0.5)

        p_img = perspective_projection(self.cam, self.obj, self.K)
        ax.plot(p_img[0,:], p_img[1,:], color='tomato', linewidth=0.8)

        # ponto principal
        ox, oy = self.w_px/2, self.h_px/2
        ax.plot(ox, oy, '+', color='#f38ba8', markersize=10, markeredgewidth=1.5)
        ax.text(ox+10, oy-15, 'pp', color='#f38ba8', fontsize=7)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    app = CameraApp()
