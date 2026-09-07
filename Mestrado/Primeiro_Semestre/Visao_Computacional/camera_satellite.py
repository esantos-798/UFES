import matplotlib.pyplot as plt

from matplotlib.widgets import TextBox, Button, RadioButtons
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
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
# Object: Satellite
# =============================================================================
 
NAN3 = [float('nan')] * 3
 
def _box(cx, cy, cz, dx, dy, dz):
    x0,x1 = cx-dx,cx+dx; y0,y1 = cy-dy,cy+dy; z0,z1 = cz-dz,cz+dz
    pts = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z0], NAN3,
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],[x0,y0,z1], NAN3,
        [x0,y0,z0],[x0,y0,z1], NAN3, [x1,y0,z0],[x1,y0,z1], NAN3,
        [x1,y1,z0],[x1,y1,z1], NAN3, [x0,y1,z0],[x0,y1,z1], NAN3,
    ]
    return np.array(pts).T
 
def _cylinder(cx, cy, cz_bot, cz_top, radius, seg=10):
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
    angles = np.linspace(0, 2*pi, seg+1)
    pts = []
    for a in angles:
        if normal == 'z': pts.append([cx+rx*cos(a), cy+ry*sin(a), cz])
        elif normal == 'x': pts.append([cx, cy+rx*cos(a), cz+ry*sin(a)])
        elif normal == 'y': pts.append([cx+rx*cos(a), cy, cz+ry*sin(a)])
    pts.append(NAN3)
    return np.array(pts).T
 
def _line(*points):
    return np.array(list(points) + [NAN3]).T
 
def _join(*parts):
    nan_sep = np.full((3,1), np.nan)
    return np.hstack([p for part in parts for p in (part, nan_sep)])
 
def _finalize(xyz, scale=1.0):
    xyz = xyz * scale
    xyz[2,:] -= np.nanmin(xyz[2,:])
    hom = np.where(np.isnan(xyz[0,:]), np.nan, 1.0)
    return np.vstack([xyz, hom])
 
def build_satellite(scale=3.5):
    parts = []
    parts.append(_box(0, 0, 0,  0.5, 0.5, 0.6))
    parts.append(_cylinder(0, 0, 0.6, 1.1, 0.04, 6))
    parts.append(_circle(0, 0, 1.1, 0.18, 0.18, 16))
    for side in (+1, -1):
        px = side * 1.6
        parts.append(_box(px, 0, 0, 0.9, 0.02, 0.45))
        for frac in [-0.3, 0, 0.3]:
            parts.append(_line([px+frac*0.9,-0.02,-0.45],[px+frac*0.9,-0.02,0.45]))
        parts.append(_line([side*0.5,0,0],[side*0.7,0,0]))
    parts.append(_cylinder(0,-0.5,0,0,0.06,6))
    parts.append(_line([0,-0.5,0],[0,-1.0,0]))
    r_dish=0.55; depth=0.20; seg=24
    for angle_deg in range(0, 180, 20):
        xs,zs,ys = [],[],[]
        for t in np.linspace(0, 2*pi, seg+1):
            x=r_dish*cos(t); z=r_dish*sin(t)
            y=-1.0-depth*(1-(x**2+z**2)/r_dish**2)
            xs.append(x); ys.append(y); zs.append(z)
        xs.append(float('nan')); ys.append(float('nan')); zs.append(float('nan'))
        parts.append(np.array([xs,ys,zs]))
    parts.append(_circle(0,-1.0,0,r_dish,r_dish,32,normal='x'))
    for a in [0,pi/2,pi,3*pi/2]:
        parts.append(_line([0,-1.0,0],[r_dish*cos(a),-1.0-depth*0.3,r_dish*sin(a)]))
    parts.append(_cylinder(0,0.5,0,0,0.12,8))
    parts.append(_circle(0,0.5,0,0.12,0.12,16,normal='y'))
    for a in np.linspace(0,2*pi,8,endpoint=False):
        parts.append(_line([0.10*cos(a),0.5,0.10*sin(a)],[0.22*cos(a),0.75,0.22*sin(a)]))
    return _finalize(_join(*parts), scale=scale)
 
# =============================================================================
# Perspective Projection
# =============================================================================
 
def perspective_projection(cam, obj, K):
    G     = np.linalg.inv(cam)
    Mproj = K @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) @ G

    valid = ~np.isnan(obj[0,:])
    ping  = np.full((3, obj.shape[1]), np.nan)
    ping[:,valid] = Mproj @ obj[:,valid]

    p_img = np.full_like(ping, np.nan)
    nz = valid & (np.abs(ping[2,:]) > 1e-6)
    p_img[:,nz] = ping[:,nz] / ping[2,nz]
    return p_img
 
def draw_arrows(point, base, axis, length=1.5):
    for col, ci in [('red',0),('green',1),('blue',2)]:
        axis.quiver(point[0],point[1],point[2],
                    base[0,ci],base[1,ci],base[2,ci],
                    color=col, pivot='tail', length=length)
 
# =============================================================================
# Application
# =============================================================================
 
class CameraApp:
 
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
 
        self.world = np.eye(4)
        cam = np.eye(4)
        cam = own_move(0, -10, 6, cam)
        cam = own_x_rotation(80, cam)
        self.cam_initial = cam.copy()
        self.cam = cam.copy()
 
        self.obj   = build_satellite(scale=5.0)
        self.f     = 15.0
        self.mx    = 30 
        self.my    = 25
        self.w_px  = 1280
        self.h_px  = 720
        self.skew  = 0.0
        self.K     = self._make_K()
        self.ref   = 'world'
 
        self._build_ui()
        self._refresh()
        plt.show()
 
    def _make_K(self):
        # f in mm, mx/my in pixels/mm
        fx = self.f * self.mx
        fy = self.f * self.my
        ox = self.w_px / 2.0
        oy = self.h_px / 2.0
        return np.array([[fx, self.skew, ox], [0, fy, oy], [0, 0, 1]])
 
    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(17, 9))
        self.fig.patch.set_facecolor('#f5f5f5')
 
        # ── Plots ──────────────────────────────────────────────────────────
        self.ax3d   = self.fig.add_axes([0.02, 0.50, 0.60, 0.47], projection='3d')
        self.ax_img = self.fig.add_axes([0.02, 0.04, 0.60, 0.42])
        self.ax3d.set_facecolor('#ffffff')
        self.ax_img.set_facecolor('#ffffff')
 
        PX  = 0.645 
        PW  = 0.345   
        LW  = 0.072   
        FW  = 0.262   
        BH  = 0.034   
        GAP = 0.007  
        BW  = PW * 0.55  
        fc  = '#e8e8e8'
        tc  = '#222222'
        lc  = '#1a5276'
        btn = '#d0d0d0'
 
        def ry(i):
            return 0.965 - i*(BH+GAP)
 
        def lbl(i, text, color=tc, bold=False, full=False):
            ax = self.fig.add_axes([PX, ry(i), PW if full else LW, BH])
            ax.set_facecolor('#f5f5f5'); ax.axis('off')
            ax.text(0.5, 0.5, text, ha='center', va='center', color=color,
                    fontsize=8, fontweight='bold' if bold else 'normal',
                    transform=ax.transAxes)
 
        def flbl(i, text, color=tc):
            ax = self.fig.add_axes([PX, ry(i), LW, BH])
            ax.set_facecolor('#f5f5f5'); ax.axis('off')
            ax.text(0.95, 0.5, text, ha='right', va='center', color=color,
                    fontsize=8, fontweight='bold', transform=ax.transAxes)
 
        def tb(i, val='0'):
            ax = self.fig.add_axes([PX+LW+0.006, ry(i), FW, BH])
            t  = TextBox(ax, '', initial=val, color=fc, hovercolor='#d0d0d0', label_pad=0)
            t.label.set_color(tc); t.text_disp.set_color(tc)
            return t
 
        def btn_w(i, text, x=None, w=None):
            ax = self.fig.add_axes([x if x else PX, ry(i), w if w else PW, BH])
            b  = Button(ax, text, color=btn, hovercolor='#aaaaaa')
            b.label.set_color(tc); b.label.set_fontsize(8)
            return b
 
        def radio(i, labels):
            ax = self.fig.add_axes([PX+LW+0.006, ry(i), FW, BH*2.0])
            ax.set_facecolor('#f5f5f5')
            rb = RadioButtons(ax, labels, activecolor='#2e86c1')
            for l in rb.labels: l.set_color(tc); l.set_fontsize(8)
            return rb
 
        def box_rect(row_top, row_bot):
            top    = ry(row_top) + BH + 0.010
            bottom = ry(row_bot) - 0.006
            self.fig.add_artist(mpatches.FancyBboxPatch(
                (PX-0.008, bottom), PW+0.016, top-bottom,
                boxstyle="round,pad=0.005",
                linewidth=1.2, edgecolor='#1a5276', facecolor='#eaf2fb',
                transform=self.fig.transFigure, clip_on=False, zorder=0))
 
        lbl(0, 'Camera Controls', color='#1a3a5c', bold=True, full=True)
 
        lbl(1, 'Reference:', color=lc, bold=True)
        self.radio_ref = radio(1, ('World', 'Camera'))
 
        box_rect(3, 13)
 
        lbl(3, '                    --- Translation ---', color=lc, bold=True, full=True)
        flbl(4, 'dx');  self.tb_dx = tb(4)
        flbl(5, 'dy');  self.tb_dy = tb(5)
        flbl(6, 'dz');  self.tb_dz = tb(6)
 
        lbl(8, '                    --- Rotation ---', color=lc, bold=True, full=True)
        flbl(9,  'Rx (°)'); self.tb_rx = tb(9)
        flbl(10, 'Ry (°)'); self.tb_ry = tb(10)
        flbl(11, 'Rz (°)'); self.tb_rz = tb(11)
 
        RW = FW * 0.70
        self.btn_reset = btn_w(13, 'Reset Transformations',
                               x=PX+LW+0.006+FW-RW, w=RW)
 
        box_rect(15, 20)
 
        lbl(15, '                   --- Intrinsics ---', color=lc, bold=True, full=True)
        flbl(16, 'f (mm)'); self.tb_f   = tb(16, str(self.f))
        flbl(17, 'mx (px/mm)'); self.tb_mx = tb(17, f"{self.mx:.2f}")
        flbl(18, 'my (px/mm)'); self.tb_my = tb(18, f"{self.my:.2f}")

        self.btn_reset_intrinsic = btn_w(20, 'Reset Intrinsic Parameters',
                               x=PX+LW+0.006+FW-RW, w=RW)
 
        ox, oy = self.w_px/2, self.h_px/2
        ax_pp = self.fig.add_axes([PX, ry(19), PW, BH])
        ax_pp.set_facecolor('#eaf2fb'); ax_pp.axis('off')
        self.txt_pp = ax_pp.text(0.5, 0.5,
            f'Principal Point: ({ox:.0f}, {oy:.0f})',
            ha='center', va='center', color=lc,
            fontsize=8, transform=ax_pp.transAxes)
 
        self.btn_info = btn_w(22, 'Info (terminal)',
                              x=PX+(PW-BW)/2, w=BW)
 
        ax_st = self.fig.add_axes([PX, ry(23), PW, BH])
        ax_st.set_facecolor('#f5f5f5'); ax_st.axis('off')
        self.txt_status = ax_st.text(
            0.5, 0.5, 'Ready.', ha='center', va='center',
            color='#196f3d', fontsize=8, transform=ax_st.transAxes)
 
        self.btn_reset.on_clicked(self._reset)
        self.btn_reset_intrinsic.on_clicked(self._reset_intrinsic)
        self.btn_info.on_clicked(self._show_info)
        self.radio_ref.on_clicked(self._set_ref)
        self.tb_dx.on_submit(self._apply_translation)
        self.tb_dy.on_submit(self._apply_translation)
        self.tb_dz.on_submit(self._apply_translation)
        self.tb_rx.on_submit(self._apply_rotation)
        self.tb_ry.on_submit(self._apply_rotation)
        self.tb_rz.on_submit(self._apply_rotation)
        self.tb_f.on_submit(self._apply_intrinsics)
        self.tb_mx.on_submit(self._apply_intrinsics) 
        self.tb_my.on_submit(self._apply_intrinsics) 
 
    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
 
    def _set_ref(self, label):
        self.ref = 'world' if label == 'World' else 'camera'
        self._status(f'Reference: {label}')
 
    def _read(self, t, default=0.0):
        try:    return float(t.text)
        except: return default
 
    def _apply_translation(self, _=None):
        dx=self._read(self.tb_dx); dy=self._read(self.tb_dy); dz=self._read(self.tb_dz)
        if self.ref == 'camera':
            self.cam = own_move(dx, dy, dz, self.cam)
        else:
            self.cam = move(dx, dy, dz) @ self.cam
        self._status(f'Translation ({dx:.2f}, {dy:.2f}, {dz:.2f}) [{self.ref}]')
        self._refresh()
 
    def _apply_rotation(self, _=None):
        rx=self._read(self.tb_rx); ry=self._read(self.tb_ry); rz=self._read(self.tb_rz)
        if self.ref == 'camera':
            if rx: self.cam = own_x_rotation(rx, self.cam)
            if ry: self.cam = own_y_rotation(ry, self.cam)
            if rz: self.cam = own_z_rotation(rz, self.cam)
        else:
            if rx: self.cam = x_rotation(rx) @ self.cam
            if ry: self.cam = y_rotation(ry) @ self.cam
            if rz: self.cam = z_rotation(rz) @ self.cam
        self._status(f'Rotation Rx={rx:.1f}° Ry={ry:.1f}° Rz={rz:.1f}° [{self.ref}]')
        self._refresh()
 
    def _apply_intrinsics(self, _=None):
        try:
            self.f  = float(self.tb_f.text)
            self.mx = float(self.tb_mx.text) 
            self.my = float(self.tb_my.text) 
            
            self.K = self._make_K()
            
            ox, oy = self.w_px / 2, self.h_px / 2
            self.txt_pp.set_text(f'Principal Point: ({ox:.0f}, {oy:.0f})')
            
            self._status(f'K updated. f={self.f:.1f}, mx={self.mx:.1f}')
            self._refresh()
        except Exception as e:
            self._status(f'Error: {e}')
 
    def _reset(self, _=None):
        self.cam = self.cam_initial.copy()
        self.tb_dx.set_val('0')
        self.tb_dy.set_val('0')
        self.tb_dz.set_val('0')
        self.tb_rx.set_val('0')
        self.tb_ry.set_val('0')
        self.tb_rz.set_val('0')
        self._status('Transformations reset.')
        self._refresh()

    def _reset_intrinsic(self, _=None):
        self.cam = self.cam_initial.copy()
        self.tb_f.set_val('15')
        self.tb_mx.set_val('35')
        self.tb_my.set_val('25')
        self._status('Intrinsics reset.')
        self._refresh()    
 
    def _show_info(self, _=None):
        G = np.linalg.inv(self.cam)
        print('\n=== Current State ===')
        print(f'Reference  : {self.ref}')
        print(f'cam:\n{self.cam}')
        print(f'G=inv(cam):\n{G}')
        print(f'K:\n{self.K}')
        self._status('Info printed to terminal.')
 
    def _status(self, msg):
        self.txt_status.set_text(msg)
        self.fig.canvas.draw_idle()
 
    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------
 
    def _refresh(self):
        self._draw_3d()
        self._draw_image()
        self.fig.canvas.draw_idle()
 
    def _draw_3d(self):
        ax = self.ax3d; ax.cla()
        ax.set_facecolor('#ffffff')
        lim = [-15,20]
        ax.set_xlim(lim); ax.set_xlabel('x', color='#333333', fontsize=8)
        ax.set_ylim(lim); ax.set_ylabel('y', color='#333333', fontsize=8)
        ax.set_zlim(lim); ax.set_zlabel('z', color='#333333', fontsize=8)
        ax.set_title('3D Scene', color='#1a1a1a', fontsize=10, pad=4)
        ax.tick_params(colors='#444444', labelsize=6)
        ax.plot3D(self.obj[0,:], self.obj[1,:], self.obj[2,:],
                  '#c0392b', linewidth=0.8)
        draw_arrows(self.world[:,-1],       self.world[:,0:3],       ax, 3.0)
        draw_arrows(self.cam_initial[:,-1], self.cam_initial[:,0:3], ax, 1.5)
        draw_arrows(self.cam[:,-1],         self.cam[:,0:3],         ax, 1.5)
 
    def _draw_image(self):
        ax = self.ax_img; ax.cla()
        ax.set_facecolor('#ffffff')
        ax.set_title('Camera Image', color='#1a1a1a', fontsize=10, pad=4)
        ax.set_xlim([0, self.w_px]); ax.set_ylim([self.h_px, 0])
        ax.set_xlabel('u (px)', color='#333333', fontsize=8)
        ax.set_ylabel('v (px)', color='#333333', fontsize=8)
        ax.tick_params(colors='#444444', labelsize=6)
        ax.grid(True, color='#cccccc', linewidth=0.5)
        p_img = perspective_projection(self.cam, self.obj, self.K)
        ax.plot(p_img[0,:], p_img[1,:], color='#c0392b', linewidth=0.8)
        ox,oy = self.w_px/2, self.h_px/2
        ax.plot(ox, oy, '+', color='#2e86c1', markersize=10, markeredgewidth=1.5)
        ax.text(ox+10, oy-15, 'pp', color='#2e86c1', fontsize=7)
 
# =============================================================================
if __name__ == '__main__':
    app = CameraApp()
 