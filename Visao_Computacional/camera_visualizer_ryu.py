"""
Camera Visualizer - Assignment 1  (objeto: Ryu do Street Fighter)
Visualizes 3D camera/object positions and generates camera images.
Supports extrinsic (position/orientation) and intrinsic parameter modification.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rot_x(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [1, 0,          0         ],
        [0, np.cos(a), -np.sin(a) ],
        [0, np.sin(a),  np.cos(a) ],
    ])

def rot_y(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [ np.cos(a), 0, np.sin(a)],
        [ 0,         1, 0        ],
        [-np.sin(a), 0, np.cos(a)],
    ])

def rot_z(angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1],
    ])

def make_rotation(rx, ry, rz):
    """Extrinsic XYZ rotation (world axes)."""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)

def homogeneous(R, t):
    """Build 4x4 SE(3) matrix from 3x3 R and 3-vector t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ---------------------------------------------------------------------------
# Object definition – Ryu do Street Fighter
# ---------------------------------------------------------------------------
# Importa a geometria do Ryu definida em ryu_object.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ryu_object import OBJECT_VERTS, OBJECT_FACES, OBJECT_COLORS
#from objects_3d import build_rocket, build_mug, build_robot

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Visualizer – Assignment 1")
        self.resizable(True, True)

        # ---- Camera extrinsic state ----
        self.cam_tx = tk.DoubleVar(value=0.0)
        self.cam_ty = tk.DoubleVar(value=-8.0)
        self.cam_tz = tk.DoubleVar(value=2.5)
        self.cam_rx = tk.DoubleVar(value=80.0)
        self.cam_ry = tk.DoubleVar(value=0.0)
        self.cam_rz = tk.DoubleVar(value=0.0)
        self.ref_frame = tk.StringVar(value="world")  # "world" or "camera"

        # ---- Camera intrinsic state ----
        self.focal_x  = tk.DoubleVar(value=600.0)
        self.focal_y  = tk.DoubleVar(value=600.0)
        self.img_w    = tk.IntVar(value=640)
        self.img_h    = tk.IntVar(value=480)

        # Derived principal point (auto from image size)
        self.cx = self.img_w.get() / 2.0
        self.cy = self.img_h.get() / 2.0

        self._build_ui()
        self._update()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Top-level paned layout ----
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6,
                               sashrelief=tk.RAISED, bg="#2b2b2b")
        paned.pack(fill=tk.BOTH, expand=True)

        # Left panel: controls
        ctrl_frame = tk.Frame(paned, bg="#1e1e2e", width=310)
        ctrl_frame.pack_propagate(False)
        paned.add(ctrl_frame, minsize=280)

        # Right panel: plots
        plot_frame = tk.Frame(paned, bg="#1e1e2e")
        paned.add(plot_frame, minsize=600)

        self._build_controls(ctrl_frame)
        self._build_plots(plot_frame)

    def _build_controls(self, parent):
        style_kw = dict(bg="#1e1e2e", fg="#cdd6f4")
        lbl_kw   = dict(bg="#1e1e2e", fg="#89b4fa", font=("Helvetica", 10, "bold"))
        sep_kw   = dict(bg="#313244")

        canvas = tk.Canvas(parent, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#1e1e2e")
        scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        pad = dict(padx=10, pady=3)

        # Title
        tk.Label(scroll_frame, text="🎥 Camera Visualizer",
                 font=("Helvetica", 13, "bold"), bg="#1e1e2e", fg="#cba6f7"
                 ).pack(pady=(12,4))

        # ---- Reference frame ----
        tk.Frame(scroll_frame, height=1, **sep_kw).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(scroll_frame, text="Reference Frame", **lbl_kw).pack(**pad)
        rf_frame = tk.Frame(scroll_frame, bg="#1e1e2e")
        rf_frame.pack()
        for text, val in [("World", "world"), ("Camera", "camera")]:
            tk.Radiobutton(rf_frame, text=text, variable=self.ref_frame,
                           value=val, command=self._update,
                           bg="#1e1e2e", fg="#cdd6f4",
                           selectcolor="#313244",
                           activebackground="#1e1e2e").pack(side=tk.LEFT, padx=8)

        # ---- Extrinsic: Translation ----
        tk.Frame(scroll_frame, height=1, **sep_kw).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(scroll_frame, text="Translation (m)", **lbl_kw).pack(**pad)
        self._slider_row(scroll_frame, "Tx", self.cam_tx, -10, 10)
        self._slider_row(scroll_frame, "Ty", self.cam_ty, -10, 10)
        self._slider_row(scroll_frame, "Tz", self.cam_tz, -10, 10)

        # ---- Extrinsic: Rotation ----
        tk.Frame(scroll_frame, height=1, **sep_kw).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(scroll_frame, text="Rotation (°)", **lbl_kw).pack(**pad)
        self._slider_row(scroll_frame, "Rx", self.cam_rx, -180, 180)
        self._slider_row(scroll_frame, "Ry", self.cam_ry, -180, 180)
        self._slider_row(scroll_frame, "Rz", self.cam_rz, -180, 180)

        # ---- Incremental move buttons ----
        tk.Frame(scroll_frame, height=1, **sep_kw).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(scroll_frame, text="Step Move", **lbl_kw).pack(**pad)
        self._step_controls(scroll_frame)

        # ---- Intrinsic parameters ----
        tk.Frame(scroll_frame, height=1, **sep_kw).pack(fill=tk.X, padx=8, pady=4)
        tk.Label(scroll_frame, text="Intrinsic Parameters", **lbl_kw).pack(**pad)
        self._slider_row(scroll_frame, "fx (px)", self.focal_x, 100, 2000, resolution=10)
        self._slider_row(scroll_frame, "fy (px)", self.focal_y, 100, 2000, resolution=10)
        self._int_slider_row(scroll_frame, "Width",  self.img_w,  160, 1920, step=16)
        self._int_slider_row(scroll_frame, "Height", self.img_h,  120, 1080, step=16)

        # Principal point display
        self.pp_label = tk.Label(scroll_frame,
            text=f"Principal point: cx={self.cx:.1f}, cy={self.cy:.1f}",
            bg="#1e1e2e", fg="#a6e3a1", font=("Helvetica", 9))
        self.pp_label.pack(pady=(2,6))

        # Reset button
        tk.Frame(scroll_frame, height=1, **sep_kw).pack(fill=tk.X, padx=8, pady=4)
        tk.Button(scroll_frame, text="↺  Reset All", command=self._reset,
                  bg="#313244", fg="#cdd6f4", relief=tk.FLAT,
                  font=("Helvetica", 10, "bold"),
                  activebackground="#45475a").pack(pady=8, ipadx=12, ipady=4)

    def _slider_row(self, parent, label, var, lo, hi, resolution=0.1):
        row = tk.Frame(parent, bg="#1e1e2e")
        row.pack(fill=tk.X, padx=10, pady=1)
        tk.Label(row, text=f"{label:>8}", width=9, anchor="e",
                 bg="#1e1e2e", fg="#cdd6f4", font=("Courier", 9)).pack(side=tk.LEFT)
        sl = tk.Scale(row, from_=lo, to=hi, resolution=resolution,
                      orient=tk.HORIZONTAL, variable=var,
                      command=lambda _: self._update(),
                      bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
                      highlightthickness=0, bd=0, length=155)
        sl.pack(side=tk.LEFT)
        ent = tk.Entry(row, textvariable=var, width=7,
                       bg="#313244", fg="#cdd6f4",
                       insertbackground="#cdd6f4", relief=tk.FLAT)
        ent.pack(side=tk.LEFT, padx=4)
        ent.bind("<Return>", lambda _: self._update())

    def _int_slider_row(self, parent, label, var, lo, hi, step=1):
        row = tk.Frame(parent, bg="#1e1e2e")
        row.pack(fill=tk.X, padx=10, pady=1)
        tk.Label(row, text=f"{label:>8}", width=9, anchor="e",
                 bg="#1e1e2e", fg="#cdd6f4", font=("Courier", 9)).pack(side=tk.LEFT)
        sl = tk.Scale(row, from_=lo, to=hi, resolution=step,
                      orient=tk.HORIZONTAL, variable=var,
                      command=lambda _: self._update(),
                      bg="#313244", fg="#cdd6f4", troughcolor="#45475a",
                      highlightthickness=0, bd=0, length=155)
        sl.pack(side=tk.LEFT)
        ent = tk.Entry(row, textvariable=var, width=7,
                       bg="#313244", fg="#cdd6f4",
                       insertbackground="#cdd6f4", relief=tk.FLAT)
        ent.pack(side=tk.LEFT, padx=4)
        ent.bind("<Return>", lambda _: self._update())

    def _step_controls(self, parent):
        """Buttons for incremental translation/rotation."""
        grid = tk.Frame(parent, bg="#1e1e2e")
        grid.pack(pady=4)
        self.step_t = tk.DoubleVar(value=0.5)
        self.step_r = tk.DoubleVar(value=5.0)
        btn_kw = dict(bg="#313244", fg="#cdd6f4", relief=tk.FLAT,
                      activebackground="#45475a", width=5)

        axes = ["X", "Y", "Z"]
        vars_t = [self.cam_tx, self.cam_ty, self.cam_tz]
        vars_r = [self.cam_rx, self.cam_ry, self.cam_rz]

        tk.Label(grid, text="Δt:", bg="#1e1e2e", fg="#cdd6f4").grid(row=0, column=0)
        tk.Entry(grid, textvariable=self.step_t, width=5,
                 bg="#313244", fg="#cdd6f4", relief=tk.FLAT).grid(row=0, column=1)
        tk.Label(grid, text="Δr°:", bg="#1e1e2e", fg="#cdd6f4").grid(row=0, column=2)
        tk.Entry(grid, textvariable=self.step_r, width=5,
                 bg="#313244", fg="#cdd6f4", relief=tk.FLAT).grid(row=0, column=3)

        for i, (ax, vt, vr) in enumerate(zip(axes, vars_t, vars_r)):
            r = i + 1
            tk.Label(grid, text=ax, bg="#1e1e2e", fg="#89b4fa",
                     font=("Helvetica",9,"bold")).grid(row=r, column=0, padx=4)
            tk.Button(grid, text=f"t-",
                      command=lambda v=vt: self._step(v, -self.step_t.get()),
                      **btn_kw).grid(row=r, column=1, padx=2, pady=1)
            tk.Button(grid, text=f"t+",
                      command=lambda v=vt: self._step(v, +self.step_t.get()),
                      **btn_kw).grid(row=r, column=2, padx=2)
            tk.Button(grid, text=f"r-",
                      command=lambda v=vr: self._step(v, -self.step_r.get()),
                      **btn_kw).grid(row=r, column=3, padx=2)
            tk.Button(grid, text=f"r+",
                      command=lambda v=vr: self._step(v, +self.step_r.get()),
                      **btn_kw).grid(row=r, column=4, padx=2)

    def _step(self, var, delta):
        if self.ref_frame.get() == "world":
            var.set(round(var.get() + delta, 6))
            self._update()
        else:
            # Move/rotate in camera frame
            self._apply_camera_frame_step(var, delta)

    def _apply_camera_frame_step(self, var, delta):
        """
        For camera-frame moves: transform delta into world frame then apply.
        For rotations, compose on the right.
        """
        # Determine which variable was stepped
        tx_v = [self.cam_tx, self.cam_ty, self.cam_tz]
        rx_v = [self.cam_rx, self.cam_ry, self.cam_rz]

        R_cw = make_rotation(self.cam_rx.get(), self.cam_ry.get(), self.cam_rz.get())

        if var in tx_v:
            idx = tx_v.index(var)
            cam_dir = np.zeros(3); cam_dir[idx] = delta
            world_dir = R_cw @ cam_dir
            self.cam_tx.set(round(self.cam_tx.get() + world_dir[0], 6))
            self.cam_ty.set(round(self.cam_ty.get() + world_dir[1], 6))
            self.cam_tz.set(round(self.cam_tz.get() + world_dir[2], 6))
        else:
            idx = rx_v.index(var)
            # Compose rotation: R_new = R_cw @ R_delta_camera
            delta_vec = np.zeros(3); delta_vec[idx] = delta
            R_delta = make_rotation(*delta_vec)
            R_new = R_cw @ R_delta
            # Extract Euler angles (XYZ extrinsic)
            sy = np.sqrt(R_new[0,0]**2 + R_new[1,0]**2)
            if sy > 1e-6:
                rx = np.degrees(np.arctan2(R_new[2,1], R_new[2,2]))
                ry = np.degrees(np.arctan2(-R_new[2,0], sy))
                rz = np.degrees(np.arctan2(R_new[1,0], R_new[0,0]))
            else:
                rx = np.degrees(np.arctan2(-R_new[1,2], R_new[1,1]))
                ry = np.degrees(np.arctan2(-R_new[2,0], sy))
                rz = 0
            self.cam_rx.set(round(rx, 4))
            self.cam_ry.set(round(ry, 4))
            self.cam_rz.set(round(rz, 4))

        self._update()

    def _build_plots(self, parent):
        self.fig = plt.Figure(figsize=(11, 6), facecolor="#1e1e2e")
        self.ax3d  = self.fig.add_subplot(121, projection="3d")
        self.ax_img = self.fig.add_subplot(122)
        self.fig.subplots_adjust(left=0.02, right=0.98, wspace=0.08)

        canvas = FigureCanvasTkAgg(self.fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    # ------------------------------------------------------------------
    # Camera math
    # ------------------------------------------------------------------

    def _get_camera_matrix(self):
        """Build intrinsic matrix K (3x3)."""
        fx = self.focal_x.get()
        fy = self.focal_y.get()
        return np.array([
            [fx,  0, self.cx],
            [ 0, fy, self.cy],
            [ 0,  0,      1],
        ])

    def _get_extrinsic(self):
        """
        Returns R_cw (3x3) and t_cw (3,):
        transforms world points to camera coords.
        p_cam = R_cw @ p_world + t_cw
        """
        tx = self.cam_tx.get()
        ty = self.cam_ty.get()
        tz = self.cam_tz.get()
        rx = self.cam_rx.get()
        ry = self.cam_ry.get()
        rz = self.cam_rz.get()

        # Camera pose in world: position t_wc, orientation R_wc
        t_wc = np.array([tx, ty, tz])
        R_wc = make_rotation(rx, ry, rz)

        # Extrinsic: world -> camera
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        return R_cw, t_cw

    def _project_points(self, pts_world):
        """
        Project (N,3) world points → (N,2) pixel coords.
        Returns pts_img (N,2) and mask of points in front of camera.
        """
        K = self._get_camera_matrix()
        R, t = self._get_extrinsic()

        pts_cam = (R @ pts_world.T).T + t   # (N,3)
        in_front = pts_cam[:, 2] > 0

        pts_h = (K @ pts_cam.T).T            # (N,3) homogeneous image
        with np.errstate(divide='ignore', invalid='ignore'):
            pts_img = pts_h[:, :2] / pts_h[:, 2:3]

        return pts_img, in_front

    def _camera_axes_in_world(self):
        """Return camera position and direction vectors in world coords."""
        tx = self.cam_tx.get(); ty = self.cam_ty.get(); tz = self.cam_tz.get()
        rx = self.cam_rx.get(); ry = self.cam_ry.get(); rz = self.cam_rz.get()
        origin = np.array([tx, ty, tz])
        R_wc = make_rotation(rx, ry, rz)
        return origin, R_wc   # columns are x,y,z axes of camera in world

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_3d(self):
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor("#1e1e2e")
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.set_xlabel("X", color="#89b4fa"); ax.set_ylabel("Y", color="#89b4fa")
        ax.set_zlabel("Z", color="#89b4fa")
        ax.tick_params(colors="#585b70")
        ax.set_title("3D Scene – Ryu", color="#cdd6f4", pad=6, fontsize=11)

        # World axes
        for vec, col, lbl in [([1,0,0],"#f38ba8","X"),
                                ([0,1,0],"#a6e3a1","Y"),
                                ([0,0,1],"#89b4fa","Z")]:
            ax.quiver(0,0,0,*vec,color=col,length=0.8,normalize=False,linewidth=1.5)
            ax.text(*(np.array(vec)*0.9), lbl, color=col, fontsize=8)

        # Object faces
        faces_pts = []
        for face in OBJECT_FACES:
            faces_pts.append([OBJECT_VERTS[i] for i in face])
        poly = Poly3DCollection(faces_pts, alpha=0.55,
                                facecolor=OBJECT_COLORS,
                                edgecolor="#45475a", linewidth=0.5)
        ax.add_collection3d(poly)

        # Object vertices
        ax.scatter(*OBJECT_VERTS.T, color="#fab387", s=18, zorder=5)

        # Camera body
        cam_pos, R_wc = self._camera_axes_in_world()
        ax.scatter(*cam_pos, color="#cba6f7", s=60, zorder=6, marker="^")
        ax.text(*(cam_pos + np.array([0.1,0.1,0.1])), "CAM",
                color="#cba6f7", fontsize=8, fontweight="bold")

        # Camera local axes
        scale = 1.2
        colors_cam = ["#f38ba8","#a6e3a1","#89b4fa"]
        labels_cam = ["Xc","Yc","Zc"]
        for i, (col, lbl) in enumerate(zip(colors_cam, labels_cam)):
            d = R_wc[:, i] * scale
            ax.quiver(*cam_pos, *d, color=col, linewidth=1.5,
                      arrow_length_ratio=0.2)
            ax.text(*(cam_pos + d * 1.05), lbl, color=col, fontsize=7)

        # Camera frustum (draw image plane corners as pyramid)
        fx = self.focal_x.get(); fy = self.focal_y.get()
        w = self.img_w.get(); h = self.img_h.get()
        depth = 2.5
        # Image corners in camera frame
        corners_cam = np.array([
            [(0 - self.cx) / fx, (0 - self.cy) / fy, 1],
            [(w - self.cx) / fx, (0 - self.cy) / fy, 1],
            [(w - self.cx) / fx, (h - self.cy) / fy, 1],
            [(0 - self.cx) / fx, (h - self.cy) / fy, 1],
        ]) * depth
        # Transform to world
        corners_world = (R_wc @ corners_cam.T).T + cam_pos

        for c in corners_world:
            ax.plot(*zip(cam_pos, c), color="#f9e2af", linewidth=0.8, alpha=0.7)
        # Draw image plane rectangle
        for i in range(4):
            ax.plot(*zip(corners_world[i], corners_world[(i+1)%4]),
                    color="#f9e2af", linewidth=0.8, alpha=0.7)

        lim = 6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.5, lim)
        ax.view_init(elev=20, azim=-60)

    def _draw_image(self):
        ax = self.ax_img
        ax.cla()
        ax.set_facecolor("#11111b")
        ax.set_title("Camera Image", color="#cdd6f4", pad=6, fontsize=11)

        W = self.img_w.get(); H = self.img_h.get()
        ax.set_xlim(0, W); ax.set_ylim(H, 0)   # origin top-left, Y down
        ax.set_xlabel("u (px)", color="#89b4fa", fontsize=8)
        ax.set_ylabel("v (px)", color="#89b4fa", fontsize=8)
        ax.tick_params(colors="#585b70", labelsize=7)

        # Image boundary
        rect = patches.Rectangle((0,0), W, H, linewidth=1.5,
                                  edgecolor="#585b70", facecolor="none")
        ax.add_patch(rect)

        # Project and draw each face
        for face, col in zip(OBJECT_FACES, OBJECT_COLORS):
            verts_w = OBJECT_VERTS[face]
            pts_2d, in_front = self._project_points(verts_w)

            # Only draw if all vertices are in front
            if not np.all(in_front):
                continue

            xs, ys = pts_2d[:, 0], pts_2d[:, 1]

            # Clip check: at least one vertex inside extended image
            margin = max(W, H) * 2
            if (np.all(xs < -margin) or np.all(xs > W + margin) or
                    np.all(ys < -margin) or np.all(ys > H + margin)):
                continue

            poly_pts = np.column_stack([xs, ys])
            poly2d = plt.Polygon(poly_pts, closed=True,
                                 facecolor=col, edgecolor="#45475a",
                                 linewidth=0.8, alpha=0.7)
            ax.add_patch(poly2d)

        # Project vertices
        pts_2d, in_front = self._project_points(OBJECT_VERTS)
        visible = in_front & (pts_2d[:,0] >= 0) & (pts_2d[:,0] <= W) & \
                              (pts_2d[:,1] >= 0) & (pts_2d[:,1] <= H)
        if visible.any():
            ax.scatter(pts_2d[visible, 0], pts_2d[visible, 1],
                       color="#fab387", s=20, zorder=5)

        # Principal point cross-hair
        ax.plot(self.cx, self.cy, "+", color="#f38ba8", markersize=12, markeredgewidth=1.5)
        ax.text(self.cx + 5, self.cy - 8, "pp", color="#f38ba8", fontsize=7)

    # ------------------------------------------------------------------
    # Update callback
    # ------------------------------------------------------------------

    def _update(self, *_):
        # Update principal point from image size
        self.cx = self.img_w.get() / 2.0
        self.cy = self.img_h.get() / 2.0
        self.pp_label.config(
            text=f"Principal point: cx={self.cx:.1f}, cy={self.cy:.1f}")

        self._draw_3d()
        self._draw_image()
        self.canvas.draw()

    def _reset(self):
        self.cam_tx.set(0.0); self.cam_ty.set(-8.0); self.cam_tz.set(2.5)
        self.cam_rx.set(80.0); self.cam_ry.set(0.0); self.cam_rz.set(0.0)
        self.focal_x.set(600.0); self.focal_y.set(600.0)
        self.img_w.set(640); self.img_h.set(480)
        self.ref_frame.set("world")
        self._update()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = CameraApp()
    app.mainloop()
