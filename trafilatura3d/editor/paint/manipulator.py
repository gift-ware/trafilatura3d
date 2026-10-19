"""The 3D projection manipulator and contextual toolbar (Fill projections pages, "3D projection contextual toolbar
and manipulator") and the Warp projection's grid edition.

With a fill (layer or fill effect) using a 3D projection selected, the viewport shows the projection's frame and a
manipulator at its origin: Translate (W), Rotate (E), Scale (R) along X / Y / Z (red, green, blue), Surface
(Shift+W, Planar and Warp: the projection snaps on the model, looking at it), in Local or World space (T), shown or
hidden (Q), bigger or smaller (+ / -). The contextual toolbar also mirrors the transform on X / Y / Z and resets it.
Grid steps and Angle steps snap translation / scale and rotation while Shift is held.

Warp projection: Transform warp (the manipulator) or Edit vertices (Shift+V): click a grid point to select it
(Shift adds, Ctrl removes), drag it on the surface (the other selected points follow); Split warp crosswise /
horizontally / vertically doubles the grid's cells; Rows and columns while no point was moved; Automatic tangents;
Reset: the global transformation, all vertices, the selected vertices.

Ours where the documentation is silent: the gizmo's look and sizes, the picking distance, the default steps.
Not offered: the tangents' manipulators (Rotation / Scale on a point), rectangle selection of points, placing a
split interactively (a split doubles every row or column).
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QToolButton, QWidget

from trafilatura3d.paint.model import MATCH_PER_TILE, warp_grid

PICK_PX = 9.0
AXIS_COLORS = ((1.0, 0.3, 0.3, 1.0), (0.3, 1.0, 0.3, 1.0), (0.35, 0.55, 1.0, 1.0))
MODES = ("Translate", "Rotate", "Scale", "Surface")


def euler_matrix(deg) -> np.ndarray:
    """The projection's rotation from its 3D Rotation (degrees about X, Y, Z: Rz @ Ry @ Rx, as stack.py)."""
    rx, ry, rz = np.radians(deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def matrix_euler(m) -> list:
    """The 3D Rotation (degrees) of a rotation matrix, inverse of `euler_matrix`."""
    sy = -m[2, 0]
    ry = math.asin(max(-1.0, min(1.0, sy)))
    if abs(sy) < 0.99999:
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: Z folded into X
        rx = math.atan2(-m[1, 2], m[1, 1])
        rz = 0.0
    return [math.degrees(rx), math.degrees(ry), math.degrees(rz)]


def axis_rotation(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-12)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class Manipulator:
    def __init__(self, ws):
        self.ws = ws
        self.visible = True
        self.mode = "Translate"
        self.space = "Local space"
        self.size = 1.0
        self.grid_step = 0.05  # bounding box units (Shift)
        self.angle_step = 15.0  # degrees (Shift)
        self.warp_edit = False  # Warp: Edit vertices instead of Transform warp
        self.selected_points: set = set()
        self.handle_size = 1.0
        self.grid_color = (0.9, 0.9, 0.2, 1.0)
        self._drag = None

    # --- what is manipulated -------------------------------------------------------------------------

    def fill(self):
        """The selected fill layer or fill effect, when its projection is a 3D one."""
        doc = self.ws.doc
        if doc is None:
            return None
        fx = doc.effect()
        obj = fx if fx is not None and fx.kind == "fill" else (doc.layer() if fx is None else None)
        if obj is None or (getattr(obj, "kind", "") not in ("fill",)) or obj.projection.mode in ("UV projection", MATCH_PER_TILE):
            return None
        return obj

    def _frame(self, pr):
        """(origin, rotation, axes lengths) of the projection in the scene (paint/stack.py `_projection`)."""
        _maps, lo, hi = self.ws.doc.session.geometry(self.ws.doc.ts, self.ws.doc.ts.tiles[0])
        unit = float(np.max(hi - lo)) or 1.0
        origin = (lo + hi) / 2 + np.asarray(pr.offset3d) * unit
        return origin, euler_matrix(pr.rotation3d), np.asarray(pr.scale3d, float) * unit, unit, (lo + hi) / 2

    def _local_to_world(self, pr, p):
        origin, rot, scale, _u, _c = self._frame(pr)
        return origin + rot @ (np.asarray(p) * scale)

    def _world_to_local(self, pr, w):
        origin, rot, scale, _u, _c = self._frame(pr)
        return (rot.T @ (np.asarray(w) - origin)) / np.maximum(np.abs(scale), 1e-9)

    def edit(self, label, fn, merge=None):
        uid = self.fill().uid

        def do():
            fn(self.ws.properties._obj(uid).projection)
        self.ws.doc.edit(label, do, merge=merge, structure=False)
        self.ws.refresh_materials()
        self.refresh()

    # --- drawing ----------------------------------------------------------------------------------------

    def refresh(self):
        r = self.ws.view3d.renderer
        r.extra_lines = self.lines()
        self.ws.view3d.refresh()

    def _gizmo_axes(self, pr):
        origin, rot, scale, unit, _c = self._frame(pr)
        axes = rot if self.space == "Local space" else np.eye(3)
        length = 0.25 * unit * self.size
        return origin, axes, length

    def lines(self):
        obj = self.fill()
        if obj is None or not self.visible:
            return None
        pr = obj.projection
        segs = []

        def seg(a, b, color):
            s = np.empty((2, 7), np.float32)
            s[0, :3], s[1, :3], s[:, 3:] = a, b, color
            segs.append(s)
        # the projection's frame: the square (Planar, Warp) or the cube of its shape
        corners = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        frame_col = (0.85, 0.85, 0.85, 0.7)
        zs = (0.0,) if pr.mode in ("Planar projection", "Warp projection") else (-0.5, 0.5)
        for z in zs:
            for i in range(4):
                a, b = corners[i], corners[(i + 1) % 4]
                seg(self._local_to_world(pr, (*a, z)), self._local_to_world(pr, (*b, z)), frame_col)
        if len(zs) == 2:
            for a in corners:
                seg(self._local_to_world(pr, (*a, -0.5)), self._local_to_world(pr, (*a, 0.5)), frame_col)
        if pr.mode == "Warp projection":
            segs += self._grid_lines(pr)
        if not (pr.mode == "Warp projection" and self.warp_edit):
            origin, axes, length = self._gizmo_axes(pr)
            for k in range(3):
                c = AXIS_COLORS[k]
                if self.mode == "Rotate":
                    u, v = axes[:, (k + 1) % 3], axes[:, (k + 2) % 3]
                    t = np.linspace(0, 2 * math.pi, 49)
                    pts = origin + length * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
                    for i in range(48):
                        seg(pts[i], pts[i + 1], c)
                else:
                    end = origin + axes[:, k] * length
                    seg(origin, end, c)
                    if self.mode == "Scale":  # a small square at the end
                        u = axes[:, (k + 1) % 3] * length * 0.06
                        seg(end - u, end + u, c)
        return np.concatenate(segs) if segs else None

    def _grid_lines(self, pr):
        from trafilatura3d.paint import warp
        k = 8  # samples per cell along a line
        grid = warp.points(pr)
        interp = warp._catmull_rom if pr.warp_tangents else warp._linear
        rows, cols = pr.warp_rows, pr.warp_cols
        # the rows' curves (along u) and the columns' (along v), through the grid's points
        row_lines = interp(np.swapaxes(grid, 0, 1), np.linspace(0, cols, cols * k + 1))  # (n, rows + 1, 3)
        col_lines = interp(grid, np.linspace(0, rows, rows * k + 1))  # (n, cols + 1, 3)
        lines = [row_lines[:, r] for r in range(rows + 1)] + [col_lines[:, c] for c in range(cols + 1)]
        out = []
        for local in lines:
            line = np.array([self._local_to_world(pr, p) for p in local])
            s = np.empty(((len(line) - 1) * 2, 7), np.float32)
            s[0::2, :3], s[1::2, :3] = line[:-1], line[1:]
            s[:, 3:] = self.grid_color
            out.append(s)
        if self.warp_edit:  # the points as crosses (selected: white)
            pts = warp.points(pr).reshape(-1, 3)
            r = self._frame(pr)[3] * 0.008 * self.handle_size
            for i, p in enumerate(pts):
                w = self._local_to_world(pr, p)
                col = (1, 1, 1, 1) if i in self.selected_points else self.grid_color
                s = np.empty((6, 7), np.float32)
                for k, ax in enumerate(np.eye(3)):
                    s[2 * k, :3], s[2 * k + 1, :3] = w - ax * r, w + ax * r
                s[:, 3:] = col
                out.append(s)
        return out

    # --- interaction -------------------------------------------------------------------------------------

    def _screen(self, p):
        view, _cam = self.ws._camera()
        return self.ws._screen(view, p)

    def _pick_axis(self, x, y, pr):
        origin, axes, length = self._gizmo_axes(pr)
        best, best_d = None, PICK_PX * self.ws.view3d.get_pixel_ratio()
        for k in range(3):
            if self.mode == "Rotate":
                u, v = axes[:, (k + 1) % 3], axes[:, (k + 2) % 3]
                pts = [origin + length * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * math.pi, 33)]
            else:
                pts = [origin + axes[:, k] * length * t for t in np.linspace(0, 1, 17)]
            sp = [self._screen(p) for p in pts]
            d = min((math.hypot(s[0] - x, s[1] - y) for s in sp if s is not None), default=1e9)
            if d < best_d:
                best, best_d = k, d
        return best

    def press(self, x, y, shift=False, ctrl=False) -> bool:
        obj = self.fill()
        if obj is None:
            return False
        pr = obj.projection
        if pr.mode == "Warp projection" and self.warp_edit:
            return self._press_point(x, y, pr, shift, ctrl)
        if self.mode == "Surface":
            self._surface(x, y)
            self._drag = ("surface",)
            return True
        axis = self._pick_axis(x, y, pr)  # None: the screen plane (Translate), every axis (Scale)
        self._drag = ("axis", axis, (x, y), [list(pr.offset3d), list(pr.rotation3d), list(pr.scale3d)])
        return True

    def drag(self, x, y, shift=False):
        if self._drag is None:
            return
        if self._drag[0] == "surface":
            self._surface(x, y)
            return
        if self._drag[0] == "point":
            self._drag_point(x, y)
            return
        _kind, axis, (x0, y0), (off0, rot0, sc0) = self._drag
        pr = self.fill().projection
        origin, rot, scale, unit, center = self._frame(pr)
        _o, axes, length = self._gizmo_axes(pr)
        o0 = center + np.asarray(off0) * unit
        if self.mode == "Translate":
            if axis is None:  # the screen plane: the pixel offset turned into world units at the gizmo's depth
                a0, a1 = self._screen(o0), self._screen(o0 + axes[:, 0] * length)
                px = max(math.hypot(a1[0] - a0[0], a1[1] - a0[1]), 1e-6) / length if a0 and a1 else 1.0
                view, _cam = self.ws._camera()
                vp = np.asarray(view.view_proj, float)
                right, up = vp[0, :3] / np.linalg.norm(vp[0, :3]), vp[1, :3] / np.linalg.norm(vp[1, :3])
                delta = (right * (x - x0) - up * (y - y0)) / px
            else:
                a0, a1 = self._screen(o0), self._screen(o0 + axes[:, axis] * length)
                if a0 is None or a1 is None:
                    return
                d = np.array(a1) - np.array(a0)
                t = float(np.dot([x - x0, y - y0], d) / max(np.dot(d, d), 1e-9)) * length
                delta = axes[:, axis] * t
            off = np.asarray(off0) + delta / unit
            if shift:
                off = np.round(off / self.grid_step) * self.grid_step
            self.edit("Move projection", lambda p, off=off: setattr(p, "offset3d", [float(v) for v in off]),
                      merge="manip.move")
        elif self.mode == "Rotate" and axis is not None:
            c = self._screen(o0)
            if c is None:
                return
            ang = math.atan2(y - c[1], x - c[0]) - math.atan2(y0 - c[1], x0 - c[0])
            if shift:
                step = math.radians(self.angle_step)
                ang = round(ang / step) * step
            r0 = euler_matrix(rot0)
            ax = (r0 if self.space == "Local space" else np.eye(3))[:, axis]
            eye = self.ws.view3d.eye()
            if np.dot(ax, eye - o0) < 0:  # turning as seen on screen
                ang = -ang
            rnew = axis_rotation(ax, -ang) @ r0
            self.edit("Rotate projection", lambda p, r=rnew: setattr(p, "rotation3d", matrix_euler(r)),
                      merge="manip.rotate")
        elif self.mode == "Scale":
            c = self._screen(o0)
            if c is None:
                return
            k = math.hypot(x - c[0], y - c[1]) / max(math.hypot(x0 - c[0], y0 - c[1]), 1e-6)
            sc = list(sc0)
            for i in (range(3) if axis is None else (axis,)):
                sc[i] = sc0[i] * k
                if shift:
                    sc[i] = max(round(sc[i] / self.grid_step) * self.grid_step, self.grid_step)
            self.edit("Scale projection", lambda p, sc=sc: setattr(p, "scale3d", [float(v) for v in sc]),
                      merge="manip.scale")

    def release(self):
        self._drag = None

    def _surface(self, x, y):
        """Surface manipulator: the projection's origin on the model under the pointer, its Z along the surface
        normal (projecting onto it)."""
        view, cam = self.ws._camera()
        got = cam.hit(x, y)
        if got is None or got[0][3] <= 0:
            return
        p, n = got[0][:3].astype(float), got[1].astype(float)
        pr = self.fill().projection
        _o, rot, _s, unit, center = self._frame(pr)
        z = n / max(np.linalg.norm(n), 1e-12)
        xa = rot[:, 0] - z * float(np.dot(rot[:, 0], z))
        if np.linalg.norm(xa) < 1e-6:
            xa = np.cross([0.0, 1.0, 0.0], z)
        xa /= np.linalg.norm(xa)
        m = np.stack([xa, np.cross(z, xa), z], 1)
        off = (p - center) / unit

        def do(pp):
            pp.offset3d = [float(v) for v in off]
            pp.rotation3d = matrix_euler(m)
        self.edit("Move projection on surface", do, merge="manip.surface")

    # --- the Warp grid -----------------------------------------------------------------------------------

    def _press_point(self, x, y, pr, shift, ctrl) -> bool:
        from trafilatura3d.paint import warp
        pts = warp.points(pr).reshape(-1, 3)
        sp = [self._screen(self._local_to_world(pr, p)) for p in pts]
        d = [math.hypot(s[0] - x, s[1] - y) if s is not None else 1e9 for s in sp]
        i = int(np.argmin(d))
        if d[i] > PICK_PX * self.ws.view3d.get_pixel_ratio() * self.handle_size:
            if not shift and not ctrl:
                self.selected_points.clear()
            self.refresh()
            return True
        if ctrl:
            self.selected_points.discard(i)
        elif shift:
            self.selected_points.add(i)
        elif i not in self.selected_points:
            self.selected_points = {i}
        self._drag = ("point", i, pts.copy())
        self.refresh()
        return True

    def _drag_point(self, x, y):
        _k, i, start = self._drag
        if i not in self.selected_points:
            return
        view, cam = self.ws._camera()
        got = cam.hit(x, y)
        if got is None or got[0][3] <= 0:
            return
        pr = self.fill().projection
        target = self._world_to_local(pr, got[0][:3].astype(float))
        delta = target - start[i]
        pts = start.copy()
        for j in self.selected_points:
            pts[j] = start[j] + delta
        self.edit("Move warp points", lambda p, pts=pts: setattr(p, "warp_points", pts.tolist()),
                  merge="manip.warp")

    def split(self, rows: bool, cols: bool):
        """Split warp: every row and / or column of cells cut in two, the surface kept."""
        from trafilatura3d.paint import warp
        pr = self.fill().projection
        nr, nc = pr.warp_rows * (2 if rows else 1), pr.warp_cols * (2 if cols else 1)
        if nr > 32 or nc > 32:
            self.ws.log("Warp: at most 32 rows and columns")
            return
        grid = warp.points(pr)
        interp = warp._catmull_rom if pr.warp_tangents else warp._linear
        along = interp(np.swapaxes(grid, 0, 1), np.linspace(0, pr.warp_cols, nc + 1))
        new = np.swapaxes(interp(np.swapaxes(along, 0, 1), np.linspace(0, pr.warp_rows, nr + 1)), 0, 1)
        pts = np.swapaxes(new, 0, 1).reshape(-1, 3)

        def do(p):
            p.warp_rows, p.warp_cols, p.warp_points = nr, nc, pts.tolist()
        self.selected_points.clear()
        self.edit("Split warp", do)

    def reset(self, what: str):
        """Reset: "transform" (the global transformation), "all" (every vertex) or "selected" vertices."""
        def do(p):
            if what == "transform":
                p.offset3d, p.rotation3d, p.scale3d = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
            elif what == "all":
                p.warp_points = []
            else:
                flat = warp_grid(p.warp_rows, p.warp_cols)
                pts = [list(q) for q in (p.warp_points or flat)]
                for i in self.selected_points:
                    pts[i] = flat[i]
                p.warp_points = pts
        self.edit("Reset transformation", do)

    def mirror(self, axis: int):
        """Mirror on X / Y / Z: the projection flipped along its own axis."""
        def do(p):
            sc = list(p.scale3d)
            sc[axis] = -sc[axis]
            p.scale3d = sc
        self.edit("Mirror projection", do)


class ManipulatorBar(QWidget):
    """The contextual toolbar of 3D projections (left side of the viewport's top bar)."""

    def __init__(self, m: Manipulator):
        super().__init__()
        self.m = m
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.show_btn = self._button("Show", "Show/Hide manipulator (Q)", self.toggle_visible, checkable=True)
        self.show_btn.setChecked(True)
        row.addWidget(self.show_btn)
        self.mode_box = QComboBox()
        self.mode_box.addItems(MODES)
        self.mode_box.setToolTip("Translation (W) / Rotation (E) / Scale (R) / Surface (Shift+W) manipulator")
        self.mode_box.currentTextChanged.connect(self.set_mode)
        row.addWidget(self.mode_box)
        self.space_box = QComboBox()
        self.space_box.addItems(("Local space", "World space"))
        self.space_box.setToolTip("Manipulator space (T)")
        self.space_box.currentTextChanged.connect(lambda t: (setattr(m, "space", t), m.refresh()))
        row.addWidget(self.space_box)
        for k, name in enumerate("XYZ"):
            row.addWidget(self._button(f"Mirror {name}", f"Mirror on {name}", lambda _=False, k=k: m.mirror(k)))
        row.addWidget(self._button("Reset", "Reset transformation", lambda: m.reset("transform")))
        self.warp_box = QComboBox()
        self.warp_box.addItems(("Transform warp", "Edit vertices"))
        self.warp_box.setToolTip("Warp edition (Shift+V)")
        self.warp_box.currentTextChanged.connect(lambda t: (setattr(m, "warp_edit", t == "Edit vertices"), m.refresh()))
        row.addWidget(self.warp_box)
        self.warp_split = QComboBox()
        self.warp_split.addItems(("Split…", "Split warp crosswise", "Split warp horizontally", "Split warp vertically"))
        self.warp_split.activated.connect(self._split)
        row.addWidget(self.warp_split)
        self.warp_reset = QComboBox()
        self.warp_reset.addItems(("Reset…", "Restore the global transformation", "Reset all vertices",
                                  "Reset selected vertices"))
        self.warp_reset.activated.connect(self._reset)
        row.addWidget(self.warp_reset)
        self.label = QLabel()
        row.addWidget(self.label)

    def _button(self, text, tip, fn, checkable=False):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        (b.toggled if checkable else b.clicked).connect(fn)
        return b

    def toggle_visible(self, on=None):
        """The button's state, or (Q) the other one."""
        self.m.visible = (not self.m.visible) if on is None else bool(on)
        self.show_btn.blockSignals(True)
        self.show_btn.setChecked(self.m.visible)
        self.show_btn.blockSignals(False)
        self.m.refresh()

    def set_mode(self, mode):
        self.m.mode = mode
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentText(mode)
        self.mode_box.blockSignals(False)
        self.m.refresh()

    def _split(self, i):
        if i > 0:
            self.m.split(rows=i in (1, 2), cols=i in (1, 3))
        self.warp_split.setCurrentIndex(0)

    def _reset(self, i):
        if i > 0:
            self.m.reset(("transform", "all", "selected")[i - 1])
        self.warp_reset.setCurrentIndex(0)

    def sync(self):
        obj = self.m.fill()
        warp = obj is not None and obj.projection.mode == "Warp projection"
        for w in (self.warp_box, self.warp_split, self.warp_reset):
            w.setVisible(warp)
        surface_ok = obj is not None and obj.projection.mode in ("Planar projection", "Warp projection")
        if not surface_ok and self.m.mode == "Surface":
            self.set_mode("Translate")
