"""The Path tool in the workspace (Path tool, Filled path, Ribbon path pages): editing paths in the 3D viewport,
their overlay, the Paths panel and the path's properties.

In the viewport: click the surface to start a path and keep clicking to add points; Enter or Escape ends the
editing; click a path to select it, click on its line to insert a point, drag a point to move it on the surface;
Delete removes the selected point; double click a point toggles corner / smooth. Every change is a History step
(a drag is one) and the path paints again.

The Paths panel (beside the viewport while the Path tool is on): the selected painting's paths with their
visibility (applied to texturing or not), rename (double click), duplicate, delete, reverse, open / close, copy and
paste of properties or positions.

Paths belong to the selected painting: a paint layer's content, a layer's mask (any layer), a paint effect of the
content or of the mask stack. On a mask a path paints the tool's grayscale.

The selected point shows its tangent handles (red): dragging one moves it on the point's tangent plane, the other
opposite (Alt breaks the tangents, Ctrl scales both); its Pressure, Corner and Custom tangents are in the
Properties. Display settings: interface, normals, tangents, direction, handle size, colors. Symmetry per path.

Not offered: Path width (one pixel lines), excluded geometry for paths, path presets.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QToolButton, QVBoxLayout, QWidget)

from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.editor.paint.widgets import PSlider, header
from trafilatura3d.paint import paths as pathcore
from trafilatura3d.paint.brush import BrushSettings
from trafilatura3d.paint.model import PATH_MODES, ChannelSource, PaintPath, new_uid

PICK_PX = 8.0  # screen distance to grab a point or the line (ours)
HANDLE_PX = 5.0  # half size of a point's handle on screen (ours)


class PathTool:
    def __init__(self, ws):
        self.ws = ws
        self.selected: str | None = None  # path uid
        self.vertex: int | None = None  # selected point
        self.editing = False  # clicks add points to the selected path
        self._drag = None  # (path uid, point index) while dragging a point
        self.template = PaintPath("", "", brush=asdict(BrushSettings()))  # the properties of the next path
        self.clipboard = None  # ("properties" | "positions", data)
        # Display settings (Path page): the overlay shown, handle size (pixels), colors, the points' normals (the
        # projection direction), every point's tangents (else the selected one's), the direction arrow.
        self.show_interface = True
        self.handle_px = HANDLE_PX
        self.path_color = (1.0, 0.55, 0.1, 1.0)
        self.unselected_color = (0.75, 0.75, 0.75, 0.9)
        self.show_normals = False
        self.show_tangents = False
        self.show_direction = True

    # --- model -------------------------------------------------------------------------------------------

    def owner(self):
        """(what holds the paths, its PaintStore uid, on a mask) of the selected painting: a paint layer's content,
        a layer's mask, a paint effect (of the content or of the mask); None when the selection is none of these.
        Resolved again at every call: edits restore the project from snapshots."""
        doc = self.ws.doc
        if doc is None or doc.ts is None:
            return None
        layer, fx = doc.layer(), doc.effect()
        if fx is not None:
            return (fx, fx.uid, doc.ts.owner_of(fx.uid)[2]) if fx.kind == "paint" else None
        if layer is None:
            return None
        if doc.mask_selected:
            return (layer.mask, layer.uid, True) if layer.mask is not None else None
        return (layer, layer.uid, False) if layer.kind == "paint" else None

    def paths(self) -> list:
        o = self.owner()
        return o[0].paths if o is not None else []

    def path(self) -> PaintPath | None:
        return next((p for p in self.paths() if p.uid == self.selected), None)

    def target(self) -> PaintPath:
        """What the Properties edit: the selected path, or the next path's properties."""
        return self.path() or self.template

    def _find(self, uid):
        """The path `uid` of the current layer, looked up again (edits restore the project from snapshots)."""
        return next((p for p in self.paths() if p.uid == uid), None)

    def edit(self, label, fn, merge=None, structure=False):
        self.ws.doc.edit(label, fn, merge=merge, structure=structure)
        self.ws.refresh_materials()
        self.refresh_overlay()

    def set_attr(self, attr, value, final=True):
        p = self.path()
        if p is None:
            setattr(self.template, attr, value)
            return
        uid = p.uid

        def do():
            setattr(self._find(uid), attr, value)
        self.edit("Path property", do, merge=None if final else f"path.{uid}.{attr}")

    # --- viewport ------------------------------------------------------------------------------------------

    def _hit(self, x, y):
        """(point [x y z nx ny nz pressure corner], material) of the surface under a pixel, or None."""
        view, cam = self.ws._camera()
        got = cam.hit(x, y)
        if got is None or got[0][3] <= 0:
            return None
        p, n = got[0][:3].astype(float), got[1].astype(float)
        return [*p, *n, 1.0, 0.0], int(round(got[0][3])) - 1

    def _screen_points(self, pts):
        view, _cam = self.ws._camera()
        out = []
        for p in pts:
            s = self.ws._screen(view, p[:3])
            out.append(s if s is not None else (-1e9, -1e9))
        return np.asarray(out, float)

    def _pick_vertex(self, x, y):
        """(path uid, point index) under the pointer, the selected path first."""
        for p in sorted(self.paths(), key=lambda p: p.uid != self.selected):
            if not p.points:
                continue
            sp = self._screen_points(p.points)
            d = np.hypot(sp[:, 0] - x, sp[:, 1] - y)
            i = int(np.argmin(d))
            if d[i] <= PICK_PX * self.ws.view3d.get_pixel_ratio():
                return p.uid, i
        return None

    def _pick_line(self, x, y):
        """(path uid, index to insert at) when the pointer is on a path's line."""
        diag = self.ws.doc.session.bbox_diagonal()
        for p in sorted(self.paths(), key=lambda p: p.uid != self.selected):
            if len(p.points) < 2:
                continue
            k = len(p.points) if p.closed else len(p.points) - 1
            for i in range(k):  # each span alone, so the insert index is the span's
                span = [p.points[i], p.points[(i + 1) % len(p.points)]]
                c = pathcore.curve(span, False, diag * 4e-3)
                sp = self._screen_points(c.pos)
                if np.min(np.hypot(sp[:, 0] - x, sp[:, 1] - y)) <= PICK_PX * self.ws.view3d.get_pixel_ratio():
                    return p.uid, i + 1
        return None

    # --- tangents ----------------------------------------------------------------------------------------

    def _handle_points(self, p: PaintPath, i: int):
        """(in handle, out handle) world positions of point i."""
        h_in, h_out = pathcore.handles(p.points, p.closed)
        q = np.asarray(p.points[i][:3], float)
        return q + h_in[i], q + h_out[i]

    def _pick_handle(self, x, y):
        """0 (in) or 1 (out) when the pointer is on the selected point's handle."""
        p = self.path()
        if p is None or self.vertex is None or self.vertex >= len(p.points):
            return None
        sp = self._screen_points(list(self._handle_points(p, self.vertex)))
        d = np.hypot(sp[:, 0] - x, sp[:, 1] - y)
        k = int(np.argmin(d))
        return k if d[k] <= PICK_PX * self.ws.view3d.get_pixel_ratio() else None

    def _ray_plane(self, x, y, origin, normal):
        """The point of the plane (origin, normal) under the pixel, or None."""
        view, _cam = self.ws._camera()
        inv = np.linalg.inv(np.asarray(view.view_proj, float))
        ndc = (x / view.width * 2 - 1, 1 - y / view.height * 2)
        a = inv @ np.array([*ndc, 0.0, 1.0])
        b = inv @ np.array([*ndc, 1.0, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        den = float(np.dot(d, normal))
        if abs(den) < 1e-12:
            return None
        return a + d * float(np.dot(np.asarray(origin) - a, normal)) / den

    def _drag_handle(self, x, y):
        """A handle dragged on the point's tangent plane (the page: tangents are constrained to the plane of the
        point's normal): the other one stays opposite (smooth), Alt moves it alone (breaks the tangents), Ctrl
        scales both keeping their directions."""
        from PySide6.QtWidgets import QApplication
        mods = QApplication.keyboardModifiers()
        uid, i, which = self._drag[1:]
        p = self._find(uid)
        if p is None:
            return
        pt = p.points[i]
        q, n = np.asarray(pt[:3], float), np.asarray(pt[3:6], float)
        n /= max(np.linalg.norm(n), 1e-12)
        w = self._ray_plane(x, y, q, n)
        if w is None:
            return
        h = w - q
        h_in, h_out = pathcore.handles(p.points, p.closed)
        cur = [h_in[i], h_out[i]]
        other = cur[1 - which]
        if mods & Qt.ControlModifier:  # scale both
            ln = float(np.linalg.norm(h))
            new = [c / max(np.linalg.norm(c), 1e-12) * ln for c in cur]
        elif mods & Qt.AltModifier:  # break: this one alone
            new = list(cur)
            new[which] = h
        else:  # smooth: the other one opposite, its length kept
            new = [None, None]
            new[which] = h
            new[1 - which] = -h / max(np.linalg.norm(h), 1e-12) * float(np.linalg.norm(other))
        point = list(pt[:8]) + [*new[0], *new[1], 1.0]

        def do():
            self._find(uid).points[i] = point
        self.edit("Move path tangent", do, merge=f"path.{uid}.tangent.{i}")

    def set_custom_tangents(self, on: bool):
        """Custom tangents of the selected point: on keeps its current handles editable, off goes back to the
        automatic ones."""
        p = self.path()
        if p is None or self.vertex is None:
            return
        uid, i = p.uid, self.vertex
        h_in, h_out = pathcore.handles(p.points, p.closed)
        point = list(p.points[i][:8]) + ([*h_in[i], *h_out[i], 1.0] if on else [])

        def do():
            self._find(uid).points[i] = point
        self.edit("Custom tangents", do)

    def set_pressure(self, value: float, final: bool = True):
        """The selected point's pressure (Pressure per vertex)."""
        p = self.path()
        if p is None or self.vertex is None:
            return
        uid, i = p.uid, self.vertex

        def do():
            self._find(uid).points[i][6] = float(value)
        self.edit("Path point pressure", do, merge=None if final else f"path.{uid}.pressure.{i}")

    def press(self, x, y) -> bool:
        if self.owner() is None:
            self.ws.log("The Path tool works on a paint layer, a mask or a paint effect")
            return False
        k = self._pick_handle(x, y)
        if k is not None:  # a tangent handle of the selected point
            self._drag = ("handle", self.selected, self.vertex, k)
            return True
        v = self._pick_vertex(x, y)
        if v is not None:
            self.selected, self.vertex = v
            self._drag = v
            self.ws.properties.rebuild()
            self.paths_panel_refresh()
            self.refresh_overlay()
            return True
        line = self._pick_line(x, y)
        if line is not None and not self.editing:
            uid, at = line
            hit = self._hit(x, y)
            if hit is not None:
                def do():
                    self._find(uid).points.insert(at, hit[0])
                self.edit("Insert path point", do)
                self.selected, self.vertex = uid, at
                self._drag = (uid, at)
            return True
        hit = self._hit(x, y)
        if hit is None or self.ws.doc.session.materials[hit[1]] != self.ws.doc.ts.material:
            return False
        point = hit[0]
        if self.editing and self.path() is not None:  # one more point at the end
            uid = self.selected

            def do():
                self._find(uid).points.append(point)
            self.edit("Add path point", do)
            self.vertex = len(self.path().points) - 1
            self._drag = (uid, self.vertex)
            return True
        p = self._new_path(point)  # a new path

        def do():
            self.owner()[0].paths.append(p)
        self.edit("New path", do, structure=True)
        self.selected, self.vertex, self.editing = p.uid, 0, True
        self._drag = (p.uid, 0)
        self.paths_panel_refresh()
        self.ws.properties.rebuild()
        return True

    def _new_path(self, point) -> PaintPath:
        st, t = self.ws.tools, self.template
        ts = self.ws.doc.ts
        p = copy.deepcopy(t)
        p.uid = new_uid()
        p.name = f"Path {len(self.paths()) + 1}"
        p.points = [point]
        p.brush = asdict(st.brushes["path"])
        p.alpha = st.alphas["path"]
        p.values = {c: list(st.value(c)) for c in ts.channel_ids()}
        p.disabled = sorted(st.disabled)
        p.mask_value = st.mask_value  # on a mask: the tool's grayscale
        return p

    def drag(self, x, y):
        if self._drag is None:
            return
        if self._drag[0] == "handle":
            self._drag_handle(x, y)
            return
        uid, i = self._drag
        hit = self._hit(x, y)
        p = self._find(uid)
        if hit is None or p is None or i >= len(p.points):
            return
        point = hit[0] + list(p.points[i][8:])  # its custom handles move with it
        point[6:8] = p.points[i][6:8]  # keep its pressure and corner
        if len(point) > pathcore.CUSTOM:  # the handles on the new tangent plane
            n = np.asarray(point[3:6], float)
            n /= max(np.linalg.norm(n), 1e-12)
            for base in (8, 11):
                h = np.asarray(point[base:base + 3], float)
                point[base:base + 3] = list(h - n * float(np.dot(h, n)))

        def do():
            self._find(uid).points[i] = point
        self.edit("Move path point", do, merge=f"path.{uid}.move.{i}")

    def release(self):
        self._drag = None

    def double_click(self, x, y):
        v = self._pick_vertex(x, y)
        if v is not None:
            self.toggle_corner(*v)

    def toggle_corner(self, uid, i):
        def do():  # a corner breaks the tangents (zero), smooth aligns them (automatic again)
            pts = self._find(uid).points
            corner = 0.0 if pts[i][7] > 0.5 else 1.0
            pts[i] = list(pts[i][:7]) + [corner]
        self.edit("Toggle corner / smooth", do)

    def key(self, key) -> bool:
        k = str(key)
        if k in ("Escape", "Enter", "Return"):
            self.editing = False
            self.refresh_overlay()
            return True
        if k in ("Delete", "Backspace") and self.path() is not None and self.vertex is not None:
            self.delete_vertex()
            return True
        return False

    def delete_vertex(self):
        uid, i = self.selected, self.vertex

        def do():
            del self._find(uid).points[i]
        self.edit("Delete path point", do)
        self.vertex = None

    # --- overlay -----------------------------------------------------------------------------------------

    def refresh_overlay(self, pos=None):
        """The overlay again; `pos`: the pointer, for the next segment while drawing a path."""
        r = self.ws.view3d.renderer
        r.extra_lines = self.overlay(pos) if self.ws.tools.tool == "path" else r.extra_lines
        self.ws.view3d.refresh()

    def _handle_radius(self, p) -> float:
        """World radius of a point's handle so that it is HANDLE_PX pixels on screen."""
        v = self.ws.view3d
        view = v.brush_view()
        dist = float(np.linalg.norm(np.asarray(p) - view.eye))
        return dist * np.tan(view.fov / 2) * 2 * self.handle_px * v.get_pixel_ratio() / max(view.height, 1)

    def overlay(self, pos=None):
        """The paths of the layer as lines over the mesh: the curve (selected: orange, others: gray, hidden: dark),
        a square handle per point of a fixed size on screen (the selected one yellow), and while drawing the segment
        from the last point to the pointer."""
        if self.ws.doc is None:
            return None
        diag = self.ws.doc.session.bbox_diagonal()
        segs = []
        p = self.path()
        if self.editing and p is not None and p.points and pos is not None:  # the next segment, dashed (ours)
            hit = self._hit(*pos)
            if hit is not None:
                a, b = np.asarray(p.points[-1][:3]), np.asarray(hit[0][:3])
                n = np.asarray(hit[0][3:6])
                t = np.linspace(0, 1, 21)
                pts = a + (b - a) * t[:, None] + n * diag * 2e-3
                for i in range(0, 20, 2):
                    s = np.empty((2, 7), np.float32)
                    s[0, :3], s[1, :3], s[:, 3:] = pts[i], pts[i + 1], (1.0, 0.55, 0.1, 0.8)
                    segs.append(s)
        if not self.show_interface:
            return np.concatenate(segs) if segs else None

        def line(a, b, col):
            s = np.empty((2, 7), np.float32)
            s[0, :3], s[1, :3], s[:, 3:] = a, b, col
            segs.append(s)

        def square(q, n, col):  # a handle: a square on the surface filled with lines, a fixed size on screen
            r = self._handle_radius(q)
            u = np.cross(n, [0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(n, u)
            quad = np.empty((2 * 9, 7), np.float32)
            for k, f in enumerate(np.linspace(-1, 1, 9)):
                quad[2 * k, :3], quad[2 * k + 1, :3] = q + (u * -1 + w * f) * r, q + (u + w * f) * r
            quad[:, 3:] = col
            segs.append(quad)
        tangent_col = (1.0, 0.2, 0.2, 1.0)  # the page: tangents in red
        for p in self.paths():
            color = self.path_color if p.uid == self.selected else (
                self.unselected_color if p.visible else (0.35, 0.35, 0.35, 0.8))
            shapes = [p.points] + [pts for pts, _flip in pathcore.symmetric(p)]
            for k_shape, pts in enumerate(shapes):
                col = color if k_shape == 0 else (*color[:3], color[3] * 0.5)  # the mirror image dimmer
                c = pathcore.curve(pts, p.closed, diag * 2e-3, p.reversed) if len(pts) >= 2 else None
                if c is None:
                    continue
                pos = c.pos + c.nrm * diag * 2e-3
                seg = np.empty(((len(pos) - 1) * 2, 7), np.float32)
                seg[0::2, :3], seg[1::2, :3] = pos[:-1], pos[1:]
                seg[:, 3:] = col
                segs.append(seg)
                if self.show_direction and k_shape == 0 and len(pos) > 2:  # an arrow head at the end
                    end, back = pos[-1], pos[-1] - c.tangent[-1] * self._handle_radius(pos[-1]) * 4
                    side = np.cross(c.nrm[-1], c.tangent[-1]) * self._handle_radius(pos[-1]) * 2
                    line(end, back + side, col)
                    line(end, back - side, col)
            h_in, h_out = pathcore.handles(p.points, p.closed) if p.points else (None, None)
            for i, pt in enumerate(p.points):
                n = np.asarray(pt[3:6], float)
                n /= max(np.linalg.norm(n), 1e-12)
                q = np.asarray(pt[:3]) + n * diag * 2e-3
                selected = p.uid == self.selected and i == self.vertex
                square(q, n, (1.0, 0.95, 0.2, 1.0) if selected else color)
                if self.show_normals:  # the projection direction
                    line(q, q + n * diag * 0.03, (0.35, 0.6, 1.0, 1.0))
                if selected or (self.show_tangents and p.uid == self.selected):
                    for h in (h_in[i], h_out[i]):
                        if np.linalg.norm(h) > 1e-9:
                            line(q, q + h, tangent_col)
                            square(q + h, n, tangent_col)
        return np.concatenate(segs) if segs else None

    # --- the Paths panel ---------------------------------------------------------------------------------

    def paths_panel_refresh(self):
        panel = getattr(self.ws, "paths_panel", None)
        if panel is not None:
            panel.rebuild()

    def duplicate(self, uid):
        p = self._find(uid)
        if p is None:
            return
        q = copy.deepcopy(p)
        q.uid, q.name = new_uid(), f"{p.name} copy"

        def do():
            self.owner()[0].paths.append(q)
        self.edit("Duplicate path", do, structure=True)
        self.paths_panel_refresh()

    def delete(self, uid):
        def do():
            holder = self.owner()[0]
            holder.paths[:] = [p for p in holder.paths if p.uid != uid]
        self.edit("Delete path", do, structure=True)
        if self.selected == uid:
            self.selected, self.vertex, self.editing = None, None, False
        self.paths_panel_refresh()
        self.ws.properties.rebuild()

    def toggle(self, uid, attr):
        def do():
            p = self._find(uid)
            setattr(p, attr, not getattr(p, attr))
        self.edit({"visible": "Show / hide path", "reversed": "Reverse path direction",
                   "closed": "Open / close path"}[attr], do)
        self.paths_panel_refresh()

    def rename(self, uid, name):
        def do():
            self._find(uid).name = name
        self.edit("Rename path", do)
        self.paths_panel_refresh()

    PROPERTY_FIELDS = ("mode", "brush", "alpha", "values", "sources", "disabled", "mask_value", "projection_depth",
                       "up_axis", "opacity", "ribbon")

    def copy_properties(self, uid):
        p = self._find(uid)
        self.clipboard = ("properties", {f: copy.deepcopy(getattr(p, f)) for f in self.PROPERTY_FIELDS})

    def copy_positions(self, uid):
        p = self._find(uid)
        self.clipboard = ("positions", (copy.deepcopy(p.points), p.closed))

    def paste(self, uid):
        """Properties are copied only to a path of the same painting mode; positions to any path."""
        if self.clipboard is None:
            return
        kind, data = self.clipboard
        p = self._find(uid)
        if kind == "properties" and data["mode"] != p.mode:
            self.ws.log("Path properties paste only between paths of the same mode")
            return

        def do():
            q = self._find(uid)
            if kind == "properties":
                for f, v in data.items():
                    setattr(q, f, copy.deepcopy(v))
            else:
                q.points, q.closed = copy.deepcopy(data[0]), data[1]
        self.edit("Paste path " + kind, do)


class PathsPanel(QWidget):
    """The Paths panel at the top of the 3D viewport (shown with the Path tool): the paths in a row, their
    actions beside."""

    def __init__(self, tool: PathTool):
        super().__init__()
        self.tool = tool
        self.setFixedHeight(40)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        title = QLabel("<b>PATHS</b>")
        outer.addWidget(title)
        self.list = QListWidget()
        self.list.setFlow(QListWidget.LeftToRight)
        self.list.setWrapping(False)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.itemClicked.connect(self._clicked)
        self.list.itemDoubleClicked.connect(self._rename)
        outer.addWidget(self.list, 1)
        lay = QHBoxLayout()
        outer.addLayout(lay)
        row = QHBoxLayout()
        for ic, tip, fn in (("eye", "Show / hide (applied to texturing)", lambda u: tool.toggle(u, "visible")),
                            ("random", "Reverse path direction", lambda u: tool.toggle(u, "reversed")),
                            ("lines_menu", "Open / close path", lambda u: tool.toggle(u, "closed")),
                            ("add", "Duplicate", tool.duplicate), ("delete", "Delete", tool.delete)):
            b = QToolButton()
            b.setIcon(picon(ic))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(lambda _=False, fn=fn: self._on_selected(fn))
            row.addWidget(b)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        for text, fn in (("Copy props", tool.copy_properties), ("Copy pos", tool.copy_positions), ("Paste", tool.paste)):
            b = QToolButton()
            b.setText(text)
            b.clicked.connect(lambda _=False, fn=fn: self._on_selected(fn))
            row2.addWidget(b)
        lay.addLayout(row2)

    def _on_selected(self, fn):
        if self.tool.selected is not None:
            fn(self.tool.selected)

    def rebuild(self):
        self.list.clear()
        for p in self.tool.paths():
            it = QListWidgetItem(picon("eye" if p.visible else "eye_off"), f"{p.name}  ({p.mode})")
            it.setData(Qt.UserRole, p.uid)
            self.list.addItem(it)
            if p.uid == self.tool.selected:
                it.setSelected(True)

    def _clicked(self, it):
        self.tool.selected = it.data(Qt.UserRole)
        self.tool.vertex = None
        self.tool.ws.properties.rebuild()
        self.tool.refresh_overlay()

    def _rename(self, it):
        uid = it.data(Qt.UserRole)
        p = self.tool._find(uid)
        name, ok = QInputDialog.getText(self, "Rename path", "Name", text=p.name)
        if ok and name:
            self.tool.rename(uid, name)


def build_properties(panel, tool: PathTool):
    """The Path tool's Properties: Mode, then the mode's parameters (the selected path's, or the next path's)."""
    p = tool.target()
    lay = panel.lay
    if tool.owner() is None:  # why clicking does nothing
        warn = QLabel("Paths are drawn on a painting: select a paint layer, a mask or a paint effect.")
        warn.setStyleSheet("color: #e0a040;")
        warn.setWordWrap(True)
        lay.addWidget(warn)
    lay.addWidget(header("Path"))
    form = QFormLayout()
    mode = QComboBox()
    mode.addItems(PATH_MODES)
    mode.setCurrentText(p.mode)
    mode.currentTextChanged.connect(lambda t: (tool.set_attr("mode", t), panel.rebuild()))
    form.addRow("Mode", mode)
    lay.addLayout(form)
    if tool.path() is None:
        hint = QLabel("Click the mesh to start a path, keep clicking to add points, Enter to finish.")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)
    else:
        path = tool.path()
        sym = QCheckBox("Symmetry (the current symmetry settings)")
        sym.setChecked(bool(path.symmetry))
        st = tool.ws.tools
        spec = lambda: {"type": st.sym_type, "axis": st.sym_axis,  # noqa: E731
                        "position": st.sym_position if st.sym_type == "Mirror" else list(st.radial_position),
                        "copies": st.radial_copies, "flip": st.radial_flip}
        sym.toggled.connect(lambda on: tool.set_attr("symmetry", spec() if on else {}))
        lay.addWidget(sym)
        if tool.vertex is not None and tool.vertex < len(path.points):  # the selected point
            pt = path.points[tool.vertex]
            lay.addWidget(header("Point"))
            pr = PSlider("Pressure", pt[6], 0.0, 1.0)
            pr.edited.connect(lambda v, final: tool.set_pressure(v, final))
            lay.addWidget(pr)
            corner = QCheckBox("Corner (double click toggles corner / smooth)")
            corner.setChecked(pt[7] > 0.5)
            corner.toggled.connect(lambda _on: tool.toggle_corner(path.uid, tool.vertex))
            lay.addWidget(corner)
            custom = QCheckBox("Custom tangents (drag the red handles; Alt breaks, Ctrl scales)")
            custom.setChecked(len(pt) > pathcore.CUSTOM and pt[pathcore.CUSTOM] > 0.5)
            custom.toggled.connect(tool.set_custom_tangents)
            lay.addWidget(custom)
            row = QHBoxLayout()
            for text, fn in (("Delete vertex", tool.delete_vertex),):
                b = QToolButton()
                b.setText(text)
                b.clicked.connect(fn)
                row.addWidget(b)
            lay.addLayout(row)

    def slider(label, attr, lo, hi, integer=False, owner=None):
        """A slider on the path's brush (owner "brush"), ribbon (owner "ribbon") or own attribute."""
        if owner is None:
            value = getattr(p, attr)
        else:
            value = getattr(p, owner).get(attr, getattr(pathcore.RibbonSettings() if owner == "ribbon"
                                                        else BrushSettings(), attr))
        s = PSlider(label, value, lo, hi, integer)

        def edited(v, final, attr=attr, owner=owner):
            if owner is None:
                tool.set_attr(attr, v, final)
            else:
                tool.set_attr(owner, {**getattr(tool.target(), owner), attr: int(v) if integer else v}, final)
        s.edited.connect(edited)
        lay.addWidget(s)

    def combo(label, attr, items, owner):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(getattr(p, owner).get(attr, getattr(pathcore.RibbonSettings(), attr)))
        c.currentTextChanged.connect(lambda t: tool.set_attr(owner, {**getattr(tool.target(), owner), attr: t}))
        f = QFormLayout()
        f.addRow(label, c)
        lay.addLayout(f)

    def check(label, attr, owner):
        box = QCheckBox(label)
        box.setChecked(bool(getattr(p, owner).get(attr, getattr(pathcore.RibbonSettings(), attr))))
        box.toggled.connect(lambda on: tool.set_attr(owner, {**getattr(tool.target(), owner), attr: on}))
        lay.addWidget(box)

    if p.mode in ("Paint along path", "Erase along path", "Smudge along path"):
        lay.addWidget(header("Brush"))
        if tool.path() is None:  # the next path takes the Path tool's brush (and its contextual toolbar)
            st = tool.ws.tools.brushes["path"]
            for label, attr, lo, hi in (("Size", "size", 0.0, 100.0), ("Flow", "flow", 0.0, 100.0),
                                        ("Stroke opacity", "stroke_opacity", 0.0, 100.0),
                                        ("Spacing", "spacing", 1.0, 200.0), ("Angle", "angle", 0.0, 360.0)):
                s = PSlider(label, getattr(st, attr), lo, hi)
                s.edited.connect(lambda v, _f, attr=attr: setattr(st, attr, v))
                lay.addWidget(s)
            follow = QCheckBox("Follow Path")
            follow.setChecked(st.follow_path)
            follow.toggled.connect(lambda on: setattr(st, "follow_path", on))
            lay.addWidget(follow)
        else:
            for label, attr, lo, hi in (("Size", "size", 0.0, 100.0), ("Flow", "flow", 0.0, 100.0),
                                        ("Stroke opacity", "stroke_opacity", 0.0, 100.0),
                                        ("Spacing", "spacing", 1.0, 200.0), ("Angle", "angle", 0.0, 360.0)):
                slider(label, attr, lo, hi, owner="brush")
            follow = QCheckBox("Follow Path")
            follow.setChecked(bool(p.brush.get("follow_path", False)))
            follow.toggled.connect(lambda on: tool.set_attr("brush", {**tool.target().brush, "follow_path": on}))
            lay.addWidget(follow)
        slider("Projection depth", "projection_depth", 0.1, 10.0)

        def pick_alpha(name):
            if tool.path() is None:  # the next path's
                tool.ws.tools.alphas["path"] = name
            else:
                tool.set_attr("alpha", name)
            panel.rebuild()
        panel.alpha_picker(p.alpha or tool.ws.tools.alphas["path"], pick_alpha)
        f = QFormLayout()
        up = QComboBox()
        up.addItems(pathcore.UP_AXES)
        up.setCurrentText(p.up_axis)
        up.currentTextChanged.connect(lambda t: tool.set_attr("up_axis", t))
        f.addRow("Up axis", up)
        lay.addLayout(f)
    elif p.mode == "Filled path":
        slider("Opacity", "opacity", 0.0, 1.0)
        slider("Projection depth", "projection_depth", 0.1, 10.0)
    else:  # Ribbon path
        lay.addWidget(header("Ribbon"))
        slider("Stroke width", "stroke_width", 0.1, 50.0, owner="ribbon")
        slider("Stroke opacity", "stroke_opacity", 0.0, 100.0, owner="ribbon")
        combo("Image orientation", "image_orientation", pathcore.RIBBON_ORIENTATIONS, "ribbon")
        check("Flip image", "flip_image", "ribbon")
        combo("Corner", "corner", pathcore.CORNERS, "ribbon")
        check("Omit ends when closed", "omit_ends_when_closed", "ribbon")
        lay.addWidget(header("Stretching & Tiling"))
        combo("Stretching & Tiling mode", "stretching_mode", pathcore.STRETCH_MODES, "ribbon")
        r = pathcore.RibbonSettings.of(p.ribbon)
        if r.stretching_mode == "Stretch along path":
            check("Stretch between offsets only", "stretch_between_offsets", "ribbon")
            slider("Start offset", "start_offset", 0.0, 0.5, owner="ribbon")
            slider("End offset", "end_offset", 0.0, 0.5, owner="ribbon")
            combo("Tiling mode", "tiling_mode", pathcore.TILING_MODES, "ribbon")
            slider("Tiling amount", "tiling_amount", 1, 64, integer=True, owner="ribbon")
        else:
            combo("Ratio", "ratio", pathcore.RATIOS, "ribbon")
            check("Remove clipped tiles", "remove_clipped_tiles", "ribbon")
            combo("Alignment", "alignment", pathcore.ALIGNMENTS, "ribbon")
        check("Mirror every 2nd tile", "mirror_every_2nd_tile", "ribbon")
        slider("Aspect ratio factor", "aspect_ratio_factor", 0.1, 10.0, owner="ribbon")
    if p.mode not in ("Erase along path", "Smudge along path"):
        _material(panel, tool, p)
    _display(panel, tool)


def _display(panel, tool):
    """Display settings of the paths (the page's list; Path width is not offered: lines are one pixel)."""
    from trafilatura3d.editor.paint.widgets import Swatch
    lay = panel.lay
    lay.addWidget(header("Display settings"))
    for label, attr in (("Show viewport interface", "show_interface"), ("Normals", "show_normals"),
                        ("Tangents", "show_tangents"), ("Path direction", "show_direction")):
        box = QCheckBox(label)
        box.setChecked(getattr(tool, attr))
        box.toggled.connect(lambda on, attr=attr: (setattr(tool, attr, on), tool.refresh_overlay()))
        lay.addWidget(box)
    hs = PSlider("Handle size", tool.handle_px, 2.0, 16.0)
    hs.edited.connect(lambda v, _f: (setattr(tool, "handle_px", v), tool.refresh_overlay()))
    lay.addWidget(hs)
    form = QFormLayout()
    for label, attr in (("Path color", "path_color"), ("Unselected path color", "unselected_color")):
        sw = Swatch(getattr(tool, attr), label)
        sw.edited.connect(lambda rgb, _f, attr=attr: (setattr(tool, attr, (*rgb, 1.0)), tool.refresh_overlay()))
        form.addRow(label, sw)
    lay.addLayout(form)


def _material(panel, tool, p):
    """The path's channel values (Ribbon: a resource per channel too)."""
    from trafilatura3d.paint.channels import FORMATS
    ts = tool.ws.doc.ts
    owner = tool.owner()
    if owner is not None and owner[2]:  # on a mask: one grayscale
        panel.lay.addWidget(header("Grayscale"))
        g = PSlider("Grayscale", p.mask_value, 0.0, 1.0)
        g.edited.connect(lambda v, final: tool.set_attr("mask_value", v, final))
        panel.lay.addWidget(g)
        return
    panel.lay.addWidget(header("Material"))
    for c in ts.channel_ids():
        if c in p.disabled:
            continue
        value = p.values.get(c) or list(tool.ws.tools.value(c))

        def set_value(v, final, c=c):
            tool.set_attr("values", {**tool.target().values, c: list(v)}, final)
        if p.mode == "Ribbon path":
            src = ChannelSource(**p.sources[c]) if c in p.sources else ChannelSource(value)

            def pick(name, c=c):
                s = ChannelSource(**tool.target().sources.get(c, {"color": value}))
                if name.startswith("node:"):
                    s.procedural, s.image, s.values = name, "", {}
                else:
                    s.procedural, s.image = "", name
                tool.set_attr("sources", {**tool.target().sources, c: asdict(s)})
                panel.rebuild()
            panel._value_row(c, FORMATS[ts.format_of(c)][0], value, set_value, source=src, pick=pick)
        else:
            panel._value_row(c, FORMATS[ts.format_of(c)][0], value, set_value)
