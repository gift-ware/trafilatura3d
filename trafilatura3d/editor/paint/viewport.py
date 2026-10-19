"""The Paint viewports (Viewport, 3D view and 2D view pages): the 3D view draws the mesh with its Texture
Sets through the editor's renderer and is where the brush paints; the 2D view shows the selected Texture Set
in texture space.

3D navigation (Viewport and Shortcuts pages): Alt + left drag rotates (Alt + Shift snaps every 90°; macOS
Shift + Cmd), Alt + middle drag or Ctrl + Alt + left drag translates (macOS Option + Cmd + left), Alt + right
drag zooms (macOS: middle drag), the wheel zooms, Shift + right drag rotates the environment, Ctrl + right drag
changes the tool size, Ctrl + left drag its flow (horizontal) and angle (vertical). Left drag paints with the
current tool. Display mode: Material, a solo channel or a solo mesh map.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import wgpu
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QInputDevice

from trafilatura3d.editor.canvas import Canvas
from trafilatura3d.editor.paint import shortcuts
from trafilatura3d.editor.render3d import (MAX_FPS, Look, Renderer, look_at, orthographic, perspective,
                                           present_method)
from trafilatura3d.editor.view2d import Viewport
from trafilatura3d.paint.model import tile_uv, udim_at
from trafilatura3d.paint.brush import View

MAC = sys.platform == "darwin"
VIEW2D_BINDINGS = ("Use tool / paint", "Camera translate", "Camera translate (alternative)", "Camera zoom")


def _axis_rotation(axis, angle: float) -> np.ndarray:
    """The rotation by `angle` radians around the unit `axis` (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class PenPressure:
    """The pen's pressure for the viewports (Pen pressure page). Qt sends a tablet's pen first as tablet events,
    which carry the pressure, and makes the usual mouse events of them when the widget leaves them unhandled:
    the pressure is kept from the tablet event and the mouse events paint with it (rendercanvas, which turns the 3D
    view's Qt events into its own, reads no tablet events). A mouse paints at full pressure.

    The pen's last pressure outlives its release: the canvas handles the mouse events later, once per frame, and
    the moves of a fast stroke still queued at the release painted at full pressure (a last stamp at full
    intensity). A press of a real mouse (not the pen's) goes back to full pressure."""

    def tabletEvent(self, e):  # noqa: N802 - Qt's name
        if e.type() in (QEvent.TabletPress, QEvent.TabletMove):
            self._pen = float(e.pressure())
        e.ignore()  # Qt goes on with the mouse events, which paint

    def mousePressEvent(self, e):  # noqa: N802 - Qt's name
        if e.pointingDevice().type() != QInputDevice.DeviceType.Stylus:
            self._pen = None
        super().mousePressEvent(e)

    def pen_pressure(self) -> float:
        """The pressure of the stroke being drawn: the pen's, 1 with a mouse."""
        p = getattr(self, "_pen", None)
        return 1.0 if p is None else p


class PaintView3D(PenPressure, Canvas):
    """The 3D view: renderer, Paint navigation, brush cursor and strokes (through `owner`)."""

    def __init__(self, device, owner, parent=None):
        super().__init__(parent, present_method=present_method(), update_mode="ondemand", max_fps=MAX_FPS)
        self.device, self.owner = device, owner
        self.yaw, self.pitch, self.distance = 0.6, 0.3, 3.4
        self.target = np.zeros(3, np.float32)
        self.fov = 40.0
        self.orthographic = False  # Perspective Mode
        # Camera Rotation Mode: Constrained (the documentation's default) orbits by yaw and pitch around the world Y;
        # Free turns the camera around its own axes (X, Y and Z: the roll comes from the drags, as a trackball),
        # `orient` = columns right, up, back in world
        self.orient = None
        self.look = Look(enable_displacement=False, ground=False, tonemap=2)
        self._drag = None
        self._pivot = None  # the orbit's center for the current drag (surface under the cursor)
        self._action = None  # the mouse binding of the current press (shortcuts.MOUSE)
        # General preferences > camera options (set by the workspace): multipliers, negative = inverse direction
        self.rotation_speed = self.zoom_speed = self.wheel_speed = 1.0
        self._painting = False
        self.held: set = set()  # keys held down (S: projection / stencil drags, V: clone source…)
        self.camera_cache = None  # (camera key, (texture, positions, normals)) of the last camera buffer
        ctx = self.get_wgpu_context()
        fmt = ctx.get_preferred_format(device.adapter).replace("-srgb", "")
        ctx.configure(device=device, format=fmt)
        self.renderer = Renderer(device, fmt)
        self.add_event_handler(self._on_event, "wheel", "pointer_down", "pointer_move", "pointer_up", "key_down",
                               "key_up", "pointer_leave", "double_click")
        self.request_draw(self._draw)

    # --- camera ---------------------------------------------------------------------------------------

    def refresh(self):
        self.request_draw(self._draw)

    @property
    def free_rotation(self) -> bool:
        return self.orient is not None

    @free_rotation.setter
    def free_rotation(self, on: bool) -> None:
        if on == self.free_rotation:
            return
        if on:
            self.orient = self.frame_axes()
        else:  # back to the orbit: the view direction kept, the roll dropped
            b = self.orient[:, 2]
            self.yaw = math.atan2(b[0], b[2])
            self.pitch = min(max(math.asin(min(max(b[1], -1.0), 1.0)), -1.55), 1.55)
            self.orient = None
        self.refresh()

    def frame_axes(self) -> np.ndarray:
        """The camera's axes in world, columns right, up, back."""
        if self.orient is not None:
            return self.orient.copy()
        cp = math.cos(self.pitch)
        back = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        right = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        return np.column_stack([right, np.cross(back, right), back])

    def eye(self):
        if self.orient is not None:
            return self.target + self.distance * self.orient[:, 2].astype(np.float32)
        cp = math.cos(self.pitch)
        return self.target + self.distance * np.array([cp * math.sin(self.yaw), math.sin(self.pitch),
                                                       cp * math.cos(self.yaw)], np.float32)

    def up(self):
        return (0, 1, 0) if self.orient is None else tuple(self.orient[:, 1])

    def _rotate_free(self, dx: float, dy: float, snap: bool) -> None:
        """Free rotation: a horizontal drag turns around the camera's up, a vertical one around its right (the
        same directions as the orbit); Shift snaps the orientation to the nearest axis-aligned one."""
        o = self.orient
        r = _axis_rotation(o[:, 1], -dx * 0.01) @ _axis_rotation(o[:, 0], -dy * 0.01)
        o = r @ o
        u, _, vt = np.linalg.svd(o)  # orthonormal again (drift of the accumulated products)
        o = u @ vt
        if snap:
            back = np.zeros(3)
            back[np.argmax(np.abs(o[:, 2]))] = np.sign(o[np.argmax(np.abs(o[:, 2])), 2])
            up = o[:, 1] - back * float(np.dot(o[:, 1], back))
            i = int(np.argmax(np.abs(up)))
            up = np.zeros(3)
            up[i] = np.sign(o[i, 1])
            o = np.column_stack([np.cross(up, back), up, back])
        self.orient = o

    def view_proj(self, w, h):
        aspect = w / max(h, 1)
        if self.orthographic:
            half = self.distance * math.tan(math.radians(self.fov) / 2)
            proj = orthographic(half, 0.01, 1000.0)
            proj = np.diag([1 / aspect, 1, 1, 1]).astype(np.float32) @ proj
        else:
            proj = perspective(math.radians(self.fov), aspect, 0.01, 1000.0)
        return proj @ look_at(self.eye(), self.target, self.up())

    def camera_key(self, w, h):
        return (round(self.yaw, 6), round(self.pitch, 6), round(self.distance, 6), tuple(np.round(self.target, 6)),
                self.fov, self.orthographic, None if self.orient is None else tuple(np.round(self.orient, 6).flat), w,
                h, self.owner.mesh_revision)

    def brush_view(self) -> View:
        w, h = self.physical_size()
        return View(self.view_proj(w, h), self.eye().astype(np.float64), math.radians(self.fov), w, h)

    def physical_size(self):
        r = self.get_pixel_ratio()
        return max(1, int(self.width() * r)), max(1, int(self.height() * r))

    def frame(self, lo, hi):
        center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
        self.target = center.astype(np.float32)
        self.distance = max(radius / math.tan(math.radians(self.fov) / 2) * 1.15, 0.05)
        self.refresh()

    def render_image(self, width: int, height: int) -> np.ndarray:
        """The view rendered offscreen (HxWx4 uint8): for tests and checks (grab() can return an old frame)."""
        fmt = self.renderer.format
        t = self.device.create_texture(size=(width, height, 1), format=fmt,
                                       usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        self.renderer.draw(t, self.eye(), self.view_proj(width, height), self.project_look())
        row = (width * 4 + 255) // 256 * 256
        data = self.device.queue.read_texture({"texture": t, "origin": (0, 0, 0)}, {"bytes_per_row": row},
                                              (width, height, 1))
        a = np.frombuffer(data, np.uint8).reshape(height, row)[:, :width * 4].reshape(height, width, 4).copy()
        if fmt.startswith("bgra"):
            a = a[..., [2, 1, 0, 3]]
        return a

    def _draw(self):
        cur = self.get_wgpu_context().get_current_texture()
        if not self.renderer.meshes:  # no project: the viewport's background only
            enc = self.device.create_command_encoder()
            rp = enc.begin_render_pass(color_attachments=[{"view": cur.create_view(), "load_op": "clear",
                                                           "store_op": "store", "clear_value": (0.137, 0.137, 0.137, 1)}])
            rp.end()
            self.device.queue.submit([enc.finish()])
            return
        f = self.renderer.falloff_overlay
        if f is not None and f.get("key") != self.camera_key(*self.physical_size()):  # the camera moved
            self.owner.update_falloff()
        self.renderer.draw(cur, self.eye(), self.view_proj(cur.size[0], cur.size[1]), self.project_look())

    def project_look(self):
        """The look with the project's normal map format: the stack's normal channel is in it (Project settings >
        Normal Map Format), not in the Graph workspace's project setting the 3D view of graphs follows."""
        doc = getattr(self.owner, "doc", None)
        if doc is not None and doc.project is not None:
            self.look.opengl_normals = not doc.project.directx
        return self.look

    # --- events ---------------------------------------------------------------------------------------

    def _on_event(self, e):
        kind = e["event_type"]
        mods = set(e.get("modifiers", ()))
        if kind in ("key_down", "key_up"):
            k = str(e.get("key", "")).lower()
            (self.held.add if kind == "key_down" else self.held.discard)(k)
            if kind == "key_down":
                self.owner.key(e.get("key", ""), mods)
            if k == shortcuts.binding("Ignore stencil mask").lower():  # held: the stencil is ignored (and hidden)
                self.owner.update_projection_preview()
            key = str(e.get("key", ""))
            if key in ("Control", "Shift", "Alt", "Meta"):
                self.owner.keyboard_helper(mods | {key} if kind == "key_down" else mods - {key})
            return
        if kind == "pointer_leave":
            self.owner.hover(None)
            self.owner.keyboard_helper(set())
            return
        r = self.get_pixel_ratio()
        if kind == "double_click":
            self.owner.double_click(e["x"] * r, e["y"] * r)
            return
        if kind == "wheel":
            self.distance = min(max(self.distance * 1.001 ** (e["dy"] * self.wheel_speed), 0.02), 200.0)
            self.refresh()
            return
        x, y = e["x"], e["y"]
        if kind == "pointer_down":
            self._drag = (x, y, e["button"], frozenset(mods), x, y)
            self._action = self._binding_for(e["button"], mods)
            a = self._action
            # Orbit/rotate turns around the surface under the cursor (Viewport page), the target without one
            self._pivot = self.owner.surface_point(x * r, y * r) if a in ("Camera rotate", "Camera snap rotate") else None
            if a == "Set Clone tool source location":
                self.owner.set_clone_source(x * r, y * r)
                self._drag = None
            elif a == "Texture set selection":
                self.owner.select_texture_set_at(x * r, y * r)
                self._drag = None
            elif a in ("Use tool / paint", "Draw straight lines", "Draw straight lines with snapping"):
                # straight lines from the last stroke's end; with snapping every 5 degrees
                self._painting = self.owner.stroke_begin(x * r, y * r, self.pen_pressure(), shift=a != "Use tool / paint",
                                                         ctrl=a == "Draw straight lines with snapping")
            return
        if kind == "pointer_move":
            if self._painting:
                self.owner.stroke_add(x * r, y * r, self.pen_pressure())
                return
            if self._drag is None:
                self.owner.hover((x * r, y * r))
                return
            x0, y0, b, mods0, sx, sy = self._drag
            dx, dy = x - x0, y - y0
            self._drag = (x, y, b, mods0, sx, sy)
            a = self._action
            if a and a.startswith("Stencil"):  # S + left rotates (Shift snaps), right scales, middle moves
                self.owner.drag_projection(dx, dy, {"Stencil zoom": 2, "Stencil translation": 3}.get(a, 1),
                                           a == "Stencil snap rotate")
                return
            if a in ("Camera translate", "Camera translate (alternative)"):
                self._pan(dx, dy)
            elif a in ("Camera rotate", "Camera snap rotate"):
                self._orbit(dx * self.rotation_speed, dy * self.rotation_speed, a == "Camera snap rotate")
            elif a == "Camera zoom":
                self.distance = min(max(self.distance * 1.01 ** (dy * self.zoom_speed), 0.02), 200.0)
            elif a == "Rotate environment":
                self.look.env_rotation += dx * 0.01
            elif a == "Change tool Size / Hardness":
                self.owner.drag_brush(dx, dy, "size")
                self.owner.hover((sx * r, sy * r))
                return
            elif a == "Change tool Flow / Rotation":
                self.owner.drag_brush(dx, dy, "flow")
                return
            else:
                return
            self.owner.hover(None)
            self.refresh()
            return
        if kind == "pointer_up":
            if self._painting:
                self._painting = False
                self.owner.stroke_end()
            a = self._action
            if self._drag is not None and a and a.startswith("Stencil"):  # the image moved: its Properties follow
                self.owner.properties.rebuild()
            if (self._drag is not None and a == "Context menu"
                    and abs(x - self._drag[4]) + abs(y - self._drag[5]) < 4):  # a click, not a drag: the quick menu
                self.owner.quick_properties(self.mapToGlobal(QPointF(x, y).toPoint()))
            self._drag, self._action = None, None

    def stencil_ignored(self) -> bool:
        """The Ignore stencil mask key held down (Shortcuts page, N by default)."""
        key = shortcuts.binding("Ignore stencil mask").lower()
        return bool(key) and key in self.held

    def _binding_for(self, button, mods):
        """The mouse binding a press starts (Shortcuts page), the first that applies: the stencil ones need a
        stencil or a projection image, the clone source the Clone tool."""
        for name in shortcuts.match_mouse(button, mods, self.held):
            if name.startswith("Stencil") and self.owner._xf_target() is None:
                continue
            if name == "Set Clone tool source location" and self.owner.tools.tool != "clone":
                continue
            return name
        return None

    def _orbit(self, dx, dy, snap):
        before = self.frame_axes()
        if self.orient is not None:
            self._rotate_free(dx, dy, snap)
        else:
            self.yaw -= dx * 0.01
            self.pitch = min(max(self.pitch + dy * 0.01, -1.55), 1.55)
            if snap:
                q = math.pi / 2
                self.yaw = round(self.yaw / q) * q
                self.pitch = round(self.pitch / q) * q if abs(self.pitch) > q / 2 else 0.0
        if self._pivot is not None:  # the camera's turn applied around the pivot: the target follows
            turn = self.frame_axes() @ before.T
            self.target = (self._pivot + turn @ (self.target - self._pivot)).astype(np.float32)

    def _pan(self, dx, dy):
        axes = self.frame_axes()  # the camera's right and up (rolled too, in Free rotation)
        self.target = (self.target + (-axes[:, 0] * dx + axes[:, 1] * dy) * self.distance * 0.0015).astype(np.float32)




def circle_lines(center, normal, radius, color=(1.0, 1.0, 1.0, 0.9), n=48) -> np.ndarray:
    """A circle on the surface plane at a hit: line segments (position + color) for the renderer."""
    nrm = np.asarray(normal, np.float64)
    nrm /= max(np.linalg.norm(nrm), 1e-9)
    a = np.cross(nrm, [0, 1, 0]) if abs(nrm[1]) < 0.9 else np.cross(nrm, [1, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(nrm, a)
    t = np.linspace(0, 2 * math.pi, n + 1)
    pts = center + nrm * radius * 0.02 + radius * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)
    seg = np.empty((n * 2, 7), np.float32)
    seg[0::2, :3], seg[1::2, :3] = pts[:-1], pts[1:]
    seg[:, 3:] = color
    return seg


# 7-segment digits for the UDIM numbers (the overlay draws lines only): segments a-g of each digit.
SEGMENTS = {"a": ((0, 0), (1, 0)), "b": ((1, 0), (1, 0.5)), "c": ((1, 0.5), (1, 1)), "d": ((0, 1), (1, 1)),
            "e": ((0, 0.5), (0, 1)), "f": ((0, 0), (0, 0.5)), "g": ((0, 0.5), (1, 0.5))}
DIGITS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc",
          "7": "abc", "8": "abcdefg", "9": "abcdfg"}


class PaintView2D(PenPressure, Viewport):
    """The 2D view: the selected Texture Set's texture space with the displayed channel and the UV wireframe,
    drawn by the GPU from the stack's textures (the Graph 2D view's Viewport: no readback). Each UDIM tile is
    its own texture, laid at its place in UV space (1001 at 0-1, U to the right, V up); UV Tile information shows
    the tiles' grid and UDIM numbers. Left drag paints with the current tool inside the Texture Set's tiles, the
    stamp size relative to the view (ours: Size is a percentage of a quarter of the view's short side);
    the camera translate and zoom bindings of the Shortcuts page (as in the 3D view) and the wheel move and zoom
    the view. Rotating the UV space is not offered."""
    UV_COLOR = (1.0, 1.0, 1.0, 60 / 255)
    GRID, NUMBERS = (1.0, 1.0, 1.0, 0.35), (0.85, 0.85, 0.85, 1.0)
    CURSOR, ERASER = (1.0, 1.0, 1.0, 220 / 255), (1.0, 120 / 255, 120 / 255, 220 / 255)
    U_AXIS, V_AXIS, LETTERS = (0.878, 0.314, 0.314, 1.0), (0.314, 0.753, 0.314, 1.0), (0.733, 0.733, 0.733, 1.0)

    def __init__(self, device, owner=None):
        super().__init__(device)
        self.owner = owner
        self.mouse = None  # the cursor (logical pixels)
        self._painting = False
        self._uv_segments = None
        self.tiles: list = []  # [(UDIM, source)]: what each tile shows (`source`: callable -> (texture, is_gray))
        self.tile_info = True  # UV Tile information: the tiles' grid and UDIM numbers
        self.checker = False  # the stack's colors as they are
        self.uv_color = self.UV_COLOR
        self.setMinimumSize(100, 100)
        self.zoom_speed = self.wheel_speed = 1.0  # General preferences (set by the workspace)

    def set_tiles(self, tiles: list) -> None:
        """[(UDIM, source)]: each tile's `source`, a callable giving (texture, is_gray) of what it shows."""
        self.tiles = tiles
        self.source = tiles[0][1] if tiles else None
        self.refresh()

    def unit(self) -> tuple | None:
        """The Texture Set's size: the view's units (a tile with its own resolution is scaled to it)."""
        doc = self.owner.doc if self.owner is not None else None
        return tuple(doc.ts.size) if doc is not None and doc.ts is not None and self.tiles else None

    def shown(self) -> list:
        unit = self.unit()
        if unit is None:  # no Texture Set (a document being set)
            return []
        out = []
        for udim, source in self.tiles:
            got = source()
            if got is not None:
                du, dv = tile_uv(udim)
                out.append((got[0], got[1], (du * unit[0], -dv * unit[1]), got[0].width / unit[0]))
        return out

    def extent(self) -> tuple:
        """The tiles' rectangle (view units)."""
        w, h = self.unit()
        cells = [tile_uv(u) for u, _s in self.tiles] or [(0, 0)]
        return (min(c[0] for c in cells) * w, -max(c[1] for c in cells) * h,
                (max(c[0] for c in cells) + 1) * w, (1 - min(c[1] for c in cells)) * h)

    def set_uvs(self, segments) -> None:
        if segments is not self._uv_segments:  # a new UV layout: its line buffer
            self._uv_segments = segments
            self.set_uv_lines(segments)

    def _short(self) -> float:
        return min(self.width(), self.height())

    def _clamp_zoom(self) -> None:
        """The view's scale between 5% and 50 times its short side (in the view's units)."""
        size = self.unit()
        if size is None or self.zoom is None:
            return
        side = self._short() * self.get_pixel_ratio() / size[0]
        self.zoom = min(max(self.zoom, 0.05 * side), 50.0 * side)

    def brush_radius_px(self) -> float:
        return self.owner.tools.brush.size / 100 * self._short() / 4

    def _uv(self, x, y):
        """The stored UV (U, 1 - V: 1001 is 0-1) under a logical position (None before the first draw)."""
        size = self.unit()
        if size is None or self.zoom is None:
            return None
        r = self.get_pixel_ratio()
        w, h = self.get_physical_size()
        return (((x * r - w / 2) / self.zoom + self.center[0]) / size[0],
                ((y * r - h / 2) / self.zoom + self.center[1]) / size[1])

    def _texel(self, x, y):
        """(x, y) in texels of the Texture Set's grid (the 1001 tile's texels, Stroke.add_texture), or None
        outside its tiles."""
        uv = self._uv(x, y)
        if uv is None or not 0 <= uv[0] < 10 or udim_at(uv[0], uv[1]) not in self.owner.doc.ts.tiles:
            return None
        w, h = self.owner.doc.ts.size
        return uv[0] * w, uv[1] * h

    def _radius_texels(self) -> float:
        return self.brush_radius_px() * self.get_pixel_ratio() / self.zoom

    def _on_event(self, e):
        kind = e["event_type"]
        if kind == "pointer_leave":
            self.mouse = None
            self.refresh()
            return
        if kind == "wheel":
            if self.zoom is not None:
                self.zoom *= 1.0015 ** (-e["dy"] * self.wheel_speed)
                self._clamp_zoom()
                self.refresh()
            return
        if kind not in ("pointer_down", "pointer_move", "pointer_up"):
            return
        x, y = e["x"], e["y"]
        self.mouse = (x, y)
        if kind == "pointer_down":
            names = [n for n in shortcuts.match_mouse(e["button"], set(e.get("modifiers", ())), set())
                     if n in VIEW2D_BINDINGS]
            action = names[0] if names else None
            self._drag = (x, y, action)
            if action == "Use tool / paint" and self.owner is not None and self.owner.doc is not None:
                t = self._texel(x, y)
                if t is not None:
                    self._painting = self.owner.stroke2d_begin(*t, self.pen_pressure(), self._radius_texels())
        elif kind == "pointer_move":
            if self._painting:
                t = self._texel(x, y)
                if t is not None:
                    self.owner.stroke2d_add(*t, self.pen_pressure(), self._radius_texels())
            elif self._drag is not None and self.zoom is not None:
                x0, y0, action = self._drag
                self._drag = (x, y, action)
                r = self.get_pixel_ratio()
                if action in ("Camera translate", "Camera translate (alternative)"):
                    self.center = (self.center[0] - (x - x0) * r / self.zoom,
                                   self.center[1] - (y - y0) * r / self.zoom)
                elif action == "Camera zoom":
                    self.zoom *= 1.01 ** (-(y - y0) * self.zoom_speed)
                    self._clamp_zoom()
        else:
            self._drag = None
            if self._painting:
                self._painting = False
                self.owner.stroke_end()
        self.refresh()

    def overlay(self) -> np.ndarray:
        """The UV Tile information (the tiles' borders, their UDIM numbers at the top left), the brush cursor and the
        axis information (bottom right: U and V), in physical pixels."""
        r = self.get_pixel_ratio()
        segs = []

        def line(a, b, color):
            segs.append([a[0] * r, a[1] * r, *color])
            segs.append([b[0] * r, b[1] * r, *color])

        size = self.unit()
        if self.tile_info and size is not None and self.zoom is not None and len(self.tiles) > 0:
            w, h = self.get_physical_size()
            for udim, _source in self.tiles:
                du, dv = tile_uv(udim)
                x0 = ((du * size[0] - self.center[0]) * self.zoom + w / 2) / r
                y0 = ((-dv * size[1] - self.center[1]) * self.zoom + h / 2) / r
                x1, y1 = x0 + size[0] * self.zoom / r, y0 + size[1] * self.zoom / r
                for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                    line(a, b, self.GRID)
                for k, digit in enumerate(str(udim)):  # 6 x 10 px digits, 4 px apart (a 1 centered in its cell)
                    ox, oy = x0 + 6 + k * 10 - (3 if digit == "1" else 0), y0 + 6
                    for seg in DIGITS[digit]:
                        (ax, ay), (bx, by) = SEGMENTS[seg]
                        line((ox + ax * 6, oy + ay * 10), (ox + bx * 6, oy + by * 10), self.NUMBERS)

        if self.mouse is not None and self.owner is not None and self.owner.doc is not None:
            color = self.ERASER if self.owner.tools.tool == "eraser" else self.CURSOR
            rad, (mx, my) = self.brush_radius_px(), self.mouse
            t = np.linspace(0.0, 2 * math.pi, 49)
            pts = np.stack([mx + rad * np.cos(t), my + rad * np.sin(t)], 1)
            for a, b in zip(pts[:-1], pts[1:]):
                line(a, b, color)
        bx, by = self.width() - 30, self.height() - 14
        line((bx, by), (bx + 18, by), self.U_AXIS)
        line((bx, by), (bx, by - 18), self.V_AXIS)
        ux, uy = bx + 21, by - 4  # the letters as strokes: U right of the U axis, V above the V axis
        for a, b in (((ux, uy), (ux, uy + 6)), ((ux, uy + 6), (ux + 2, uy + 8)), ((ux + 2, uy + 8), (ux + 4, uy + 8)),
                     ((ux + 4, uy + 8), (ux + 6, uy + 6)), ((ux + 6, uy + 6), (ux + 6, uy)),
                     ((bx - 3, by - 30), (bx, by - 22)), ((bx, by - 22), (bx + 3, by - 30))):
            line(a, b, self.LETTERS)
        return np.asarray(segs, np.float32)
