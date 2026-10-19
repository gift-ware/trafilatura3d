"""The 3D View (the view, its renderers, the displacement pop-up and the scene browser): menu bar (Scene,
Materials, Lights, Camera, Environment, Display, Renderer), the scene toolbar on the left, the display toolbar at the bottom, the scene
browser, mesh selection, point lights, renders and state files. Drawn by render3d.

Outputs reach a material's channels by their Usage attribute (outputs without a usage match by
identifier), unless the material's Edit properties connect another output or node.

Navigation: left drag orbits, middle drag (or Ctrl+right) pans, wheel or right drag
zooms, Shift+right drag rotates the environment, Ctrl+Shift+left drag orbits point light 1,
Ctrl+Shift+right drag moves it closer or farther, Shift+left click selects a mesh, right click on
the selection opens its menu, Escape unselects, F frames, R resets the camera, Shift+R also its
properties, Alt+S saves a render, Alt+C copies it."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import wgpu
from PySide6.QtCore import QSettings, QSize, Qt, Signal
from PySide6.QtGui import QActionGroup, QCursor, QImage
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMenu,
                               QMenuBar, QSizePolicy, QSplitter, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from trafilatura3d import bake
from trafilatura3d import shaders
from trafilatura3d.editor import theme
from trafilatura3d.editor.canvas import Canvas
from trafilatura3d.editor.icons import icon
from trafilatura3d.editor.meshes import EXTENDED, PRIMITIVES, STANDARD
from trafilatura3d.editor.render3d import (DIAGNOSTICS, SHADOW_MODES, SLOTS, TONEMAPS, Look, Renderer, look_at,
                                     MAX_FPS, perspective, present_method, uv_to_dir)
from trafilatura3d.editor.scene3d import (AUTO, EXPORT_FORMATS, NONE, STATE_SUFFIX, CameraProps, Lights, Material,
                                    Scene)
from trafilatura3d.shaders import SHADERS
from trafilatura3d.meshio import MESH_FILTER, MESH_SUFFIXES, split_by_tile

# Outputs without a usage: matched by identifier (ours), per channel usage.
NAME_HINTS = {"baseColor": ("basecolor", "base_color", "albedo", "color"), "diffuse": ("diffuse",),
              "normal": ("normal",), "roughness": ("roughness", "rough"), "specularRoughness": ("roughness", "rough"),
              "glossiness": ("glossiness", "gloss"), "metallic": ("metallic", "metalness", "metal"),
              "metalness": ("metallic", "metalness", "metal"), "specular": ("specular", "specularlevel"),
              "height": ("height", "displacement"), "ambientOcclusion": ("ambientocclusion", "ao", "occlusion"),
              "emissive": ("emissive", "emission"), "emissionColor": ("emissive", "emission"),
              "opacity": ("opacity", "alpha")}

# Daylight presets made with the Physical Sun/Sky node: sun position (longitude, latitude: 0 top,
# 0.5 horizon) and the key light's tint.
SUN_SKY = {"Sun & Sky: morning": ((0.30, 0.42), (1.0, 0.82, 0.62)),
           "Sun & Sky: noon": ((0.42, 0.14), (1.0, 0.97, 0.92)),
           "Sun & Sky: sunset": ((0.62, 0.47), (1.0, 0.58, 0.32))}
ENVIRONMENTS = ("Studio", *SUN_SKY)
# Camera > Viewpoints: name -> (yaw, pitch) in radians
VIEWPOINTS = {"Default": (0.6, 0.35), "Back": (math.pi, 0.0), "Bottom": (0.0, -1.49), "Front": (0.0, 0.0),
              "Left": (-math.pi / 2, 0.0), "Right": (math.pi / 2, 0.0), "Top": (0.0, 1.49)}
HDRI_MAX_WIDTH = 4096
CHANNEL_NAMES = ("RGB", "Red", "Green", "Blue")
RENDER_FILTER = "Images (*.png *.jpg *.tif *.tiff *.exr)"
DEFAULT_STATE_KEY = "view3d/default_state"

# The settings shown in the Properties dock; the camera, environment and light parameters are ours.
ENVIRONMENT_ROWS = [("Environment", [
    ("Environment", "environment", "enum", ENVIRONMENTS, None),
    ("Show environment", "show_env", "bool", None, None),
    ("Exposure (EV)", "exposure", "float", -5.0, 5.0),
    ("Rotation", "env_rotation", "float", 0.0, 360.0),
    ("Intensity", "env_intensity", "float", 0.0, 4.0),
    ("Background blur", "background_blur", "float", 0.0, 6.0)])]
CAMERA_ROWS = [("Camera", [
    ("Field of view", "cam.fov", "float", 10.0, 120.0),
    ("Near clip", "cam.near", "float", 0.001, 1.0),
    ("Far clip", "cam.far", "float", 10.0, 1000.0),
    ("Tone mapping", "tonemap", "enum", TONEMAPS, None)]),
    ("Render", [
    ("Width", "cam.render_width", "int", 64, 8192),
    ("Height", "cam.render_height", "int", 64, 8192)])]
RENDERER_ROWS = [
    ("Parameters", [
        ("Samples", "samples", "enum", ("4 (MSAA)",), None),
        ("Ambient occlusion opacity", "ao_opacity", "float", 0.0, 1.0),
        ("Enable displacement", "enable_displacement", "bool", None, None),
        ("Enable backface culling", "backface_culling", "bool", None, None),
        ("Diagnostic mode", "diagnostic", "enum", DIAGNOSTICS, None),
        ("Rasterizer shadow mode", "shadow_mode", "enum", SHADOW_MODES, None),
        ("Rasterizer shadow opacity", "shadow_opacity", "float", 0.0, 1.0),
        ("Rasterizer order independent transparency enabled", "order_independent_transparency", "bool", None, None)]),
    ("Ground plane", [
        ("Enabled", "ground", "bool", None, None),
        ("Height", "ground_height", "float", -1.0, 1.0),
        ("Shadow intensity", "ground_shadow", "float", 0.0, 1.0)])]
PATHTRACER_ROWS = [
    ("Parameters", [
        ("Samples", "pt_samples", "int", 1, 4096),
        ("Enable displacement", "enable_displacement", "bool", None, None),
        ("Max bounces", "pt_bounces", "int", 1, 16)]),
    ("Ground plane", [
        ("Enabled", "ground", "bool", None, None),
        ("Height", "ground_height", "float", -1.0, 1.0),
        ("Shadow intensity", "ground_shadow", "float", 0.0, 1.0),
        ("Enable local lights", "ground_local_lights", "bool", None, None),
        ("Enable reflections", "ground_reflections", "bool", None, None),
        ("Reflections opacity", "reflections_opacity", "float", 0.0, 1.0),
        ("Reflections roughness", "reflections_roughness", "float", 0.0, 1.0)])]
DISPLACEMENT_ROWS = [("Displacement", [
    ("Height scale", "displacement", "float", 0.0, 0.5),
    ("Height level", "height_level", "float", 0.0, 1.0),
    ("Tessellation", "tessellation", "int", 0, 3),
    ("Parallax depth", "parallax", "float", 0.0, 0.15)])]
ENVIRONMENT_KEYS = ("show_env", "exposure", "env_rotation", "env_intensity", "background_blur")


class Viewport3D(Canvas):
    """The drawing surface: camera orbit, interaction modes and the renderer."""

    def __init__(self, device, owner, parent=None):
        super().__init__(parent, present_method=present_method(), update_mode="ondemand", max_fps=MAX_FPS)
        self.device = device
        self.owner = owner
        self.yaw, self.pitch, self.distance = 0.6, 0.35, 4.2
        self.target = np.zeros(3, np.float32)
        self.look = Look()
        self.mode = "camera"  # toolbar: "select", "camera" or "light"
        self._drag = None
        ctx = self.get_wgpu_context()
        fmt = ctx.get_preferred_format(device.adapter).replace("-srgb", "")  # the shader encodes sRGB
        ctx.configure(device=device, format=fmt)
        self.renderer = Renderer(device, fmt)
        self.setAcceptDrops(True)
        self.add_event_handler(self._on_event, "wheel", "pointer_down", "pointer_move", "pointer_up", "key_down")
        self.request_draw(self._draw)

    def refresh(self):
        self.request_draw(self._draw)

    def eye(self):
        cp = math.cos(self.pitch)
        return self.target + self.distance * np.array([cp * math.sin(self.yaw), math.sin(self.pitch),
                                                       cp * math.cos(self.yaw)], np.float32)

    def view_proj(self, w, h):
        cam = self.owner.scene.camera
        return perspective(math.radians(cam.fov), w / max(h, 1), cam.near, cam.far) @ look_at(self.eye(), self.target)

    def _draw(self):
        cur = self.get_wgpu_context().get_current_texture()
        vp = self.view_proj(cur.size[0], cur.size[1])
        if self.look.pathtracer:
            samples = self.owner.pathtrace(cur, self.eye(), vp)
            if samples < self.look.pt_samples:
                self.request_draw(self._draw)  # keep accumulating
        else:
            self.renderer.draw(cur, self.eye(), vp, self.look)
        self.owner.frame_drawn(cur.size[0], cur.size[1])

    def reset_camera(self):
        self.yaw, self.pitch, self.distance = 0.6, 0.35, 4.2
        self.target = np.zeros(3, np.float32)
        self.refresh()

    def ray(self, x, y):
        """World ray through a point of the widget (logical pixels)."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        inv = np.linalg.inv(self.view_proj(w, h))
        ndc = np.array([2 * x / w - 1, 1 - 2 * y / h])
        near = inv @ np.array([*ndc, 0.0, 1.0])
        far = inv @ np.array([*ndc, 1.0, 1.0])
        near, far = near[:3] / near[3], far[:3] / far[3]
        d = far - near
        return near, d / np.linalg.norm(d)

    def _on_event(self, e):
        kind = e["event_type"]
        mods = tuple(e.get("modifiers", ()))
        if kind == "key_down":
            self.owner.key(e.get("key", ""), mods)
            return
        if kind == "wheel":
            from trafilatura3d.editor.preferences import zoom_sign
            self.distance = min(max(self.distance * 1.001 ** (e["dy"] * zoom_sign()), 0.3), 60.0)
        elif kind == "pointer_down":
            self._drag = (e["x"], e["y"], e["button"], mods, e["x"], e["y"])
            if e["button"] == 1 and ("Shift" in mods and "Control" not in mods or self.mode == "select"):
                self.owner.pick(e["x"], e["y"])
                self._drag = None
            return
        elif kind == "pointer_move" and self._drag:
            x0, y0, button, mods0, sx, sy = self._drag
            dx, dy = e["x"] - x0, e["y"] - y0
            self._drag = (e["x"], e["y"], button, mods0, sx, sy)
            light = ("Control" in mods0 and "Shift" in mods0) or self.mode == "light"
            if button == 1 and light:
                self.owner.orbit_light(dx * 0.01, dy * 0.01, 1.0)
            elif button == 2 and light:
                self.owner.orbit_light(0.0, 0.0, 1.01 ** dy)
            elif button == 1:
                self.yaw -= dx * 0.01
                self.pitch = min(max(self.pitch + dy * 0.01, -1.5), 1.5)
            elif button == 3 or (button == 2 and "Control" in mods0):
                eye = self.eye()
                f = (self.target - eye) / np.linalg.norm(self.target - eye)
                s = np.cross(f, [0, 1, 0])
                s /= np.linalg.norm(s)
                up = np.cross(s, f)
                self.target = self.target + (-s * dx + up * dy) * self.distance * 0.0015
            elif button == 2 and "Shift" in mods0:
                self.look.env_rotation += dx * 0.01
            elif button == 2:
                self.distance = min(max(self.distance * 1.01 ** dy, 0.3), 60.0)
        elif kind == "pointer_up":
            drag, self._drag = self._drag, None
            if drag and drag[2] == 2 and abs(e["x"] - drag[4]) + abs(e["y"] - drag[5]) < 3:
                self.owner.context_menu()
            return
        else:
            return
        self.refresh()

    def _dropped(self, mime) -> str | None:
        """A dropped file this view opens (a 3D scene or a saved state); anything else goes on to the window."""
        for url in mime.urls():
            path = url.toLocalFile()
            if path.lower().endswith((*MESH_SUFFIXES, STATE_SUFFIX)):
                return path
        return None

    def dragEnterEvent(self, e):
        if self._dropped(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        path = self._dropped(e.mimeData())
        if path is None:
            e.ignore()
            return
        if path.endswith(STATE_SUFFIX):
            self.owner.load_state(path, with_mesh=True)
        else:
            self.owner.load_mesh(path)
        e.acceptProposedAction()


class View3D(QWidget):
    title_changed = Signal(str)
    settings_requested = Signal(str, object, object, object)  # title, rollouts, get, set (Properties)
    show_uvs = Signal(object)  # (uv, triangles) of the selected mesh, or None, for the 2D View

    def __init__(self, document):
        super().__init__()
        self.doc = document
        self.package, self.graph_id = None, ""
        self.overrides: dict[str, str] = {}  # binding slot -> node whose output replaces it (all materials)
        self.preview_source: str | None = None  # node shown alone and unlit
        self.environment = "Studio"
        self.scene = Scene()
        self.selected: int | None = None
        self.pathtracer = None  # made on first use
        self.pt_dirty = True  # meshes, materials or displacement changed since the last BVH
        self.env_version = 0
        self._editing = None  # the settings section shown in Properties
        self.toggles = {}  # look attribute -> checkable actions showing it
        self.stats = {"resolution": False, "scene_stats": False, "render_time": False, "samples": False}
        # The 3D scene resource on view (package, id): its baked maps go on the low, its bake setup
        # (high poly, cage) can be shown (Baking menu).
        self.bake_resource = None
        self.baked_preview: str | None = None  # a baker whose map is shown alone and unlit
        self._shown: dict = {}  # (material, slot) -> (job key, output) the renderer shows (`show_ready`)
        self._bake_meshes = {}  # (path, mtime) -> meshes in their own units
        self._overlay_key = None
        self.udim_tiles = [(0, 0)]  # the UDIM tiles of the meshes on view (_render_objects)
        self._subdivisions = {}  # (id(object), levels) -> (object, subdivided): _subdivided
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.menubar = QMenuBar()
        self.menubar.setNativeMenuBar(False)  # on macOS it would move to the global menu bar
        self.menubar.setVisible(True)  # the native mode hid it explicitly
        self.menubar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # narrower: its own "»" for the rest
        lay.addWidget(self.menubar)
        body = QHBoxLayout()
        body.setSpacing(0)
        side = theme.divider(QToolBar(), "right")
        side.setOrientation(Qt.Vertical)
        side.setIconSize(QSize(16, 16))
        body.addWidget(side)
        self.split = QSplitter(Qt.Horizontal)
        self.browser = QTreeWidget()
        self.browser.setHeaderHidden(True)
        self.browser.setContextMenuPolicy(Qt.CustomContextMenu)
        self.browser.customContextMenuRequested.connect(self._browser_menu)
        self.browser.itemClicked.connect(self._browser_clicked)
        self.browser.itemChanged.connect(self._browser_changed)
        self.browser.hide()
        self.view = Viewport3D(self.doc.ev.device, self)
        self.split.addWidget(self.browser)
        self.split.addWidget(self.view)
        self.split.setSizes([180, 600])
        body.addWidget(self.split, 1)
        holder = QWidget()
        holder.setLayout(body)
        lay.addWidget(holder, 1)
        self._display_toolbar(lay)
        self._menus()
        self._scene_toolbar(side)
        self.doc.changed.connect(self._doc_changed)
        self.set_primitive("Rounded Cube")
        self._apply_default_state()
        self.fill_color_spaces()
        self.view.setCursor(self.MODE_CURSORS["camera"])

    # --- UI ---------------------------------------------------------------------------------------

    def _display_toolbar(self, lay):
        bottom = QHBoxLayout()
        bottom.setContentsMargins(6, 2, 6, 2)
        self.aov = QComboBox()
        self.aov.setToolTip("3D render AOVs")
        self.aov.addItems(["Beauty", *DIAGNOSTICS[1:]])
        self.aov.currentIndexChanged.connect(lambda i: self.set_setting("diagnostic", i))
        bottom.addWidget(self.aov)
        self.channels = QComboBox()
        self.channels.setToolTip("Color channels")
        self.channels.addItems(list(CHANNEL_NAMES))
        self.channels.currentIndexChanged.connect(lambda i: self.set_setting("channel", i))
        bottom.addWidget(self.channels)
        self.cs = QComboBox()
        self.cs.setToolTip("Color space (display / view)")
        self.cs.currentIndexChanged.connect(lambda _i: self._apply_color())
        bottom.addWidget(self.cs)
        bottom.addStretch(1)
        self.info = QLabel("")
        self.info.setObjectName("dim")
        bottom.addWidget(self.info)
        b = theme.divider(QWidget(), "top")
        b.setLayout(bottom)
        lay.addWidget(theme.Overflow(b))

    def _menus(self):
        mb = self.menubar
        scene = mb.addMenu("Scene")
        scene.addAction("Edit", lambda: self.edit_settings("scene"))
        std = scene.addMenu("Standard primitives")
        for name in STANDARD:
            std.addAction(name, lambda n=name: self.set_primitive(n))
        ext = scene.addMenu("Extended primitives")
        for name in EXTENDED:
            ext.addAction(name, lambda n=name: self.set_primitive(n))
        scene.addAction("Load 3D scene...", self._load_mesh_dialog)
        scene.addSeparator()
        scene.addAction("Display UV's in 2D View", self._display_uvs)
        scene.addAction("Create 3D resource from current scene...", self.create_resource)
        scene.addSeparator()
        scene.addAction("Load state file...", lambda: self._load_state_dialog(False))
        scene.addAction("Load state file with mesh...", lambda: self._load_state_dialog(True))
        scene.addAction("Save state file...", self._save_state_dialog)
        scene.addAction("Save current state as default", self.save_default_state)
        scene.addSeparator()
        scene.addAction("Export scene...", self._export_dialog)
        a = scene.addAction("Export scene with layers...")
        a.setEnabled(False)
        a.setToolTip("Layered USD export of a loaded USD scene: not available")
        a = scene.addAction("Export tessellated geometry...")
        a.setEnabled(False)
        a.setToolTip("OpenGL renderer only, not available")
        scene.addSeparator()
        scene.addAction("Reset scene", self.reset_scene)
        self.materials_menu = mb.addMenu("Materials")
        lights = mb.addMenu("Lights")
        lights.addAction("Edit", lambda: self.edit_settings("lights"))
        lights.addAction("Reset Lights", self.reset_lights)
        cam = mb.addMenu("Camera")
        cam.addAction("Edit properties", lambda: self.edit_settings("camera"))
        cam.addAction("Focus (F)", self.focus)
        self.scene_cameras = cam.addMenu("Scene cameras")
        views = cam.addMenu("Viewpoints")
        for name, (yaw, pitch) in VIEWPOINTS.items():
            views.addAction(name, lambda y=yaw, p=pitch: self._camera(y, p))
        cam.addSeparator()
        a = cam.addAction("Save render... (Alt+S)", self.save_render_dialog)
        a = cam.addAction("Copy render to clipboard (Alt+C)", self.copy_render)
        cam.addSeparator()
        cam.addAction("Reset position (R)", self.view.reset_camera)
        cam.addAction("Reset selected (Shift+R)", self.reset_camera_full)
        env = mb.addMenu("Environment")
        env.addAction("Edit properties", lambda: self.edit_settings("environment"))
        env.addSeparator()
        for name in ENVIRONMENTS:
            env.addAction(name, lambda n=name: self._environment(n))
        env.addAction("Load HDRI...", self._load_hdri)
        env.addSeparator()
        env.addAction("Reset environment", self.reset_environment)
        disp = mb.addMenu("Display")
        for label, attr in (("Axis", "axis"), ("Grid", "grid")):
            self._check(disp.addAction(label), attr)
        for label, key in (("Resolution", "resolution"), ("Scene stats", "scene_stats"),
                           ("Render time", "render_time"), ("Samples", "samples")):
            self._stat_check(disp.addAction(label), key)
        disp.addSeparator()
        for label, attr in (("Backface culling", "backface_culling"), ("Bounding box", "bounding_box"),
                            ("Wireframe", "wireframe"), ("Vertex tangent space", "tangent_space"),
                            ("Light", "light_helpers")):
            self._check(disp.addAction(label), attr)
        disp.addSeparator()
        self._check(disp.addAction("OpenGL normal format"), "opengl_normals")
        rend = mb.addMenu("Renderer")
        group = QActionGroup(rend)
        r = rend.addAction("Rasterizer")
        r.setCheckable(True)
        r.setChecked(True)
        group.addAction(r)
        pt = rend.addAction("GPU Pathtracer")
        pt.setCheckable(True)
        group.addAction(pt)
        r.toggled.connect(lambda on: self.set_setting("pathtracer", not on) if on == self.view.look.pathtracer else None)
        self.toggles.setdefault("pathtracer", []).append(pt)
        pt.toggled.connect(lambda on: self.set_setting("pathtracer", on) if on != self.view.look.pathtracer else None)
        rend.addSeparator()
        rend.addAction("Edit properties", lambda: self.edit_settings("renderer"))
        # Baking visualization: the scene resource's bake setup over its low mesh.
        self.baking_menu = mb.addMenu("Baking")
        self.baking_menu.aboutToShow.connect(self._fill_baking_menu)

    def _fill_baking_menu(self):
        m = self.baking_menu
        m.clear()
        res = self._bake_config()
        if res is None:
            a = m.addAction("Open a 3D scene resource to see its bake")
            a.setEnabled(False)
            return
        m.addAction("Bake model information...", self._open_baker)
        m.addSeparator()
        for label, attr in (("Show high definition meshes", "show_high"), ("Show cage", "show_cage"),
                            ("Show rear distance", "show_rear"),
                            ("High poly the bake misses in red", "outside_red"),
                            ("Hide low definition mesh", "hide_low")):
            a = m.addAction(label)
            a.setCheckable(True)
            a.setChecked(getattr(self.view.look, attr))
            a.toggled.connect(lambda on, attr=attr: self._set_bake_look(attr, on))
        opacity = m.addMenu("Cage opacity")
        for pct in (10, 30, 50, 75):
            a = opacity.addAction(f"{pct}%")
            a.setCheckable(True)
            a.setChecked(abs(self.view.look.cage_opacity - pct / 100) < 1e-3)
            a.triggered.connect(lambda _c, v=pct / 100: self._set_bake_look("cage_opacity", v))
        m.addSeparator()
        shown = m.addMenu("Show baked map")
        files = self._baked_files()
        a = shown.addAction("Material (baked maps on the low)")
        a.setCheckable(True)
        a.setChecked(self.baked_preview is None)
        a.triggered.connect(lambda: self._show_baked(None))
        for baker, path in files.items():
            a = shown.addAction(bake.BAKERS[baker])
            a.setCheckable(True)
            a.setChecked(self.baked_preview == baker)
            a.triggered.connect(lambda _c, b=baker: self._show_baked(b))
        if not files:
            a = shown.addAction("Nothing baked yet")
            a.setEnabled(False)

    def _scene_toolbar(self, side):
        b = side.addAction(icon("explorer"), "Scene browser")
        b.setCheckable(True)  # toggled, not addAction's slot: that one is called without the state
        b.toggled.connect(self.browser.setVisible)
        modes = QActionGroup(side)
        for ic, tip, mode in (("search", "Select", "select"), ("camera", "Camera", "camera"),
                              ("highlight", "Light", "light")):
            if mode == "light":
                self._check(side.addAction(icon("environment"), "Show environment"), "show_env")
            a = side.addAction(icon(ic), tip, lambda m=mode: self.set_mode(m))
            a.setCheckable(True)
            a.setChecked(mode == "camera")
            modes.addAction(a)
        side.addAction(icon("renderer"), "Renderer settings", lambda: self.edit_settings("renderer"))
        self._check(side.addAction(icon("palette"), "Enable Pathtracer"), "pathtracer")
        self._check(side.addAction(icon("shadows"), "Enable shadows"), "shadows")
        self._check(side.addAction(icon("ground"), "Enable ground plane"), "ground")
        side.addAction(icon("displacement"), "Displacement", lambda: self.edit_settings("displacement"))
        side.addSeparator()
        self._check(side.addAction(icon("tile"), "Grid"), "grid")
        self._stat_check(side.addAction(icon("info"), "Scene stats"), "scene_stats")
        self._check(side.addAction(icon("align"), "Axis"), "axis")
        side.addSeparator()
        self._check(side.addAction(icon("view2d"), "Backface culling"), "backface_culling")
        self._check(side.addAction(icon("focus"), "Bounding Box"), "bounding_box")
        self._check(side.addAction(icon("chain"), "Vertex tangent space"), "tangent_space")
        self._check(side.addAction(icon("graph"), "Wireframe"), "wireframe")

    MODE_HINTS = {"select": "Select: click a mesh (Escape unselects, right click for its menu)",
                  "camera": "",  # the default: no hint
                  "light": "Light: left drag orbits point light 1, right drag moves it closer or farther"}
    MODE_CURSORS = {"select": Qt.PointingHandCursor, "camera": Qt.OpenHandCursor, "light": Qt.CrossCursor}

    def set_mode(self, mode):
        """The scene toolbar's Select / Camera / Light: what a left drag does in the view."""
        self.view.mode = mode
        self.view.setCursor(self.MODE_CURSORS[mode])
        self.mode_hint = self.MODE_HINTS[mode]
        if mode == "light" and not self.scene.lights.points[0].enabled:
            self.scene.lights.points[0].enabled = True  # visible (and its helper) as soon as it can be moved
            self._push_lights()
        self.frame_drawn(self.view.width(), self.view.height())  # show the hint now
        self.view.refresh()

    def _check(self, action, attr):
        action.setCheckable(True)
        action.setChecked(getattr(self.view.look, attr))
        action.toggled.connect(lambda on: self.set_setting(attr, on) if getattr(self.view.look, attr) != on else None)
        self.toggles.setdefault(attr, []).append(action)

    def _stat_check(self, action, key):
        action.setCheckable(True)
        action.toggled.connect(lambda on: self._set_stat(key, on))
        self.toggles.setdefault(f"stat.{key}", []).append(action)

    def _set_stat(self, key, on):
        self.stats[key] = on
        for a in self.toggles[f"stat.{key}"]:
            if a.isChecked() != on:
                a.setChecked(on)
        self.view.refresh()

    def fill_color_spaces(self):
        """Legacy: the renderer's own tone mapping and sRGB; OCIO: the config's displays and views
        (the image goes through the chosen view, tone mapping included)."""
        from trafilatura3d import colormanagement as cm
        self.cs.blockSignals(True)
        self.cs.clear()
        if cm.active():
            d0, v0 = cm.default_display_view()
            for d in cm.displays():
                for v in cm.views(d):
                    self.cs.addItem(f"{d} / {v}", (d, v))
            self.cs.setCurrentIndex(max(0, self.cs.findText(f"{d0} / {v0}")))
        else:
            self.cs.addItem("sRGB (default)", None)
        self.cs.blockSignals(False)
        self._apply_color()

    def _apply_color(self):
        from trafilatura3d import colormanagement as cm
        dv = self.cs.currentData()
        r = self.view.renderer
        if not cm.active() or dv is None:
            r.set_display_lut(None)
            if self.view.look.tonemap == TONEMAPS.index("OCIO view"):
                self.view.look.tonemap = Look.tonemap
        else:
            try:
                r.set_display_lut(cm.display_lut(cm.working_space(), *dv, scene_linear=True))
                self.view.look.tonemap = TONEMAPS.index("OCIO view")
            except Exception as e:  # noqa: BLE001 - a bad config falls back to Legacy and says why
                self.doc.error.emit(f"Color management: {e}")
                r.set_display_lut(None)
        self.view.refresh()

    def frame_drawn(self, w, h):
        r = self.view.renderer
        parts = [getattr(self, "mode_hint", "")] if getattr(self, "mode_hint", "") else []
        if self.stats["resolution"]:
            parts.append(f"{w} x {h}")
        if self.stats["scene_stats"]:
            verts = sum(len(o.pos) for o in self.scene.objects if o.visible)
            faces = sum(o.idx.size // 3 for o in self.scene.objects if o.visible)
            parts.append(f"{verts:,} vertices · {faces:,} faces · {len(self.scene.materials)} materials")
        if self.stats["render_time"]:
            parts.append(f"{r.last_draw_ms:.1f} ms")
        if self.stats["samples"]:
            if self.view.look.pathtracer and self.pathtracer is not None:
                parts.append(f"{self.pathtracer.samples} / {self.view.look.pt_samples} samples")
            else:
                parts.append("4 samples (MSAA)")
        self.info.setText("   ".join(parts))

    def _title(self):
        self.title_changed.emit(f"{self.scene.mesh_name} - {self.environment} - 3D View")

    # --- settings (Properties dock) ---------------------------------------------------------------

    def edit_settings(self, section, index=None):
        if section == "environment":
            title, rollouts = "Environment - Properties", ENVIRONMENT_ROWS
        elif section == "camera":
            title, rollouts = "Camera - Properties", CAMERA_ROWS
        elif section == "renderer" and self.view.look.pathtracer:
            title, rollouts = "GPU Pathtracer - Properties", PATHTRACER_ROWS
        elif section == "renderer":
            title, rollouts = "Rasterizer - Properties", RENDERER_ROWS
        elif section == "displacement":
            title, rollouts = "Displacement - Properties", DISPLACEMENT_ROWS
        elif section == "lights":
            title, rollouts = "Lights - Properties", self._light_rows()
        elif section == "scene":
            title, rollouts = "Scene - Properties", self._scene_rows()
        elif section == "mesh":
            title, rollouts = f"{self.scene.objects[index].name} - Properties", self._mesh_rows(index)
        elif section == "material":
            title, rollouts = f"{self.scene.materials[index].name} - Properties", self._material_rows(index)
        else:
            raise ValueError(section)
        self._editing = (section, index)
        self.settings_requested.emit(title, rollouts, self.setting, self.set_setting)

    def _light_rows(self):
        rows = [("Ambient light", [("Color", "lights.ambient_color", "color", None, None),
                                   ("Intensity", "lights.ambient_intensity", "float", 0.0, 2.0)])]
        for i in range(2):
            rows.append((f"Point light {i + 1}", [
                ("Enabled", f"point.{i}.enabled", "bool", None, None),
                ("Position", f"point.{i}.position", "vec3", -10.0, 10.0),
                ("Color", f"point.{i}.color", "color", None, None),
                ("Intensity", f"point.{i}.intensity", "float", 0.0, 100.0)]))
        rows.append(("Environment key light", [("Intensity", "light_intensity", "float", 0.0, 4.0)]))
        return rows

    def _scene_rows(self):
        return [("Meshes", [(o.name, f"visible.{i}", "bool", None, None) for i, o in enumerate(self.scene.objects)])]

    def _mesh_rows(self, index):
        o = self.scene.objects[index]
        names = tuple(m.name for m in self.scene.materials)
        rows = [("Visible", f"visible.{index}", "bool", None, None)]
        rows += [(f"Material (part {k + 1})" if len(o.submeshes) > 1 else "Material", f"submesh.{index}.{k}", "enum",
                  names, None) for k in range(len(o.submeshes))]
        return [("Mesh", rows)]

    def _material_rows(self, index):
        """The material's shader, its parameters grouped as the documentation lists them (shaders.TABLE), and
        where each of its channels comes from."""
        m = self.scene.materials[index]
        choices = self._source_choices()
        rollouts = {"Material": [("Name", f"mat.{index}.name", "text", None, None),
                                 ("Shader", f"mat.{index}.shader", "enum", SHADERS, None)]}
        for p in shaders.BY_NAME[m.shader].params:
            kind = {"float2": "vec3"}.get(p.kind, p.kind)
            a, b = (p.choices, None) if p.kind == "enum" else (p.lo, p.hi)
            rollouts.setdefault(p.group, []).append((p.label, f"par.{index}.{p.key}", kind, a, b))
        channels = [(ch, f"src.{index}.{ch}", "enum", tuple(label for label, _ in choices), None)
                    for ch in shaders.BY_NAME[m.shader].channels]
        if channels:
            rollouts["Channels"] = channels
        return list(rollouts.items())

    def _source_choices(self):
        """Channel sources for a material: by usage, nothing, the graph's outputs, its nodes."""
        out = [("Auto (by usage)", AUTO), ("None", NONE)]
        if self.package is not None and self.graph_id in self.package.graphs:
            g = self.package.graphs[self.graph_id]
            for node in g.output_nodes():
                ident = self.package.definition(node).values(node)["identifier"]
                out.append((f"Output: {ident}", f"out:{ident}"))
            out += [(f"Node: {nid}", nid) for nid in g.nodes]
        return out

    def setting(self, key):
        look, sc = self.view.look, self.scene
        if key == "environment":
            return ENVIRONMENTS.index(self.environment) if self.environment in ENVIRONMENTS else 0
        if key == "env_rotation":
            return math.degrees(look.env_rotation) % 360.0
        if key == "shadow_mode":
            return int(look.shadows)
        if key == "samples":
            return 0
        head, _, rest = key.partition(".")
        if head == "cam":
            return getattr(sc.camera, rest)
        if head == "lights":
            return getattr(sc.lights, rest)
        if head == "point":
            i, attr = rest.split(".")
            return getattr(sc.lights.points[int(i)], attr)
        if head == "visible":
            return sc.objects[int(rest)].visible
        if head == "submesh":
            o, k = map(int, rest.split("."))
            return sc.objects[o].submeshes[k].material
        if head == "mat":
            i, attr = rest.split(".")
            m = sc.materials[int(i)]
            if attr == "shader":
                return SHADERS.index(m.shader)
            return getattr(m, attr)
        if head == "par":
            i, k = rest.split(".")
            m = sc.materials[int(i)]
            p = next(p for p in shaders.BY_NAME[m.shader].params if p.key == k)
            if p.target[0] == "look":  # Height level / scale: the Displacement window's, for every material
                return getattr(look, p.target[1])
            v = m.value(k)
            return list(v) if isinstance(v, (tuple, list)) else v
        if head == "src":
            i, ch = rest.split(".")
            values = [v for _, v in self._source_choices()]
            src = sc.materials[int(i)].source(ch)
            return values.index(src) if src in values else 0
        return getattr(look, key)

    def use_template(self, template) -> None:
        """A new graph's template (shaders.Template): the first material takes its shader and blending mode, as the
        Materials menu and the Properties would set them."""
        self.set_setting("mat.0.shader", shaders.index(template.shader))
        if template.blending is not None:
            self.set_setting("par.0.blending", template.blending)

    def set_setting(self, key, value):
        look, sc = self.view.look, self.scene
        head, _, rest = key.partition(".")
        material_changed = False
        if key == "environment":
            self._environment(ENVIRONMENTS[value])
            return
        if key == "samples":
            return
        if key == "env_rotation":
            look.env_rotation = math.radians(value)
        elif key == "shadow_mode":
            self.set_setting("shadows", bool(value))
            return
        elif head == "cam":
            setattr(sc.camera, rest, value)
        elif head == "lights":
            setattr(sc.lights, rest, value)
            self._push_lights()
        elif head == "point":
            i, attr = rest.split(".")
            setattr(sc.lights.points[int(i)], attr, value)
            self._push_lights()
        elif head == "visible":
            sc.objects[int(rest)].visible = bool(value)
            self.view.renderer.set_visibility([o.visible for o in sc.objects])
            self.pt_dirty = True
            self._fill_browser()
        elif head == "submesh":
            o, k = map(int, rest.split("."))
            sc.objects[o].submeshes[k].material = int(value)
            self.view.renderer.set_meshes(sc.objects)
            self._fill_browser()
        elif head == "mat":
            i, attr = rest.split(".")
            m = sc.materials[int(i)]
            if attr == "shader":
                m.shader = SHADERS[value]
                self._reopen()
            else:
                setattr(m, attr, value)
            if attr == "name":
                self._fill_materials_menu()
                self._fill_browser()
            material_changed = True
        elif head == "par":
            i, k = rest.split(".")
            m = sc.materials[int(i)]
            p = next(p for p in shaders.BY_NAME[m.shader].params if p.key == k)
            if p.target[0] == "look":
                self.set_setting(p.target[1], value)
                return
            m.values[k] = value
            material_changed = True
        elif head == "src":
            i, ch = rest.split(".")
            sc.materials[int(i)].sources[ch] = self._source_choices()[value][1]
            material_changed = True
        else:
            setattr(look, key, value)
            for a in self.toggles.get(key, ()):
                if a.isChecked() != bool(value):
                    a.setChecked(bool(value))
            if key == "diagnostic" and self.aov.currentIndex() != value:
                self.aov.setCurrentIndex(value)
            if key == "channel" and self.channels.currentIndex() != value:
                self.channels.setCurrentIndex(value)
            if key in ("tessellation", "enable_displacement"):
                self._upload_meshes()
            if key in ("displacement", "height_level", "enable_displacement", "tessellation"):
                self.pt_dirty = True
            if key == "pathtracer" and self._editing and self._editing[0] == "renderer":
                self._reopen()
        if material_changed:
            self.refresh()
            return
        self.view.refresh()

    def _reopen(self):
        """Show the settings being edited again (rows that depend on a value, e.g. the shader)."""
        section = getattr(self, "_editing", None)
        if section:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.edit_settings(*section))

    # --- scene ------------------------------------------------------------------------------------

    def set_primitive(self, name):
        self.bake_resource = self.baked_preview = None
        self.scene.set_primitive(name)
        self._scene_changed()

    def load_mesh(self, path):
        try:
            self.scene.load(path)
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: {e}")
            return
        self.bake_resource = self.baked_preview = None
        self._scene_changed()

    def _load_mesh_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load 3D scene", "", MESH_FILTER)
        if path:
            self.load_mesh(path)

    def _scene_changed(self):
        self.selected = None
        self.view.renderer.selected = None
        self._upload_meshes()
        self.scene_cameras.clear()
        for c in self.scene.cameras:
            self.scene_cameras.addAction(c.name, lambda c=c: self._scene_camera(c))
        self.scene_cameras.setEnabled(bool(self.scene.cameras))
        self._fill_materials_menu()
        self._fill_browser()
        self._title()
        self.refresh()

    def _render_objects(self):
        """The meshes as the renderers draw them: subdivided for displacement, and with UVs on several
        UDIM tiles split by tile, material m of tile t becoming renderer material m * tiles + t (each tile
        shows its own baked maps; refresh() builds the materials in that order)."""
        objects = self.scene.objects
        if self.view.look.tessellation and self.view.look.enable_displacement:
            objects = [self._subdivided(o, self.view.look.tessellation) for o in objects]
        self.udim_tiles = bake.udim_tiles(objects) if not self.scene.source.startswith("primitive:") else [(0, 0)]
        if len(self.udim_tiles) > 1:
            objects = [split_by_tile(o, self.udim_tiles) for o in objects]
        return objects

    def _subdivided(self, o, levels):
        """`o` subdivided, kept for the next call (the pathtracer, a setting toggled back): only the geometry is kept,
        visibility and materials are read from `o` each time (they change in place)."""
        hit = self._subdivisions.get((id(o), levels))
        if hit is None or hit[0] is not o:
            hit = (o, _subdivided(o, levels))
            self._subdivisions = {k: v for k, v in self._subdivisions.items() if v[0] in self.scene.objects}
            self._subdivisions[(id(o), levels)] = hit
        sub = hit[1]
        sub.visible = o.visible
        for a, b in zip(sub.submeshes, o.submeshes):
            a.material = b.material
        return sub

    def _upload_meshes(self):
        self.view.renderer.set_meshes(self._render_objects())
        self.pt_dirty = True

    def _scene_camera(self, c):
        eye, target = np.array(c.eye, np.float32), np.array(c.target, np.float32)
        d = eye - target
        self.view.target = target
        self.view.distance = float(np.linalg.norm(d))
        self.view.pitch = math.asin(float(d[1]) / max(self.view.distance, 1e-6))
        self.view.yaw = math.atan2(float(d[0]), float(d[2]))
        self.scene.camera.fov = c.fov
        self.view.refresh()

    def reset_scene(self):
        self.view.reset_camera()
        self._environment("Studio")
        fresh = Look()
        for f in fields(Look):
            if f.name != "preview":
                self.set_setting(f.name, getattr(fresh, f.name))
        self.scene.lights, self.scene.camera = Lights(), CameraProps()
        self.scene.materials = [Material()]
        self._push_lights()
        self.set_primitive("Rounded Cube")

    def reset_lights(self):
        self.scene.lights = Lights()
        self.set_setting("light_intensity", Look.light_intensity)
        self._push_lights()

    def reset_environment(self):
        self._environment("Studio")
        for k in ENVIRONMENT_KEYS:
            self.set_setting(k, getattr(Look(), k))

    def reset_camera_full(self):
        self.view.reset_camera()
        self.scene.camera = CameraProps()
        self.set_setting("tonemap", Look.tonemap)

    def focus(self):
        if self.selected is not None:
            lo, hi = self.scene.objects[self.selected].bounds()
        else:
            lo, hi = self.scene.bounds()
        center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
        self.view.target = center.astype(np.float32)
        self.view.distance = max(radius / math.tan(math.radians(self.scene.camera.fov) / 2) * 1.1, 0.3)
        self.view.refresh()

    def _camera(self, yaw, pitch):
        self.view.yaw, self.view.pitch = yaw, pitch
        self.view.refresh()

    def key(self, key, mods):
        k = key.lower()
        if k == "f":
            self.focus()
        elif k == "r" and "Shift" in mods:
            self.reset_camera_full()
        elif k == "r":
            self.view.reset_camera()
        elif k == "s" and "Alt" in mods:
            self.save_render_dialog()
        elif k == "c" and "Alt" in mods:
            self.copy_render()
        elif key == "Escape":
            self.select(None)

    # --- selection and browser --------------------------------------------------------------------

    def pick(self, x, y):
        o, d = self.view.ray(x, y)
        self.select(self.scene.pick(o, d))

    def select(self, index):
        self.selected = index
        self.view.renderer.selected = index
        self.view.refresh()
        if index is not None:
            self.edit_settings("mesh", index)

    def context_menu(self):
        if self.selected is None:
            return
        i = self.selected
        menu = QMenu(self)
        menu.addAction("Edit", lambda: self.edit_settings("mesh", i))
        menu.addAction("Hide", lambda: self.set_setting(f"visible.{i}", False))
        menu.addAction("Show only", lambda: self._show_only(i))
        menu.addAction("Show all", self._show_all)
        mats = menu.addMenu("Material")
        for k, m in enumerate(self.scene.materials):
            mats.addAction(m.name, lambda k=k: [self.set_setting(f"submesh.{i}.{s}", k)
                                                for s in range(len(self.scene.objects[i].submeshes))])
        menu.addAction("Display UV's in 2D View", self._display_uvs)
        menu.addSeparator()
        menu.addAction("Unselect", lambda: self.select(None))
        menu.exec(QCursor.pos())

    def _show_only(self, i):
        for k in range(len(self.scene.objects)):
            self.scene.objects[k].visible = k == i
        self.view.renderer.set_visibility([o.visible for o in self.scene.objects])
        self._fill_browser()
        self.view.refresh()

    def _show_all(self):
        for o in self.scene.objects:
            o.visible = True
        self.view.renderer.set_visibility([True] * len(self.scene.objects))
        self._fill_browser()
        self.view.refresh()

    def _fill_browser(self):
        """Rebuild the scene browser once control returns to the event loop: rebuilding inside one
        of its own signals (a check box click) would delete the item being clicked."""
        from PySide6.QtCore import QTimer
        if not getattr(self, "_browser_pending", False):
            self._browser_pending = True
            QTimer.singleShot(0, self._rebuild_browser)

    def _rebuild_browser(self):
        """The scene browser: objects added by the view in bold (camera, environment, point lights),
        then the meshes with their eye (check box) and the connected materials."""
        self._browser_pending = False
        tree = self.browser
        tree.blockSignals(True)
        tree.clear()
        bold = tree.font()
        bold.setBold(True)

        def add(parent, text, data, checkable=None):
            it = QTreeWidgetItem([text])
            it.setData(0, Qt.UserRole, data)
            if checkable is not None:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if checkable else Qt.Unchecked)
            (parent.addChild(it) if parent is not None else tree.addTopLevelItem(it))
            return it
        for text, data in (("Camera", ("camera",)), ("Environment", ("environment",))):
            add(None, text, data).setFont(0, bold)
        for i, p in enumerate(self.scene.lights.points):
            if p.enabled:
                add(None, f"Point light {i + 1}", ("light", i)).setFont(0, bold)
        root = add(None, self.scene.mesh_name, ("scene",))
        for i, o in enumerate(self.scene.objects):
            add(root, o.name, ("mesh", i), o.visible)
        used = sorted({s.material for o in self.scene.objects for s in o.submeshes})
        mats = add(None, "Materials", ("materials",))
        for i, m in enumerate(self.scene.materials):
            add(mats, m.name + ("" if i in used else "  (unused)"), ("material", i))
        tree.expandAll()
        tree.blockSignals(False)

    def _browser_clicked(self, item, _col):
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if data[0] == "camera":
            self.edit_settings("camera")
        elif data[0] == "environment":
            self.edit_settings("environment")
        elif data[0] == "light":
            self.edit_settings("lights")
        elif data[0] == "scene":
            self.edit_settings("scene")
        elif data[0] == "mesh":
            self.select(data[1])
        elif data[0] == "material":
            self.edit_settings("material", data[1])

    def _browser_changed(self, item, _col):
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if data[0] == "mesh":
            self.set_setting(f"visible.{data[1]}", item.checkState(0) == Qt.Checked)

    def _browser_menu(self, pos):
        item = self.browser.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        menu = QMenu(self)
        if data[0] == "mesh":
            i = data[1]
            menu.addAction("Hide", lambda: self.set_setting(f"visible.{i}", False))
            menu.addAction("Show", lambda: self.set_setting(f"visible.{i}", True))
            menu.addAction("Show only", lambda: self._show_only(i))
            menu.addAction("Show all", self._show_all)
        elif data[0] == "scene":
            menu.addAction("Hide children", lambda: [self.set_setting(f"visible.{i}", False)
                                                     for i in range(len(self.scene.objects))])
            menu.addAction("Show children", self._show_all)
            menu.addAction("Expand all children", self.browser.expandAll)
            menu.addAction("Collapse all children", self.browser.collapseAll)
        elif data[0] == "material":
            self._material_actions(menu, data[1])
        elif data[0] == "light":
            menu.addAction("Hide", lambda: self.set_setting(f"point.{data[1]}.enabled", False) or self._fill_browser())
        elif data[0] == "environment":
            menu.addAction("Hide", lambda: self.set_setting("show_env", False))
            menu.addAction("Show", lambda: self.set_setting("show_env", True))
        if not menu.isEmpty():
            menu.exec(self.browser.viewport().mapToGlobal(pos))

    # --- materials --------------------------------------------------------------------------------

    def _fill_materials_menu(self):
        menu = self.materials_menu
        menu.clear()
        for i, m in enumerate(self.scene.materials):
            self._material_actions(menu.addMenu(m.name), i)
        menu.addSeparator()
        menu.addAction("Add", self._add_material)
        reset = menu.addAction("Reset and view outputs", lambda: self.show_graph(self.package, self.graph_id, reset=True))
        reset.setEnabled(bool(self.overrides) or self.preview_source is not None)

    def _material_actions(self, menu, i):
        m = self.scene.materials[i]
        menu.addAction("Edit", lambda: self.edit_settings("material", i))
        shaders = menu.addMenu("Shaders")
        group = QActionGroup(shaders)
        for k, name in enumerate(SHADERS):
            a = shaders.addAction(name, lambda k=k: self.set_setting(f"mat.{i}.shader", k))
            a.setCheckable(True)
            a.setChecked(name == m.shader)
            group.addAction(a)
        a = menu.addAction("Load Definition...")
        a.setEnabled(False)
        a.setToolTip("GLSLFX shaders are for a deprecated OpenGL renderer")
        menu.addAction("Reset common parameters", lambda: self._reset_material(i, common_only=True))
        menu.addAction("Rename", lambda: self._rename_material(i))
        menu.addAction("Reset material", lambda: self._reset_material(i))
        r = menu.addAction("Reset material to scene state", lambda: self._reset_material_to_scene(i))
        r.setEnabled(i < len(self.scene.scene_materials))

    def _add_material(self):
        self.scene.add_material()
        self._fill_materials_menu()
        self._fill_browser()
        self.refresh()

    def _rename_material(self, i):
        name, ok = QInputDialog.getText(self, "Rename material", "Label", text=self.scene.materials[i].name)
        if ok and name:
            self.set_setting(f"mat.{i}.name", name)

    def _reset_material(self, i, common_only=False):
        m, fresh = self.scene.materials[i], Material(name=self.scene.materials[i].name)
        if common_only:  # the parameters back to the shader's defaults, the channels' sources kept
            m.values = {}
        else:
            self.scene.materials[i] = fresh
        self._fill_materials_menu()
        self.refresh()

    def _reset_material_to_scene(self, i):
        import copy
        self.scene.materials[i] = copy.deepcopy(self.scene.scene_materials[i])
        self._fill_materials_menu()
        self._fill_browser()
        self.refresh()

    # --- lights -----------------------------------------------------------------------------------

    def _push_lights(self):
        L = self.scene.lights
        r = self.view.renderer
        r.ambient = tuple(float(c) * L.ambient_intensity for c in L.ambient_color)
        r.point_lights = [(p.position, p.color, p.intensity) if p.enabled else None for p in L.points]
        r.point_lights = [p for p in r.point_lights if p is not None]
        self._fill_browser()
        self.view.refresh()

    def orbit_light(self, dyaw, dpitch, scale):
        p = self.scene.lights.points[0]
        p.enabled = True
        x, y, z = p.position
        dist = math.sqrt(x * x + y * y + z * z) or 1.0
        yaw, pitch = math.atan2(x, z) - dyaw, math.asin(max(-1.0, min(1.0, y / dist))) + dpitch
        pitch = max(-1.5, min(1.5, pitch))
        dist = max(0.2, dist * scale)
        p.position = [dist * math.cos(pitch) * math.sin(yaw), dist * math.sin(pitch), dist * math.cos(pitch) * math.cos(yaw)]
        self._push_lights()

    # --- environment ------------------------------------------------------------------------------

    def _environment(self, name):
        """Studio (procedural) or a daylight panorama cooked by the Physical Sun/Sky node."""
        from trafilatura3d.sample import sample_job
        r = self.view.renderer
        try:
            if name == "Studio":
                r.set_studio()
            else:
                pos, tint = SUN_SKY[name]
                job = sample_job("physical_sun_sky", size_log2=10, params={"sun_position": list(pos)})
                r.set_environment(self.doc.ev.result(job, "output"), uv_to_dir(*pos), tint, 3.0)
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: {e}")
            return
        self.environment = name
        self.env_version += 1
        self._title()
        self.view.refresh()

    def _load_hdri(self, path=None):
        import OpenImageIO as oiio
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load HDRI", "", "HDR images (*.hdr *.exr *.tif *.tiff)")
        if not path:
            return
        buf = oiio.ImageBuf(path)
        if buf.has_error:
            self.doc.error.emit(f"3D view: {buf.geterror()}")
            return
        spec = buf.spec()
        if spec.width > HDRI_MAX_WIDTH:
            w = HDRI_MAX_WIDTH
            buf = oiio.ImageBufAlgo.resize(buf, roi=oiio.ROI(0, w, 0, max(1, spec.height * w // spec.width), 0, 1,
                                                             0, spec.nchannels))
        self.view.renderer.set_environment_image(np.asarray(buf.get_pixels(oiio.FLOAT), np.float32))
        self.environment = path.rsplit("/", 1)[-1]
        self.hdri_path = path
        self.env_version += 1
        self._title()
        self.view.refresh()

    # --- GPU Pathtracer ---------------------------------------------------------------------------

    def pathtrace(self, target, eye, view_proj, linear=False, samples=None):
        """Accumulate path traced samples into `target`; returns the samples done so far."""
        from trafilatura3d.editor.pathtrace import PathTracer
        look = self.view.look
        if self.pathtracer is None:
            self.pathtracer = PathTracer(self.view.renderer)
        if self.pt_dirty:
            self.pathtracer.set_scene(self._render_objects(), look)
            self.pt_dirty = False
        refl = (look.ground_reflections, look.reflections_opacity, look.reflections_roughness)
        return self.pathtracer.draw(target, eye, view_proj, look, self.view.renderer.point_lights, self.env_version,
                                    samples or look.pt_samples, linear=linear, spp=samples or 1,
                                    max_bounces=look.pt_bounces, reflections=refl,
                                    local_lights=look.ground_local_lights)

    # --- renders ----------------------------------------------------------------------------------

    def configure(self, mesh=None, viewpoint=None, yaw=None, pitch=None, distance=None, environment=None,
                  env_rotation=None, env_intensity=None, key_light=None, light=None) -> dict:
        """Set the view as its menus and settings do (for scripts and the remote bridge): `mesh` a primitive
        (Scene menu), `viewpoint` a Camera > Viewpoints name, `yaw`/`pitch` in degrees and `distance` (meters from
        the target) to orbit the camera, `environment` a name of the Environment menu or an HDRI file, its rotation
        (degrees: turns the key light too) and intensity, `key_light` the key light's intensity, `light` point light
        1 as {azimuth, elevation (degrees; low = grazing), distance, intensity, color, enabled}. Returns the state."""
        if mesh is not None:
            if mesh not in PRIMITIVES:
                raise ValueError(f"unknown mesh {mesh!r} (meshes: {', '.join(PRIMITIVES)})")
            self.set_primitive(mesh)
        if viewpoint is not None:
            if viewpoint not in VIEWPOINTS:
                raise ValueError(f"unknown viewpoint {viewpoint!r} (viewpoints: {', '.join(VIEWPOINTS)})")
            self._camera(*VIEWPOINTS[viewpoint])
        v = self.view
        if yaw is not None or pitch is not None:
            self._camera(math.radians(yaw) if yaw is not None else v.yaw,
                         max(-1.49, min(1.49, math.radians(pitch))) if pitch is not None else v.pitch)
        if distance is not None:
            v.distance = max(0.3, float(distance))
        if environment is not None:
            if environment in ENVIRONMENTS:
                self._environment(environment)
            elif Path(environment).is_file():
                self._load_hdri(str(environment))
            else:
                raise ValueError(f"unknown environment {environment!r} (environments: {', '.join(ENVIRONMENTS)}, "
                                 f"or an HDRI file)")
        if env_rotation is not None:
            self.set_setting("env_rotation", float(env_rotation))
        if env_intensity is not None:
            self.set_setting("env_intensity", max(0.0, float(env_intensity)))
        if key_light is not None:
            self.set_setting("light_intensity", max(0.0, float(key_light)))
        if light is not None:
            p = self.scene.lights.points[0]
            unknown = set(light) - {"azimuth", "elevation", "distance", "intensity", "color", "enabled"}
            if unknown:
                raise ValueError(f"light: unknown {', '.join(sorted(unknown))} (azimuth, elevation, distance, "
                                 f"intensity, color, enabled)")
            x, y, z = p.position
            dist = math.sqrt(x * x + y * y + z * z) or 1.0
            az = math.radians(light.get("azimuth", math.degrees(math.atan2(x, z))))
            el = math.radians(light.get("elevation", math.degrees(math.asin(max(-1.0, min(1.0, y / dist))))))
            dist = float(light.get("distance", dist))
            self.set_setting("point.0.position", [dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                                                  dist * math.cos(el) * math.cos(az)])
            for key in ("intensity", "color"):
                if key in light:
                    self.set_setting(f"point.0.{key}", light[key])
            self.set_setting("point.0.enabled", bool(light.get("enabled", True)))
        p = self.scene.lights.points[0]
        return {"mesh": self.scene.mesh_name, "yaw": round(math.degrees(v.yaw), 1),
                "pitch": round(math.degrees(v.pitch), 1), "distance": round(float(v.distance), 2),
                "environment": self.environment, "env_rotation": round(self.setting("env_rotation"), 1),
                "env_intensity": v.look.env_intensity, "key_light": v.look.light_intensity,
                "light": {"enabled": p.enabled, "position": [round(c, 2) for c in p.position],
                          "intensity": p.intensity, "color": p.color}}

    def render_image(self, width, height, hdr=False) -> np.ndarray:
        """The current view at a resolution: (h, w, 4) uint8, or float32 linear when `hdr`."""
        fmt = "rgba16float" if hdr else "rgba8unorm"
        t = self.doc.ev.device.create_texture(size=(width, height, 1), format=fmt,
                                              usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        v = self.view
        if v.look.pathtracer:
            if self.pathtracer is not None:
                self.pathtracer.key = None  # a fresh accumulation at this resolution
            self.pathtrace(t, v.eye(), v.view_proj(width, height), linear=hdr, samples=v.look.pt_samples)
            self.pathtracer.key = None
        else:
            v.renderer.draw(t, v.eye(), v.view_proj(width, height), v.look, linear=hdr)
        bpp = 8 if hdr else 4
        data = self.doc.ev.device.queue.read_texture({"texture": t}, {"bytes_per_row": width * bpp}, (width, height, 1))
        img = np.frombuffer(data, np.float16 if hdr else np.uint8).reshape(height, width, 4)
        return img.astype(np.float32) if hdr else img.copy()

    def save_render_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save render", "render.png", RENDER_FILTER)
        if path:
            self.save_render(path)

    def save_render(self, path):
        import OpenImageIO as oiio
        cam = self.scene.camera
        hdr = path.lower().endswith(".exr")
        img = self.render_image(cam.render_width, cam.render_height, hdr)
        h, w = img.shape[:2]
        buf = oiio.ImageBuf(oiio.ImageSpec(w, h, 4, oiio.HALF if hdr else oiio.UINT8))
        buf.set_pixels(oiio.ROI(), np.ascontiguousarray(img))
        if not buf.write(path):
            self.doc.error.emit(f"3D view: {buf.geterror()}")

    def copy_render(self):
        cam = self.scene.camera
        img = self.render_image(cam.render_width, cam.render_height)
        h, w = img.shape[:2]
        q = QImage(img.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        QApplication.clipboard().setImage(q)

    # --- state files and resources ----------------------------------------------------------------

    def state(self):
        data = self.scene.state(self.view)
        data["environment"] = self.environment if self.environment in ENVIRONMENTS else getattr(self, "hdri_path", "Studio")
        data["stats"] = dict(self.stats)
        return data

    def apply_state(self, data, with_mesh=False):
        self.scene.apply_state(data, self.view, with_mesh)
        env = data.get("environment", "Studio")
        self._environment(env) if env in ENVIRONMENTS else self._load_hdri(env)
        for k, v in data.get("stats", {}).items():
            if k in self.stats:
                self._set_stat(k, v)
        for attr, actions in self.toggles.items():
            if not attr.startswith("stat.") and hasattr(self.view.look, attr):
                for a in actions:
                    a.setChecked(bool(getattr(self.view.look, attr)))
        self.aov.setCurrentIndex(self.view.look.diagnostic)
        self.channels.setCurrentIndex(self.view.look.channel)
        self._push_lights()
        self._scene_changed()

    def load_state(self, path, with_mesh=False):
        try:
            self.apply_state(json.loads(open(path, encoding="utf-8").read()), with_mesh)
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: {e}")

    def _load_state_dialog(self, with_mesh):
        path, _ = QFileDialog.getOpenFileName(self, "Load state file", "", f"3D view state (*{STATE_SUFFIX})")
        if path:
            self.load_state(path, with_mesh)

    def _save_state_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save state file", f"scene{STATE_SUFFIX}",
                                              f"3D view state (*{STATE_SUFFIX})")
        if path:
            open(path, "w", encoding="utf-8").write(json.dumps(self.state(), indent=1))

    def save_default_state(self):
        QSettings().setValue(DEFAULT_STATE_KEY, json.dumps(self.state()))

    def _apply_default_state(self):
        raw = QSettings().value(DEFAULT_STATE_KEY)
        if raw:
            try:
                self.apply_state(json.loads(raw), with_mesh=True)
            except Exception:  # noqa: BLE001 - a stale default is ignored
                pass

    def create_resource(self):
        """A 3D scene resource in the current package: the mesh source and the whole state."""
        from trafilatura3d.package import Resource
        pkg = self.package
        if pkg is None:
            self.doc.error.emit("3D view: open a graph of the package that will hold the 3D scene resource")
            return
        name, ok = QInputDialog.getText(self, "Create 3D resource", "Identifier", text="scene")
        if not ok or not name:
            return
        source = self.scene.source
        if not source.startswith("primitive:") and pkg.path is not None:
            import os
            source = os.path.relpath(source, pkg.path.parent).replace(os.sep, "/")
        state = self.state()
        state["source"] = source
        self.doc.edit(pkg, "Create 3D resource", lambda: pkg.resources.__setitem__(name, Resource("scene", source, state)))

    def load_resource(self, pkg, rid):
        """A 3D scene resource: with a saved state (Create 3D resource) the whole scene, a linked mesh
        alone into the current scene settings."""
        r = pkg.resources[rid]
        if not r.path.startswith("primitive:") and pkg.path is not None:
            source = str((pkg.path.parent / r.path).resolve())
        else:
            source = r.path
        self.bake_resource, self.baked_preview, self._overlay_key = (pkg, rid), None, None
        if r.state:
            self.apply_state({**r.state, "source": source}, with_mesh=True)
            return
        try:
            if source.startswith("primitive:"):
                self.scene.set_primitive(source.split(":", 1)[1])
            else:
                self.scene.load(source)
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: {e}")
            return
        self._scene_changed()

    # --- baking: the resource's baked maps and bake setup -----------------------------------------

    def _doc_changed(self, pkg):
        """The graph shown is refreshed by the Graph workspace's `LiveCook` (its steps call `refresh`)."""
        if self.bake_resource is None or pkg is not self.bake_resource[0]:
            return
        if pkg is not self.package:
            self.refresh()  # baking settings or a new bake: maps and overlay follow
            return
        try:  # the overlay at once (the cage follows a dragged distance), the maps with the refresh
            self._update_bake_overlay()
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: baked maps: {e}")
        self.view.refresh()

    def _bake_config(self):
        """(package, resource, settings, low mesh path) of the scene resource on view, or None."""
        if self.bake_resource is None:
            return None
        pkg, rid = self.bake_resource
        r = pkg.resources.get(rid)
        if r is None or r.kind != "scene" or r.path.startswith("primitive:") or pkg.path is None:
            return None
        from trafilatura3d.editor.bakers import settings_from
        return pkg, r, settings_from((r.baking or {}).get("settings")), (pkg.path.parent / r.path).resolve()

    def _baked_files(self, tile=None) -> dict:
        """Baker -> the file it last wrote for this resource, for a UDIM tile (default: the first)."""
        res = self._bake_config()
        if res is None or not res[1].baking:
            return {}
        pkg, r, settings, low = res
        cfg = r.baking
        tiles = self.udim_tiles
        out = {}
        for b in cfg.get("bakers", []):
            if b["id"] == "convert_uv_to_svg":
                continue
            path = bake.output_path(low, b["id"], settings, cfg.get("folder", ""), cfg.get("name", bake.NAME_TEMPLATE),
                                    tile or tiles[0], len(tiles) > 1)
            if path.exists():
                out[b["id"]] = path
        return out

    def _baked_texture(self, path, flip_green=False):
        from trafilatura3d.nodes.atomic.raster import _read_image

        def read():
            img = _read_image(str(path))
            if flip_green:
                img[..., 1] = 1.0 - img[..., 1]
            return img
        return self.doc.ev.upload(f"baked:{path}:{path.stat().st_mtime}:{int(flip_green)}", read)

    def _baked_slots(self, tile=None) -> dict:
        """Material slot -> (texture, gray) from the baked maps of a UDIM tile: the low shows its bake where
        the graph gives the channel nothing (mesh maps on the model)."""
        files = self._baked_files(tile)
        res = self._bake_config()
        out = {}
        normal = files.get("normal_map_from_mesh")
        params = {b["id"]: bake.resolved(b["id"], b.get("params")) for b in (res[1].baking or {}).get("bakers", [])} \
            if res else {}
        if normal is not None and params["normal_map_from_mesh"]["map_type"] == "Tangent Space":
            dx = params["normal_map_from_mesh"]["normal_orientation"] == "DirectX"
            out["normal"] = (self._baked_texture(normal, flip_green=dx == self.view.look.opengl_normals), False)
        ao = files.get("ambient_occlusion_from_mesh") or files.get("ambient_occlusion")
        if ao is not None:
            out["ambientOcclusion"] = (self._baked_texture(ao), True)
        return out

    def _show_baked(self, baker):
        self.baked_preview = baker
        self.refresh()

    def _set_bake_look(self, attr, value):
        setattr(self.view.look, attr, value)
        self._update_bake_overlay()
        self.view.refresh()

    def _open_baker(self):
        from trafilatura3d.editor.bakers import open_baker
        pkg, rid = self.bake_resource
        open_baker(self.doc, pkg, rid, self)

    def _update_bake_overlay(self):
        """The high poly, the cage (from the file, or the implicit one) and the rear distance surface in
        the view's space, with the high poly's vertices no bake ray reaches flagged; rebuilt when the bake
        setup changes (live while the Baking window is edited)."""
        look, renderer = self.view.look, self.view.renderer
        res = self._bake_config()
        if res is None or not (look.show_high or look.show_cage or look.show_rear):
            if res is None:
                renderer.set_bake_overlay([], [], [], [])
                self._overlay_key = None
            return
        pkg, r, s, low_path = res
        paths = [p for p in ([] if s.use_low_as_high else s.high_meshes)]
        cage_path = s.cage_file if s.use_cage and s.cage_file else ""
        mtimes = tuple(Path(p).stat().st_mtime if Path(p).exists() else 0 for p in [*paths, cage_path] if p)
        key = (str(low_path), tuple(paths), cage_path, mtimes, s.frontal, s.rear, s.relative_to_bounding_box,
               s.average_normals, tuple(np.round(self.scene.fit[0], 6)), self.scene.fit[1])
        if key == self._overlay_key:
            return
        self._overlay_key = key
        try:
            high = [o for p in paths for o in self._fitted(p)] if paths else [_copy_mesh(o) for o in self.scene.objects]
            # Distances in the view's space: scene units scale with the fit, relative ones don't change.
            k = 1.0 if s.relative_to_bounding_box else self.scene.fit[1]
            fitted = dataclasses.replace(s, frontal=s.frontal * k, rear=s.rear * k)
            low = self.scene.objects
            cage_file = self._fitted(cage_path) if cage_path else None
            cage = cage_file or bake.implicit_cage(low, fitted)
            rear = [] if cage_file else bake.rear_surface(low, fitted)
            outside = bake.unreached(self.doc.ev.device, high, low, fitted, cage_file)
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: bake setup: {e}")
            renderer.set_bake_overlay([], [], [], [])
            return
        renderer.set_bake_overlay(high, outside, cage, rear)

    def _fitted(self, path):
        """A mesh file of the bake in the view's space (the low's fit transform)."""
        from trafilatura3d import meshio
        p = Path(path)
        key = (str(p), p.stat().st_mtime)
        if key not in self._bake_meshes:
            self._bake_meshes = {key: meshio.load(p)[0], **{k: v for k, v in self._bake_meshes.items() if k[0] != str(p)}}
        objects = [_copy_mesh(o) for o in self._bake_meshes[key]]
        meshio.apply_fit(objects, self.scene.fit)
        return objects

    # --- export -----------------------------------------------------------------------------------

    def _export_dialog(self):
        filt = ";;".join(f"{f.upper()} (*.{f})" for f in EXPORT_FORMATS)
        path, _ = QFileDialog.getSaveFileName(self, "Export scene", "scene.glb", filt)
        if path:
            self.export(path)

    def export(self, path):
        from trafilatura3d.editor.scene3d import export_scene
        textures = {}
        for i in range(len(self.scene.materials)):
            tex = {}
            for slot in ("baseColor", "normal", "roughness", "metallic", "ambientOcclusion", "emissive", "opacity"):
                # With UDIM tiles the renderer has a material per tile: the export carries the first one's.
                r, index = self.view.renderer, i * len(self.udim_tiles)
                if r.material_texture(index, slot).size[0] > 1:
                    tex[slot] = (np.clip(r.read_material(index, slot), 0, 1) * 255 + 0.5).astype(np.uint8)
            textures[i] = tex
        try:
            export_scene(path, self.scene.objects, self.scene.materials, textures)
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: {e}")

    # --- UVs in the 2D View -----------------------------------------------------------------------

    def _display_uvs(self):
        i = self.selected if self.selected is not None else 0
        if not self.scene.objects:
            return
        o = self.scene.objects[i]
        self.show_uvs.emit((o.uv, o.triangles))

    # --- graph outputs on the materials -----------------------------------------------------------

    def show_graph(self, package, graph_id, reset=False):
        """Follow a graph: its outputs reach the materials by usage (applied when a
        graph opens). `reset` drops per-node overrides and previews."""
        if reset or (package, graph_id) != (self.package, self.graph_id):
            self.overrides, self.preview_source = {}, None
        self.package, self.graph_id = package, graph_id
        self.refresh()

    def show_node(self, package, graph_id, node_id, slot=None):
        """A node's output in the 3D view: alone and unlit (slot None, Preview) or on one channel of
        the materials (the others stay from the graph's outputs)."""
        if (package, graph_id) != (self.package, self.graph_id):
            self.package, self.graph_id, self.overrides = package, graph_id, {}
        if slot is None:
            self.preview_source = node_id
        else:
            self.preview_source = None
            self.overrides[slot] = node_id
        self.refresh()

    def _resolve(self, compiled, graph, material, channel):
        src = material.source(channel)
        if src == NONE:
            return None
        if src.startswith("out:"):
            return compiled.outputs.get(src[4:])
        if src != AUTO:
            outs = compiled.nodes.get(src)
            return next(iter(outs.values())) if outs else None
        outputs = [self.package.definition(node).values(node) for node in graph.output_nodes()]
        outputs = [v for v in outputs if v["identifier"] in compiled.outputs]
        for v in outputs:  # by usage, legacy usages matched
            if v["usage"] and shaders.channel_of(material.shader, v["usage"]) == channel:
                return compiled.outputs[v["identifier"]]
        for v in outputs:  # outputs without a usage, by identifier
            if not v["usage"] and v["identifier"].lower() in NAME_HINTS.get(channel, ()):
                return compiled.outputs[v["identifier"]]
        return None

    def sources(self) -> list:
        """The (job, output) of every material channel shown: what `refresh` cooks (`LiveCook` steps)."""
        return [src for _, assigned in self._materials()[0] for src in assigned.values()]

    def _materials(self) -> tuple[list, bool]:
        """(spec, slot -> (job, output)) per scene material, of what the graph shows (`Document.shown`), and whether
        a node is shown alone, unlit (Preview source)."""
        unlit = False
        compiled = graph = None
        if self.package is not None and self.graph_id in self.package.graphs:
            compiled = self.doc.shown(self.package, self.graph_id)
            graph = self.package.graphs[self.graph_id]
        materials = []
        for m in self.scene.materials:
            spec = m.spec()
            assigned = {}
            if compiled is not None:
                for ch, slot in shaders.BY_NAME[m.shader].channels.items():
                    found = self._resolve(compiled, graph, m, ch)
                    if found is not None:
                        assigned[slot] = found
                for slot, node_id in self.overrides.items():
                    outs = compiled.nodes.get(node_id)
                    if outs:
                        assigned[slot] = next(iter(outs.values()))
                if self.preview_source in compiled.nodes and compiled.nodes[self.preview_source]:
                    assigned = {"baseColor": next(iter(compiled.nodes[self.preview_source].values()))}
                    spec["shader"] = shaders.index("Unlit")
                    unlit = True
            materials.append((spec, assigned))
        return materials, unlit

    def show_ready(self, job) -> None:
        """The channels `job` gives shown now, the others as they are (a refresh in steps: each channel as soon as it
        is cooked; `refresh` makes the whole view after them). Only on the materials the view already has."""
        r, tiles = self.view.renderer, self.udim_tiles
        materials, _ = self._materials()
        if len(r.materials) != len(materials) * len(tiles) or job.key not in self.doc.ev.cache:
            return
        for m, (_spec, assigned) in enumerate(materials):
            slots = {slot: (self.doc.ev.texture(j, out), j.is_gray(out)) for slot, (j, out) in assigned.items()
                     if j.key == job.key and self._shown.get((m, slot)) != (j.key, out)}
            for t in range(len(tiles)) if slots else ():
                r.update_slots(m * len(tiles) + t, slots)
            self._shown.update({(m, slot): (job.key, assigned[slot][1]) for slot in slots})
        self.view.refresh()

    def shows_all(self) -> bool:
        """Every channel the view should show is the one it shows (`show_ready` brought them all): no refresh. With
        baked maps on the view, never (a bake changes them, not the graph's channels)."""
        if self.bake_resource is not None or self.baked_preview:
            return False
        materials, unlit = self._materials()
        if len(self.view.renderer.materials) != len(materials) * len(self.udim_tiles) or unlit:
            return False
        want = {(m, slot): (job.key, out) for m, (_, assigned) in enumerate(materials)
                for slot, (job, out) in assigned.items()}
        return want == self._shown

    def refresh(self):
        tiles = self.udim_tiles
        try:  # per UDIM tile: the baked maps on the low, or the one shown alone
            baked = [self._baked_slots(t) for t in tiles]
            preview = [self._baked_files(t).get(self.baked_preview) if self.baked_preview else None for t in tiles]
            self._update_bake_overlay()
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: baked maps: {e}")
            baked, preview = [{} for _ in tiles], [None for _ in tiles]
        materials, self.view.look.preview = self._materials()
        self._shown = {(m, slot): (job.key, out) for m, (_, assigned) in enumerate(materials)
                       for slot, (job, out) in assigned.items()}
        try:  # every material's channels cooked together: one by one, a cook could give back the ones before
            textures = iter(self.doc.ev.results([src for _, assigned in materials for src in assigned.values()]))
            for spec, assigned in materials:
                for slot, (job, out) in assigned.items():
                    spec[slot] = (next(textures), job.is_gray(out))
        except Exception as e:  # noqa: BLE001 - shown in the status bar
            self.doc.error.emit(f"3D view: {e}")
        specs = []
        for spec, _ in materials:
            for t in range(len(tiles)):  # renderer material m * tiles + t (_render_objects)
                tile_spec = dict(spec)
                if not self.view.look.preview:
                    for slot, value in baked[t].items():  # the bake fills what the graph leaves empty
                        tile_spec.setdefault(slot, value)
                if preview[t] is not None:  # Baking > Show baked map: that map alone, unlit
                    tile_spec = {"shader": shaders.index("Unlit"),
                                 "baseColor": (self._baked_texture(preview[t]), False)}
                    self.view.look.preview = True
                specs.append(tile_spec)
        self.view.renderer.set_materials(specs)
        self.pt_dirty = True
        self._fill_materials_menu()
        self.view.refresh()


def _copy_mesh(o):
    """A mesh with its own positions (the bake overlay moves them into the view's space)."""
    return dataclasses.replace(o, pos=o.pos.copy())


def _subdivided(o, levels):
    """A mesh subdivided `levels` times (each triangle into four) for displacement."""
    import trimesh
    from trafilatura3d.editor.scene3d import MeshObject, Submesh
    pos = o.pos
    attrs = np.concatenate([o.nrm, o.tan, o.uv], 1)
    subs = []
    out_faces, out_attr, out_pos = [], [], []
    base = 0
    for s in o.submeshes:
        used, faces = np.unique(o.idx[s.start:s.start + s.count], return_inverse=True)  # only this part's vertices
        v, f = pos[used], faces.reshape(-1, 3)
        a = attrs[used]
        for _ in range(levels):
            v, f, extra = trimesh.remesh.subdivide(v, f, vertex_attributes={"a": a})
            a = extra["a"]
        out_pos.append(v)
        out_attr.append(a)
        out_faces.append(f + base)
        subs.append(Submesh(sum(len(x) for x in out_faces[:-1]) * 3, len(f) * 3, s.material))
        base += len(v)
    pos = np.concatenate(out_pos).astype(np.float32)
    a = np.concatenate(out_attr).astype(np.float32)
    nrm = a[:, :3] / np.maximum(np.linalg.norm(a[:, :3], axis=1, keepdims=True), 1e-9)
    tan = a[:, 3:7]
    tan[:, 3] = np.where(tan[:, 3] < 0, -1.0, 1.0)
    return MeshObject(o.name, pos, nrm, tan, a[:, 7:9], np.concatenate(out_faces).reshape(-1).astype(np.uint32), subs,
                      o.visible)

