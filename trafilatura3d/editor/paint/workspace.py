"""The Paint workspace: the painting main window inside Trafilatura 3D (decision 23), shown while a Paint project is
open. Layout after the application (Interface, Toolbars, Viewport pages and screenshots): the Tools toolbar
on the left, the Assets dock, the viewports in the middle with the contextual toolbar on top (brush sliders on
the left; viewport mode, perspective and camera rotation on the right), the Texture Set list, Layer stack /
Texture Set settings and Properties docks on the right, the Docks toolbar on the right edge (Display settings,
History, Log and the others). The main menu (File, Edit, Window, Viewport, Help) replaces the Graph one while the
workspace is shown.

Tools (Shortcuts page): 1 Paint, 2 Eraser. Viewport: F1 3D/2D, F2 3D only, F3 2D only, F4 Swap 3D/2D, F5
perspective, F6 orthographic, F show entire mesh. Tools only work on a Paint layer or a mask (Toolbars page).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QSplitter, QToolBar, QToolButton, QUndoView, QVBoxLayout,
                               QWidget)

from trafilatura3d import shaders
from trafilatura3d.editor import theme
from trafilatura3d.editor.keys import variants
from trafilatura3d.editor.live import PREVIEW_SIZE, LiveCook, Plan
from trafilatura3d.editor.paint.assets import AssetsPanel
from trafilatura3d.editor.paint.display import DisplaySettings
from trafilatura3d.editor.paint.document import PaintDocument
from trafilatura3d.editor.paint.icons import picon
from trafilatura3d.editor.paint.layers import LayerStackPanel
from trafilatura3d.editor.paint.properties import PropertiesPanel, ToolState
from trafilatura3d.editor.paint.texture_sets import TextureSetList, TextureSetSettings
from trafilatura3d.editor.paint.viewport import PaintView2D, PaintView3D, circle_lines
from trafilatura3d.editor.paint.widgets import PressureButton, PSlider
from trafilatura3d.paint.brush import TILING_MODES, Stroke, Target, TileTargets, stamp_radius
from trafilatura3d.paint.channels import CHANNELS, FORMATS, render_slots
from trafilatura3d.paint.model import MESH_MAPS, PROJECT_SUFFIX, ProjectError, local_uv, udim_at
from trafilatura3d.paint.paint import MASK, storage_for
from trafilatura3d.paint.session import IMAGE_FILTER, Session

PROJECT_FILTER = f"Paint project (*{PROJECT_SUFFIX})"
OPEN_FILTER = f"Paint projects (*{PROJECT_SUFFIX} *.spp);;{PROJECT_FILTER};;.spp project (*.spp)"
VIEWPORT_MODES = ("3D/2D", "3D only", "2D only", "Swap 3D/2D")
LAYOUT_KEY = "paint/workspace/3"  # bump when the default layout changes
# Ours: while painting, every slot depending on the painting updates unless Fast painting (General preferences) is
# on, or a stroke turns it on by itself: its chain does not fit the cache budget, or FAST_FRAMES frames in a row take
# more than FAST_PAINTING_MS (it stays on for the selected layer until another one is selected).
FAST_PAINTING_MS = 100
FAST_FRAMES = 3
# Constants the renderer reads for the slots a Texture Set lacks (lib-sampler defaults).
SLOT_CONSTANTS = {"baseColor": (0.5, 0.5, 0.5, 1.0), "roughness": (0.3, 0.3, 0.3, 1.0), "metallic": (0, 0, 0, 1),
                  "emissive": (0, 0, 0, 1), "ambientOcclusion": (1, 1, 1, 1), "normal": (0.5, 0.5, 1, 1)}


class PaintWorkspace(QMainWindow):
    title_changed = Signal(str)

    def __init__(self, main, evaluator):
        super().__init__()
        self.main, self.ev = main, evaluator
        self.doc: PaintDocument | None = None
        self.tools = ToolState()
        self.mesh_revision = 0
        self.stroke: Stroke | None = None
        self._last_end = None  # screen point where the last stroke ended (Shift + click draws from it)
        self._last_view = None
        self._line = False
        self._lazy = (0.0, 0.0)
        self.display = "material"  # "material", "channel:<id>" or "mesh:<slot>"
        self.mode = "painting"  # Mode menu: painting | baking
        self._quick = None  # the viewport's quick Properties popup (Mouse right)
        self.mask_clipboard = None  # Copy mask content: (Mask, PaintStore snapshot)
        self.clipboard = None  # Copy layer: ("layers", [Layer], snapshot) or ("effect", Effect, snapshot)
        self.content_clipboard = None  # Copy layer content: (layer uid, [Effect], snapshot)
        self._full_submeshes = []
        self._uv_cache = {}
        self.geometry_edit: str | None = None  # the layer whose Geometry mask the viewport clicks edit
        self.geometry_clipboard = None  # Copy geometry mask: (UV Tiles type, the excluded tiles or names)
        self.hide_excluded = False  # Hide excluded geometry (Alt+H): the selected layer's
        self._quick_masks: dict = {}  # (Texture Set name, tile) -> quick mask (paint.Pages)
        self._slots: list = []  # renderer material slot -> (mesh material index, UDIM tile)
        self._quick_rev = 0
        self.quick_edit = False
        self._before_picker = "paint"
        self.setDockNestingEnabled(True)

        self.view3d = PaintView3D(evaluator.device, self)
        self.view2d = PaintView2D(evaluator.device, self)
        from trafilatura3d.editor.paint.widgets import accept_assets
        accept_assets(self.view3d, ("texture", "alpha", "procedural"), self.drop_on_viewport)
        from trafilatura3d.editor.paint.path_tool import PathsPanel, PathTool
        self.path_tool = PathTool(self)
        self.paths_panel = PathsPanel(self.path_tool)
        self.paths_panel.hide()
        self.setCentralWidget(self._center())
        self._tools_toolbar()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.assets = AssetsPanel(self)
        self.texture_sets = TextureSetList(self)
        self.layers = LayerStackPanel(self)
        self.ts_settings = TextureSetSettings(self)
        self.properties = PropertiesPanel(self)
        self.display_settings = DisplaySettings(self)
        from trafilatura3d.editor.paint.shader_settings import ShaderSettingsPanel
        self.shader_settings = ShaderSettingsPanel(self)
        self.history = QUndoView()
        from trafilatura3d.editor.paint.baking_mode import BakingPanels
        self.baking = BakingPanels(self)
        self.docks = {}
        for key, title, widget, ic in [("assets", "Assets", self.assets, "assets"),
                                       ("texture_sets", "Texture Set List", self.texture_sets, "texture_sets"),
                                       ("layers", "Layers", self.layers, "paint_layer"),
                                       ("ts_settings", "Texture Set Settings", self.ts_settings, "texture_sets"),
                                       ("properties", "Properties", self.properties, "properties"),
                                       ("display", "Display Settings", self.display_settings, "display_settings"),
                                       ("shader", "Shader Settings", self.shader_settings, "properties"),
                                       ("history", "History", self.history, "history"),
                                       ("log", "Log", self.log_view, "log"),
                                       ("bakers", "Mesh map bakers", self.baking.bakers, "texture_sets"),
                                       ("common", "Common settings", self.baking.common, "properties"),
                                       ("map_settings", "Mesh map settings", self.baking.settings, "properties"),
                                       ("baking_log", "Baking log", self.baking.log, "log")]:
            d = QDockWidget(title, self)
            d.setObjectName(f"paint_{key}")
            d.setWidget(widget)
            d.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
                          | QDockWidget.DockWidgetFloatable)
            from trafilatura3d.editor.mainwindow import DockTitle
            d.setTitleBarWidget(DockTitle(d, ic, main))
            self.docks[key] = d
        self.properties.title_changed.connect(lambda t: self.docks["properties"].titleBarWidget().set_title(t))
        self.properties.tool_changed.connect(self._sync_context)
        self._default_layout()
        self.default_state = self.saveState()
        state = QSettings().value(LAYOUT_KEY)
        if state is not None:
            self.restoreState(state)
        self._docks_toolbar()
        self.menus = self._menus()
        self.apply_preferences()
        from trafilatura3d.editor.paint import color_picker
        color_picker.set_sampler(self.eyedrop_at)  # the eyedroppers' picks over the 3D view
        # The refresh: a material (a tile) per step, at PREVIEW_SIZE while a value is dragged.
        self.live = LiveCook(self, self.ev, self._plan_refresh, self._refresh_step,
                             paused=lambda: self.doc is None or self.stroke is not None,
                             done=lambda: self._view2d_timer.start(), settled=self._content_changed)
        self._frozen: dict = {}  # a stroke's jobs per tile, computed again over the brush's rectangle (`_freeze_stroke`)
        # (renderer material, slot) -> the key of the result the renderer has (None: something else, a stroke's);
        # renderer material -> (shader, emissive intensity, slots) it was made with
        self._shown: dict = {}
        self._shown_sig: dict = {}
        self._fast = False  # this stroke updates only the painted channels' slots (`_painting_slots`)
        self._fast_layer = None  # the layer Fast painting turned on by itself for (until another is selected)
        self._fast_warned = False  # the session's warning given
        self._slow_frames = 0
        # While painting the display follows at most once per frame, only the painted slots, without padding.
        self._stroke_timer = QTimer(self, singleShot=True, interval=16, timeout=self._refresh_stroke)
        self._view2d_timer = QTimer(self, singleShot=True, interval=120, timeout=self._update_view2d)

    # --- layout ---------------------------------------------------------------------------------------

    def _center(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = theme.divider(QWidget(), "bottom")  # the contextual toolbar over the viewports
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 2, 8, 2)
        # The contextual toolbar's left side: the brush's (painting tools) or the 3D projection's manipulator.
        self.brush_bar = QWidget()
        brow = QHBoxLayout(self.brush_bar)
        brow.setContentsMargins(0, 0, 0, 0)
        self.context_sliders, self.context_pressure = {}, {}
        for name, attr, hi in (("Size", "size", 100.0), ("Flow", "flow", 100.0),
                               ("Stroke opacity", "stroke_opacity", 100.0), ("Spacing", "spacing", 200.0)):
            s = PSlider(name, getattr(self.tools.brush, attr), 1.0 if attr == "spacing" else 0.0, hi, compact=True)
            s.setFixedWidth(150)
            s.edited.connect(lambda v, _f, attr=attr: (setattr(self.tools.brush, attr, v), self._brush_edited()))
            brow.addWidget(s)
            self.context_sliders[attr] = s
            if attr in ("size", "flow"):  # pressure capable (Pen pressure page): the toggle beside the slider
                pb = PressureButton(getattr(self.tools.brush, f"{attr}_pressure"), name)
                pb.toggled.connect(lambda on, attr=attr: (setattr(self.tools.brush, f"{attr}_pressure", on),
                                                          self._brush_edited()))
                brow.addWidget(pb, 0, Qt.AlignBottom)
                self.context_pressure[attr] = pb
        brow.addSpacing(8)
        self.lazy_button = self._lazy_button()
        brow.addWidget(self.lazy_button)
        self.sym_button = self._symmetry_button()
        brow.addWidget(self.sym_button)
        row.addWidget(self.brush_bar)
        from trafilatura3d.editor.paint.manipulator import Manipulator, ManipulatorBar
        self.manip = Manipulator(self)
        self.manip_bar = ManipulatorBar(self.manip)
        self.manip_bar.hide()
        row.addWidget(self.manip_bar)
        self.hide_excluded_button = QToolButton()
        self.hide_excluded_button.setIcon(picon("geometry_mask"))
        self.hide_excluded_button.setText("Hide excluded")
        self.hide_excluded_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.hide_excluded_button.setToolTip("Hide excluded geometry (Alt+H): the parts the selected layer's "
                                             "Geometry mask excludes, to paint what they cover")
        self.hide_excluded_button.setCheckable(True)
        self.hide_excluded_button.toggled.connect(lambda on: on != self.hide_excluded and self.set_hide_excluded(on))
        row.addWidget(self.hide_excluded_button)
        row.addStretch(1)
        self.vp_mode = QComboBox()
        self.vp_mode.addItems(VIEWPORT_MODES)
        self.vp_mode.setToolTip("Viewport Mode (F1-F4)")
        self.vp_mode.currentTextChanged.connect(self.set_viewport_mode)
        self.persp = QComboBox()
        self.persp.addItems(["Perspective view", "Orthographic view"])
        self.persp.setToolTip("Perspective Mode (F5/F6)")
        self.persp.currentIndexChanged.connect(lambda i: (setattr(self.view3d, "orthographic", i == 1),
                                                          self.view3d.refresh()))
        self.rotation = QComboBox()
        self.rotation.addItems(["Constrained rotation", "Free rotation"])
        self.rotation.setToolTip("Camera Rotation Mode")
        self.rotation.currentIndexChanged.connect(lambda i: setattr(self.view3d, "free_rotation", i == 1))
        for c in (self.vp_mode, self.persp, self.rotation):
            row.addWidget(c)
        lay.addWidget(theme.Overflow(bar))
        self.split = QSplitter(Qt.Horizontal)
        self.panel3d = self._view_panel(self.view3d, "display3d")
        self.panel2d = self._view_panel(self.view2d, "display2d")
        self.split.addWidget(self.panel3d)
        self.split.addWidget(self.panel2d)
        self.split.setSizes([700, 500])
        from trafilatura3d.editor.paint.baking_mode import BakingVisualization
        self.bake_vis = BakingVisualization(self, self.view3d)  # over the 3D view, in the Baking mode
        self.bake_vis.hide()
        lay.addWidget(self.split, 1)
        # Baking mode: the bar under the viewport (Bake button, Return to painting mode).
        self.bake_bar = theme.divider(QWidget(), "top")  # under the viewport in the Baking mode
        brow = QHBoxLayout(self.bake_bar)
        brow.setContentsMargins(8, 4, 8, 4)
        brow.addStretch(1)
        from PySide6.QtWidgets import QPushButton
        back = QPushButton("Return to painting mode")
        back.clicked.connect(lambda: self.set_mode("painting"))
        bake = QPushButton("Bake selected textures")
        bake.setStyleSheet("background: #2f80ed; color: white; padding: 4px 16px;")
        bake.clicked.connect(self.bake)
        brow.addWidget(back)
        brow.addWidget(bake)
        self.bake_bar.hide()
        lay.addWidget(self.bake_bar)
        from trafilatura3d.editor.paint.autosave import Autosave
        self.autosave = Autosave(self)  # its warning bar (progress, Snooze) under the viewport
        lay.addWidget(self.autosave)
        return w

    def _lazy_button(self):
        """Lazy Mouse (Lazy mouse page): the paint follows the cursor at a distance; the menu sets it."""
        from PySide6.QtWidgets import QWidgetAction
        b = _menu_toggle("Lazy Mouse", "Lazy Mouse (D)")
        b.toggled.connect(lambda on: setattr(self.tools, "lazy", on))
        m = QMenu(b)
        wa = QWidgetAction(m)
        dist = PSlider("Lazy Mouse distance", self.tools.lazy_distance, 1.0, 200.0, integer=True)
        dist.setMinimumWidth(200)
        dist.edited.connect(lambda v, _f: setattr(self.tools, "lazy_distance", v))
        wa.setDefaultWidget(dist)
        m.addAction(wa)
        b.setMenu(m)
        return b

    def _symmetry_button(self):
        """Symmetry (Symmetry, Mirror symmetry, Radial symmetry pages): the toggle and its settings menu: the type,
        the scene axis; Mirror: the axis position, Show Plane, Show Intersection; Radial: the axis position X Y Z,
        the number of copies (ours), Flip copy, Show Axis, Show Intersection; both: Show Cursor, Hide While
        Painting. The position manipulator is not offered (the sliders are)."""
        b = _menu_toggle("Symmetry", "Symmetry (L)")
        b.toggled.connect(lambda on: setattr(self.tools, "symmetry", on))
        m = QMenu(b)
        m.aboutToShow.connect(lambda: self._fill_symmetry_menu(m))
        b.setMenu(m)
        return b

    def _fill_symmetry_menu(self, m):
        """The settings menu, built when it opens (its rows depend on the type)."""
        from PySide6.QtWidgets import QSpinBox, QWidgetAction
        from trafilatura3d.paint.brush import FLIP_COPIES, SYMMETRY_TYPES
        t = self.tools
        m.clear()

        def widget(w):
            wa = QWidgetAction(m)
            w.setMinimumWidth(220)
            wa.setDefaultWidget(w)
            m.addAction(wa)

        def checks(group_items, current, set_value):
            group = QActionGroup(m)
            for value, label in group_items:
                a = m.addAction(label)
                a.setCheckable(True)
                a.setChecked(value == current)
                a.triggered.connect(lambda _=False, v=value: (set_value(v), self.view3d.refresh()))
                group.addAction(a)
        checks([(k, k) for k in SYMMETRY_TYPES], t.sym_type, lambda v: setattr(t, "sym_type", v))
        m.addSeparator()
        checks([(i, f"Axis {name}") for i, name in enumerate("XYZ")], t.sym_axis, lambda v: setattr(t, "sym_axis", v))
        m.addSeparator()
        if t.sym_type == "Mirror":
            pos = PSlider("Axis position", t.sym_position, -1.0, 1.0)
            pos.edited.connect(lambda v, _f: setattr(t, "sym_position", v))
            widget(pos)
            toggles = (("Show Plane", "sym_show_plane"), ("Show Intersection", "sym_show_intersection"))
        else:
            for i, name in enumerate("XYZ"):
                s = PSlider(f"Axis position: {name}", t.radial_position[i], -1.0, 1.0)
                s.edited.connect(lambda v, _f, i=i: t.radial_position.__setitem__(i, v))
                widget(s)
            copies = QSpinBox()
            copies.setRange(2, 64)
            copies.setValue(t.radial_copies)
            copies.setPrefix("Copies: ")
            copies.valueChanged.connect(lambda v: setattr(t, "radial_copies", v))
            widget(copies)
            flip = QComboBox()
            flip.addItems([f"Flip copy: {f}" for f in FLIP_COPIES])
            flip.setCurrentIndex(FLIP_COPIES.index(t.radial_flip))
            flip.currentIndexChanged.connect(lambda i: setattr(t, "radial_flip", FLIP_COPIES[i]))
            widget(flip)
            toggles = (("Show Axis", "sym_show_axis"), ("Show Intersection", "sym_show_intersection"))
        m.addSeparator()
        for label, attr in toggles + (("Show Cursor", "sym_show_cursor"), ("Hide While Painting", "sym_hide_painting")):
            a = m.addAction(label)
            a.setCheckable(True)
            a.setChecked(getattr(t, attr))
            a.toggled.connect(lambda on, attr=attr: setattr(t, attr, on))

    def _symmetry_display(self) -> list:
        """The symmetry's display over the mesh (lines): Mirror's plane (its outline across the scene) and
        Radial's axis (Show Plane / Show Axis), and the Intersection (where the plane or the axis meets the mesh)."""
        t = self.tools
        if not t.symmetry or self.doc is None:
            return []
        lo, hi = self.doc.session.bounds()
        size = float(np.max(hi - lo))
        out = []

        def seg(a, b, col):
            s = np.empty((2, 7), np.float32)
            s[0, :3], s[1, :3], s[:, 3:] = a, b, col
            out.append(s)
        col = (0.4, 0.7, 1.0, 0.7)
        ax = t.sym_axis
        if t.sym_type == "Mirror":
            center = (lo + hi) / 2
            center[ax] = t.sym_position
            u, v = np.eye(3)[(ax + 1) % 3] * size * 0.6, np.eye(3)[(ax + 2) % 3] * size * 0.6
            corners = [center + u + v, center - u + v, center - u - v, center + u - v]
            if t.sym_show_plane:
                for i in range(4):
                    seg(corners[i], corners[(i + 1) % 4], col)
                seg(corners[0], corners[2], (*col[:3], 0.3))
                seg(corners[1], corners[3], (*col[:3], 0.3))
            if t.sym_show_intersection:  # the mesh's edges crossing the plane
                for o in objs:
                    tri = o.pos[o.idx.reshape(-1, 3)]
                    d = tri[:, :, ax] - t.sym_position
                    cross = (d.min(1) < 0) & (d.max(1) > 0)
                    for tr, dd in zip(tri[cross][:4000], d[cross][:4000]):
                        pts = [tr[i] + (tr[j] - tr[i]) * dd[i] / (dd[i] - dd[j])
                               for i, j in ((0, 1), (1, 2), (2, 0)) if (dd[i] < 0) != (dd[j] < 0)]
                        if len(pts) == 2:
                            seg(pts[0], pts[1], (1.0, 0.85, 0.2, 0.9))
        else:
            p = np.asarray(t.radial_position, float)
            a = np.eye(3)[ax]
            if t.sym_show_axis:
                seg(p - a * size, p + a * size, col)
            if t.sym_show_intersection:  # the axis' crossings of the mesh: small crosses
                for o in objs:
                    tri = o.pos[o.idx.reshape(-1, 3)].astype(float)
                    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
                    h = np.cross(a, e2)
                    det = np.sum(e1 * h, 1)
                    ok = np.abs(det) > 1e-12
                    f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
                    s = p - tri[:, 0]
                    uu = f * np.sum(s * h, 1)
                    q = np.cross(s, e1)
                    vv = f * (q @ a)
                    tt = f * np.sum(e2 * q, 1)
                    hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1)
                    r = size * 0.01
                    for w in (p + a * tt[hit][:, None])[:16]:
                        for k in range(3):
                            e = np.eye(3)[k] * r
                            seg(w - e, w + e, (1.0, 0.85, 0.2, 0.9))
        return out

    def _view_panel(self, view, attr) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        header = theme.divider(QWidget(), "bottom")  # the view's header (its display mode) over the view
        top = QHBoxLayout(header)
        top.setContentsMargins(4, 2, 4, 2)
        top.addStretch(1)
        combo = QComboBox()
        combo.setMinimumWidth(170)
        combo.activated.connect(lambda i, c=combo: self.set_display(c.itemData(i)))
        setattr(self, attr, combo)
        top.addWidget(combo)
        if view is self.view2d:  # next to the Display mode, in UV Tile projects only
            self.tile_info_button = QToolButton()
            self.tile_info_button.setIcon(picon("uv_tiles"))
            self.tile_info_button.setToolTip("UV Tile information")
            self.tile_info_button.setCheckable(True)
            self.tile_info_button.setChecked(True)
            self.tile_info_button.toggled.connect(self._set_tile_info)
            self.tile_info_button.hide()
            top.addWidget(self.tile_info_button)
        lay.addWidget(header)
        if view is self.view3d:  # the Paths panel at the top of the 3D viewport (Path tool)
            lay.addWidget(self.paths_panel)
        lay.addWidget(view, 1)
        return w

    def _set_tile_info(self, on: bool) -> None:
        self.view2d.tile_info = on
        self.view2d.refresh()

    def _tools_toolbar(self):
        bar = QToolBar("Tools")
        bar.setObjectName("paint_tools")
        bar.setOrientation(Qt.Vertical)
        bar.setIconSize(QSize(18, 18))
        bar.setMovable(True)
        theme.divider(bar, "right")  # the tools, left of the panels and viewports
        self.tool_group = QActionGroup(self)
        self.tool_actions = {}
        self.named_actions = {}  # Shortcuts page name -> action (`apply_shortcuts`)
        for key, ic, tip, name in (("paint", "brush", "Paint", "Select Paint tool"),
                                   ("eraser", "eraser", "Eraser", "Select Eraser tool"),
                                   ("path", "path", "Path", None),
                                   ("projection", "projection", "Projection", "Select Projection tool"),
                                   ("polygon_fill", "polygon_fill", "Polygon Fill", "Select Polygon Fill"),
                                   ("smudge", "smudge", "Smudge", "Select Smudge tool"),
                                   ("clone", "clone", "Clone", "Select Clone tool (relative source)"),
                                   ("material_picker", "picker", "Material Picker", "Pick stroke material")):
            a = QAction(picon(ic), tip, self)
            a.setData(tip)  # the tooltip's text; `apply_shortcuts` adds the shortcut
            if name:
                self.named_actions[name] = a
            a.setCheckable(True)
            a.setChecked(key == self.tools.tool)
            if key == "clone":  # 6: relative source (Shortcuts page)
                a.triggered.connect(lambda _=False: self.set_clone(True))
            else:
                a.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            self.tool_group.addAction(a)
            bar.addAction(a)
            self.tool_actions[key] = a
        bar.addSeparator()
        self.quick_action = QAction(picon("quick_mask"), "Quick Mask (U: edit, Y: clear, I: invert)", self)
        self.quick_action.setCheckable(True)
        self.quick_action.triggered.connect(lambda _=False: self.toggle_quick_mask())
        bar.addAction(self.quick_action)
        self.addToolBar(Qt.LeftToolBarArea, bar)
        self.tools_bar = bar

    def _docks_toolbar(self):
        bar = QToolBar("Docks")
        bar.setObjectName("paint_docks")
        bar.setOrientation(Qt.Vertical)
        bar.setIconSize(QSize(18, 18))
        theme.divider(bar, "left")  # the panels' buttons, right of the panels
        from trafilatura3d.editor.icons import icon
        bar.addAction(icon("home"), "Home screen", self.main.show_home)  # ours: back to the Home (Help menu too)
        bar.addSeparator()
        for key, d in self.docks.items():
            if key in BAKING_DOCKS:  # the Baking mode's panels come with the mode
                continue
            a = d.toggleViewAction()
            a.setIcon(picon(_DOCK_ICONS[key]))
            bar.addAction(a)
        self.addToolBar(Qt.RightToolBarArea, bar)
        self.docks_bar = bar

    def _default_layout(self):
        d = self.docks
        self.addDockWidget(Qt.LeftDockWidgetArea, d["assets"])
        self.addDockWidget(Qt.RightDockWidgetArea, d["texture_sets"])
        self.splitDockWidget(d["texture_sets"], d["layers"], Qt.Vertical)
        self.splitDockWidget(d["layers"], d["properties"], Qt.Vertical)
        self.tabifyDockWidget(d["layers"], d["ts_settings"])
        for key in ("display", "shader", "history", "log"):
            self.addDockWidget(Qt.RightDockWidgetArea, d[key])
            d[key].setFloating(True)
            d[key].hide()
        # The Baking mode panels are out of the layout until the Mode menu shows them (_place_baking_docks), and
        # hidden: a dock outside the layout is still a child widget, drawn at the top left of the window.
        for key in BAKING_DOCKS:
            d[key].hide()
        d["layers"].raise_()
        self.resizeDocks([d["assets"]], [230], Qt.Horizontal)
        self.resizeDocks([d["texture_sets"]], [340], Qt.Horizontal)
        self.resizeDocks([d["texture_sets"], d["layers"], d["properties"]], [160, 360, 420], Qt.Vertical)

    def reset_layout(self):
        for d in self.docks.values():
            d.setFloating(False)
        self.restoreState(self.default_state)

    def hide_ui(self, on):
        """Window > Hide UI: every dock hidden, the viewports maximized (again: back)."""
        if on:
            self._before_hide = self.saveState()
            for d in self.docks.values():
                d.hide()
            self.tools_bar.hide()
            self.docks_bar.hide()
        elif getattr(self, "_before_hide", None) is not None:
            self.restoreState(self._before_hide)
            self._before_hide = None

    def save_layout(self):
        QSettings().setValue(LAYOUT_KEY, self.saveState())

    # --- menus ----------------------------------------------------------------------------------------

    def _menus(self) -> list[QMenu]:
        self.shortcut_actions = []

        def act(menu, text, slot, name=None, checkable=False):
            """A menu action; `name`: its entry of the Shortcuts page (the binding from `shortcuts`)."""
            a = QAction(text, self)
            if name:
                self.named_actions[name] = a
                self.shortcut_actions.append(a)
            a.setCheckable(checkable)
            (a.toggled if checkable else a.triggered).connect(slot)
            menu.addAction(a)
            return a

        f = QMenu("File", self)
        act(f, "New...", self.new_dialog, "New project")
        act(f, "Open...", self.open_dialog, "Open project")
        self.recent_menu = f.addMenu("Recent files")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        act(f, "Close", self.close_project, "Close project")
        f.addSeparator()
        act(f, "Save", lambda: self.save(), "Save project")
        act(f, "Save as...", lambda: self.save(ask=True), "Save as project")
        act(f, "Save as copy...", self.save_copy)
        act(f, "Clean", self.clean)
        f.addSeparator()
        act(f, "Import resources...", self.import_resources_dialog)
        act(f, "Export mesh...", self.export_mesh_dialog)
        act(f, "Export textures...", self.export_dialog, "Export textures")
        f.addSeparator()
        act(f, "Quit", self.main.close, "Quit application")
        e = QMenu("Edit", self)
        act(e, "Undo", lambda: self.doc and self.doc.undo.undo(), "Undo")
        act(e, "Redo", lambda: self.doc and self.doc.undo.redo(), "Redo")
        act(e, "Project configuration...", self.project_configuration)
        act(e, "Settings...", self.settings_dialog)
        act(e, "Bake Mesh maps", lambda: self.set_mode("baking"), "Bake Mesh Maps")
        mode = QMenu("Mode", self)
        self.mode_actions = {
            "baking": act(mode, "Bake mesh maps", lambda: self.set_mode("baking"), "Baking"),
            "painting": act(mode, "Painting", lambda: self.set_mode("painting"), "Painting")}
        for a in self.mode_actions.values():
            a.setCheckable(True)
        self.mode_actions["painting"].setChecked(True)
        w = QMenu("Window", self)
        views = w.addMenu("Views")
        for d in self.docks.values():
            views.addAction(d.toggleViewAction())
        bars = w.addMenu("Toolbars")
        bars.addAction(self.tools_bar.toggleViewAction())
        bars.addAction(self.docks_bar.toggleViewAction())
        self.hide_ui_action = act(w, "Hide UI", self.hide_ui, "Hide UI", checkable=True)
        act(w, "Reset UI", self.reset_layout)
        v = QMenu("Viewport", self)
        act(v, "Display material", lambda: self.set_display("material"), "Display material")
        act(v, "Display next channel", lambda: self.step_display("channel", 1), "Display next channel")
        act(v, "Display previous channel", lambda: self.step_display("channel", -1), "Display previous channel")
        act(v, "Display next mesh map", lambda: self.step_display("mesh", 1), "Display next mesh map")
        act(v, "Display previous mesh map", lambda: self.step_display("mesh", -1), "Display previous mesh map")
        act(v, "Show entire mesh", self.frame, "Center the whole mesh")
        v.addSeparator()
        for m, name in zip(VIEWPORT_MODES, ("Viewport layout 3D/2D", "Viewport layout 3D only",
                                            "Viewport layout 2D only", "Swap 2D / 3D view")):
            act(v, m, lambda _=False, m=m: self.vp_mode.setCurrentText(m), name)
        act(v, "Perspective view", lambda: self.persp.setCurrentIndex(0), "Camera Perspective")
        act(v, "Orthographic view", lambda: self.persp.setCurrentIndex(1), "Camera Orthographic")
        self.shortcut_actions += list(self.tool_actions.values())
        act(v, "Texture Set isolate", lambda: self.texture_sets.focus.toggle(), "Texture Set isolate")
        t = QMenu("Tools", self)  # not a menu of the application: holds the tool shortcuts of the Shortcuts page
        act(t, "Increase tool size", lambda: self._step_size(1.1), "Increase tool size")
        act(t, "Decrease tool size", lambda: self._step_size(1 / 1.1), "Decrease tool size")
        act(t, "Invert grayscale tool", self._invert_grayscale, "Invert grayscale tool")
        act(t, "Symmetry", lambda: self.sym_button.toggle(), "Symmetry")
        act(t, "Lazy Mouse", lambda: self.lazy_button.toggle(), "Lazy Mouse")
        act(t, "Select Clone tool (absolute source)", lambda: self.set_clone(False), "Select Clone tool (absolute source)")
        act(t, "Toggle quick mask edition", self.toggle_quick_mask, "Toggle quick mask edition")
        act(t, "Hide/ignore excluded geometry", lambda: self.set_hide_excluded(not self.hide_excluded),
            "Hide/ignore excluded geometry")
        m, mb = self.manip, self.manip_bar
        act(t, "Show/Hide manipulator", lambda: self.manip_key(lambda: mb.toggle_visible()), "Show/Hide manipulator")
        act(t, "Increase the manipulator size",
            lambda: self.manip_key(lambda: (setattr(m, "size", m.size * 1.25), m.refresh())),
            "Increase the manipulator size")
        act(t, "Decrease the manipulator size",
            lambda: self.manip_key(lambda: (setattr(m, "size", m.size / 1.25), m.refresh())),
            "Decrease the manipulator size")
        act(t, "Cycle through manipulation spaces", lambda: self.manip_key(lambda: mb.space_box.setCurrentIndex(
            1 - mb.space_box.currentIndex())), "Cycle through manipulation spaces")
        act(t, "Toggle warp edition mode", lambda: self.manip_key(lambda: mb.warp_box.setCurrentIndex(
            1 - mb.warp_box.currentIndex())), "Toggle warp edition mode")
        act(t, "Translate tool", lambda: self.manip_key(lambda: mb.set_mode("Translate")), "Translate tool")
        act(t, "Rotate tool", lambda: self.manip_key(lambda: mb.set_mode("Rotate")), "Rotate tool")
        act(t, "Scale tool", lambda: self.manip_key(lambda: mb.set_mode("Scale")), "Scale tool")
        act(t, "Surface tool", lambda: self.manip_key(lambda: mb.set_mode("Surface")), "Surface tool")
        act(t, "Clear quick mask", self.clear_quick_mask, "Clear quick mask")
        act(t, "Invert quick mask", self.invert_quick_mask, "Invert quick mask")
        self.hidden_actions = t
        pl = QMenu("Plugins", self)
        auto = pl.addMenu("Autosave")
        from trafilatura3d.editor.paint.autosave import setting
        self.autosave_disable = act(auto, "Disable", lambda on: self.autosave.set_enabled(not on), checkable=True)
        self.autosave_disable.setChecked(not setting("enabled"))
        act(auto, "Configure", self.autosave.configure)
        h = QMenu("Help", self)
        act(h, "Home screen", self.main.show_home)
        act(h, "About Trafilatura 3D", self.main.about)
        self.apply_shortcuts()
        return [f, e, mode, w, v, pl, h]

    def apply_shortcuts(self) -> None:
        """The bindings of the Shortcuts page (defaults or the user's) on the named actions; tool tooltips show
        them."""
        from trafilatura3d.editor.paint import shortcuts
        for name, a in self.named_actions.items():
            seq = shortcuts.binding(name)
            a.setShortcuts(variants(seq) if seq else [])
            if a.data():
                a.setToolTip(f"{a.data()} ({shortcuts.label(seq)})" if seq else a.data())

    def settings_dialog(self):
        from trafilatura3d.editor.paint.settings import SettingsDialog
        SettingsDialog(self).exec()

    def apply_preferences(self, _key=None) -> None:
        """Edit > Settings > General preferences on the workspace."""
        from trafilatura3d.editor import preferences as editor_prefs
        from trafilatura3d.editor.paint import settings as prefs
        self.ev.budget = editor_prefs.cache_bytes()  # Engine > Cache memory budget (the Graph workspace Preferences' one)
        v, v2 = self.view3d, self.view2d
        v.rotation_speed = prefs.get("rotation_speed")
        v.zoom_speed = v2.zoom_speed = prefs.get("zoom_speed") * (-1 if prefs.get("zoom_inverse") else 1)
        v.wheel_speed = v2.wheel_speed = prefs.get("wheel_speed") * (-1 if prefs.get("wheel_inverse") else 1)
        v.look.world_axes = prefs.get("world_axes")
        v.look.background = (tuple(prefs.get("background_top")), tuple(prefs.get("background_bottom")))
        if not prefs.get("keyboard_helper"):
            self.keyboard_helper(set())
        v.refresh()
        if self.doc is not None:
            self.refresh_materials()  # Only display the selected material when painting
            self.layers.update_thumbnails()

    def libraries_changed(self) -> None:
        """Settings > Libraries configuration changed: the Assets search the libraries again."""
        self.assets._materials = None
        self.assets.refresh()

    def isolated(self) -> bool:
        """Only the selected Texture Set displayed: Texture Set isolate, or in painting mode the General
        preference "Only display the selected material when painting"."""
        from trafilatura3d.editor.paint import settings as prefs
        return self.texture_sets.focus_mode or (self.mode == "painting" and prefs.get("selected_material_only"))

    def select_texture_set_at(self, x, y) -> None:
        """Texture set selection (mouse binding): the Texture Set of the mesh under a viewport pixel."""
        if self.doc is None:
            return
        _view, cam = self._camera()
        got = cam.hit(x, y)
        if got is None or got[0][3] <= 0:
            return
        name = self.doc.session.materials[int(round(got[0][3])) - 1]
        ts = next((t for t in self.doc.project.texture_sets if t.material == name), None)
        if ts is not None and ts.name != self.doc.ts_name:
            self.doc.select_texture_set(ts.name)
            if self.isolated():
                self.refresh_materials()

    def keyboard_helper(self, mods) -> None:
        """Show keyboard helper (General preferences): with modifiers held, the shortcuts and mouse bindings that
        use them, at the bottom left of the 3D view (an image drawn over the frame)."""
        from trafilatura3d.editor.paint import settings as prefs
        from trafilatura3d.editor.paint import shortcuts
        r = self.view3d.renderer
        rows = []
        if mods and prefs.get("keyboard_helper"):
            names = {"Control": "Ctrl", "Meta": "Meta", "Alt": "Alt", "Shift": "Shift"}
            held = {names[m] for m in mods if m in names}
            for name in list(shortcuts.KEYBOARD) + list(shortcuts.MOUSE):
                seq = shortcuts.binding(name)
                parts = set(seq.split("+")[:-1]) if seq else set()
                if seq and held <= parts and parts & held and len(seq.split("+")) > 1:
                    rows.append((shortcuts.label(seq), name))
        if not rows:
            if r.hud is not None:
                r.set_hud(None)
                self.view3d.refresh()
            return
        r.set_hud(*_helper_image(rows, self.view3d.physical_size()[1], self.view3d.get_pixel_ratio()))
        self.view3d.refresh()

    def _fill_recent(self):
        self.recent_menu.clear()
        for p in QSettings().value("recent", [], type=list):
            if p.endswith(PROJECT_SUFFIX):
                self.recent_menu.addAction(p, lambda p=p: self.open_path(p))

    # --- project --------------------------------------------------------------------------------------

    def log(self, text: str, error: bool = False):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {'ERROR: ' if error else ''}{text}")
        self.main.status(text)

    def new_dialog(self):
        from trafilatura3d.editor.paint.new_project import NewProjectDialog
        from trafilatura3d.editor.preferences import suggested_dir
        if not self.close_project():
            return
        dlg = NewProjectDialog(self.main, suggested_dir())
        if not dlg.exec():
            return
        v = dlg.values()
        try:
            session = Session.new(v["mesh"], v["resolution"], v["normal_format"], v["baked_maps"], evaluator=self.ev,
                                  uv_tiles=v["uv_tiles"], template=v["template"])
        except (OSError, ValueError, ProjectError) as e:
            QMessageBox.critical(self.main, "Trafilatura 3D", f"Cannot create the project:\n{e}")
            return
        self.set_document(PaintDocument(session))
        self.doc.dirty = True
        self.log(f"New project from {Path(v['mesh']).name}: {len(session.project.texture_sets)} Texture Sets")

    def open_dialog(self):
        from trafilatura3d.editor.preferences import suggested_dir
        name, _ = QFileDialog.getOpenFileName(self.main, "Open project", suggested_dir(), OPEN_FILTER)
        if name:
            self.open_path(name)

    def open_path(self, path) -> bool:
        if self.doc is not None and self.doc.session.path == Path(path).resolve():
            self.main.show_paint()
            return True
        if not self.close_project():
            return False
        imported = str(path).lower().endswith(".spp")  # imported into a new unsaved project
        try:
            session = self._import_spp(path) if imported else Session.open(path, evaluator=self.ev)
        except (OSError, ValueError, KeyError, ProjectError) as e:
            QMessageBox.critical(self.main, "Trafilatura 3D", f"Cannot open {path}:\n{e}")
            return False
        self.set_document(PaintDocument(session))
        if imported:
            self.doc.dirty = True
            self.log(f"Imported {Path(path).name}: {len(session.load_warnings)} import notes")
        else:
            self.main.remember(session.path)
            self.log(f"Opened {session.path}")
        for w in session.load_warnings:  # what the file named but the library no longer has
            self.log(f"Warning: {w}")
        return True

    def _import_spp(self, path) -> Session:
        """A .spp project imported (its strokes replayed: minutes on a large project), the steps in the Log."""
        from PySide6.QtWidgets import QApplication

        from trafilatura3d.spp import import_spp

        def progress(text):
            self.log(text)
            QApplication.processEvents()
        return import_spp(path, evaluator=self.ev, progress=progress)

    def save(self, ask=False) -> bool:
        if self.doc is None:
            return False
        path = self.doc.session.path
        if ask or path is None:
            from trafilatura3d.editor.preferences import suggested_dir
            start = str(path) if path else str(Path(suggested_dir()) / f"{Path(self.doc.project.mesh).stem}{PROJECT_SUFFIX}")
            name, _ = QFileDialog.getSaveFileName(self.main, "Save project", start, PROJECT_FILTER)
            if not name:
                return False
            path = Path(name)
        try:
            self.doc.save(path)
        except OSError as e:
            QMessageBox.critical(self.main, "Trafilatura 3D", str(e))
            return False
        self.main.remember(self.doc.session.path)
        self.log(f"Saved {self.doc.session.path}")
        self._title()
        return True

    def save_copy(self):
        if self.doc is None:
            return
        from trafilatura3d.editor.preferences import suggested_dir
        name, _ = QFileDialog.getSaveFileName(self.main, "Save as copy", suggested_dir(), PROJECT_FILTER)
        if not name:
            return
        keep = self.doc.session.path
        try:
            self.doc.session.save(name)
        except OSError as e:
            QMessageBox.critical(self.main, "Trafilatura 3D", str(e))
        finally:
            self.doc.session.path = keep
        self.log(f"Saved a copy to {name}")

    def clean(self):
        if self.doc is None:
            return
        gone = self.doc.session.clean()
        self.doc.dirty = True
        self.assets.refresh()
        self.log(f"Clean: {len(gone)} unused resources removed (at the next save)")

    def close_project(self) -> bool:
        """Close the open project (asking to save its changes); False if the user cancels."""
        if self.doc is None:
            return True
        if self.doc.dirty:
            name = self.doc.session.path.name if self.doc.session.path else "the new project"
            answer = QMessageBox.question(self.main, "Trafilatura 3D", f"Save the changes to {name}?",
                                          QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self.save()):
                return False
        self.doc = None
        self.view3d.renderer.set_meshes([])
        self.view3d.renderer.extra_lines = None
        self.history.setStack(None)
        self._rebuild_panels()
        self.main.paint_closed()
        return True

    def set_document(self, doc: PaintDocument):
        for q in self._quick_masks.values():  # the previous document's temporary states
            q.destroy()
        self._quick_masks, self.quick_edit, self.geometry_edit = {}, False, None
        self.quick_action.setChecked(False)
        self.path_tool.selected, self.path_tool.vertex, self.path_tool.editing = None, None, False
        self.doc = doc
        self.mesh_revision += 1
        self._uv_cache = {}
        self.view3d.camera_cache = None
        self._set_meshes()
        self.tile_info_button.setVisible(bool(doc.project.uv_tiles))
        doc.structure_changed.connect(self._rebuild_panels)
        doc.selection_changed.connect(self._selection_changed)
        doc.content_changed.connect(self._content_changed)
        doc.content_previewed.connect(self._content_previewed)
        doc.error.connect(lambda e: self.log(e, True))
        doc.undo.cleanChanged.connect(self._title)  # bound methods: disconnected when the window goes
        doc.undo.indexChanged.connect(self._title)
        self.history.setStack(doc.undo)
        self._rebuild_panels()
        self.frame()
        # shown at once at the preview resolution, a tile per turn of the event loop, then at full resolution (as
        # after a drag): the window answers while the project opens
        self.live.start(preview=True)
        self.main.show_paint()
        self._title()

    def _title(self, *_):
        if self.doc is None:
            self.title_changed.emit("Trafilatura 3D")
            return
        p = self.doc.session.path
        name = p.stem if p else "Untitled"
        self.title_changed.emit(f"Trafilatura 3D Paint - {name}{' *' if self.doc.dirty else ''}")

    def import_resource(self, path) -> str | None:
        try:
            name = self.doc.session.import_image(path)
        except (OSError, ValueError) as e:
            self.log(f"Import {Path(path).name}: {e}", True)
            return None
        self.doc.dirty = True
        self.assets.refresh()
        tiles = len(self.doc.session.resources.sequence(name))
        self.log(f"Imported {Path(path).name} into the project as "
                 + (f"the image sequence {name} ({tiles} tiles)" if tiles else name))
        return name

    def import_resources_dialog(self) -> str | None:
        """File > Import resources: images into the project (the Session / Shelf destinations of the
        application are not offered). Returns the last imported name."""
        if self.doc is None:
            return None
        from trafilatura3d.editor.preferences import suggested_dir
        names, _ = QFileDialog.getOpenFileNames(self.main, "Import resources", suggested_dir(), IMAGE_FILTER)
        from trafilatura3d.paint.session import UDIM_FILE
        last, seen = None, set()
        for n in names:
            m = UDIM_FILE.match(Path(n).stem)
            seq = m is not None and int(m.group(3)) >= 1001
            key = (str(Path(n).parent), m.group(1), m.group(2), Path(n).suffix) if seq else n
            if key in seen:  # a file of a sequence already brought in whole
                continue
            seen.add(key)
            last = self.import_resource(n) or last
        return last

    def use_asset(self, asset: dict):
        """Double click on an asset: an alpha goes to the tool, a material becomes a fill layer, a smart material
        is added, a filter or a generator goes on the selected layer (or its mask)."""
        if self.doc is None:
            return
        kind = asset["kind"]
        if kind == "alpha":
            self.tools.alphas[self.tools.tool] = asset["name"]
            self.properties.rebuild()
        elif kind == "material":
            self.layers.add_material_layer(asset["package"], asset["graph"], asset.get("label", asset["graph"]))
        elif kind == "smart":
            self.layers.add_smart_material(asset["path"])
        elif kind in ("filter", "generator"):
            self.layers.add_effect(kind, asset["id"], asset.get("label", ""))
        else:
            self.log(f"Drag the {kind} onto a channel of the Properties")

    def export_dialog(self):
        if self.doc is None:
            return
        from trafilatura3d.editor.paint.export_dialog import ExportDialog
        ExportDialog(self).exec()

    # Export mesh formats: the documentation's (FBX, OBJ, DAE, PLY, GLTF, USD) that Trafilatura 3D writes (no FBX: Autodesk's
    # SDK is proprietary; no DAE).
    MESH_FORMATS = ("obj", "ply", "glb", "gltf", "usd", "usda", "usdc", "usdz")

    def export_mesh_dialog(self):
        if self.doc is None:
            return
        from trafilatura3d.editor.preferences import suggested_dir
        filt = ";;".join(f"{f.upper()} (*.{f})" for f in self.MESH_FORMATS)
        start = str(Path(suggested_dir()) / f"{Path(self.doc.project.mesh).stem}.obj")
        name, _ = QFileDialog.getSaveFileName(self.main, "Export mesh", start, filt)
        if name:
            self.export_mesh(name)

    def export_mesh(self, path):
        """File > Export mesh, Without displacement/tessellation (the option offered; the mesh is already the
        triangles the project uses): the meshes in their file's coordinates, materials by name."""
        from types import SimpleNamespace
        from trafilatura3d.editor.scene3d import export_scene
        s = self.doc.session
        try:
            export_scene(path, s.mesh_in_file_space(), [SimpleNamespace(name=m) for m in s.materials])
        except Exception as e:  # noqa: BLE001 - writer errors are reported
            self.log(f"Export mesh: {e}", error=True)
            return False
        self.log(f"Exported mesh {path}")
        return True

    def project_configuration(self):
        if self.doc is None:
            return
        from trafilatura3d.editor.paint.new_project import ProjectConfiguration
        ProjectConfiguration(self).exec()

    def mesh_changed(self):
        """The session's mesh was replaced (reimport): viewport meshes, UV caches, panels."""
        self.mesh_revision += 1
        self._uv_cache = {}
        self.view3d.camera_cache = None
        self._set_meshes()
        self._rebuild_panels()
        self.frame()
        self.refresh_materials()

    # --- panels ---------------------------------------------------------------------------------------

    def _rebuild_panels(self):
        if self.mode == "baking":
            QTimer.singleShot(0, self._update_bake_overlay)
        self.baking.rebuild()
        self.texture_sets.rebuild()
        self.layers.rebuild()
        self.ts_settings.rebuild()
        self.properties.rebuild()
        self.shader_settings.rebuild()
        self.assets.refresh()
        self._fill_display_combos()
        self.live.schedule()

    def _selection_changed(self):
        from trafilatura3d.editor.paint import color_picker
        layer = self.doc.layer()
        if layer is None or layer.uid != self._fast_layer:  # Fast painting turned on by itself: for its layer only
            self._fast_layer = None
        color_picker.close_context()  # semi-persistent: another layer or effect is another context
        if self.mode == "baking":
            QTimer.singleShot(0, self._update_bake_overlay)
        self.baking.rebuild()
        self.layers.sync_selection()  # not a rebuild: the row under the mouse stays (drags, clicks)
        self.properties.rebuild()
        self.ts_settings.rebuild()
        self.texture_sets.rebuild()
        self._fill_display_combos()
        if self.geometry_edit is not None and self.doc.selected != self.geometry_edit:  # another part: edition ends
            self.geometry_edit = None
        self.sync_context_bar()
        if self.hide_excluded:  # the hidden parts follow the selected layer
            self.view3d.camera_cache = None
        if self.tools.tool == "path":  # the selected painting's paths
            if self.path_tool.path() is None:
                self.path_tool.selected, self.path_tool.vertex, self.path_tool.editing = None, None, False
            self.paths_panel.rebuild()
            self.path_tool.refresh_overlay()
        self.live.schedule()

    def sync_context_bar(self):
        """The contextual toolbar follows the selection: a fill with a 3D projection shows the manipulator."""
        manip = self.manip.fill() is not None
        self.manip_bar.setVisible(manip)
        self.brush_bar.setVisible(not manip)
        if manip:
            self.manip_bar.sync()
            self.manip.refresh()
        elif self.tools.tool != "path":
            self.view3d.renderer.extra_lines = None
            self.view3d.refresh()

    def manip_key(self, fn):
        """A manipulator shortcut, only while a 3D projection is selected."""
        if self.manip.fill() is not None:
            fn()

    def _content_changed(self):
        self.live.changed()
        self.layers.schedule_thumbnails()

    def _content_previewed(self):
        self.live.previewed()
        self.layers.schedule_thumbnails()

    def set_tool(self, tool):
        if tool != self.tools.tool:
            from trafilatura3d.editor.paint import color_picker
            color_picker.close_context()  # another tool is another context
        if tool == "material_picker" and self.tools.tool != "material_picker":
            self._before_picker = self.tools.tool  # a temporary tool: the previous one comes back after picking
        self.tools.tool = tool
        self.tool_actions[tool].setChecked(True)
        self.properties.rebuild()
        self._sync_context()
        self.update_projection_preview()
        self.paths_panel.setVisible(tool == "path")
        if tool == "path":
            self.paths_panel.rebuild()
            self.path_tool.refresh_overlay()
        else:
            self.path_tool.editing = False
            self.view3d.renderer.extra_lines = None
            self.view3d.refresh()

    # --- Projection tool --------------------------------------------------------------------------------

    def source_texture(self, src, ts):
        """A tool's resource as a texture (the same in every tile: computed in the first one's)."""
        return self.doc.session.source_texture(src, ts, ts.tiles[0])

    def projection_resource(self, asset: dict) -> str:
        """The Projection tool's resource reference of an asset: a project texture by name, a procedural by its id;
        an alpha becomes a project texture ("alpha <name>", its grayscale) the first time."""
        kind = asset["kind"]
        if kind == "procedural":
            return asset["id"]
        if kind == "alpha":
            from trafilatura3d.paint.alphas import ALPHAS
            name = f"alpha {asset['name']}"
            res = self.doc.session.resources
            if name not in res.images:
                res.add(name, np.asarray(ALPHAS[asset["name"]](1024), np.float32))
            return name
        return asset["name"]

    def drop_on_viewport(self, asset: dict, channel: str | None = None):
        """A texture, an alpha or a procedural dropped on the 3D viewport: the Projection tool's image (the Base
        color's, or `channel`'s; on a mask its luminance is painted), shown over the view at once. Painting projects
        it into the selected painting (paint layer, mask, paint effect)."""
        if self.doc is None:
            return
        ids = self.doc.ts.channel_ids()
        if channel is None:
            channel = "basecolor" if "basecolor" in ids else ids[0]
        ref = self.projection_resource(asset)
        src = self.tools.proj_source(channel)
        if ref.startswith("node:"):
            src.procedural, src.image, src.values = ref, "", {}
        else:
            src.procedural, src.image = "", ref
        self.tools.proj_disabled.discard(channel)
        self.set_tool("projection")
        self.log(f"Projection: {ref} (S + drag or the Properties to place it, paint to project)")

    def _preview_channel(self, ts):
        """The channel the projection preview shows (Viewport settings > Projection preview channel: ours, the
        first enabled one with a resource, Base color first)."""
        st = self.tools
        chans = [c for c in ts.channel_ids() if c not in st.proj_disabled]
        chans.sort(key=lambda c: c != "basecolor")
        return next((c for c in chans if st.proj_source(c).image or st.proj_source(c).procedural), None)

    def _xf_target(self):
        """The screen space image S + drag transforms: the Projection tool's, or the tool's stencil."""
        if self.tools.tool == "projection":
            return self.tools.projection
        if self.tools.stencil() is not None:
            return self.tools.stencil_settings()
        return None

    def update_projection_preview(self):
        """The Projection tool's image (or the tool's stencil) over the viewport, through its transform (half
        opacity, ours); N held hides the stencil."""
        r = self.view3d.renderer
        overlay = None
        if self.doc is not None:
            ts, st = self.doc.ts, self.tools
            got, xf = None, None
            if st.tool == "projection":
                ch = self._preview_channel(ts)
                got = self.source_texture(st.proj_source(ch), ts) if ch else None
                xf = st.projection
            elif st.stencil() is not None and not self.view3d.stencil_ignored():
                got = self.source_texture(st.stencil(), ts)
                xf = st.stencil_settings()
            if got is not None:
                overlay = {"texture": got[0], "single_channel": got[1], "xf": xf.xf(),
                           "tiling": xf.tiling_flags(), "opacity": xf.opacity}
        r.screen_overlay = overlay
        self.update_falloff()
        self.view3d.refresh()

    def update_falloff(self):
        """The Projection tool's falloff in red over the view (Show falloff), from the view's camera buffer: made
        again when the camera moved (the viewport calls it before drawing)."""
        r, v, st = self.view3d.renderer, self.view3d, self.tools
        if self.doc is None or st.tool != "projection" or not st.projection.show_falloff:
            r.falloff_overlay = None
            return
        view, cam = self._camera()
        key = v.camera_key(view.width, view.height)
        r.falloff_overlay = {"position": cam.texture, "normal": cam.normals, "eye": tuple(float(x) for x in view.eye),
                             "fade": st.projection.fade(), "opacity": 0.45, "key": key}

    def _stencil_texture(self, view, udim: int):
        """The tool's stencil seen from the stroke's view in a tile's texture space (None: no stencil, or N
        held)."""
        st = self.tools
        src = st.stencil()
        if src is None or self.view3d.stencil_ignored():
            return None
        s, ts = self.doc.session, self.doc.ts
        got = self.source_texture(src, ts)
        if got is None:
            return None
        return s.brush_engine().project(s.uv_maps(ts, udim), view, st.stencil_settings(), (1.0, 1.0, 1.0), got[0],
                                        got[1], gray=True)

    def _projection_sources(self, tiles, view, camera):
        """Each target's values seen through the projection from the stroke's view, with its falloff (fade by angle,
        occlusion): a channel's resource or its uniform value; a mask (or the quick mask) takes the projected
        image's luminance."""
        s, ts, st = self.doc.session, self.doc.ts, self.tools
        engine = s.brush_engine()
        for tile in tiles:
            for tg in tile.targets:
                ch = tg.key[1]
                mask = ch not in ts.channel_ids()  # a mask entry or the quick mask
                src = st.proj_source(self._preview_channel(ts) or "basecolor") if mask else st.proj_source(ch)
                got = self.source_texture(src, ts)
                tex, single = got if got is not None else (None, False)
                tg.source = engine.project(tile.maps, view, st.projection, tg.value if mask else src.color, tex,
                                           single, gray=mask or FORMATS[ts.format_of(ch)][0], camera=camera,
                                           falloff=True, diag=s.bbox_diagonal())

    def drag_projection(self, dx, dy, button, shift):
        """S + drag in the viewport (Projection page, Stencil): left rotates (Shift snaps every 90 degrees), right
        scales, middle moves; dx, dy in viewport pixels."""
        pr = self._xf_target()
        if pr is None:
            return
        w, h = self.view3d.width(), self.view3d.height()
        if button == 1:
            pr.rotation = (pr.rotation + dx * 0.5) % 360.0
            if shift:
                pr.rotation = round(pr.rotation / 90.0) * 90.0 % 360.0
        elif button == 2:
            pr.scale = min(max(pr.scale * 1.01 ** (dx - dy), 0.01), 100.0)
        else:
            pr.offset = (pr.offset[0] + dx / max(w, 1), pr.offset[1] + dy / max(h, 1))
        self.update_projection_preview()

    def _sync_context(self):
        for attr, s in self.context_sliders.items():
            s.set_value(getattr(self.tools.brush, attr))
        for attr, pb in self.context_pressure.items():
            pb.set_on(getattr(self.tools.brush, f"{attr}_pressure"))

    def _brush_edited(self):
        self.properties.rebuild()

    def _step_size(self, k):
        b = self.tools.brush
        b.size = min(max(b.size * k, 0.1), 100.0)
        self._sync_context()
        self.properties.rebuild()

    def _invert_grayscale(self):
        """X: invert the grayscale value painted on a mask."""
        self.tools.mask_value = 1.0 - self.tools.mask_value
        self.properties.rebuild()

    def drag_brush(self, dx, dy, which):
        """Ctrl + right drag: size (horizontal); Ctrl + left drag: flow (horizontal) and angle (vertical)."""
        b = self.tools.brush
        if which == "size":
            b.size = min(max(b.size + dx * 0.2, 0.1), 100.0)
        else:
            b.flow = min(max(b.flow + dx * 0.3, 0.0), 100.0)
            b.angle = (b.angle + dy) % 360
        self._sync_context()

    # --- modes ----------------------------------------------------------------------------------------

    def set_mode(self, mode):
        """Mode menu: the Baking mode shows its panels instead of the painting ones (the Texture Set list,
        Log, Display settings and History stay); painting tools are off while baking."""
        if self.doc is None or mode == self.mode:
            self.mode_actions[self.mode].setChecked(True)
            return
        if mode == "baking":
            self._painting_state = self.saveState()
            for key in PAINTING_DOCKS:
                self.docks[key].hide()
            self._place_baking_docks()
            self.tools_bar.hide()
        else:
            for key in BAKING_DOCKS:  # out of the layout (hidden docks can leave a tab bar behind)
                self.removeDockWidget(self.docks[key])
            if getattr(self, "_painting_state", None) is not None:
                self.restoreState(self._painting_state)
            for key in PAINTING_DOCKS:
                self.docks[key].show()
            self.docks["layers"].raise_()
            self.tools_bar.show()
        self.mode = mode
        for k, a in self.mode_actions.items():
            a.setChecked(k == mode)
        self.bake_bar.setVisible(mode == "baking")
        self.bake_vis.setVisible(mode == "baking")
        self.bake_vis.move(8, 8)
        self.bake_vis.raise_()
        self.bake_vis.apply(mode == "baking")
        self._overlay_key = None
        self._update_bake_overlay()  # built in the Baking mode, emptied back in painting
        self.view3d.renderer.extra_lines = None
        self.texture_sets.rebuild()
        self.baking.rebuild()
        from trafilatura3d.editor.paint import settings as prefs
        if prefs.get("selected_material_only"):  # the other Texture Sets come back (or go) with the mode
            self.refresh_materials()
        self.view3d.refresh()

    def _update_bake_overlay(self):
        """The selected Texture Set's high poly meshes (vertices no bake ray reaches flagged), cage (the file, or
        the low pushed out by Max Frontal Distance) and rear surface in the viewport, rebuilt when the baking
        setup changes (the baking visualization of decision 22)."""
        import dataclasses
        from trafilatura3d import bake as bk
        from trafilatura3d import meshio
        from trafilatura3d.paint import baking
        r = self.view3d.renderer
        if self.doc is None or self.mode != "baking":
            r.set_bake_overlay([], [], [], [])
            return
        s, ts = self.doc.session, self.doc.ts
        cfg = baking.config(ts)
        st = baking.settings_of(cfg["common"], 0)
        paths = [] if st.use_low_as_high else list(st.high_meshes)
        cage_path = st.cage_file if st.use_cage and st.cage_file else ""
        stamp = lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0
        key = (ts.name, tuple(paths), cage_path, tuple(stamp(p) for p in [*paths, cage_path] if p), st.frontal, st.rear,
               st.relative_to_bounding_box, st.average_normals)
        if key == getattr(self, "_overlay_key", None):
            return
        self._overlay_key = key
        try:
            low = baking.low_of(s, ts)
            load = lambda p: [o for o in meshio.load(p)[0]]

            def fitted(p):
                objs = load(p)
                meshio.apply_fit(objs, s.fit)
                return objs
            high = [o for p in paths for o in fitted(p)] if paths else [
                meshio.MeshObject(o.name, o.pos.copy(), o.nrm, o.tan, o.uv, o.idx, o.submeshes) for o in s.objects]
            k = 1.0 if st.relative_to_bounding_box else s.fit[1]
            fs = dataclasses.replace(st, frontal=st.frontal * k, rear=st.rear * k)
            cage_file = fitted(cage_path) if cage_path else None
            cage = cage_file or bk.implicit_cage(low, fs)
            outside = bk.unreached(self.ev.device, high, low, fs, cage_file)
        except Exception as e:  # noqa: BLE001 - shown in the Log
            self.log(f"Baking visualization: {e}", True)
            r.set_bake_overlay([], [], [], [])
            return
        r.set_bake_overlay(high, outside, cage, [])
        self.view3d.refresh()

    def _place_baking_docks(self):
        """Mesh map bakers and Common settings on the left, Mesh map settings and Baking log on the right."""
        d = self.docks
        self.addDockWidget(Qt.LeftDockWidgetArea, d["bakers"])
        self.splitDockWidget(d["bakers"], d["common"], Qt.Vertical)
        self.addDockWidget(Qt.RightDockWidgetArea, d["map_settings"])
        self.splitDockWidget(d["map_settings"], d["baking_log"], Qt.Vertical)
        for key in BAKING_DOCKS:
            d[key].setFloating(False)
            d[key].show()
        self.resizeDocks([d["bakers"], d["common"]], [300, 500], Qt.Vertical)
        # Widths as the painting layout's: the titles no longer widen a panel (they are cut).
        self.resizeDocks([d["bakers"], d["map_settings"]], [300, 340], Qt.Horizontal)
        self.resizeDocks([d["map_settings"], d["baking_log"]], [520, 160], Qt.Vertical)

    def bake(self):
        """Bake selected textures: the checked mesh maps of each checked Texture Set (one History step)."""
        from PySide6.QtWidgets import QApplication, QProgressDialog
        from trafilatura3d.paint import baking
        doc = self.doc
        checked = self.texture_sets.checked()
        names = [n for n, _tiles in checked]
        if doc is None or not names:
            self.baking.log.appendPlainText("No Texture Set checked.")
            return
        dlg = QProgressDialog("Baking…", "Cancel", 0, 100, self.main)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        log = self.baking.log
        stopped = []

        def do():
            for k, (name, tiles) in enumerate(checked):
                ts = doc.project.texture_set(name)

                def progress(done, total, label, k=k):
                    dlg.setValue(int((k + done / max(total, 1)) / len(names) * 100))
                    dlg.setLabelText(f"{name}: {label}" if label else name)
                    QApplication.processEvents()
                    if dlg.wasCanceled():
                        stopped.append(1)
                        return False
                    return True
                t0 = time.perf_counter()
                try:
                    out = baking.bake_texture_set(doc.session, ts, progress=progress, tiles=tiles)
                except OSError as e:  # a high poly or cage file that does not load
                    raise ValueError(f"{name}: {e}") from None
                log.appendPlainText(f"{name}: {', '.join(out) or 'nothing checked'} ({time.perf_counter() - t0:.1f} s)")
                if stopped:
                    break
        try:
            doc.edit("Bake mesh maps", do)
        except Exception as e:  # noqa: BLE001 - shown in the Baking log
            log.appendPlainText(f"ERROR: {e}")
        dlg.close()
        doc.content_changed.emit()
        self.assets.refresh()
        self.log("Bake finished" + (" (canceled)" if stopped else ""))

    # --- display --------------------------------------------------------------------------------------

    def _fill_display_combos(self):
        for combo in (self.display3d, self.display2d):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Material", "material")
            if self.doc is not None and self.doc.ts is not None:
                for c in self.doc.ts.channel_ids():
                    combo.addItem(CHANNELS[c].label, f"channel:{c}")
                for slot, label, _ in MESH_MAPS:
                    if slot in self.doc.ts.mesh_maps:
                        combo.addItem(label, f"mesh:{slot}")
            if self.display.startswith("mask:") and self.doc is not None and self.doc.ts.find(self.display[5:]):
                layer = self.doc.ts.find(self.display[5:])
                combo.addItem(f"Mask: {layer.name}", self.display)  # Alt+click on a mask thumbnail
            i = combo.findData(self.display)
            combo.setCurrentIndex(max(i, 0))
            if i < 0:
                self.display = "material"
            combo.blockSignals(False)

    def set_display(self, mode):
        self.display = mode
        self._fill_display_combos()
        self.refresh_materials()

    def isolate_mask(self, uid):
        """Alt+click on a mask: the viewports show that mask alone (again: back to the material)."""
        key = f"mask:{uid}"
        self.set_display("material" if self.display == key else key)

    def step_display(self, kind, step):
        """Viewport > Display next/previous channel or mesh map."""
        if self.doc is None:
            return
        ts = self.doc.ts
        items = ts.channel_ids() if kind == "channel" else [s for s, _, _ in MESH_MAPS if s in ts.mesh_maps]
        if not items:
            return
        cur = self.display.split(":", 1)[1] if self.display.startswith(f"{kind}:") else None
        i = (items.index(cur) + step) % len(items) if cur in items else (0 if step > 0 else len(items) - 1)
        self.set_display(f"{kind}:{items[i]}")

    def set_viewport_mode(self, mode):
        if mode == "3D/2D" or mode == "Swap 3D/2D":
            self.panel3d.show()
            self.panel2d.show()
            first = self.panel2d if mode == "Swap 3D/2D" else self.panel3d
            self.split.insertWidget(0, first)
        elif mode == "3D only":
            self.panel3d.show()
            self.panel2d.hide()
        else:
            self.panel3d.hide()
            self.panel2d.show()
        self._view2d_timer.start()

    def frame(self):
        if self.doc is None:
            return
        lo, hi = self.doc.session.bounds()
        self.view3d.frame(lo, hi)
        self.view2d.fit()

    def _visible_sets(self):
        doc = self.doc
        focus = self.isolated()
        return {ts.material: ts for ts in doc.project.texture_sets
                if ts.visible and (not focus or ts.name == doc.ts_name)}

    def view_compiler(self, udim: int):
        """The stack compiler of what the viewports show: the Texture Set's resolution, at most PREVIEW_SIZE while a
        value is dragged (`LiveCook.previewing`)."""
        return self.doc.session.compiler(udim, PREVIEW_SIZE if self.live.previewing else None)

    def _slot_jobs(self, ts, udim: int, padded=True, only=None, full=False) -> dict:
        """Renderer slot -> job for a Texture Set's tile in the current display mode (`only`: just those slots;
        `padded` False: without the UV padding, while painting; `full`: at full resolution even in preview)."""
        sc = self.doc.session.compiler(udim) if full else self.view_compiler(udim)
        self.doc.session.uv_mask(ts, udim)  # the UV mask the padding needs
        ids = ts.channel_ids()
        if self.display == "material":
            jobs = {}
            want = lambda slot: only is None or slot in only
            skipped = ("normal", "height", "ambientocclusion") + (
                () if self.shader_spec()["uniforms"]["blending"] != shaders.OPAQUE else ("opacity",))
            for ch, slot in render_slots(self.template().shader).items():  # opacity: for a shader that sees through
                if ch in ids and ch not in skipped and want(slot):
                    jobs[slot] = sc.channel(ts, ch)
            if self.quick_edit and ts.name == self.doc.ts_name and want("baseColor"):
                base = jobs.get("baseColor") or sc.constant((0.5, 0.5, 0.5, 1.0), ts.size_of(udim), "16f")
                jobs["baseColor"] = self._quick_overlay(ts, udim, base)
            if want("normal"):
                jobs["normal"] = sc.normal(ts)
            sh = self.doc.project.shader
            if "height" in ids and want("height") and (sh.scale_amount != 0.0 or sh.pom_enable):
                # displacement and parallax read the renderer's convention: 0.5 = flat (the Paint height is signed)
                jobs["height"] = sc.to_material(ts, sc.channel(ts, "height"), "height", False)[0]
            ao = sc.ambient_occlusion(ts) if want("ambientOcclusion") else None
            if ao is not None:
                jobs["ambientOcclusion"] = ao
        else:
            kind, name = self.display.split(":", 1)
            if kind == "mask":
                layer = ts.find(name)
                job = sc.mask(ts, layer) if layer is not None else None
            else:
                job = sc.channel(ts, name) if kind == "channel" and name in ids else sc.mesh_map(ts, name)
            jobs = {"baseColor": job} if job is not None else {}
        if only is not None:
            jobs = {slot: job for slot, job in jobs.items() if slot in only}
        jobs = {slot: sc.uv_only(ts, job) for slot, job in jobs.items()}  # the islands are what a viewport shows
        return {slot: (sc.padded(ts, job) if padded else job) for slot, job in jobs.items()}

    def _below_selection(self, visible) -> list:
        """The stack below the selected layer and below each folder holding it, in every channel and tile shown:
        what an edit of the selection restarts from (the cache keeps them, else a full cook gives them up first as
        new results and every edit computes the whole stack). A constant (the base of a channel, 1x1) is left out:
        it is the same job as every equal constant of the stack (a mask's black background was the black base under
        the selection, and every mask's steps became restart points: 3.3 GB instead of 1.6 on a 63 layer project)."""
        doc = self.doc
        layer = doc.layer()
        ts = doc.ts
        if layer is None or ts is None or visible.get(ts.material) is not ts:
            return []
        parent = {c.uid: f for f in ts.walk() for c in f.children}
        uids = [layer.uid]
        while uids[-1] in parent:
            uids.append(parent[uids[-1]].uid)
        out = []
        for udim in ts.tiles:  # at full resolution always (what the release restarts from), and in preview
            for sc in {self.doc.session.compiler(udim), self.view_compiler(udim)}:
                for ch in ts.channel_ids():
                    out += [sc.uv_only(ts, j) for uid in uids if (j := sc.channel_at(ts, ch, uid)) is not None
                            and any(j.inputs.values())]
        return out

    def refresh_materials(self, only_current=False):
        """Cook what the 3D view shows and give it to the renderer (one material per mesh material and tile), now."""
        self.live.refresh_now()

    def _plan_refresh(self) -> Plan | None:
        """The refresh (`LiveCook`): a material (a tile) per step, so the interface answers between them and each
        tile shows as soon as it is ready."""
        if self.doc is None:
            return None
        with self.doc.session.planning():  # every channel and mask compiled once for all the slots asking
            return self._plan()

    def _plan(self) -> Plan:
        r = self.view3d.renderer
        visible = self._visible_sets()
        specs, full_keys = [], []
        for m, udim in self._slots:
            ts = visible.get(self.doc.session.materials[m])
            if ts is None or udim not in ts.tiles:
                specs.append({})
                full_keys.append({})
                continue
            try:
                jobs = self._slot_jobs(ts, udim)
                # in preview, a slot whose full resolution result is the one shown stays as it is (a dragged value
                # lowers only what it changes)
                full_keys.append({slot: j.key for slot, j in self._slot_jobs(ts, udim, full=True).items()}
                                 if self.live.previewing else {})
            except Exception as e:  # noqa: BLE001 - a broken stack shows its defaults and is logged
                self.log(f"{ts.name}: {e}", True)
                jobs = {}
                full_keys.append({})
            specs.append(jobs)
        below = self._below_selection(visible)
        if len(r.materials) != max(len(specs), 1):  # another layout: every material shows the defaults until cooked
            r.materials = [r.material({}) for _ in range(max(len(specs), 1))]
            self._shown, self._shown_sig = {}, {}
        self._apply_shader()
        lines = self.doc.session.log_lines
        while lines:  # what the stack could not do (a missing material…)
            self.log(lines.pop(0), True)
        # hidden Texture Sets and hidden excluded geometry (Geometry mask) are not drawn
        materials = self.doc.session.materials
        for i, (m, subs) in enumerate(zip(r.meshes, self._full_submeshes)):
            m.submeshes = [s for s in subs if materials[self._slots[s[2]][0]] in visible
                           and not self._hidden(i, *self._slots[s[2]])]
        # One material (a tile) at a time: cooked, copied by the renderer, given back (the renderer's copies are
        # what both views draw), so only one tile's results wait in the cache instead of every tile's.
        return Plan([(i, jobs, full_keys[i]) for i, jobs in enumerate(specs)],
                    [j for jobs in specs for j in jobs.values()], below)

    def _refresh_step(self, item):
        """A material of the refresh: cooked, copied by the renderer, its results given back. Paused while a stroke
        is painted (its frames update the renderer; replacing a material would erase them): the stroke's end resumes
        it."""
        index, jobs, full_keys = item
        r = self.view3d.renderer
        spec = self.shader_spec() if self.display == "material" else {"shader": shaders.index("Unlit"),
                                                                       "constants": dict(SLOT_CONSTANTS)}
        sig = (spec["shader"], tuple(sorted(spec.get("uniforms", {}).items())), frozenset(jobs))
        if sig != self._shown_sig.get(index):  # another shader or other slots: the material made again
            self.live.cook(jobs.values())
            for slot, job in jobs.items():
                spec[slot] = (self.ev.texture(job, "output"), job.is_gray("output"))
            r.materials[index] = r.material(spec)
            self._shown_sig[index] = sig
            changed = jobs
        else:  # only the slots whose result changed: the others are what the renderer has (their copies stay)
            changed = {slot: job for slot, job in jobs.items()
                       if self._shown.get((index, slot)) not in (job.key, full_keys.get(slot))}
            if changed:
                self.live.cook(changed.values())
                r.update_slots(index, {slot: (self.ev.texture(j, "output"), j.is_gray("output"))
                                       for slot, j in changed.items()})
        self._shown.update({(index, slot): job.key for slot, job in changed.items()})
        self.ev.drop(changed.values())
        self.view3d.refresh()

    POM_DEPTH = 0.05  # parallax depth (tangent space) of Strength 1: ours, the documentation gives no unit

    def template(self) -> shaders.Template:
        """The project's shader with its blending mode (Shader settings > Shader, the New Project's template)."""
        return shaders.TEMPLATES[self.doc.project.shader.shader]

    def shader_spec(self) -> dict:
        """The renderer material (without textures) of the project's shader (`template`): its model with its default
        parameters, its blending mode and Alpha threshold, the Texture Set's missing channels at lib-sampler's
        defaults, Emissive Intensity on the emission (OpenPBR's emission is in nits / 100: x 100)."""
        sh, t = self.doc.project.shader, self.template()
        name = t.shader
        uniforms = shaders.uniforms(name, t.values(sh.alpha_threshold))
        uniforms["emission"] = sh.emissive_intensity * (100.0 if name == "OpenPBR" else 1.0)
        return {"shader": shaders.index(name), "uniforms": uniforms,
                "constants": {**shaders.constants(name, {}), **SLOT_CONSTANTS}}

    def _apply_shader(self):
        """Shader settings to the viewport: AO intensity, parallax, displacement (Normalized: a height of 1 moves
        the surface by Scale amount x the bounding box's largest side), tessellation (meshes subdivided for the
        display; painting hits the surface without displacement)."""
        sh, look = self.doc.project.shader, self.view3d.look
        look.ao_opacity = sh.ao_intensity
        look.parallax = sh.pom_strength * self.POM_DEPTH if sh.pom_enable else 0.0
        lo, hi = self.doc.session.bounds()
        size = float(np.max(hi - lo))
        look.enable_displacement = sh.scale_amount != 0.0
        look.displacement = 2.0 * sh.scale_amount * size  # the height slot is 0.5 + signed height / 2
        look.height_level = 0.5
        if getattr(self, "_tessellation", 0) != sh.subdivision_count:
            self._set_meshes()

    def _set_meshes(self):
        """The session's meshes to the renderer, subdivided by the Shader settings' Subdivision Count, their parts
        split by UDIM tile: a renderer material per (mesh material, tile) (`_slots`), each tile its own textures."""
        from trafilatura3d.bake import udim
        from trafilatura3d.editor.view3d import _subdivided
        from trafilatura3d.meshio import split_by_tile
        from trafilatura3d.paint.model import tile_uv
        sh = self.doc.project.shader
        objs = self.doc.session.objects
        if sh.subdivision_count:
            objs = [_subdivided(o, sh.subdivision_count) for o in objs]
        tiles = sorted({tile_uv(t) for ts in self.doc.project.texture_sets for t in ts.tiles},
                       key=lambda t: (t[1], t[0]))
        objs = [split_by_tile(o, tiles) for o in objs]
        self._slots = [(m, udim(t)) for m in range(len(self.doc.session.materials)) for t in tiles]
        self._tessellation = sh.subdivision_count
        r = self.view3d.renderer
        r.set_meshes(objs)
        self._full_submeshes = [list(m.submeshes) for m in r.meshes]

    def _slot(self, ts, udim: int) -> int:
        """The renderer material of a Texture Set's tile."""
        return self._slots.index((self.doc.session.material_index(ts), udim))

    def _update_view2d(self):
        """The 2D view shows what the 3D view draws in the base color slot (a tile each): the renderer's textures
        (a hidden Texture Set is hidden in both viewports, Texture Set list page)."""
        if self.doc is None or not self.panel2d.isVisible():
            self.view2d.set_tiles([])
            return
        ts, r = self.doc.ts, self.view3d.renderer
        tiles = []
        for udim in ts.tiles:
            key = (self.doc.session.material_index(ts), udim)
            if key in self._slots:
                index = self._slots.index(key)
                tiles.append((udim, lambda i=index: (r.materials[i]["textures"]["baseColor"], False)
                              if "baseColor" in r.materials[i]["spec"] else None))
        self.view2d.set_tiles(tiles)
        self.view2d.set_uvs(self._uv_lines(ts))

    def _uv_lines(self, ts):
        key = (ts.material, self.mesh_revision)
        if key not in self._uv_cache:
            m = self.doc.session.material_index(ts)
            segs = []
            for o in self.doc.session.objects:
                for s in o.submeshes:
                    if s.material != m:
                        continue
                    tri = o.idx[s.start:s.start + s.count].reshape(-1, 3)
                    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
                    e = np.unique(np.sort(e, 1), axis=0)
                    segs.append(o.uv[e])
            self._uv_cache[key] = np.concatenate(segs) if segs else np.zeros((0, 2, 2))
        return self._uv_cache[key]

    # --- painting -------------------------------------------------------------------------------------

    def _meshes(self):
        """The meshes the brush sees (the camera buffer): all of them, but the hidden excluded geometry."""
        r = self.view3d.renderer
        out = []
        for i, (m, subs) in enumerate(zip(r.meshes, self._full_submeshes)):  # the mesh material, not the tile's slot
            parts = [(start, count, *self._slots[slot]) for start, count, slot, _centre in subs]
            out.append((m.buffers, m.index,
                        [(a, b, mat) for a, b, mat, udim in parts if not self._hidden(i, mat, udim)]))
        return out

    def _camera(self):
        """The viewport's camera buffer, rendered again only when the camera (or the mesh) moved."""
        v = self.view3d
        view = v.brush_view()
        key = v.camera_key(view.width, view.height)
        if v.camera_cache is None or v.camera_cache[0] != key:
            engine = self.doc.session.brush_engine()
            v.camera_cache = (key, engine.camera_buffer(self._meshes(), view))
        return view, v.camera_cache[1]

    def _paint_target_reason(self):
        doc = self.doc
        if doc is None or doc.ts is None:
            return "Open a project first"
        if self.mode == "baking":
            return "Painting tools are off in the Baking mode"
        if not doc.ts.visible:
            return "The Texture Set is hidden"
        if self.quick_edit:  # painting the quick mask: no layer needed
            return None
        layer, fx = doc.layer(), doc.effect()
        if layer is None:
            return "Select a paint layer"
        if fx is not None:
            return None if fx.kind == "paint" else "Painting tools work on a paint layer, a paint effect or a mask"
        if doc.mask_selected:
            return None
        if layer.kind != "paint":
            return "Painting tools work on a paint layer, a paint effect or a mask"
        return None

    def hover(self, pos):
        r = self.view3d.renderer
        if self.manip.fill() is not None or self.geometry_edit is not None:  # their lines stay, no brush cursor
            return
        if pos is None or self._paint_target_reason() is not None:
            if r.extra_lines is not None:
                r.extra_lines = None
                self.view3d.refresh()
            return
        if self.tools.tool == "path":  # no brush cursor: the paths' overlay (and the next segment)
            self.path_tool.refresh_overlay(pos)
            return
        if self.tools.tool == "polygon_fill" and self._polygon_hover(pos):
            self.view3d.refresh()
            return
        view, cam = self._camera()
        m = self.doc.session.material_index(self.doc.ts)
        under = cam.hit(pos[0], pos[1])
        if under is not None and under[0][3] > 0 and int(round(under[0][3])) - 1 == m:
            hit, n = under[0][:3].astype(np.float64), under[1].astype(np.float64)
            rw, _ = stamp_radius(self.tools.brush, view, hit, self.doc.session.bbox_diagonal())
            color = (1.0, 0.45, 0.45, 0.9) if self.tools.tool == "eraser" else (1.0, 1.0, 1.0, 0.9)
            lines = [circle_lines(hit, n, rw, color)]
            t = self.tools
            if t.tool == "clone" and t.clone_source is not None:  # where the clone takes from (green, ours)
                src = None
                if t.clone_follow and t.clone_delta is not None:
                    got = cam.hit(pos[0] + t.clone_delta[0], pos[1] + t.clone_delta[1])
                    if got is not None and got[0][3] > 0:
                        src = (got[0][:3].astype(np.float64), got[1].astype(np.float64))
                else:
                    src = (np.asarray(t.clone_source), n)
                if src is not None:
                    lines.append(circle_lines(src[0], src[1], rw, (0.4, 1.0, 0.5, 0.9)))
            if t.symmetry and t.sym_show_cursor and not (self.stroke is not None and t.sym_hide_painting):
                for m, off, _reflect, _flip in t.symmetry_copies():  # a cursor per copy (Show Cursor)
                    lines.append(circle_lines(m @ hit + off, m @ n, rw, (color[0], color[1], color[2], 0.5)))
            lines += self._symmetry_display()
            r.extra_lines = np.concatenate(lines)
        else:
            r.extra_lines = None
        self.view3d.refresh()

    def _tile_targets(self, view=None):
        """What a stroke writes in each tile of the Texture Set (brush.TileTargets): the entries, the quick mask
        protecting them, the stencil seen from `view` (None: no stencil); None (logged) when there is nothing."""
        reason = self._paint_target_reason()
        if reason is not None:
            self.log(reason)
            return None
        s, ts = self.doc.session, self.doc.ts
        stencil = view is not None and self.tools.tool in ("paint", "eraser")
        out = []
        for udim in ts.tiles:
            targets = self._targets(udim)
            if targets:
                out.append(TileTargets(udim, s.uv_maps(ts, udim), targets, self._limit(udim),
                                       self._stencil_texture(view, udim) if stencil else None))
        if not out:
            self.log("No channel to paint: turn on a channel of the material")
            return None
        return out

    def _targets(self, udim: int) -> list:
        """The painted entries a stroke writes in a tile (the selected paint layer's or paint effect's channels, or
        the mask), created or resampled to the Texture Set's size; the Eraser skips what has nothing painted."""
        doc, s = self.doc, self.doc.session
        ts, layer, fx = doc.ts, doc.layer(), doc.effect()
        erase = self.tools.tool == "eraser"
        if self.quick_edit:  # the quick mask: the tool's grayscale, the Eraser paints 0
            value = 0.0 if erase else self.tools.mask_value
            return [Target((self.QUICK, ts.name, udim), self.quick_pages(ts, udim, create=True), (value,) * 3)]
        uid = fx.uid if fx is not None else layer.uid  # the paint effect's pixels, or the layer's (or its mask's)
        proj = self.tools.tool == "projection"
        targets = []
        if doc.mask_selected:
            if not (erase and s.store.get(uid, MASK, udim) is None):
                e = s.store.ensure(uid, MASK, udim, ts.size_of(udim), storage_for(None))
                targets.append(Target((uid, MASK, udim), e.pages, (self.tools.mask_value,) * 3, erase))
        else:
            off = (self.tools.erase_disabled if erase else self.tools.proj_disabled if proj
                   else self.tools.disabled)
            for ch in ts.channel_ids():
                if ch in off or (erase and s.store.get(uid, ch, udim) is None):
                    continue
                if self.tools.tool == "polygon_fill" and ch != "basecolor":  # the page: Base color only
                    continue
                e = s.store.ensure(uid, ch, udim, ts.size_of(udim), storage_for(ts.format_of(ch)))
                value = self.tools.proj_source(ch).color if proj else self.tools.value(ch)
                targets.append(Target((uid, ch, udim), e.pages, tuple(value), erase))
        return targets

    def _alpha_name(self) -> str:
        """The tool's alpha, ready in the brush engine (a project texture by its luminance)."""
        return self.doc.session.alpha_name(self.tools.alphas[self.tools.tool])

    def stroke_begin(self, x, y, pressure, shift=False, ctrl=False) -> bool:
        if self.geometry_edit is not None:  # editing a Geometry mask: a click toggles the mesh under it
            self._geometry_toggle(x, y)
            return False
        if self.manip.fill() is not None:  # a fill with a 3D projection: the manipulator (or the warp grid)
            return self.manip.press(x, y, shift, ctrl)
        if self.tools.tool == "path":
            return self.path_tool.press(x, y)
        if self.tools.tool == "material_picker":
            self.pick_material(x, y)
            return False
        s, ts = self.doc.session, self.doc.ts
        view, camera = self._camera()
        tiles = self._tile_targets(None if self.tools.tool == "polygon_fill" else view)
        if tiles is None:
            return False
        t = self.tools
        if t.tool == "projection":
            self._projection_sources(tiles, view, camera)
        if t.tool == "clone" and not self._clone_sources(tiles, view, camera, x, y):
            return False
        smudge = self._content_sources(tiles) if t.tool == "smudge" else None
        if t.tool == "polygon_fill":
            self.stroke = Stroke(s.brush_engine(), tiles, view, self._meshes(), s.material_index(ts), t.brush,
                                 s.bbox_diagonal(), camera=camera)
            self._stroke_keys = [k.key for k in self.stroke.targets()]
            self._freeze_stroke()
            self._filled = set()
            self._line = False
            self._polygon_fill(x, y)
            return True
        self.stroke = Stroke(s.brush_engine(), tiles, view, self._meshes(), s.material_index(ts), t.brush,
                             s.bbox_diagonal(), self._alpha_name(), seed=int(time.time() * 1000) & 0xFFFF,
                             camera=camera, symmetry=t.symmetry_copies() if smudge is None else None, smudge=smudge)
        self._stroke_keys = [k.key for k in self.stroke.targets()]
        self._freeze_stroke()
        self._lazy = (x, y)
        self._line = shift and self._last_end is not None and self._last_view == self.view3d.camera_key(view.width, view.height)
        if self._line:  # Shift + click: a straight line from the end of the last stroke (Straight line page)
            x0, y0 = self._last_end
            if ctrl:  # Shift + Ctrl: the angle snapped every 5 degrees
                ang = math.atan2(y - y0, x - x0)
                ang = round(ang / math.radians(5)) * math.radians(5)
                dist = math.hypot(x - x0, y - y0)
                x, y = x0 + math.cos(ang) * dist, y0 + math.sin(ang) * dist
            self.stroke.add([(x0, y0, pressure)])
            self._paint_points([(x, y, pressure)])
            self._last_end = (x, y)
            return True
        self.stroke_add(x, y, pressure)
        return True

    def _paint_points(self, points):
        if self.stroke.add(points):
            self._touched()
            if not self._stroke_timer.isActive():
                self._stroke_timer.start()

    def _stroke_slots(self) -> set:
        """The renderer slots a stroke changes (a mask changes every channel of its layer)."""
        out, slots = set(), render_slots(self.template().shader)
        for uid, key, _udim in self._stroke_keys:
            if uid == self.QUICK:  # shown over the base color
                out.add("baseColor")
                continue
            channels = self.doc.ts.channel_ids() if key == MASK else [key]
            for ch in channels:
                if self.display != "material":
                    out.add("baseColor")
                elif ch in ("normal", "height"):
                    out.add("normal")
                elif ch == "ambientocclusion":
                    out.add("ambientOcclusion")
                elif ch in slots:
                    out.add(slots[ch])
        return out

    def _freeze_stroke(self):
        """At the start of a stroke on the stack's paintings: per tile, the jobs from the painted entries up to the
        slots it updates (`_painting_slots`, unpadded). Every frame computes them again where the brush changed
        (Evaluator.recook_region): the steps as textures of that region only, but in place in their whole results
        the slots, what a filter reading far reads, that filter and what depends on it (computed whole: the change
        spreads). Held during the stroke, inside the cache budget: those whole results and what the chain reads from
        outside it (the masks and contents of the layers above, the stack below); when they do not fit, Fast
        painting turns on by itself."""
        from trafilatura3d.editor.paint import settings as prefs
        self._frozen, self._slow_frames = {}, 0
        layer = self.doc.layer()
        self._fast = prefs.get("fast_painting") or (layer is not None and self._fast_layer == layer.uid)
        if any(k[0] == self.QUICK for k in self._stroke_keys):  # the quick mask is drawn over the stack
            return
        ts, painted, held = self.doc.ts, {id(t.pages) for t in self.stroke.targets()}, []
        for udim in sorted({k[2] for k in self._stroke_keys}):
            jobs = self._painting_slots(ts, udim)
            memo, order = {}, []

            def reads(job):  # whether the job reads a painted entry; those that do in order (inputs first)
                if job.key not in memo:
                    read = [reads(src[0]) for src in job.inputs.values() if src]
                    memo[job.key] = id(job.extra.get("pages")) in painted or any(read)
                    if memo[job.key]:
                        order.append(job)
                return memo[job.key]

            for job in jobs.values():
                reads(job)
            chain, slot_keys = {job.key for job in order}, {j.key for j in jobs.values()}
            whole, skip, passed = set(), set(), set()
            if self._fast:  # the filters reading far seen through (their main input), the others left as they were
                for job in order:
                    main = job.node.primary_input()
                    if job.node.pointwise or job.key in slot_keys:
                        continue
                    if list(job.node.outputs) == ["output"] and main and job.inputs.get(main) \
                            and job.inputs[main][0].key in chain:
                        passed.add(job.key)
                    else:
                        skip.add(job.key)
            else:
                for job in order:  # a filter reading far, and what reads it
                    if not job.node.pointwise or any(src and src[0].key in whole for src in job.inputs.values()):
                        whole.add(job.key)
            full = slot_keys | whole | skip | {
                src[0].key for job in order if job.key in whole for src in job.inputs.values()
                if src and src[0].key in chain}
            outside = [src[0] for job in order for src in job.inputs.values() if src and src[0].key not in chain]
            kept = [job for job in order if job.key in full] + outside + list(jobs.values())
            if sum(self._result_bytes(j) for j in {j.key: j for j in held + kept}.values()) > self.ev.budget:
                if not self._fast:
                    self._go_fast("its slots with what they read take more than the cache budget")
                    return
                continue  # every frame computes the slots whole
            for job in kept:  # cached across the frames (intermediates too: a fill's content)
                job.keep = True
            held += kept
            self.ev.hold("stroke", held, counted=True)  # inside the budget: the rest makes room
            # cooked explicitly: with the slots already cached (a second stroke) a cook of the slots alone computes
            # nothing, and an input given back as an intermediate would be missing
            self.ev.cook(kept)
            self._frozen[udim] = (jobs, order, full, whole, skip, passed)
            self._show_frozen(ts, udim, jobs)

    def _painting_slots(self, ts, udim: int, padded: bool = False) -> dict:
        """The renderer slots a stroke updates (unpadded while painting): every one whose stack reads a painted
        entry (height -> normal, the filters and generators reading the painted channel…), or with Fast painting
        (`_fast`) the painted channels' only, whose filters reading far are seen through while painting (the
        painting shows at once, the filters at the release: `_freeze_stroke`)."""
        if self._fast:
            return self._slot_jobs(ts, udim, padded=padded, only=self._stroke_slots())
        painted, memo = {id(t.pages) for t in self.stroke.targets()}, {}

        def reads(job):
            if job.key not in memo:
                memo[job.key] = id(job.extra.get("pages")) in painted or any(
                    [reads(src[0]) for src in job.inputs.values() if src])
            return memo[job.key]
        return {slot: job for slot, job in self._slot_jobs(ts, udim, padded=padded).items() if reads(job)}

    def _go_fast(self, why: str) -> None:
        """Fast painting turned on by itself for the selected layer (until another is selected): the stroke frozen
        again with the painted channels' slots only, the session warned once."""
        layer = self.doc.layer()
        self._fast_layer = layer.uid if layer is not None else None
        if not self._fast_warned:
            self._fast_warned = True
            self.log(f"Fast painting turned on for this layer: {why}. While painting only the painted channels "
                     "update, the others at the release (Edit > Settings > General preferences > Fast painting).")
        self._freeze_stroke()

    def _show_frozen(self, ts, udim: int, jobs: dict) -> None:
        """A refresh not done (paused by the stroke): the stack as it is now, shown whole in the stroke's tile."""
        if self.live.pending:
            index = self._slot(ts, udim)
            self.view3d.renderer.update_slots(index, {
                slot: (self.ev.texture(j, "output"), j.is_gray("output")) for slot, j in jobs.items()})
            self._shown.update({(index, slot): None for slot in jobs})  # unpadded

    def _result_bytes(self, job) -> int:
        """What a job's results take on the GPU."""
        from trafilatura3d.evaluator import BYTES_PER_PIXEL
        from trafilatura3d.nodes.base import PortType
        w, h = job.size
        return sum(w * h * BYTES_PER_PIXEL[self.ev.storage_format(job.depths[o], job.is_gray(o),
                                                                  job.output_types[o] is PortType.GRAY_ALPHA)]
                   for o in job.node.outputs if not job.is_value(o))

    def _refresh_stroke(self, padded=False):
        """The painted slots of the current Texture Set updated in place in the renderer (unpadded while
        painting)."""
        if self.doc is None or self.stroke is None:
            return
        ts = self.doc.ts
        if ts.material not in self._visible_sets():
            return
        started = time.perf_counter()
        # while painting the texels the brush changed since the last frame, per tile; the padded refresh of the
        # stroke's end is whole (the padding changes around them)
        dirty = None if padded else self.stroke.take_dirty()
        for udim in sorted({k[2] for k in self._stroke_keys}):  # the tiles the stroke writes
            if dirty is not None and udim not in dirty:
                continue
            frozen = None if padded else self._frozen.get(udim)
            rect = None if dirty is None else dirty[udim]
            if frozen is not None:  # the brush's rectangle (and a margin for the normal's neighbors) only
                jobs, order, full, whole, skip, passed = frozen
                x0, y0, x1, y1 = rect
                w, h = ts.size_of(udim)
                self.ev.recook_region(order, (max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, w), min(y1 + 2, h)),
                                      full, whole, skip, passed)
                if any(j.key in whole for j in jobs.values()):
                    rect = None  # a filter above the painting spread the change: the slots whole
            else:
                try:
                    jobs = self._painting_slots(ts, udim, padded)
                except Exception as e:  # noqa: BLE001
                    self.log(f"{ts.name}: {e}", True)
                    return
                self.ev.cook(list(jobs.values()))
            index = self._slot(ts, udim)
            # only the changed rectangle copied (a filter above reading far shows its effect at the release, whose
            # refresh is whole)
            self.view3d.renderer.update_slots(index, {slot: (self.ev.texture(j, "output"), j.is_gray("output"))
                                                      for slot, j in jobs.items()},
                                              rect=rect)
            # the stroke's frames show unpadded results under no key; its end's padded ones are the slots' own
            self._shown.update({(index, slot): (j.key if padded else None) for slot, j in jobs.items()})
            if frozen is None:
                self.ev.drop(jobs.values())
        self.view3d.refresh()
        if not padded and not self._fast:  # FAST_FRAMES slow frames in a row: Fast painting turns on by itself
            self.ev.wait()
            self._slow_frames = self._slow_frames + 1 if time.perf_counter() - started > FAST_PAINTING_MS / 1000 else 0
            if self._slow_frames >= FAST_FRAMES:
                self._go_fast(f"its frames took more than {FAST_PAINTING_MS} ms")

    # --- Quick Mask (Quick Mask page; keys of the Shortcuts page: U edit, Y clear, I invert) -----------------
    # A temporary mask per Texture Set, not saved (ours: the documentation calls it temporary): R = how much a
    # texel is protected, painted in edit mode with the current tool (the Eraser paints 0), shown in red over the
    # material while editing (ours), and protecting the texels from every stroke once edit mode is left.
    QUICK = "quickmask"

    def quick_pages(self, ts, udim: int, create=False):
        q = self._quick_masks.get((ts.name, udim))
        if q is not None and list(q.size) != list(ts.size_of(udim)):
            q.destroy()
            q = None
        if q is None and create:
            from trafilatura3d.paint.paint import Pages
            w, h = ts.size_of(udim)
            q = Pages(self.ev.device, (w, h), "rgba16float")
            clear = np.zeros((h, w, 4), np.float32)
            clear[..., 3] = 1.0  # opaque: painting is a plain mix toward the value
            q.write(clear)
            self._quick_masks[(ts.name, udim)] = q
            self._quick_rev += 1
        return q

    def toggle_quick_mask(self):
        if self.doc is None:
            return
        self.quick_edit = not self.quick_edit
        self.quick_action.setChecked(self.quick_edit)
        if self.quick_edit:
            for udim in self.doc.ts.tiles:
                self.quick_pages(self.doc.ts, udim, create=True)
        self.log("Quick mask edition " + ("on: paint what to protect (U to leave)" if self.quick_edit else "off"))
        self.refresh_materials()

    def clear_quick_mask(self):
        for q in self._quick_masks.values():
            q.destroy()
        self._quick_masks, self.quick_edit = {}, False
        self.quick_action.setChecked(False)
        self.refresh_materials()

    def invert_quick_mask(self):
        """I: only in quick mask edition (the page)."""
        if self.doc is None or not self.quick_edit:
            return
        for udim in self.doc.ts.tiles:
            q = self.quick_pages(self.doc.ts, udim, create=True)
            a = q.read()
            a[..., :3] = 1.0 - a[..., :3]
            a[..., 3] = 1.0
            q.write(a)
        self._quick_rev += 1
        self.refresh_materials()

    def _quick_overlay(self, ts, udim, base):
        """The material's base color with the quick mask in red over it (edit mode)."""
        from trafilatura3d.paint.nodes import BLEND
        sc = self.view_compiler(udim)
        q = sc._pages_job(self.quick_pages(ts, udim, create=True), ("quick", ts.name, self._quick_rev),
                            ts.size_of(udim),
                            "16f", True)
        red = sc.constant((1.0, 0.1, 0.1, 1.0), ts.size_of(udim), "16f")
        return sc._job(BLEND, {"mode": 0, "opacity": 0.5}, {"below": (base, "output"), "top": (red, "output"),
                                                            "mask": (q, "output")}, ts.size_of(udim), "16f")

    # --- Geometry mask (Geometry mask page) -------------------------------------------------------------

    def _masking(self) -> list:
        """The selected layer and its folders (their Geometry masks mask it too)."""
        doc = self.doc
        if doc is None or doc.layer() is None:
            return []
        uid = doc.layer().uid
        return [layer for layer in doc.ts.walk()
                if layer.uid == uid or (layer.kind == "folder" and any(c.uid == uid for c in layer.walk()))]

    def _excluded_names(self) -> set:
        """The meshes excluded for the selected layer (Geometry mask, Mesh names)."""
        return {n for layer in self._masking() for n in layer.excluded_meshes()}

    def _excluded_tiles(self) -> set:
        """The UDIM tiles excluded for the selected layer (Geometry mask, UV Tiles)."""
        return {t for layer in self._masking() if layer.geometry_type == "UV Tiles" for t in layer.geometry_tiles}

    def _hidden(self, mesh_index: int, material: int, udim: int) -> bool:
        """A mesh's part in a tile hidden from the view and the brush (Hide excluded geometry)."""
        if not self.hide_excluded or self.doc is None:
            return False
        s = self.doc.session
        if material != s.material_index(self.doc.ts):
            return False
        return udim in self._excluded_tiles() or (mesh_index < len(s.objects)
                                                  and s.objects[mesh_index].name in self._excluded_names())

    def set_hide_excluded(self, on: bool):
        """Hide excluded geometry (Alt+H): the parts the selected layer excludes are hidden, so what is under
        them can be painted."""
        self.hide_excluded = on
        self.hide_excluded_button.setChecked(on)
        self.view3d.camera_cache = None
        self.refresh_materials()

    def edit_geometry_mask(self, uid):
        """Clicks in the viewport include / exclude the mesh under the pointer in the layer's Geometry mask
        (again, or Escape: the edition ends)."""
        self.geometry_edit = None if self.geometry_edit == uid else uid
        self._geometry_overlay()
        self.log("Geometry mask edition " + ("on: click the meshes to include / exclude them" if self.geometry_edit
                                             else "off"))

    def _geometry_toggle(self, x, y):
        view, cam = self._camera()
        under = cam.hit(x, y)
        s, ts = self.doc.session, self.doc.ts
        if under is None or under[0][3] <= 0 or int(round(under[0][3])) - 1 != s.material_index(ts):
            return
        tris = s.triangles(ts)
        t = tris.pick(under[0][:3].astype(np.float64))
        layer = ts.find(self.geometry_edit)
        if layer is None:
            self.geometry_edit = None
            return
        if layer.geometry_type == "UV Tiles":  # the tile under the pointer
            c = tris.uv[t].mean(0)
            self.layers.set_geometry_tiles(layer.uid, set(layer.geometry_tiles) ^ {udim_at(c[0], c[1])})
        else:
            name = s.objects[int(tris.object[t])].name
            self.layers.set_geometry_excluded(layer.uid, set(layer.geometry_excluded) ^ {name})
        self._geometry_overlay()

    def _geometry_overlay(self):
        """While editing a Geometry mask: the excluded meshes' triangles outlined in red."""
        r = self.view3d.renderer
        layer = self.doc.ts.find(self.geometry_edit) if self.doc is not None and self.geometry_edit else None
        if layer is None:
            r.extra_lines = None
            self.view3d.refresh()
            return
        s = self.doc.session
        tris = s.triangles(self.doc.ts)
        if not len(tris.object):
            sel = []
        elif layer.geometry_type == "UV Tiles":
            c = tris.uv.mean(1)
            tiles = 1001 + np.floor(c[:, 0]).astype(int) + 10 * np.floor(1.0 - c[:, 1]).astype(int)
            sel = np.where(np.isin(tiles, layer.geometry_tiles))[0]
        else:
            names = np.array([o.name for o in s.objects])
            sel = np.where(np.isin(names[tris.object], layer.excluded_meshes()))[0]
        if len(sel) > 20000:
            sel = sel[:: len(sel) // 20000 + 1]
        if not len(sel):
            r.extra_lines = None
            self.view3d.refresh()
            return
        p = tris.pos[sel]
        seg = np.empty((len(p) * 6, 7), np.float32)
        for k in range(3):
            seg[k * 2::6, :3] = p[:, k]
            seg[k * 2 + 1::6, :3] = p[:, (k + 1) % 3]
        seg[:, 3:] = (1.0, 0.25, 0.2, 0.9)
        r.extra_lines = seg
        self.view3d.refresh()

    # --- Clone and Smudge ------------------------------------------------------------------------------

    def set_clone(self, follow: bool):
        if self.tools.clone_follow != follow:
            self.tools.clone_delta = None
        self.tools.clone_follow = follow
        self.set_tool("clone")

    def set_clone_source(self, x, y) -> bool:
        """V + click: the clone source is the surface under the pixel."""
        if self.doc is None:
            return False
        view, cam = self._camera()
        under = cam.hit(x, y)
        if under is None or under[0][3] <= 0:
            return False
        self.tools.clone_source = tuple(float(v) for v in under[0][:3])
        self.tools.clone_delta = None
        self.log("Clone source set")
        self.properties.rebuild()
        return True

    def _screen(self, view, p):
        clip = np.asarray(view.view_proj, np.float64) @ np.array([*p, 1.0])
        if clip[3] <= 1e-9:
            return None
        return ((clip[0] / clip[3] * 0.5 + 0.5) * view.width, (0.5 - clip[1] / clip[3] * 0.5) * view.height)

    def _content_sources(self, tiles):
        """{target key: the stack below the layer for that channel (in its tile) when it is in Passthrough, else
        None} (what Clone and Smudge take besides the layer's own painting)."""
        doc = self.doc
        layer = doc.layer()
        out = {}
        for tg in (t for tile in tiles for t in tile.targets):
            ch, udim = tg.key[1], tg.key[2]
            below = None
            if ch != MASK and layer is not None and layer.blend_of(ch).mode == "passthrough":
                sc = doc.session.compiler(udim)
                job = sc.channel_at(doc.ts, ch, layer.uid)
                if job is not None:
                    job = sc.uv_only(doc.ts, job)
                    self.ev.cook([job])
                    below = self.ev.texture(job, "output")
            out[tg.key] = below
        return out

    def _clone_sources(self, tiles, view, camera, x, y) -> bool:
        """Each target's content seen from the clone source's offset (relative: the offset kept from the first
        stroke after V; absolute: from the V point every stroke)."""
        st, s, ts = self.tools, self.doc.session, self.doc.ts
        if st.clone_source is None:
            self.log("Clone: press V and click the mesh to set the source")
            return False
        delta = st.clone_delta if st.clone_follow else None
        if delta is None:
            src = self._screen(view, st.clone_source)
            if src is None:
                self.log("Clone: the source is behind the camera")
                return False
            delta = (src[0] - x, src[1] - y)
            if st.clone_follow:
                st.clone_delta = delta
        belows = self._content_sources(tiles)
        for tile in tiles:
            for tg in tile.targets:
                entry = tg.pages.dense()
                tg.source = s.brush_engine().resample(tile.maps, view, camera, entry, belows[tg.key], delta,
                                                      s.material_index(ts), s.bbox_diagonal(), tile.udim)
                entry.destroy()
        return True

    def quick_properties(self, global_pos) -> None:
        """Mouse right in the viewport: the Properties in a popup at the cursor (painting mode, a project open)."""
        if self.doc is None or self.mode != "painting":
            return
        self._quick = QuickProperties(self, global_pos)
        self._quick.show()

    def surface_point(self, x, y):
        """The world point of the mesh under a viewport pixel, or None."""
        if self.doc is None:
            return None
        _view, cam = self._camera()
        got = cam.hit(x, y)
        if got is None or got[0][3] <= 0:
            return None
        return got[0][:3].astype(np.float64)

    def pick_material(self, x, y) -> bool:
        """Material Picker: the Texture Set's channels (all layers) under a viewport pixel become the previous
        tool's material (its uniform values), then that tool is equipped again."""
        got = self.channel_values_at(x, y)
        if got is None:
            return False
        ts, values = got
        prev = self._before_picker
        for ch, value in values.items():
            if prev == "projection":
                self.tools.proj_source(ch).color = value
            else:
                self.tools.values[ch] = value
        self.log(f"Material picked on {ts.name}")
        self.set_tool(prev if prev in self.tool_actions and prev != "material_picker" else "paint")
        return True

    def channel_values_at(self, x, y, only=None):
        """(Texture Set, {channel: rgb}) of the stack (all layers) on the mesh under a viewport pixel, the
        channels' values as stored (gray channels repeated); `only`: just those channels. None off the mesh."""
        doc = self.doc
        if doc is None:
            return None
        _view, cam = self._camera()
        under = cam.hit(x, y)
        if under is None or under[0][3] <= 0:
            return None
        s = doc.session
        name = s.materials[int(round(under[0][3])) - 1]
        ts = next((t for t in doc.project.texture_sets if t.material == name), None)
        if ts is None:
            return None
        tris = s.triangles(ts)
        p = under[0][:3].astype(np.float64)
        uv = tris.uv_at(tris.pick(p), p)
        udim = udim_at(float(uv[0]), float(uv[1]))
        if udim not in ts.tiles:
            return None
        uv = local_uv(uv, udim)  # in the tile's texture
        values = {}
        for ch in ts.channel_ids():
            if only is not None and ch not in only:
                continue
            job = self.view_compiler(udim).channel(ts, ch)  # what the viewport shows
            self.ev.cook([job])
            raw = self.ev.read_pixel(job, "output", float(uv[0]), float(uv[1]))
            px = raw.astype(np.float64)
            if raw.dtype.kind == "u":  # unorm storage
                px /= float(np.iinfo(raw.dtype).max)
            values[ch] = [float(px[0])] * 3 if FORMATS[ts.format_of(ch)][0] else [float(v) for v in px[:3]]
        return ts, values

    def eyedrop_at(self, pos, shift, channel):
        """The color pickers' eyedropper with Shift over the 3D view (global point): the edited channel's value on
        the mesh (Color picker page: no lossy display conversion). None elsewhere (the screen's pixel is used)."""
        v = self.view3d
        if self.doc is None or not shift or not v.isVisible():
            return None
        local = v.mapFromGlobal(pos)
        if not v.rect().contains(local):
            return None
        r = v.get_pixel_ratio()
        got = self.channel_values_at(local.x() * r, local.y() * r, only=[channel or "basecolor"])
        return next(iter(got[1].values())) if got is not None and got[1] else None

    def _touched(self):
        """The stroke wrote its entries: their revisions change (the stack recooks them)."""
        for key in self._stroke_keys:
            if key[0] == self.QUICK:
                self._quick_rev += 1
            else:
                self.doc.session.store.touched(*key)

    def _limit(self, udim: int):
        """The quick mask protecting the strokes in a tile (outside its edition), whole (the stroke destroys it), or
        None."""
        q = None if self.quick_edit else self._quick_masks.get((self.doc.ts.name, udim))
        return None if q is None else q.dense()

    def _polygon_under(self, x, y):
        """The triangles the Polygon Fill tool takes under a viewport pixel (its selection mode), or None."""
        view, cam = self._camera()
        ts = self.doc.ts
        under = cam.hit(x, y)
        if under is None or under[0][3] <= 0 or int(round(under[0][3])) - 1 != self.doc.session.material_index(ts):
            return None
        tris = self.doc.session.triangles(ts)
        t = tris.pick(under[0][:3].astype(np.float64))
        return None if t is None else tris.select(t, self.tools.polygon_mode)

    def _polygon_fill(self, x, y):
        """Fill what is under the pointer that the stroke has not filled yet (dragging fills more)."""
        sel = self._polygon_under(x, y)
        if sel is None:
            return
        new = [int(i) for i in sel if int(i) not in self._filled]
        if not new:
            return
        self._filled.update(new)
        if self.stroke.fill(self.doc.session.triangles(self.doc.ts).uv[new]):
            self._touched()
            if not self._stroke_timer.isActive():
                self._stroke_timer.start()

    def _polygon_hover(self, pos) -> bool:
        """The Polygon Fill tool's cursor: the outline of what a click fills."""
        r = self.view3d.renderer
        sel = self._polygon_under(*pos)
        if sel is None:
            r.extra_lines = None
            return True
        tris = self.doc.session.triangles(self.doc.ts)
        if len(sel) > 20000:  # a whole mesh or chunk: its first triangles are enough to show it
            sel = sel[:20000]
        p = tris.pos[sel]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        p = p + n[:, None] * self.doc.session.bbox_diagonal() * 1e-3
        seg = np.empty((len(p) * 6, 7), np.float32)
        for k in range(3):
            seg[k * 2::6, :3] = p[:, k]
            seg[k * 2 + 1::6, :3] = p[:, (k + 1) % 3]
        seg[:, 3:] = (0.35, 0.65, 1.0, 0.9)
        r.extra_lines = seg
        return True

    def stroke_add(self, x, y, pressure):
        if self.manip._drag is not None:
            from PySide6.QtWidgets import QApplication
            self.manip.drag(x, y, bool(QApplication.keyboardModifiers() & Qt.ShiftModifier))
            return
        if self.tools.tool == "path":
            self.path_tool.drag(x, y)
            return
        if self.stroke is None:
            return
        if self.tools.tool == "polygon_fill":
            self._polygon_fill(x, y)
            self.hover((x, y))
            return
        if self._line:  # a straight line is drawn by the click alone
            return
        px, py = x, y
        if self.tools.lazy:  # Lazy Mouse: the paint point trails the cursor at the distance
            lx, ly = self._lazy
            d = math.hypot(x - lx, y - ly)
            r = self.tools.lazy_distance * self.view3d.get_pixel_ratio()
            if d <= r and self.stroke.last_point is not None:
                self.hover((x, y))
                return
            k = (1 - r / d) if d > r else 1.0
            px, py = lx + (x - lx) * k, ly + (y - ly) * k
            self._lazy = (px, py)
        self._paint_points([(px, py, pressure)])
        self._last_end = (px, py)
        self.hover((px, py))

    def stroke2d_begin(self, x, y, pressure, radius) -> bool:
        """A stroke in the 2D view: (x, y) and `radius` in texels of the 1001 tile's grid (Stroke.add_texture)."""
        tiles = self._tile_targets()
        if tiles is None:
            return False
        s, ts, t = self.doc.session, self.doc.ts, self.tools
        self.stroke = Stroke(s.brush_engine(), tiles, None, [], s.material_index(ts), t.brush, 1.0,
                             self._alpha_name(), seed=int(time.time() * 1000) & 0xFFFF)
        self._stroke_keys = [k.key for k in self.stroke.targets()]
        self._freeze_stroke()
        self._line = False
        self.stroke2d_add(x, y, pressure, radius)
        return True

    def stroke2d_add(self, x, y, pressure, radius):
        if self.stroke is not None and self.stroke.add_texture([(x, y, pressure)], radius, self.doc.ts.size):
            self._touched()
            if not self._stroke_timer.isActive():
                self._stroke_timer.start()
            self._view2d_timer.start()

    def stroke_end(self):
        if self.manip._drag is not None:
            self.manip.release()
            return
        if self.tools.tool == "path":
            self.path_tool.release()
            return
        if self.stroke is None:
            return
        self._stroke_timer.stop()
        stroke = self.stroke
        before = stroke.end()
        for tg in stroke.targets():  # per texel values of the stroke (Projection, Clone)
            if tg.source is not None:
                tg.source.destroy()
        for tile in stroke.tiles:
            for tex in (tile.stencil, tile.limit):
                if tex is not None:
                    tex.destroy()
        label = {"eraser": "Erase stroke", "projection": "Projection stroke", "polygon_fill": "Polygon fill",
                 "smudge": "Smudge stroke", "clone": "Clone stroke"}.get(self.tools.tool, "Paint stroke")
        self._frozen = {}
        self.ev.hold("stroke", [])
        self._refresh_stroke(padded=True)  # the painted slots with the padding again
        self.stroke = None
        if self._fast:  # the slots Fast painting left behind
            self.live.schedule()
        else:  # the refresh the stroke paused, or the cooking ahead
            self.live.resume()
        v = self.view3d
        self._last_view = v.camera_key(*v.physical_size())
        if any(k[0] == self.QUICK for k in self._stroke_keys):  # the quick mask has no History
            before.destroy()
            return
        self.doc.stroke_done(label, before)
        self.layers.schedule_thumbnails()
        self._view2d_timer.start()

    def key(self, key, mods):
        """Keys the viewport gets besides the menu actions' shortcuts: Escape ends a Geometry mask edition; the Path
        tool's (Escape, Enter, Delete)."""
        if self.geometry_edit is not None and str(key) == "Escape":
            self.edit_geometry_mask(self.geometry_edit)
            return
        if self.tools.tool == "path":
            self.path_tool.key(key)

    def double_click(self, x, y):
        if self.tools.tool == "path":
            self.path_tool.double_click(x, y)


def _menu_toggle(text: str, tip: str):
    """A toggle of the contextual toolbar with its settings menu on the arrow (Lazy Mouse, Symmetry): the arrow
    has its own room on the right, the text is never under it."""
    from PySide6.QtWidgets import QToolButton
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(True)
    b.setPopupMode(QToolButton.MenuButtonPopup)
    b.setStyleSheet("QToolButton { padding: 2px 20px 2px 6px; }"
                    "QToolButton::menu-button { width: 16px; border-left: 1px solid #3a3a3a; }")
    return b


def _helper_image(rows, view_height, ratio):
    """The keyboard helper's panel: (RGBA8 premultiplied image, x, y) at the bottom left of the view (layout ours:
    the page gives the place, not the look). Rows (binding, action)."""
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
    font = QFont()
    font.setPixelSize(round(12 * ratio))
    fm = QFontMetrics(font)
    pad, gap, line = round(8 * ratio), round(16 * ratio), fm.height()
    keys_w = max(fm.horizontalAdvance(k) for k, _ in rows)
    w = pad * 2 + keys_w + gap + max(fm.horizontalAdvance(n) for _, n in rows)
    h = pad * 2 + line * len(rows)
    img = QImage(w, h, QImage.Format_RGBA8888_Premultiplied)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(20, 20, 20, 200))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, w, h, 4 * ratio, 4 * ratio)
    p.setFont(font)
    for i, (keys, name) in enumerate(rows):
        y = pad + i * line + fm.ascent()
        p.setPen(QColor("#f0a030"))
        p.drawText(pad, y, keys)
        p.setPen(QColor("#dddddd"))
        p.drawText(pad + keys_w + gap, y, name)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4).copy()
    margin = round(12 * ratio)
    return a, margin, max(view_height - h - margin, 0)


class QuickProperties(QFrame):
    """Mouse right in the viewport: the Properties window as a popup over the viewport (Shortcuts page). The
    Properties panel itself is borrowed from its dock while
    the popup is open and given back when it closes. Size ours."""

    def __init__(self, workspace, pos):
        super().__init__(workspace, Qt.Popup)
        self.ws = workspace
        self.setFrameShape(QFrame.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: bold; padding: 2px 4px;")
        self._retitle()
        workspace.properties.title_changed.connect(self._retitle)
        lay.addWidget(self.title)
        lay.addWidget(workspace.properties)
        workspace.properties.show()
        screen = workspace.screen().availableGeometry()
        w, h = 340, min(560, screen.height() - 40)
        x = min(max(pos.x(), screen.left()), screen.right() - w)
        y = min(max(pos.y(), screen.top()), screen.bottom() - h)
        self.setGeometry(x, y, w, h)

    def _retitle(self, *_):
        self.title.setText(self.ws.docks["properties"].titleBarWidget().title.text())  # the dock's, updated first

    def hideEvent(self, e):
        super().hideEvent(e)
        self.ws.properties.title_changed.disconnect(self._retitle)
        if self.ws.properties.parent() is self:
            self.ws.docks["properties"].setWidget(self.ws.properties)
        if self.ws._quick is self:
            self.ws._quick = None
        self.deleteLater()


PAINTING_DOCKS = ("assets", "layers", "ts_settings", "properties")
BAKING_DOCKS = ("bakers", "common", "map_settings", "baking_log")
_DOCK_ICONS = {"bakers": "texture_sets", "common": "properties", "map_settings": "properties", "baking_log": "log",
               **{"assets": "assets", "texture_sets": "texture_sets", "layers": "paint_layer", "ts_settings": "texture_sets",
               "properties": "properties", "display": "display_settings", "shader": "properties", "history": "history",
               "log": "log"}}
