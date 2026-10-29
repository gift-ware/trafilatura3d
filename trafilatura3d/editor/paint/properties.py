"""The Properties panel (Properties page): its content follows the selection. A Fill layer shows its Fill
properties and Material; a Paint layer (or a mask) shows the current tool's properties: Brush, Alpha and
Material (Paint tool), Channels (Eraser), Grayscale when painting a mask. Folders have none ("No properties
found", as in the application).

Fill layers and fill effects have the Material mode: a material (a Trafilatura 3D graph, paint/materials.py) for
all the channels, with its parameters; the channels it does not provide keep their own source.

Fills project with UV, Tri-planar or Planar projection (their parameters as numbers: the 3D manipulator of the
viewport is not offered yet).

Not offered yet: the brush and material previews, Stencil, Physics, the Material mode of the brush, the
Spherical, Cylindrical and Warp projections, Scale mode Physical Size."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)
from shiboken6 import isValid

from trafilatura3d.editor import theme
from trafilatura3d.editor.paint.widgets import ChannelButtons, PressureButton, PSlider, Swatch, header
from trafilatura3d.editor.widgets import HISTOGRAM_GRAY, HISTOGRAM_RGB, LevelsEditor
from trafilatura3d.paint.brush import (ALIGNMENTS, ALPHAS, DEFAULT_ALPHA, PROJECTION_MODES, SIZE_SPACES, TILING_MODES,
                                   BrushSettings, ProjectionSettings)
from trafilatura3d.paint.channels import CHANNELS, FORMATS, is_gray
from trafilatura3d.nodes.base import Param, ParamKind, PortType
from trafilatura3d.paint.model import (FILTERING, LEVELS_COLORS, MASK_KEY, MATCH_PER_TILE, PROJECTIONS, SHAPE_CROP,
                                   UV_WRAP, ChannelSource, Levels)

TOOLS = ("paint", "eraser", "path", "projection", "polygon_fill", "smudge", "clone", "material_picker")
from trafilatura3d.paint.brush import TEXTURE_ALPHA  # noqa: E402 - a tool's alpha naming a project texture
SEED = "$seed"  # graph_nodes.SEED
SEED_PARAM = Param(ParamKind.INT, 0, "Random Seed", 0, 10000, editor="Random seed")
TOOL_LABEL = {"paint": "Paint", "eraser": "Eraser", "projection": "Projection", "polygon_fill": "Polygon fill",
              "smudge": "Smudge", "clone": "Clone", "material_picker": "Material picker", "path": "Path"}
LEVELS_ATTRS = ("in_low", "in_mid", "in_high", "out_low", "out_high")  # the levels editor's values, in order
POLYGON_MODES = ("Triangle Fill", "Polygon Fill", "Mesh Fill", "UV Chunk Fill")  # Polygon Fill page: Fill mode
HIT_TESTS = ("Intersection", "Contained center")


@dataclass
class ToolState:
    """The painting tools' settings (each tool keeps its own brush, as in the application)."""
    tool: str = "paint"
    brushes: dict = field(default_factory=lambda: {t: BrushSettings() for t in TOOLS})
    alphas: dict = field(default_factory=lambda: {t: DEFAULT_ALPHA for t in TOOLS})
    values: dict = field(default_factory=dict)  # Paint tool material: channel -> RGB
    disabled: set = field(default_factory=set)  # Paint tool channels turned off
    erase_disabled: set = field(default_factory=set)  # Eraser channels turned off
    mask_value: float = 1.0  # Grayscale painted on masks
    # Contextual toolbar: Mirror symmetry (axis of the scene, position) and Lazy Mouse (distance in pixels,
    # ours: the documentation gives no unit or default).
    symmetry: bool = False
    sym_type: str = "Mirror"  # Mirror or Radial (Symmetry pages)
    sym_axis: int = 0
    sym_position: float = 0.0  # Mirror: the plane's offset along the axis
    radial_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Radial: a point of the axis
    radial_copies: int = 4  # ours: the pages do not list the number of copies
    radial_flip: str = "None"
    sym_show_cursor: bool = True
    sym_hide_painting: bool = False
    sym_show_plane: bool = True
    sym_show_axis: bool = True
    sym_show_intersection: bool = False
    lazy: bool = False
    lazy_distance: float = 30.0
    # Projection tool: its material (a uniform value or a resource per channel) and the image's transform.
    projection: ProjectionSettings = field(default_factory=ProjectionSettings)
    proj_sources: dict = field(default_factory=dict)  # channel -> ChannelSource
    proj_disabled: set = field(default_factory=set)
    # Polygon Fill: the rectangle's Fill mode and Hit-test, its Blending, its own channels (Material). Defaults
    # ours where the page gives none.
    polygon_mode: str = "Triangle Fill"
    polygon_hit: str = "Intersection"
    polygon_visible: bool = True
    polygon_blend: str = "normal"
    polygon_opacity: float = 1.0
    polygon_disabled: set = field(default_factory=set)
    # Stencil of the Paint tool and the Eraser (Paint page): a grayscale resource ("" none, a texture, a
    # "node:" procedural) masking the strokes from the viewport, moved like the projection (S + drag).
    stencils: dict = field(default_factory=dict)  # tool -> ChannelSource
    stencil_xf: dict = field(default_factory=dict)  # tool -> ProjectionSettings
    # Clone tool: the source set with V (world position), its behavior (follow = relative, the documentation's
    # default) and the screen offset kept between strokes in relative mode.
    clone_source: tuple | None = None
    clone_follow: bool = True
    clone_delta: tuple | None = None

    @property
    def brush(self) -> BrushSettings:
        return self.brushes[self.tool]

    def symmetry_copies(self) -> list | None:
        """The copies the symmetry adds to every stamp (brush.symmetry_copies), None when it is off."""
        from trafilatura3d.paint.brush import symmetry_copies
        if not self.symmetry:
            return None
        if self.sym_type == "Mirror":
            return symmetry_copies("Mirror", self.sym_axis, self.sym_position)
        return symmetry_copies("Radial", self.sym_axis, self.radial_position, self.radial_copies, self.radial_flip)

    def stencil(self, tool: str | None = None) -> ChannelSource | None:
        """The tool's stencil, None when it has none."""
        s = self.stencils.get(tool or self.tool)
        return s if s is not None and (s.image or s.procedural) else None

    def stencil_settings(self, tool: str | None = None) -> ProjectionSettings:
        return self.stencil_xf.setdefault(tool or self.tool, ProjectionSettings(tiling="No Tiling"))

    def proj_source(self, channel: str) -> ChannelSource:
        if channel not in self.proj_sources:
            self.proj_sources[channel] = ChannelSource(list(self.value(channel)))
        return self.proj_sources[channel]

    def value(self, channel: str) -> list:
        if channel not in self.values:
            d = CHANNELS[channel].default
            self.values[channel] = list(d) * 3 if len(d) == 1 else list(d)
            if channel == "basecolor":
                self.values[channel] = [1.0, 1.0, 1.0]  # ours: a visible default color
        return self.values[channel]


def _color_field(title: str):
    """Color parameters of nodes as Paint color fields (the picker titled after the parameter)."""
    from trafilatura3d.editor.paint.widgets import ColorField
    return lambda value: ColorField(value, title)


def alpha_icon(name: str, size: int = 48) -> QIcon:
    a = np.clip(ALPHAS[name](size), 0, 1)
    g = np.ascontiguousarray((a * 255).astype(np.uint8))
    img = QImage(g.data, size, size, size, QImage.Format_Grayscale8).copy()
    return QIcon(QPixmap.fromImage(img))


class PropertiesPanel(QScrollArea):
    title_changed = Signal(str)
    tool_changed = Signal()  # the brush or the tool material changed (contextual toolbar follows)

    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self._subject = None  # what the panel shows (`rebuild`)
        self._closed: set = set()  # (node type, group path) of the parameter groups collapsed (`_grouped_params`)
        self.rebuild()

    @property
    def doc(self):
        return self.ws.doc

    def signature(self):
        """What a document edit or a selection can change in the panel (`PaintWorkspace` makes it again when this
        changes): the document, the selection, the tool, the display and mode, the whole layer holding the selection,
        the Texture Set's and the project's settings, the project's images. The tools' own settings rebuild it
        themselves."""
        import dataclasses
        doc = self.doc
        if doc is None or doc.ts is None:
            return id(doc)
        ts, p = doc.ts, doc.project
        owner = ts.owner_of(doc.selected)[0] if doc.selected else None
        return (id(doc), doc.ts_name, doc.selected, doc.mask_selected, tuple(doc.multi), self.ws.tools.tool,
                self.ws.display, self.ws.mode, repr(owner.to_dict()) if owner is not None else None,
                repr([getattr(ts, f.name) for f in dataclasses.fields(ts) if f.name != "layers"]),
                repr([getattr(p, f.name) for f in dataclasses.fields(p) if f.name != "texture_sets"]),
                doc.session.resources.signature())

    def rebuild(self):
        doc = self.doc
        # The same layer, effect or tool shown again (an edit rebuilt the panel): the scroll stays where it was.
        subject = (doc.selected, doc.mask_selected, self.ws.tools.tool) if doc else None
        keep = self.verticalScrollBar().value() if subject == self._subject else 0
        self._subject = subject
        body = theme.section_lines(QWidget())
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(10, 4, 10, 10)
        self.lay.setSpacing(4)
        layer = doc.layer() if doc else None
        effect = doc.effect() if doc else None
        if effect is not None and effect.kind == "fill":
            self.title_changed.emit("Properties - Fill")
            self._fill(effect)
        elif effect is not None and effect.kind == "levels":
            self.title_changed.emit("Properties - Levels")
            self._levels(effect)
        elif effect is not None and effect.kind in ("filter", "generator"):
            self.title_changed.emit(f"Properties - {effect.kind.capitalize()}")
            self._procedural(effect)
        elif effect is not None and effect.kind == "anchor":
            self.title_changed.emit("Properties - Anchor point")
            self._anchor(effect)
        elif layer is None or (layer.kind == "folder" and not doc.mask_selected):
            self.title_changed.emit("Properties")
            empty = QLabel("No properties found")
            empty.setObjectName("dim")
            self.lay.addWidget(empty)
        elif layer.kind == "fill" and not doc.mask_selected and effect is None:
            self.title_changed.emit("Properties - Fill")
            self._fill(layer)
        else:
            self.title_changed.emit(f"Properties - {TOOL_LABEL[self.ws.tools.tool]}")
            self._tool()
        self.lay.addStretch(1)
        self.setWidget(body)
        body.adjustSize()
        self.verticalScrollBar().setValue(keep)

    # --- tools ----------------------------------------------------------------------------------------

    def _tool(self):
        st = self.ws.tools
        b = st.brush
        if st.tool == "path":
            from trafilatura3d.editor.paint.path_tool import build_properties
            build_properties(self, self.ws.path_tool)
            return
        if st.tool == "material_picker":
            hint = QLabel("Click the mesh: the material under the cursor goes to the previous tool, which comes "
                          "back.")
            hint.setObjectName("dim")
            hint.setWordWrap(True)
            self.lay.addWidget(hint)
            return
        if st.tool == "polygon_fill":  # Polygon Fill page (a rectangle dragged, or a click), no brush
            self._polygon_fill_settings(st)
            return
        if st.tool == "projection":  # Projection page: Mode; the image moves with S + drag in the viewport
            pr = st.projection
            self.lay.addWidget(header("Projection"))
            form = QFormLayout()
            mode = QComboBox()
            mode.addItems(PROJECTION_MODES)
            mode.setCurrentText(pr.mode)
            form.addRow("Mode", mode)
            tiling = QComboBox()
            tiling.addItems(TILING_MODES)
            tiling.setCurrentText(pr.tiling)
            tiling.currentTextChanged.connect(lambda t: (setattr(pr, "tiling", t), self.ws.update_projection_preview()))
            form.addRow("Tiling Mode", tiling)
            self.lay.addLayout(form)
            if self.doc is not None and self.ws._preview_channel(self.doc.ts) is None:  # nothing to project yet
                load = QLabel("Drag a texture, an alpha or a procedural from the Assets onto the viewport (or onto a "
                              "channel below): it shows over the view, painting projects it on the mesh.")
                load.setStyleSheet("color: #e0a040;")
                load.setWordWrap(True)
                self.lay.addWidget(load)
            self._projection_image(st)
        self.lay.addWidget(header("Brush"))

        def slider(name, attr, lo, hi, integer=False, pressure=None):
            s = PSlider(name, getattr(b, attr), lo, hi, integer)
            s.edited.connect(lambda v, _f, attr=attr: (setattr(b, attr, v), self.tool_changed.emit()))
            if pressure is None:
                self.lay.addWidget(s)
                return s
            row = QHBoxLayout()  # the pen pressure button next to a pressure capable parameter (Pen pressure)
            row.addWidget(s, 1)
            pb = PressureButton(getattr(b, pressure), name)
            pb.toggled.connect(lambda on, pressure=pressure: (setattr(b, pressure, on), self.tool_changed.emit()))
            row.addWidget(pb, 0, Qt.AlignBottom)
            self.lay.addLayout(row)
            return s

        self.size = slider("Size", "size", 0.0, 100.0, pressure="size_pressure")
        self.flow = slider("Flow", "flow", 0.0, 100.0, pressure="flow_pressure")
        self.opacity = slider("Stroke opacity", "stroke_opacity", 0.0, 100.0)
        self.spacing = slider("Spacing", "spacing", 1.0, 200.0)
        slider("Angle", "angle", 0.0, 360.0)
        follow = QCheckBox("Follow Path")
        follow.setChecked(b.follow_path)
        follow.toggled.connect(lambda v: setattr(b, "follow_path", v))
        self.lay.addWidget(follow)
        for name, attr in (("Size Jitter", "size_jitter"), ("Flow Jitter", "flow_jitter"),
                           ("Angle Jitter", "angle_jitter"), ("Position Jitter", "position_jitter")):
            slider(name, attr, 0.0, 1.0)
        form = QFormLayout()
        align = QComboBox()
        align.addItems(ALIGNMENTS)
        align.setCurrentText(b.alignment)
        align.currentTextChanged.connect(lambda t: setattr(b, "alignment", t))
        form.addRow("Alignment", align)
        space = QComboBox()
        space.addItems(SIZE_SPACES)
        space.setCurrentText(b.size_space)
        space.currentTextChanged.connect(lambda t: (setattr(b, "size_space", t), self.tool_changed.emit()))
        form.addRow("Size Space", space)
        self.lay.addLayout(form)
        slider("Backface Culling", "backface_culling", 0.0, 180.0)

        self.alpha_picker(st.alphas[st.tool], lambda n: (st.alphas.__setitem__(st.tool, n), self.rebuild()))
        if st.tool in ("paint", "eraser"):
            self._stencil(st)

        ts = self.doc.ts
        if st.tool in ("smudge", "clone"):  # the content of the layer (or the stack below in Passthrough) moves
            if st.tool == "clone":
                self.lay.addWidget(header("Clone"))
                follow = QCheckBox("Clone source behavior: follow")
                follow.setToolTip("On: the source keeps its offset to the brush after each stroke (6). Off: every "
                                  "stroke starts from the point set with V (Ctrl+6).")
                follow.setChecked(st.clone_follow)
                follow.toggled.connect(lambda on: (setattr(st, "clone_follow", on), setattr(st, "clone_delta", None)))
                self.lay.addWidget(follow)
                hint = QLabel("V + click on the mesh sets the source." if st.clone_source is None
                              else "Source set (V + click to move it).")
                hint.setObjectName("dim")
                self.lay.addWidget(hint)
            if not self.doc.mask_selected:
                self.lay.addWidget(header("Channels"))
                btns = ChannelButtons(chans := [(c, CHANNELS[c].short) for c in ts.channel_ids()],
                                      {c for c, _ in chans if c not in st.disabled})
                btns.toggled.connect(lambda c, on: (st.disabled.discard(c) if on else st.disabled.add(c)))
                self.lay.addWidget(btns)
            return
        if self.doc.mask_selected:
            self.lay.addWidget(header("Grayscale"))
            g = PSlider("Grayscale", st.mask_value, 0.0, 1.0)
            g.edited.connect(lambda v, _f: setattr(st, "mask_value", v))
            self.lay.addWidget(g)
            return
        chans = [(c, CHANNELS[c].short) for c in ts.channel_ids()]
        if st.tool == "eraser":
            self.lay.addWidget(header("Channels"))
            btns = ChannelButtons(chans, {c for c, _ in chans if c not in st.erase_disabled})
            btns.toggled.connect(lambda c, on: (st.erase_disabled.discard(c) if on else st.erase_disabled.add(c)))
            self.lay.addWidget(btns)
            return
        if st.tool == "projection":  # a uniform value or a resource per channel, projected from the viewport
            self.lay.addWidget(header("Material"))
            btns = ChannelButtons(chans, {c for c, _ in chans if c not in st.proj_disabled})
            btns.toggled.connect(lambda c, on: (st.proj_disabled.discard(c) if on else st.proj_disabled.add(c),
                                                self.rebuild(), self.ws.update_projection_preview()))
            self.lay.addWidget(btns)
            for c in ts.channel_ids():
                if c in st.proj_disabled:
                    continue
                src = st.proj_source(c)
                node = None
                if src.procedural:
                    from trafilatura3d.paint.graph_nodes import node_of
                    node = node_of(src.procedural)

                def pick(name, c=c):
                    s = st.proj_source(c)
                    if name.startswith("node:"):
                        s.procedural, s.image, s.values = name, "", {}
                    else:
                        s.procedural, s.image = "", name
                    self.rebuild()
                    self.ws.update_projection_preview()
                self._value_row(c, FORMATS[ts.format_of(c)][0], src.color,
                                lambda v, _final, c=c: setattr(st.proj_source(c), "color", list(v)),
                                source=src, pick=pick,
                                on_drop=lambda a, pick=pick: pick(self.ws.projection_resource(a)),
                                kinds=("texture", "alpha", "procedural"))
                if src.procedural and node is not None:  # the procedural's own parameters
                    from trafilatura3d.editor.params_ui import make_editor
                    form = QFormLayout()
                    for name, prm in [(SEED, SEED_PARAM), *node.params.items()]:
                        def edited(v, _final, name=name, src=src):
                            src.values = {**src.values, name: v}
                            self.ws.update_projection_preview()
                        form.addRow(prm.label or name, make_editor(prm, src.values.get(name, prm.default), edited,
                                                                       _color_field(prm.label or name)))
                    self.lay.addLayout(form)
            return
        self.lay.addWidget(header("Material"))
        btns = ChannelButtons(chans, {c for c, _ in chans if c not in st.disabled})
        btns.toggled.connect(lambda c, on: (st.disabled.discard(c) if on else st.disabled.add(c), self.rebuild()))
        self.lay.addWidget(btns)
        for c in ts.channel_ids():
            if c in st.disabled:
                continue
            self._value_row(c, FORMATS[ts.format_of(c)][0], st.value(c),
                            lambda v, _final, c=c: st.values.__setitem__(c, list(v)))

    def _polygon_fill_settings(self, st):
        """Polygon Fill page, in its order: Fill mode and Hit-test (of the rectangle dragged; a click takes what is
        under it), Blending (mode and opacity, the layer stack's modes), Material (its channels and values; a
        mask: the grayscale)."""
        from trafilatura3d.paint import blending

        def combo(label, items, current, setter, data=None):
            box = QComboBox()
            for i, text in enumerate(items):
                box.addItem(text, data[i] if data else text)
            box.setCurrentIndex(max(box.findData(current), 0))
            box.currentIndexChanged.connect(lambda i: setter(box.itemData(i)))
            form = QFormLayout()
            form.addRow(label, box)
            self.lay.addLayout(form)

        self.lay.addWidget(header("Fill mode"))
        combo("Fill mode", POLYGON_MODES, st.polygon_mode, lambda v: setattr(st, "polygon_mode", v))
        self.lay.addWidget(header("Hit-test"))
        combo("Hit-test", HIT_TESTS, st.polygon_hit, lambda v: setattr(st, "polygon_hit", v))
        visible = QCheckBox("Visible polygons only")
        visible.setChecked(st.polygon_visible)
        visible.toggled.connect(lambda on: setattr(st, "polygon_visible", on))
        self.lay.addWidget(visible)
        self.lay.addWidget(header("Blending"))
        modes = [m for m in blending.IDS if m not in ("passthrough", "disable")]  # a layer's own, not a tool's
        combo("Blending mode", [blending.LABEL[m] for m in modes], st.polygon_blend,
              lambda v: setattr(st, "polygon_blend", v), modes)
        opacity = PSlider("Opacity", st.polygon_opacity * 100, 0.0, 100.0)
        opacity.edited.connect(lambda v, _f: setattr(st, "polygon_opacity", v / 100))
        self.lay.addWidget(opacity)
        if self.doc.mask_selected:
            self.lay.addWidget(header("Grayscale"))
            g = PSlider("Grayscale", st.mask_value, 0.0, 1.0)
            g.edited.connect(lambda v, _f: setattr(st, "mask_value", v))
            self.lay.addWidget(g)
            return
        ts = self.doc.ts
        self.lay.addWidget(header("Material"))
        chans = [(c, CHANNELS[c].short) for c in ts.channel_ids()]
        btns = ChannelButtons(chans, {c for c, _ in chans if c not in st.polygon_disabled})
        btns.toggled.connect(lambda c, on: (st.polygon_disabled.discard(c) if on else st.polygon_disabled.add(c),
                                            self.rebuild()))
        self.lay.addWidget(btns)
        for c in ts.channel_ids():
            if c not in st.polygon_disabled:
                self._value_row(c, FORMATS[ts.format_of(c)][0], st.value(c),
                                lambda v, _final, c=c: st.values.__setitem__(c, list(v)))

    def _projection_image(self, st):
        """The projected image's controls (ours, besides S + drag in the viewport): rotation, scale, offset, flips,
        the preview's opacity; the projection's falloff as in Mari (fade by angle, occlusion, shown in red)."""
        pr = st.projection
        ws = self.ws
        self.lay.addWidget(header("Image"))
        hint = QLabel("S + left drag: rotate (Shift snaps) · S + right drag: scale · S + middle drag: move")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        self.lay.addWidget(hint)

        def slider(label, value, lo, hi, set_value):
            s = PSlider(label, value, lo, hi)
            s.edited.connect(lambda v, _f: (set_value(v), ws.update_projection_preview()))
            self.lay.addWidget(s)
        slider("Rotation", (pr.rotation + 180.0) % 360.0 - 180.0, -180.0, 180.0, lambda v: setattr(pr, "rotation", v))
        slider("Scale", pr.scale, 0.05, 10.0, lambda v: setattr(pr, "scale", max(v, 0.01)))
        slider("Offset X", pr.offset[0], -1.0, 1.0, lambda v: setattr(pr, "offset", (v, pr.offset[1])))
        slider("Offset Y", pr.offset[1], -1.0, 1.0, lambda v: setattr(pr, "offset", (pr.offset[0], v)))
        slider("Preview opacity", pr.opacity, 0.0, 1.0, lambda v: setattr(pr, "opacity", v))
        row = QHBoxLayout()
        for label, attr in (("Flip horizontally", "flip_h"), ("Flip vertically", "flip_v")):
            box = QCheckBox(label)
            box.setChecked(getattr(pr, attr))
            box.toggled.connect(lambda on, attr=attr: (setattr(pr, attr, on), ws.update_projection_preview()))
            row.addWidget(box)
        self.lay.addLayout(row)
        self.lay.addWidget(header("Falloff"))
        slider("Fade start angle", pr.fade_start, 0.0, 90.0, lambda v: setattr(pr, "fade_start", v))
        slider("Fade end angle", pr.fade_end, 0.0, 90.0, lambda v: setattr(pr, "fade_end", max(v, pr.fade_start)))
        for label, attr, tip in (("Occlusion", "occlusion", "Project only on the surface the view sees"),
                                 ("Show falloff", "show_falloff", "Red over the view where the projection fades")):
            box = QCheckBox(label)
            box.setToolTip(tip)
            box.setChecked(getattr(pr, attr))
            box.toggled.connect(lambda on, attr=attr: (setattr(pr, attr, on), ws.update_projection_preview()))
            self.lay.addWidget(box)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: (setattr(st, "projection", ProjectionSettings(tiling=pr.tiling)),
                                       ws.update_projection_preview(), self.rebuild()))
        self.lay.addWidget(reset)

    def alpha_picker(self, current: str, pick):
        """The Alpha section: the current alpha's slot, the alphas and the project's textures (their luminance)
        from its picker or dragged from the Assets; `pick(name)`."""
        from trafilatura3d.editor.paint.assets import ResourceSlot
        self.lay.addWidget(header("Alpha"))
        if current.startswith(TEXTURE_ALPHA):
            asset = {"kind": "texture", "name": current[len(TEXTURE_ALPHA):]}
        else:
            asset = {"kind": "alpha", "name": current if current in ALPHAS else DEFAULT_ALPHA}
        self.lay.addWidget(ResourceSlot(
            self.ws.assets, ("alpha", "texture"),
            lambda a: pick(a["name"] if a["kind"] == "alpha" else TEXTURE_ALPHA + a["name"]), asset, asset["name"]))

    def _stencil(self, st):
        """Stencil (Paint page): a grayscale resource masking the stroke from the viewport, its Tiling Mode; S + left
        drag rotates (Shift snaps), S + right drag scales, S + middle drag moves; N held ignores it."""
        self.lay.addWidget(header("Stencil"))
        tool = st.tool
        src = st.stencil(tool)

        def pick(name):
            if not name:
                st.stencils.pop(tool, None)
            else:
                s = ChannelSource([1.0, 1.0, 1.0])
                if name.startswith("node:"):
                    s.procedural = name
                else:
                    s.image = name
                st.stencils[tool] = s
            self.rebuild()
            self.ws.update_projection_preview()
        from trafilatura3d.editor.paint.assets import ResourceSlot, asset_ref, source_asset
        from trafilatura3d.paint.graph_nodes import node_of
        node = node_of(src.procedural) if src is not None and src.procedural else None
        self.lay.addWidget(ResourceSlot(self.ws.assets, ("texture", "procedural"),
                                        lambda a: pick(asset_ref(a) if a else ""), source_asset(src),
                                        node.label if node else (src.image if src else ""), none="None"))
        if src is None:
            return
        xf = st.stencil_settings(tool)
        form = QFormLayout()
        tiling = QComboBox()
        tiling.addItems(TILING_MODES)
        tiling.setCurrentText(xf.tiling)
        tiling.currentTextChanged.connect(lambda t: (setattr(xf, "tiling", t), self.ws.update_projection_preview()))
        form.addRow("Tiling Mode", tiling)
        self.lay.addLayout(form)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: (st.stencil_xf.__setitem__(tool, ProjectionSettings(tiling=xf.tiling)),
                                       self.ws.update_projection_preview()))
        self.lay.addWidget(reset)

    def _value_row(self, channel, gray, value, on_edit, source=None, pick=None, label=None, on_drop=None,
                   kinds=("texture", "procedural"), stack_slot=None):
        """A channel's box: its uniform value, and with `pick(name)` its resource slot (the `source`'s texture or
        procedural; "" = uniform color) taking the `kinds` (`on_drop(asset)` instead of `pick` when given).
        `stack_slot`: (layer or effect uid, channel or slot, effect resource slot) of a slot of the stack, which
        also takes the anchor points under its layer (Anchor Point page)."""
        from trafilatura3d.editor.paint.assets import ResourceSlot, asset_ref, source_asset
        owner = None
        if stack_slot is not None:
            uid, key, is_slot = stack_slot

            def pick(name):
                self._set_image(uid, key, name, is_slot)
            owner = self.doc.ts.owner_of(uid)[0].uid
            kinds = (*kinds, "anchor")
        box = QFrame()
        box.setStyleSheet("QFrame { background: #262626; border: 1px solid #1a1a1a; border-radius: 3px; }"
                          "QLabel { background: transparent; border: none; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(6, 4, 6, 6)
        title = QLabel(f"<b>{label or CHANNELS[channel].label}</b>")
        title.setAlignment(Qt.AlignCenter)
        v.addWidget(title)
        current = source_asset(source) if pick is not None else None
        if pick is None:
            sub = QLabel("uniform color")
            sub.setAlignment(Qt.AlignCenter)
            sub.setObjectName("dim")
            v.addWidget(sub)
        else:  # textures and procedurals (and alphas, `kinds`) from the picker or dragged from the Assets
            def use(asset):
                if asset is None:
                    pick("")
                else:
                    (on_drop or (lambda a: pick(asset_ref(a))))(asset)
            shown = source.image if current else ""
            if source is not None and source.procedural:
                from trafilatura3d.paint.graph_nodes import node_of
                node = node_of(source.procedural)
                shown = node.label if node else source.procedural
            anchor = source is not None and bool(source.anchor)
            if anchor:
                fx = self.doc.ts.owner_of(source.anchor)[1]
                shown = fx.name if fx else "Deleted anchor point"
                if not any(e.uid == source.anchor for _l, e, _m in self.doc.ts.anchors(below=owner)):
                    shown += " (broken: not under this layer)"
            from trafilatura3d.editor.paint.widgets import accept_assets
            accept_assets(box, kinds, use)
            v.addWidget(ResourceSlot(self.ws.assets, kinds, use, current, shown, none="uniform color", owner=owner))
            if anchor:
                self._anchor_reference(v, source, stack_slot)
        if current is None:
            if gray:  # Height: -1..1 (Height map painting page), the other grayscale channels 0..1
                lo = -1.0 if channel in ("height", "displacement") else 0.0
                s = PSlider("", value[0], lo, 1.0)
                s.edited.connect(lambda x, final: on_edit([x, x, x], final))
                v.addWidget(s)
            else:
                sw = Swatch(value, label or CHANNELS[channel].label, channel)
                sw.edited.connect(on_edit)
                v.addWidget(sw)
        self.lay.addWidget(box)

    # --- fill layers ----------------------------------------------------------------------------------

    def _obj(self, uid):
        """The layer or effect `uid` of the current Texture Set (looked up again after every edit)."""
        layer, fx, _m = self.doc.ts.owner_of(uid)
        return fx or layer

    def _fill(self, layer):
        """Fill properties of a fill layer or a fill effect (`layer` is either)."""
        doc, uid = self.doc, layer.uid
        self.lay.addWidget(header("Fill"))
        form = QFormLayout()

        def edit_proj(attr, value, final, label):
            def do():
                setattr(self._obj(uid).projection, attr, value)
            doc.edit(label, do, merge=None if final else f"{uid}.{attr}", structure=False)

        pr = layer.projection
        mode = pr.mode
        proj = QComboBox()
        proj.addItems((MATCH_PER_TILE, *PROJECTIONS) if doc.project.uv_tiles else PROJECTIONS)
        proj.setCurrentText(mode)
        proj.currentTextChanged.connect(lambda t: (edit_proj("mode", t, True, "Projection"), self.rebuild()))
        form.addRow("Projection", proj)
        if mode == MATCH_PER_TILE:  # no dedicated settings, no deformation controls (the page)
            self.lay.addLayout(form)
        else:
            self._projection_controls(form, pr, mode, uid, edit_proj)

        ts = doc.ts
        if MASK_KEY in layer.material:  # a fill effect of a mask: one grayscale value
            self.lay.addWidget(header("Grayscale"))
            src = layer.material[MASK_KEY]

            def set_gray(v, final):
                def do():
                    self._obj(uid).material[MASK_KEY].color = list(v)
                doc.edit("Grayscale", do, merge=None if final else f"{uid}.mask", structure=False)
            self._value_row(MASK_KEY, True, src.color, set_gray, label="Grayscale", source=src,
                            stack_slot=(uid, MASK_KEY, False))
            if src.procedural:
                self._procedural_params(uid, MASK_KEY, src)
            return
        self.lay.addWidget(header("Material"))
        chans = [(c, CHANNELS[c].short) for c in ts.channel_ids()]
        btns = ChannelButtons(chans, set(layer.material))
        btns.toggled.connect(lambda c, on: self.ws.layers.toggle_fill_channel(uid, c, on))
        self.lay.addWidget(btns)
        provided = self._material_mode(layer)
        if provided is None:
            or_label = QLabel("Or")
            or_label.setAlignment(Qt.AlignCenter)
            self.lay.addWidget(or_label)
        for c in ts.channel_ids():
            src = layer.material.get(c)
            if src is None or (provided and c in provided):
                continue

            def set_color(v, final, c=c):
                def do():
                    self._obj(uid).material[c].color = list(v)
                doc.edit(f"{CHANNELS[c].label} color", do, merge=None if final else f"{uid}.{c}", structure=False)
            self._value_row(c, FORMATS[ts.format_of(c)][0], src.color, set_color, source=src,
                            stack_slot=(uid, c, False))
            if src.procedural:
                self._procedural_params(uid, c, src)

    def _projection_controls(self, form, pr, mode, uid, edit_proj):
        """The projection's settings under its dropdown (`form`): filtering, wrap, crop, its own parameters, the UV
        and 3D transformations and the symmetry."""
        filt = QComboBox()
        filt.addItems(FILTERING)
        filt.setCurrentText(pr.filtering)
        filt.currentTextChanged.connect(lambda t: edit_proj("filtering", t, True, "Filtering"))
        form.addRow("Filtering", filt)
        if mode != "Tri-planar projection":
            wrap = QComboBox()
            wrap.addItems(UV_WRAP)
            wrap.setCurrentText(pr.wrap)
            wrap.currentTextChanged.connect(lambda t: edit_proj("wrap", t, True, "UV Wrap"))
            form.addRow("UV Wrap", wrap)
        if mode != "UV projection":
            crop = QComboBox()
            crop.addItems(SHAPE_CROP)
            crop.setCurrentText(pr.shape_crop)
            crop.currentTextChanged.connect(lambda t: edit_proj("shape_crop", t, True, "Shape crop"))
            form.addRow("Shape crop", crop)
        self.lay.addLayout(form)

        def slider(label, attr, lo, hi):
            s = PSlider(label, getattr(pr, attr), lo, hi)
            s.edited.connect(lambda v, f: edit_proj(attr, v, f, label))
            self.lay.addWidget(s)

        def toggle(label, attr):
            box = QCheckBox(label)
            box.setChecked(getattr(pr, attr))
            box.toggled.connect(lambda on: (edit_proj(attr, on, True, label), self.rebuild()))
            self.lay.addWidget(box)
            return box.isChecked()

        if mode == "Tri-planar projection":
            slider("Hardness", "hardness", 0.0, 1.0)
        if mode == "Planar projection":
            if toggle("Depth culling", "depth_culling"):
                slider("Depth culling - Hardness", "depth_hardness", 0.0, 1.0)
            if toggle("Backface culling", "backface_culling"):
                slider("Backface culling - Angle", "backface_angle", 0.0, 180.0)
                slider("Backface culling - Hardness", "backface_hardness", 0.0, 1.0)
        if mode == "Cylindrical projection":
            slider("Angle", "angle", 1.0, 360.0)
            if toggle("Backface Culling", "backface_culling"):
                slider("Backface Culling - Hardness", "backface_hardness", 0.0, 1.0)
        if mode == "Warp projection":
            slider("Projection depth", "warp_depth", 0.001, 1.0)
            slider("Depth culling - Hardness", "depth_hardness", 0.0, 1.0)
            grid = QFormLayout()
            for label, attr in (("Rows", "warp_rows"), ("Columns", "warp_cols")):
                spin = QSpinBox()
                spin.setRange(1, 32)
                spin.setValue(getattr(pr, attr))
                spin.setEnabled(not pr.warp_points)  # the page: only while no point was modified
                spin.valueChanged.connect(lambda v, attr=attr: edit_proj(attr, v, True, "Rows and columns"))
                grid.addRow(label, spin)
            self.lay.addLayout(grid)
            toggle("Automatic tangents", "warp_tangents")
            m = self.ws.manip
            hs = PSlider("Handle size", m.handle_size, 0.2, 4.0)
            hs.edited.connect(lambda v, _f: (setattr(m, "handle_size", v), m.refresh()))
            self.lay.addWidget(hs)
            gc = Swatch(m.grid_color, "Grid")
            gc.edited.connect(lambda rgb, _f: (setattr(m, "grid_color", (*rgb, 1.0)), m.refresh()))
            gform = QFormLayout()
            gform.addRow("Grid color", gc)
            self.lay.addLayout(gform)
        self.lay.addWidget(header("UV transformations"))
        uv = " (UV)" if mode in ("Tri-planar projection", "Spherical projection") else ""
        slider("Scale" + uv, "tiling", 0.0, 16.0)
        slider("Rotation" + uv, "rotation", 0.0, 360.0)
        if mode != "Tri-planar projection":  # the documentation: no UV Offset with Tri-planar
            for i, axis in enumerate(("Offset U", "Offset V")):
                off = PSlider(axis, pr.offset[i], -1.0, 1.0)

                def set_off(v, f, i=i):
                    o = list(self._obj(uid).projection.offset)
                    o[i] = v
                    edit_proj("offset", o, f, "Offset")
                off.edited.connect(set_off)
                self.lay.addWidget(off)
        if mode != "UV projection":
            self.lay.addWidget(header("3D transformations"))
            for label, attr, lo, hi in (("3D Offset", "offset3d", -1.0, 1.0), ("3D Rotation", "rotation3d", -180.0, 180.0),
                                        ("3D Scale", "scale3d", 0.01, 4.0)):
                for i, axis in enumerate("XYZ"):
                    s = PSlider(f"{label} {axis}", getattr(pr, attr)[i], lo, hi)

                    def set_axis(v, f, attr=attr, i=i, label=label):
                        vec = list(getattr(self._obj(uid).projection, attr))
                        vec[i] = v
                        edit_proj(attr, vec, f, label)
                    s.edited.connect(set_axis)
                    self.lay.addWidget(s)
            # Symmetry (Symmetry page: fill layers with a 3D projection): the content repeated by the symmetry
            self.lay.addWidget(header("Symmetry"))
            st = self.ws.tools
            sym = QCheckBox("Symmetry (the current symmetry settings)")
            sym.setChecked(bool(pr.symmetry))
            spec = lambda: {"type": st.sym_type, "axis": st.sym_axis,  # noqa: E731
                            "position": st.sym_position if st.sym_type == "Mirror" else list(st.radial_position),
                            "copies": st.radial_copies, "flip": st.radial_flip}
            sym.toggled.connect(lambda on: edit_proj("symmetry", spec() if on else {}, True, "Symmetry"))
            self.lay.addWidget(sym)

    def _material_mode(self, owner) -> set | None:
        """The Material mode box (a material resource for all the channels) and the material's parameters;
        returns the channels the material provides, or None without a material."""
        doc, uid = self.doc, owner.uid
        lib = doc.session.material_library
        from trafilatura3d.editor.paint.assets import ResourceSlot
        ref = owner.resource
        title = "No Resource Selected"
        if ref:
            from pathlib import Path
            title = f"{ref['graph']}  ({Path(ref['package']).name})"
        self.lay.addWidget(header("Material mode"))
        self.lay.addWidget(ResourceSlot(
            self.ws.assets, ("material",),
            lambda a: self._set_resource(uid, a and {"package": a["package"], "graph": a["graph"], "values": {}}),
            {"kind": "material", "package": ref["package"], "graph": ref["graph"]} if ref else None, title,
            none="None"))
        if not ref:
            return None
        try:
            provided = set(lib.channels_of(ref["package"], ref["graph"]))
            params = lib.parameters(ref)
        except Exception as e:  # noqa: BLE001 - a missing package: the channels keep their own sources
            missing = QLabel(f"Material not found: {e}")
            missing.setObjectName("dim")
            missing.setWordWrap(True)
            self.lay.addWidget(missing)
            return set()
        if params:
            from trafilatura3d.editor.params_ui import make_editor
            self.lay.addWidget(header("Parameters"))
            form = QFormLayout()
            for ep in params:
                value = ref.get("values", {}).get(ep.id, ep.param.default)

                def edited(v, final, pid=ep.id):
                    def do():
                        r = dict(self._obj(uid).resource)
                        r["values"] = {**r.get("values", {}), pid: v}
                        self._obj(uid).resource = r
                    doc.edit("Material parameter", do, merge=None if final else f"{uid}.{pid}", structure=False)
                form.addRow(ep.param.label or ep.id, make_editor(ep.param, value, edited,
                                                                          _color_field(ep.param.label or ep.id)))
            self.lay.addLayout(form)
        return provided

    def _set_resource(self, uid, ref):
        from trafilatura3d.editor.paint import settings as prefs

        def do():
            obj = self._obj(uid)
            obj.resource = ref
            if ref:  # General preferences: Default UV scale for materials
                obj.projection.tiling = prefs.get("uv_scale")
        self.doc.edit("Material mode", do)
        self.rebuild()

    def _procedural(self, fx):
        """A generator's or filter's properties: the button choosing it (Filter / Generator pages) and its
        parameters (labels of the documentation)."""
        from trafilatura3d.editor.paint.assets import ResourceSlot
        from trafilatura3d.editor.params_ui import make_editor
        from trafilatura3d.paint import graph_nodes as dn
        doc, uid = self.doc, fx.uid
        node = dn.node_of(fx.procedural)
        self.lay.addWidget(header(fx.kind))
        self.lay.addWidget(ResourceSlot(self.ws.assets, (fx.kind,), lambda a: self._set_procedural(uid, a["id"]),
                                        {"kind": fx.kind, "id": fx.procedural} if node else None,
                                        node.label if node else f"Select a {fx.kind}"))
        if node is None:
            return
        if fx.kind == "generator":
            from trafilatura3d.paint.uvmaps import COMPUTED
            missing = sorted({dn.MESH_PORTS[p] for p in node.inputs if p in dn.MESH_PORTS}
                             - set(doc.ts.mesh_maps) - set(COMPUTED))
            if missing:
                note = QLabel("Needs the mesh maps: " + ", ".join(m.replace("_", " ") for m in missing)
                              + " (Baking mode).")
                note.setObjectName("dim")
                note.setWordWrap(True)
                self.lay.addWidget(note)
        self.lay.addWidget(header("Parameters"))

        def row(form, name, prm):
            def edited(v, final):
                def do():
                    self._obj(uid).values = {**self._obj(uid).values, name: v}
                doc.edit(f"{node.label} parameter", do, merge=None if final else f"{uid}.{name}", structure=False)
            form.addRow(prm.label or name, make_editor(prm, fx.values.get(name, prm.default), edited,
                                                       _color_field(prm.label or name)))
        self._grouped_params(node, [(dn.SEED, SEED_PARAM), *node.params.items()], row)
        slots = dn.resource_ports(node, fx.kind)
        if slots:  # the node's other image inputs: resources of the effect, after the parameters
            self.lay.addWidget(header("Resources"))
        for port in slots:
            src = fx.inputs.get(port) or ChannelSource([1.0, 1.0, 1.0])

            def set_color(v, final, port=port):
                def do():
                    self._src(uid, port, True).color = list(v)
                doc.edit("Effect resource", do, merge=None if final else f"{uid}.{port}", structure=False)
            gray = node.inputs[port].type is PortType.GRAY
            self._value_row(port, gray, src.color, set_color, label=node.inputs[port].label or port, source=src,
                            stack_slot=(uid, port, True))
            if src.procedural:
                self._procedural_params(uid, port, src, True)

    def _set_procedural(self, uid, sid):
        def do():
            fx = self._obj(uid)
            fx.procedural, fx.values, fx.inputs = sid, {}, {}
        self.doc.edit("Choose effect", do)
        self.rebuild()

    def _levels(self, fx):
        """Levels properties (Levels page): Affected Channel (not in a mask) and Colors on top, the levels editor
        (histogram with the shadows, midtones and highlights handles above it, black and white points under it),
        Reset, Auto, Invert and Clamp at the bottom. The histogram is the one of what the effect receives (the
        layer's content, or mask, up to it) in the first tile, in the affected component(s)."""
        doc, uid = self.doc, fx.uid
        in_mask = doc.ts.owner_of(uid)[2]
        lv = fx.levels

        def set_lv(changes: dict, final=True, merge=""):
            def do():
                for attr, v in changes.items():
                    setattr(self._obj(uid).levels, attr, v)
            doc.edit("Levels", do, merge=None if final else f"{uid}.{merge}", structure="channel" in changes)

        top = QHBoxLayout()
        if not in_mask:
            ch = QComboBox()
            for c in doc.ts.channel_ids():
                ch.addItem(CHANNELS[c].label, c)
            ch.setCurrentIndex(max(0, ch.findData(lv.channel)))
            ch.currentIndexChanged.connect(lambda i: set_lv({"channel": ch.itemData(i)}))
            top.addWidget(QLabel("Affected channel"))
            top.addWidget(ch)
        top.addStretch(1)
        colors = QComboBox()
        colors.setToolTip("Colors")
        colors.addItems(LEVELS_COLORS)
        colors.setCurrentText(lv.colors)
        # its histogram; the rebuild deletes the selector: not from inside its signal
        colors.currentTextChanged.connect(lambda t: (set_lv({"colors": t}), QTimer.singleShot(0, self, self.rebuild)))
        top.addWidget(colors)
        self.lay.addLayout(top)
        editor = LevelsEditor([getattr(lv, a) for a in LEVELS_ATTRS], dark=True)
        gray = in_mask or is_gray(lv.channel, doc.ts.format_of(lv.channel))
        comps = (0,) if gray else {"Red": (0,), "Green": (1,), "Blue": (2,)}.get(lv.colors, (0, 1, 2))

        def histogram(stats):  # now, or once computed (the panel may have been rebuilt meanwhile)
            if isValid(editor):
                editor.set_histogram([(stats[0][:, c], HISTOGRAM_GRAY if gray else HISTOGRAM_RGB[c]) for c in comps])
        self._levels_histogram(fx, histogram)
        def edited(values, final):  # a handle or a field: one value, its own History step
            now = self._obj(uid).levels
            changes = {a: v for a, v in zip(LEVELS_ATTRS, values) if getattr(now, a) != v}
            set_lv(changes, final, next(iter(changes), ""))
        editor.edited.connect(edited)
        self.lay.addWidget(editor)

        def action(changes):  # a button: the editor shows the new values (the panel stays)
            set_lv(changes)
            editor.set_levels([getattr(self._obj(uid).levels, a) for a in LEVELS_ATTRS])

        def auto():
            def apply(stats):
                if not isValid(editor):  # the panel rebuilt (another layer) while it was computed
                    return
                _, lo, hi = stats
                low, high = float(min(lo[c] for c in comps)), float(max(hi[c] for c in comps))
                if high > low:
                    action({"in_low": low, "in_high": high})
            self._levels_histogram(fx, apply)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        def invert():
            cur = self._obj(uid).levels
            action({"out_low": cur.out_high, "out_high": cur.out_low})
        defaults = Levels()
        for text, fn in (("Reset", lambda: action({a: getattr(defaults, a) for a in LEVELS_ATTRS})), ("Auto", auto),
                         ("Invert", invert)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bottom.addWidget(b)
        clamp = QCheckBox("Clamp")
        clamp.setChecked(lv.clamp)
        clamp.toggled.connect(lambda on: set_lv({"clamp": on}))
        bottom.addWidget(clamp)
        self.lay.addLayout(bottom)

    def _levels_histogram(self, fx, then) -> None:
        """`then((counts, minimum, maximum))`, the histogram of what the Levels effect `fx` receives in the Texture
        Set's first tile, now or once computed (`LiveCook.histogram`: nothing waits for the GPU); never with nothing
        below it."""
        doc = self.doc
        ts = doc.ts
        in_mask = ts.owner_of(fx.uid)[2]
        job = doc.session.compiler(ts.tiles[0]).reaching(ts, fx.uid, MASK_KEY if in_mask else fx.levels.channel)
        if job is not None:
            self.ws.live.histogram("levels", job, "output", then)

    def _src(self, uid, channel, slot=False):
        """The ChannelSource of a fill's channel, or (`slot`) of a filter's / generator's resource slot (made
        when missing)."""
        obj = self._obj(uid)
        if slot:
            return obj.inputs.setdefault(channel, ChannelSource([1.0, 1.0, 1.0]))
        return obj.material[channel]

    def _set_image(self, uid, channel, name, slot=False):
        """A channel's (or resource slot's) source: "" uniform, a project texture, a procedural ("node:<type>") or
        an anchor point (assets.ANCHOR + its uid: it reads the same channel, or the first one)."""
        from trafilatura3d.editor.paint.assets import ANCHOR

        def do():
            src = self._src(uid, channel, slot)
            src.anchor = src.anchor_channel = ""
            if name.startswith(ANCHOR):
                ids = self.doc.ts.channel_ids()
                src.procedural, src.image, src.anchor = "", "", name[len(ANCHOR):]
                src.anchor_channel = "" if self.doc.ts.owner_of(src.anchor)[2] else (
                    channel if channel in ids else ids[0])
            elif name.startswith("node:"):
                src.procedural, src.image, src.values = name, "", {}
            else:
                src.procedural, src.image = "", name
        self.doc.edit("Fill resource" if not slot else "Effect resource", do)
        self.rebuild()

    def _anchor_reference(self, v, source, stack_slot):
        """Under a slot referencing an anchor point: the channel it reads of an anchor in a layer's content
        (Referenced channel), and Go to anchor point (Find an Anchor Point)."""
        uid, key, is_slot = stack_slot
        ts = self.doc.ts
        _layer, fx, in_mask = ts.owner_of(source.anchor)
        if fx is not None and not in_mask:
            row = QHBoxLayout()
            row.addWidget(QLabel("Referenced channel"))
            ch = QComboBox()
            for c in ts.channel_ids():
                ch.addItem(CHANNELS[c].label, c)
            ch.setCurrentIndex(max(0, ch.findData(source.anchor_channel)))

            def set_channel(i):
                def do():
                    self._src(uid, key, is_slot).anchor_channel = ch.itemData(i)
                self.doc.edit("Referenced channel", do)
            ch.currentIndexChanged.connect(set_channel)
            row.addWidget(ch, 1)
            v.addLayout(row)
        if fx is not None:
            go = QPushButton("Go to anchor point")
            go.clicked.connect(lambda: self.doc.select(source.anchor))
            v.addWidget(go)

    def _anchor(self, fx):
        """An anchor point's properties: its name and the slots referencing it, each a link to its layer or
        effect (Find References for an Anchor Point)."""
        self.lay.addWidget(header("Anchor point"))
        self.lay.addWidget(QLabel(fx.name))
        self.lay.addWidget(header("References"))
        refs = self.doc.ts.references(fx.uid)
        if not refs:
            none = QLabel("No layer references this anchor point")
            none.setObjectName("dim")
            self.lay.addWidget(none)
        for layer, effect, slot in refs:
            where = layer.name if effect is None else f"{layer.name} › {effect.name}"
            label = "mask" if slot == MASK_KEY else CHANNELS[slot].label if slot in CHANNELS else slot
            b = QPushButton(f"{where} ({label})")
            b.clicked.connect(lambda _c=False, u=(effect or layer).uid: self.doc.select(u))
            self.lay.addWidget(b)

    def _procedural_params(self, uid, channel, src, slot=False):
        """The parameters of a channel's (or resource slot's) procedural (the Graph node's own)."""
        from trafilatura3d.editor.params_ui import make_editor
        from trafilatura3d.paint.graph_nodes import node_of
        node = node_of(src.procedural)
        if node is None:
            return

        def row(form, name, prm):
            def edited(v, final):
                def do():
                    s = self._src(uid, channel, slot)
                    s.values = {**s.values, name: v}
                self.doc.edit(f"{node.label} parameter", do, merge=None if final else f"{uid}.{channel}.{name}",
                              structure=False)
            form.addRow(prm.label or name, make_editor(prm, src.values.get(name, prm.default), edited,
                                                       _color_field(prm.label or name)))
        self._grouped_params(node, [(SEED, SEED_PARAM), *node.params.items()], row)

    def _grouped_params(self, node, items, row):
        """A node's parameters (`items`: [(name, Param)]): the ungrouped ones first, then each group of the node
        (`Param.group`, "/" separating sub groups) under its own collapsible title, sub groups nested in it.
        `row(form, name, param)` adds a parameter's row. A group collapsed stays so for that node type."""
        from trafilatura3d.editor.params_ui import grouped
        root = QFormLayout()
        self.lay.addLayout(root)
        forms = {(): root}
        for path, (name, prm) in grouped(items, lambda it: it[1].group):
            for depth in range(1, len(path) + 1):
                if path[:depth] not in forms:
                    forms[path[:depth]] = self._param_group(node.type, path[:depth], forms[path[:depth - 1]])
            row(forms[path], name, prm)
        # one label column for all the groups: the editors line up from a group to the next
        labels = [f.itemAt(i, QFormLayout.LabelRole).widget() for f in forms.values() for i in range(f.rowCount())
                  if f.itemAt(i, QFormLayout.LabelRole) is not None]
        width = max((lb.sizeHint().width() for lb in labels), default=0)
        for lb in labels:
            lb.setMinimumWidth(width)

    def _param_group(self, node_type, path, parent):
        """A collapsible parameter group: its title (a section of the panel at the top level, indented below)
        and the form of its rows; returns the form."""
        key = (node_type, path)
        is_open = key not in self._closed
        title = QPushButton()
        title.setCheckable(True)
        title.setChecked(is_open)
        title.setStyleSheet(f"QPushButton {{ font-weight: 600; text-align: left; border: none; border-radius: 0;"
                            f" background: transparent; padding: 8px 0 4px {12 * (len(path) - 1)}px; }}")
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 4)
        body.setVisible(is_open)

        def toggled(on):
            body.setVisible(on)
            title.setText(("▾  " if on else "▸  ") + path[-1])
            (self._closed.discard if on else self._closed.add)(key)
        title.toggled.connect(toggled)
        title.setText(("▾  " if is_open else "▸  ") + path[-1])
        if len(path) == 1:
            title.setProperty("section", True)  # the panel draws the divider under it, edge to edge
            self.lay.addWidget(title)
            self.lay.addWidget(body)
        else:
            parent.addRow(title)
            parent.addRow(body)
        return form
