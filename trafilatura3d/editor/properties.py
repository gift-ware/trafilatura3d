"""The Properties panel: context sensitive (selected node, or the graph when nothing
is selected), made of rollouts — BASE PARAMETERS, SPECIFIC PARAMETERS / INSTANCE PARAMETERS,
ATTRIBUTES; for a graph INPUT PARAMETERS (its exposed parameters) and ATTRIBUTES. Each parameter has
a menu to expose it as a graph input, link it to one, unlink it or reset it."""

from __future__ import annotations

import dataclasses
import random

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                               QPlainTextEdit, QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from trafilatura3d import evaluator
from trafilatura3d.compiler import (ABSOLUTE, BASE_PARAMS, DEPTHS, NODE_DEFAULTS, PARENT_DEFAULTS, PIXEL_RATIOS,
                              RELATIVE_INPUT, RELATIVE_PARENT, TILING_LABELS)
from trafilatura3d.editor import theme
from trafilatura3d.editor.icons import icon
from trafilatura3d.editor.params_ui import grouped, make_editor
from trafilatura3d.editor.widgets import (HISTOGRAM_GRAY, HISTOGRAM_RGB, BoolToggle, ColorButton, CurveEditor, GradientBar,
                                    LevelsEditor, NumberField, SliderField, VectorField)
from trafilatura3d.nodes.atomic.levels import LEVELS_PARAMS
from trafilatura3d.nodes.base import EDITORS, Param, ParamKind, PortType
from trafilatura3d.package import INPUT_TYPES, INSTANCE, DocumentError

ATTRIBUTES = ("identifier", "label", "description", "user_data", "group", "component", "usage", "color_space",
              "visible_if")
PRESET_SUFFIX = ".trafilatura3dpreset"  # external presets (like .sbsprs)
DEFAULT_VALUES = {ParamKind.FLOAT: 0.0, ParamKind.INT: 0, ParamKind.BOOL: False, ParamKind.FLOAT2: [0.0, 0.0],
                  ParamKind.FLOAT3: [0.0, 0.0, 0.0], ParamKind.FLOAT4: [0.0, 0.0, 0.0, 0.0], ParamKind.INT2: [0, 0],
                  ParamKind.COLOR: [0.0, 0.0, 0.0, 1.0], ParamKind.STRING: "", ParamKind.ENUM: "a"}
# The Levels Histogram's channels: label, the components of the FLOAT4 values it edits ("Luminance Channel"
# for a grayscale input; the colour labels follow it, ours).
LEVELS_GRAY = (("Luminance Channel", (0, 1, 2)),)
LEVELS_COLOR = (("RGB Channels", (0, 1, 2)), ("Red Channel", (0,)), ("Green Channel", (1,)), ("Blue Channel", (2,)),
                ("Alpha Channel", (3,)))
LEVELS_ACTIVE = "#e5c33a"  # the interface shown, on the Switch interfaces button (yellow)
MODE_LABELS = {RELATIVE_INPUT: "Relative to input", RELATIVE_PARENT: "Relative to parent", ABSOLUTE: "Absolute"}


class Rollout(QWidget):
    def __init__(self, title: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = QPushButton(f"▾  {title}")
        self.header.setStyleSheet(f"text-align: left; font-weight: 600; font-size: 11px; color: {theme.TEXT_DIM};"
                                  f"background: {theme.BG}; border: none; border-radius: 0; padding: 7px 8px;")
        theme.divider(self, "bottom")  # a section ends with the divider (one line between sections, none doubled)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.body = QWidget()
        self.form = QVBoxLayout(self.body)
        self.form.setContentsMargins(10, 4, 8, 8)
        self.form.setSpacing(6)
        self.header.toggled.connect(lambda on: (self.body.setVisible(on),
                                                self.header.setText(("▾  " if on else "▸  ") + title)))
        lay.addWidget(self.header)
        lay.addWidget(self.body)

    def add(self, label: str, editor: QWidget, tools: list[QWidget] = (), dim=False) -> None:
        row = QWidget()
        v = QVBoxLayout(row)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        top = QHBoxLayout()
        name = QLabel(label)
        name.setStyleSheet("font-weight: bold;" + (f"color: {theme.TEXT_DIM};" if dim else ""))
        top.addWidget(name, 1)
        for t in tools:
            top.addWidget(t)
        v.addLayout(top)
        if editor is not None:
            v.addWidget(editor)
        self.form.addWidget(row)


def tool_button(icon_name: str, tip: str) -> QToolButton:
    b = QToolButton()
    b.setIcon(icon(icon_name))
    b.setToolTip(tip)
    b.setAutoRaise(True)
    return b


class PropertiesPanel(QScrollArea):
    title_changed = Signal(str)
    open_function = Signal(str, str, bool)  # node id, parameter, is a parameter function
    open_fxmap = Signal(str, str)  # node id, parameter
    open_function_target = Signal(object)  # an FxFunction (function of an FX-Map node parameter)
    rename_graph = Signal(object, str, str)  # package, graph id, new id

    def __init__(self, document):
        super().__init__()
        self.doc = document
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.package, self.graph_id, self.node_id = None, "", None
        self._editing = False
        self._gesture = 0
        self._inner = None
        self._expanded: set[str] = set()  # graph inputs shown with all their properties
        self._levels_sliders = False  # Levels: the sliders interface instead of the histogram (Switch interfaces)
        self._levels_channel: dict = {}  # Levels: (package, graph, node) -> the histogram's channel
        self.doc.changed.connect(self._doc_changed)
        self.show_target(None, "", None)

    # --- target -------------------------------------------------------------------------------

    @property
    def graph(self):
        return self.package.graphs.get(self.graph_id) if self.package else None

    def show_target(self, package, graph_id: str, node_id):
        self._inner = None  # not showing a node of a function or FX-Map graph
        self.package, self.graph_id, self.node_id = package, graph_id, node_id
        self.rebuild()

    def show_settings(self, title: str, rollouts, get, set_):
        """Settings that are not part of a package (the 3D View's): rollouts of rows
        (label, key, kind, a, b) with kind "float" / "int" (a, b = range), "bool", "enum" (a = choices),
        "text", "color" (rgb) or "vec3" (a, b = range); `get(key)` reads a value, `set_(key, value)`
        applies it."""
        self._inner = None
        self.package, self.graph_id, self.node_id = None, "", None  # no document edits from here
        root = QWidget()
        self.col = QVBoxLayout(root)
        self.col.setContentsMargins(0, 0, 0, 0)
        self.col.setSpacing(0)
        for rollout_title, rows in rollouts:
            r = Rollout(rollout_title)
            for label, key, kind, a, b in rows:
                value = get(key)
                if kind == "bool":
                    editor = BoolToggle(value)
                    editor.edited.connect(lambda v, final, k=key: set_(k, v))
                elif kind == "enum":
                    editor = QComboBox()
                    editor.addItems(list(a))
                    editor.setCurrentIndex(value)
                    editor.currentIndexChanged.connect(lambda i, k=key: set_(k, i))
                elif kind == "text":
                    editor = QLineEdit(value)
                    editor.editingFinished.connect(lambda e=editor, k=key: set_(k, e.text()))
                elif kind == "color":
                    editor = ColorButton([*value[:3], 1.0])
                    editor.edited.connect(lambda v, final, k=key: set_(k, list(v[:3])))
                elif kind == "vec3":
                    editor = VectorField(value, a, b)
                    editor.edited.connect(lambda v, final, k=key: set_(k, list(v)))
                else:
                    editor = SliderField(value, a, b, integer=kind == "int")
                    editor.edited.connect(lambda v, final, k=key: set_(k, v))
                r.add(label, editor)
            self.col.addWidget(r)
        self.col.addStretch(1)
        self.setWidget(root)
        self.title_changed.emit(title)

    def _doc_changed(self, package):
        if package is not self.package or self._editing:
            return
        if self._inner is not None:  # a node of a function or FX-Map graph: show it again
            self._inner()
            return
        if self.node_id is not None and (self.graph is None or self.node_id not in self.graph.nodes):
            self.node_id = None
        self.rebuild()

    def rebuild(self):
        scroll = self.verticalScrollBar().value()
        root = QWidget()
        self.col = QVBoxLayout(root)
        self.col.setContentsMargins(0, 0, 0, 0)
        self.col.setSpacing(0)
        g = self.graph
        if g is None:
            self.col.addWidget(QLabel("  Nothing selected"))
            self.title_changed.emit("Properties")
        elif self.node_id is None:
            self._graph_properties(g)
        else:
            self._node_properties(g, g.nodes[self.node_id])
        self.col.addStretch(1)
        self.setWidget(root)
        self.verticalScrollBar().setValue(scroll)

    def _missing_properties(self, node, error: Exception):
        """A node whose definition can't be found (a missing dependency): why, and the values it keeps (read
        only: they have no parameter to edit until the dependency exists)."""
        self.title_changed.emit(f"{node.ref.rsplit('#', 1)[-1] or node.type} - Properties")
        r = Rollout("Missing dependency")
        reason = QLabel(str(error))
        reason.setWordWrap(True)
        reason.setStyleSheet("color: #e05050;")
        r.add(node.ref or node.type, reason)
        for key, value in node.params.items():
            v = QLabel(str(value))
            v.setWordWrap(True)
            r.add(key, v, dim=True)
        for key in node.functions:
            r.add(key, QLabel("function"), dim=True)
        self.col.addWidget(r)

    # --- editing ------------------------------------------------------------------------------

    def _edit(self, label: str, fn, final: bool, key: str):
        """Apply an edit; a slider drag (final False) is one undo step closed by its final value."""
        self._editing = True
        try:
            self.doc.edit(self.package, label, fn, merge=f"{key}#{self._gesture}", drag=not final)
        finally:
            self._editing = False
        if final:
            self._gesture += 1

    # --- node ---------------------------------------------------------------------------------

    def _node_properties(self, g, node):
        try:
            d = g.definition(node.id)
        except (DocumentError, OSError, ValueError) as e:
            self._missing_properties(node, e)
            return
        values = d.values(node)
        io = node.type == "output" or node.type in INPUT_TYPES
        label = (values.get("label") or values.get("identifier")) if io else d.label
        self.title_changed.emit(f"{label} - Properties")
        if d.outputs or node.type == "output":
            self._base_rollout(g, node)
        specific = [(k, p) for k, p in d.params.items() if not (k in ATTRIBUTES and (node.type == "output"
                                                                                    or node.type in INPUT_TYPES))]
        if node.type == INSTANCE and d.graph.presets:
            self._instance_presets(g, node, d)
        if specific:
            r = Rollout("Instance parameters" if node.type == INSTANCE else "Specific parameters")
            exposable = [(k, p) for k, p in specific if p.kind not in (ParamKind.GRADIENT, ParamKind.CURVE,
                                                                       ParamKind.FUNCTION, ParamKind.FXMAP)]
            levels = all(d.params.get(k) is not None and d.params[k].kind is ParamKind.FLOAT4 for k in LEVELS_PARAMS)
            # the histogram editor edits the values themselves: linked or computed ones need their own rows
            histogram = levels and not self._levels_sliders and not any(
                k in node.links or k in node.functions for k in LEVELS_PARAMS)
            tools = self._levels_tools(g, node, histogram) if levels else []
            if exposable:
                multi = tool_button("links", "Multi-expose")
                mm = QMenu(multi)
                mm.addAction("Batch expose parameters…", lambda: self._batch_expose(g, node, exposable, values))
                multi.setMenu(mm)
                multi.setPopupMode(QToolButton.InstantPopup)
                tools.append(multi)
            if tools:
                line = QHBoxLayout()
                line.addStretch(1)
                for t in tools:
                    line.addWidget(t)
                hb = QWidget()
                hb.setLayout(line)
                r.form.addWidget(hb)
            groups = {}
            for path, (name, param) in grouped(specific, lambda it: it[1].group):
                if node.type == INSTANCE and not self._param_visible(node, param, values):
                    continue
                if histogram and name in LEVELS_PARAMS:
                    if name == next(k for k in d.params if k in LEVELS_PARAMS):  # in place of the first of them
                        self._levels_row(self._group(r, path, groups), g, node, d)
                    continue
                self._param_row(self._group(r, path, groups), g, node, name, param, values[name])
            self.col.addWidget(r)
        attrs = [(k, p) for k, p in d.params.items() if k in ATTRIBUTES and (node.type == "output"
                                                                            or node.type in INPUT_TYPES)]
        if attrs:
            r = Rollout("Attributes")
            for name, param in attrs:
                self._param_row(r, g, node, name, param, values[name], exposable=False)
            self.col.addWidget(r)

    @staticmethod
    def _param_visible(node, param, values) -> bool:
        """Visible if of a graph instance's parameter. Ignored (visible) when a parameter it reads is
        exposed or driven by a function on this node."""
        import re
        from trafilatura3d.visibility import visible
        if not param.visible_if:
            return True
        used = set(re.findall(r"input(?:\.([A-Za-z_]\w*)|\[[\"']([^\"']+)[\"']\])", param.visible_if))
        names = {a or b for a, b in used}
        if names & (set(node.links) | set(node.functions)):
            return True
        return visible(param.visible_if, values)

    def _instance_presets(self, g, node, d):
        """Embedded presets of the instanced graph, and external preset files."""
        r = Rollout("Presets")
        line = QHBoxLayout()
        combo = QComboBox()
        combo.addItem("Choose a preset…")
        for pr in d.graph.presets:
            combo.addItem(pr["name"])

        def apply(i):
            if i <= 0:
                return
            values = d.graph.presets[i - 1]["values"]
            self._edit("Apply preset", lambda: [g.set_param(node.id, k, v) for k, v in values.items() if k in d.params],
                       True, "preset")
        combo.activated.connect(apply)
        line.addWidget(combo, 1)
        holder = QWidget()
        holder.setLayout(line)
        r.form.addWidget(holder)
        files = QHBoxLayout()
        save = QPushButton("Save preset…")
        save.clicked.connect(lambda: self._save_preset_file({k: d.values(node)[k] for k in d.params}))
        load = QPushButton("Load preset…")
        load.clicked.connect(lambda: self._load_preset_into_node(g, node, d))
        files.addWidget(save)
        files.addWidget(load)
        h2 = QWidget()
        h2.setLayout(files)
        r.form.addWidget(h2)
        self.col.addWidget(r)

    def _load_preset_into_node(self, g, node, d):
        values = self._read_preset_file()
        if values is not None:
            self._edit("Load preset", lambda: [g.set_param(node.id, k, v) for k, v in values.items() if k in d.params],
                       True, "preset")

    def _edit_exposed(self, pid):
        """Show the graph's Properties with this input expanded (Edit exposed graph input)."""
        self._expanded.add(pid)
        self.show_target(self.package, self.graph_id, None)

    def _batch_expose(self, g, node, params, values):
        """Batch expose parameters: choose parameters, edit identifiers and groups, add a
        prefix, a suffix and a group to all of them."""
        from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QGridLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Batch expose parameters")
        lay = QVBoxLayout(dlg)
        top = QHBoxLayout()
        all_b, none_b = QPushButton("All"), QPushButton("None")
        prefix, suffix, group = QLineEdit(), QLineEdit(), QLineEdit()
        for w, ph in ((prefix, "Prefix"), (suffix, "Suffix"), (group, "Input group")):
            w.setPlaceholderText(ph)
        for w in (all_b, none_b, prefix, suffix, group):
            top.addWidget(w)
        lay.addLayout(top)
        grid = QGridLayout()
        rows = []
        for i, (name, param) in enumerate(params):
            chk = QCheckBox(param.label or name)
            chk.setChecked(name not in node.links)
            chk.setEnabled(name not in node.links)
            ident = QLineEdit(name)
            grp = QLineEdit(param.group)
            grid.addWidget(chk, i, 0)
            grid.addWidget(ident, i, 1)
            grid.addWidget(grp, i, 2)
            rows.append((name, param, chk, ident, grp))
        lay.addLayout(grid)
        all_b.clicked.connect(lambda: [r[2].setChecked(True) for r in rows if r[2].isEnabled()])
        none_b.clicked.connect(lambda: [r[2].setChecked(False) for r in rows])
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if not dlg.exec():
            return

        def do():
            for name, param, chk, ident, grp in rows:
                if not chk.isChecked():
                    continue
                pid, base, n = prefix.text() + ident.text() + suffix.text(), None, 1
                base = pid
                while pid in g.parameters:
                    n += 1
                    pid = f"{base}{n}"
                groups = "/".join(x for x in (group.text(), grp.text()) if x)
                g.expose(pid, dataclasses.replace(param, default=values[name], label=param.label or name, group=groups))
                g.link(node.id, name, pid)
        self._edit("Batch expose", do, True, "expose")

    def _param_row(self, rollout, g, node, name, param: Param, value, exposable=True):
        from trafilatura3d.functions import PARAM_TYPES, constant, from_param
        linked = node.links.get(name)
        menu_btn = tool_button("menu", "Parameter options")
        menu = QMenu(menu_btn)
        if param.kind is ParamKind.FUNCTION:
            edit = QPushButton("Edit function…  (Ctrl+E)")
            edit.clicked.connect(lambda: self.open_function.emit(node.id, name, False))
            rollout.add(param.label or name, edit)
            return
        if param.kind is ParamKind.FXMAP:
            edit = QPushButton("Edit Stamp Map graph…  (Ctrl+E)")
            edit.clicked.connect(lambda: self.open_fxmap.emit(node.id, name))
            rollout.add(param.label or name, edit)
            return
        if exposable and param.kind in PARAM_TYPES:
            if name in node.functions:
                menu.addAction("Edit function", lambda: self.open_function.emit(node.id, name, True))
                menu.addAction("Remove function", lambda: self._edit(
                    "Remove function", lambda: g.set_function(node.id, name, None), True, "function"))
            else:
                def set_function():
                    self._edit("Set function", lambda: g.set_function(
                        node.id, name, constant(PARAM_TYPES[param.kind], from_param(value, param))), True, "function")
                    self.open_function.emit(node.id, name, True)
                menu.addAction("Set function…", set_function)
        if name in node.functions:
            edit = QPushButton("ƒ  computed by a function — Edit…")
            edit.clicked.connect(lambda: self.open_function.emit(node.id, name, True))
            menu_btn.setMenu(menu)
            menu_btn.setPopupMode(QToolButton.InstantPopup)
            rollout.add(param.label or name, edit, [menu_btn])
            return
        if exposable and param.kind not in (ParamKind.GRADIENT, ParamKind.CURVE):
            menu.addAction("Expose as new graph input", lambda: self._expose(g, node, name, param))
            same = [pid for pid, p in g.parameters.items() if p.param.kind is param.kind]
            if same:
                sub = menu.addMenu("Link to graph input")
                for pid in same:
                    sub.addAction(pid, lambda pid=pid: self._edit("Link", lambda: g.link(node.id, name, pid), True, "link"))
            if linked:
                menu.addAction("Unlink", lambda: self._edit("Unlink", lambda: g.link(node.id, name, None), True, "link"))
                menu.addAction("Edit exposed graph input", lambda: self._edit_exposed(linked))
        menu.addAction("Reset to default", lambda: self._edit(
            "Reset", lambda: node.params.pop(name, None), True, f"{node.id}.{name}"))
        menu_btn.setMenu(menu)
        menu_btn.setPopupMode(QToolButton.InstantPopup)
        if linked:
            info = QLabel(f"↳ linked to graph input “{linked}”")
            info.setObjectName("dim")
            rollout.add(param.label or name, info, [menu_btn], dim=True)
            return
        editor = self._editor(param, value, lambda v, final: self._edit(
            f"Edit {param.label or name}", lambda: g.set_param(node.id, name, v), final, f"{node.id}.{name}"))
        rollout.add(param.label or name, editor, [menu_btn])

    def _editor(self, param: Param, value, on_edit) -> QWidget:
        return make_editor(param, value, on_edit)

    # --- levels -------------------------------------------------------------------------------

    def _levels_input(self, g, node, d):
        """(job, output) of what the node's first input receives, or None (not connected, not compiled)."""
        port = next(iter(d.inputs), None)
        src = node.inputs.get(port) if port else None
        compiled = self.doc.compiled(self.package, self.graph_id) if src else None
        return compiled.nodes.get(src[0], {}).get(src[1]) if compiled else None

    def _levels_histogram(self, g, node, d):
        """(counts, minimum, maximum, depth) of the image entering the node (`Evaluator.histogram`), or None."""
        source = self._levels_input(g, node, d)
        if source is None:
            return None
        job, out = source
        try:
            return (*self.doc.ev.histogram(job, out), job.depths[out])
        except Exception:  # noqa: BLE001 - a failing node upstream must not stop the editor (as the thumbnails)
            return None

    def _levels_channels(self, g, node, d):
        return LEVELS_GRAY if g.output_type(node.id, next(iter(d.outputs))) is PortType.GRAY else LEVELS_COLOR

    def _levels_tools(self, g, node, histogram: bool) -> list[QToolButton]:
        """The Levels node's quick actions in the rollout's header: Invert (swaps Level out low and high), Auto level
        (Level in low and high = the darkest and brightest values of the image), Switch interfaces (histogram or
        sliders)."""
        d = g.definition(node.id)

        def invert():
            v = d.values(g.nodes[node.id])
            g.set_param(node.id, "level_out_low", v["level_out_high"])
            g.set_param(node.id, "level_out_high", v["level_out_low"])

        def auto_level():
            stats = self._levels_histogram(g, node, d)
            if stats is None:
                return
            _, lo, hi, _ = stats
            v = d.values(g.nodes[node.id])
            low, high = list(v["level_in_low"]), list(v["level_in_high"])
            for c in range(3):  # a grayscale image is (v, v, v, 1); alpha is not a tone
                if hi[c] > lo[c]:
                    low[c], high[c] = float(lo[c]), float(hi[c])
            g.set_param(node.id, "level_in_low", low)
            g.set_param(node.id, "level_in_high", high)

        def switch():
            self._levels_sliders = not self._levels_sliders
            QTimer.singleShot(0, self.rebuild)  # the rebuild deletes the button: not from inside its signal

        def action(label, fn):
            self._edit(label, fn, True, "levels")
            QTimer.singleShot(0, self.rebuild)  # `_edit` keeps the panel (a drag's widgets stay): show the new values

        buttons = []
        for name, tip, fn in (("invert", "Invert", lambda: action("Invert", invert)),
                              ("auto_level", "Auto level", lambda: action("Auto level", auto_level)),
                              ("histogram" if histogram else "renderer", "Switch interfaces", switch)):
            b = QToolButton()
            b.setIcon(icon(name, LEVELS_ACTIVE if fn is switch else "#cfcfcf"))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(fn)
            buttons.append(b)
        return buttons

    def _levels_row(self, rollout, g, node, d):
        """The Levels Histogram: the levels editor on one channel (the selector's), the histogram of the
        image entering the node. A channel edits its components of the five FLOAT4 values (RGB the three colour
        ones), a drag is one undo step."""
        channels = self._levels_channels(g, node, d)
        key = (id(self.package), self.graph_id, node.id)
        index = min(self._levels_channel.get(key, 0), len(channels) - 1)
        comps = channels[index][1]
        values = d.values(node)
        editor = LevelsEditor([values[k][comps[0]] for k in LEVELS_PARAMS], [c for c, _ in channels], index)
        stats = self._levels_histogram(g, node, d)
        if stats is not None:
            counts, _, _, depth = stats
            top = evaluator.DEPTHS[depth][0]  # the depth's largest value; float ones: 1
            shown = [(counts[:, c], HISTOGRAM_GRAY if channels is LEVELS_GRAY or c == 3 else HISTOGRAM_RGB[c])
                     for c in (comps if channels is LEVELS_COLOR else comps[:1])]
            editor.set_histogram(shown, ("0", f"{top:g}" if top else "1"))

        def edited(levels, final):
            def do():
                now = d.values(g.nodes[node.id])
                for k, v in zip(LEVELS_PARAMS, levels):
                    if now[k][comps[0]] != v:
                        g.set_param(node.id, k, [v if c in comps else x for c, x in enumerate(now[k])])
            self._edit("Edit Levels", do, final, f"{node.id}.levels")

        def channel(i):
            self._levels_channel[key] = i
            QTimer.singleShot(0, self.rebuild)  # not from inside the selector's signal: the rebuild deletes it
        editor.edited.connect(edited)
        editor.channel_changed.connect(channel)
        rollout.add("Levels Histogram", editor)

    def _expose(self, g, node, name, param):
        """The Expose parameter dialog: Identifier, Label and Group, then OK."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
        pid, n = name, 1
        while pid in g.parameters:
            n += 1
            pid = f"{name}{n}"
        dlg = QDialog(self)
        dlg.setWindowTitle("Expose parameter")
        form = QFormLayout(dlg)
        ident, label, group = QLineEdit(pid), QLineEdit(param.label or name), QLineEdit(param.group)
        form.addRow("Identifier", ident)
        form.addRow("Label", label)
        form.addRow("Group", group)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if not dlg.exec():
            return

        self._edit("Expose parameter", lambda: g.expose_node_param(node.id, name, ident.text(), label.text(),
                                                                   group.text()), True, "expose")

    # --- function nodes -----------------------------------------------------------------------

    def show_function_node(self, panel, node_id):
        """The parameters of a node of the function graph being edited."""
        from trafilatura3d.functions import FNODES
        self.package, self.graph_id, self.node_id = panel.package, panel.graph_id, None
        self._inner = lambda: self.show_function_node(panel, node_id)
        root = QWidget()
        self.col = QVBoxLayout(root)
        self.col.setContentsMargins(0, 0, 0, 0)
        self.col.setSpacing(0)
        data = panel.data()
        n = next((m for m in data["nodes"] if m["id"] == node_id), None)
        if n is None or n["type"] not in FNODES:  # deleted: back to the graph's properties
            self._inner = None
            self.rebuild()
            return
        f = FNODES[n["type"]]
        self.title_changed.emit(f"{f.label} - Properties")
        r = Rollout("Function node")
        for pname, param in f.params.items():
            value = n.get("params", {}).get(pname, param.default)
            r.add(param.label, self._editor(param, value, lambda v, final, pname=pname: self._function_edit(
                panel, node_id, pname, v, final)))
        if not f.params:
            r.add("No parameters", None, dim=True)
        out = QPushButton("Set as output node")
        out.clicked.connect(lambda: panel.set_output(node_id))
        r.form.addWidget(out)
        self.col.addWidget(r)
        self.col.addStretch(1)
        self.setWidget(root)

    def show_fx_node(self, panel, fx_id):
        """The parameters of a node of the FX-Map graph being edited, each possibly a function."""
        from trafilatura3d.editor.graph.function_panel import FxFunction
        from trafilatura3d.functions import PARAM_TYPES, constant, from_param
        from trafilatura3d.fxmap import FX_PARAMS
        self.package, self.graph_id, self.node_id = panel.package, panel.graph_id, None
        self._inner = lambda: self.show_fx_node(panel, fx_id)
        root = QWidget()
        self.col = QVBoxLayout(root)
        self.col.setContentsMargins(0, 0, 0, 0)
        self.col.setSpacing(0)
        n = next((m for m in panel.data()["nodes"] if m["id"] == fx_id), None)
        if n is None:  # deleted: back to the graph's properties
            self._inner = None
            self.rebuild()
            return
        self.title_changed.emit(f"{n['type'].capitalize()} - Properties")
        r = Rollout(f"{n['type']} parameters")
        for pname, param in FX_PARAMS[n["type"]].items():
            value = n.get("params", {}).get(pname, param.default)
            target = FxFunction(panel.package, panel.graph_id, panel.node_id, fx_id, pname, panel.param)
            menu_btn = tool_button("menu", "Parameter options")
            menu = QMenu(menu_btn)
            has_fn = pname in n.get("functions", {})
            if param.kind in PARAM_TYPES:
                if has_fn:
                    menu.addAction("Edit function", lambda t=target: self.open_function_target.emit(t))
                    menu.addAction("Remove function", lambda p=pname: panel.remove_fx_function(fx_id, p))
                else:
                    def set_fn(p=pname, prm=param, v=value, t=target):
                        panel.set_fx_function(fx_id, p, constant(PARAM_TYPES[prm.kind], from_param(v, prm)))
                        self.open_function_target.emit(t)
                    menu.addAction("Set function…", set_fn)
            menu_btn.setMenu(menu)
            menu_btn.setPopupMode(QToolButton.InstantPopup)
            if has_fn:
                edit = QPushButton("ƒ  computed by a function — Edit…")
                edit.clicked.connect(lambda _=False, t=target: self.open_function_target.emit(t))
                r.add(param.label, edit, [menu_btn])
            else:
                r.add(param.label, self._editor(param, value, lambda v, final, p=pname: self._fx_edit(
                    panel, fx_id, p, v, final)), [menu_btn])
        self.col.addWidget(r)
        self.col.addStretch(1)
        self.setWidget(root)

    def _fx_edit(self, panel, fx_id, name, value, final):
        self._editing = True
        try:
            panel.set_fx_param(fx_id, name, value, merge=f"fx.{fx_id}.{name}#{self._gesture}")
        finally:
            self._editing = False
        if final:
            self._gesture += 1

    def _function_edit(self, panel, node_id, name, value, final):
        self._editing = True
        try:
            panel.set_function_param(node_id, name, value, merge=f"fn.{node_id}.{name}#{self._gesture}")
        finally:
            self._editing = False
        if final:
            self._gesture += 1

    # --- base parameters ----------------------------------------------------------------------

    def _base_rollout(self, g, node):
        """Base parameters of a node (node set) or of the graph (node None)."""
        r = Rollout("Base parameters")
        target = node.base if node is not None else g.base
        if node is None:
            defaults = PARENT_DEFAULTS
        else:
            d = g.definition(node.id)
            defaults = {**(PARENT_DEFAULTS if node.type in INPUT_TYPES else NODE_DEFAULTS), **d.base_defaults_for(
                d.values(node), d.extra(d.values(node), self.package) if node.type != INSTANCE else {})}
        current = {name: target.get(name, defaults[name]) for name in BASE_PARAMS}
        key = (node.id if node else "graph") + ".base"

        def setter(name):
            def set_value(value, final=True):
                self._edit(f"Edit {BASE_PARAMS[name].label}", lambda: g.set_base(node.id if node else None, name, value),
                           final, f"{key}.{name}")
            return set_value

        def set_mode(name, mode):
            setter(name)(self._convert_mode(name, current[name], mode))
            QTimer.singleShot(0, self.rebuild)  # `_edit` keeps the panel: the row shows its new mode (not from the menu)

        def mode_tool(name):
            b = tool_button("inherit", f"Inheritance: {MODE_LABELS[current[name]['mode']]}")
            m = QMenu(b)
            modes = QActionGroup(m)  # one mode at a time
            for mode, text in MODE_LABELS.items():
                act = m.addAction(text, lambda mode=mode: set_mode(name, mode))
                act.setCheckable(True)
                act.setChecked(mode == current[name]["mode"])
                modes.addAction(act)
            b.setMenu(m)
            b.setPopupMode(QToolButton.InstantPopup)
            return b

        # Output Size: width and height, log2; relative values read "Parent x 2", "Input x 1/2".
        size = current["output_size"]
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        for axis, axis_name in enumerate(("Width", "Height")):
            row = QHBoxLayout()
            lab = QLabel(axis_name)
            lab.setObjectName("dim")
            lab.setFixedWidth(42)
            rel = size["mode"] != ABSOLUTE
            f = SliderField(size["value"][axis], -12 if rel else 0, 12 if rel else 13, integer=True)
            text = QLabel(self._size_text(size["mode"], size["value"][axis]))
            text.setFixedWidth(82)

            def changed(val, final, axis=axis, text=text):
                vals = list(size["value"])
                vals[axis] = int(val)
                text.setText(self._size_text(size["mode"], vals[axis]))
                setter("output_size")({"mode": size["mode"], "value": vals}, final)
            f.edited.connect(changed)
            row.addWidget(lab)
            row.addWidget(f, 1)
            row.addWidget(text)
            v.addLayout(row)
        r.add("Output Size", box, [mode_tool("output_size")])

        compiled = self.doc.compiled(self.package, g.id)
        outs = compiled.nodes.get(node.id) if compiled and node is not None else None
        ctx = (next(iter(outs.values()))[0].extra.get("context") if outs else None) if node is not None \
            else (compiled.graph if compiled else None)

        def shown(name):
            """The value a dropdown shows: its own when absolute, the inherited one (as compiled) otherwise."""
            if current[name]["mode"] == ABSOLUTE or ctx is None:
                return current[name]["value"]
            return getattr(ctx, {"output_format": "depth"}.get(name, name))

        fmt = QComboBox()
        fmt.addItems(list(DEPTHS.values()))
        fmt.setCurrentText(DEPTHS[shown("output_format")])
        fmt.setEnabled(current["output_format"]["mode"] == ABSOLUTE)
        fmt.currentIndexChanged.connect(lambda i: setter("output_format")(
            {"mode": ABSOLUTE, "value": list(DEPTHS)[i]}))
        r.add("Output Format", fmt, [mode_tool("output_format")])

        ps = VectorField(current["pixel_size"]["value"], 0.0, 4.0)
        ps.edited.connect(lambda vals, final: setter("pixel_size")(
            {"mode": current["pixel_size"]["mode"], "value": [max(x, 1e-3) for x in vals]}, final))
        r.add("Pixel Size", ps, [mode_tool("pixel_size")])

        ratio = QComboBox()
        ratio.addItems(["Stretch", "Square"])
        ratio.setCurrentIndex(PIXEL_RATIOS.index(shown("pixel_ratio")))
        ratio.setEnabled(current["pixel_ratio"]["mode"] == ABSOLUTE)
        ratio.currentIndexChanged.connect(lambda i: setter("pixel_ratio")({"mode": ABSOLUTE, "value": PIXEL_RATIOS[i]}))
        r.add("Pixel Ratio", ratio, [mode_tool("pixel_ratio")])

        tiling = QComboBox()
        tiling.addItems(list(TILING_LABELS.values()))
        tiling.setCurrentText(TILING_LABELS[shown("tiling")])
        tiling.setEnabled(current["tiling"]["mode"] == ABSOLUTE)
        tiling.currentIndexChanged.connect(lambda i: setter("tiling")({"mode": ABSOLUTE, "value": list(TILING_LABELS)[i]}))
        r.add("Tiling Mode", tiling, [mode_tool("tiling")])

        seed = SliderField(current["seed"]["value"], 0, 10000, integer=True)
        seed.edited.connect(lambda val, final: setter("seed")({"mode": current["seed"]["mode"], "value": int(val)}, final))
        dice = tool_button("random", "Random seed")
        dice.clicked.connect(lambda: setter("seed")({"mode": current["seed"]["mode"], "value": random.randint(0, 10000)}))
        r.add("Random Seed", seed, [dice, mode_tool("seed")])
        self.col.addWidget(r)

    @staticmethod
    def _size_text(mode, v):
        if mode == ABSOLUTE:
            return f"{1 << max(0, min(13, v))} px"
        who = "Input" if mode == RELATIVE_INPUT else "Parent"
        return f"{who} x {1 << v}" if v >= 0 else f"{who} x 1/{1 << -v}"

    @staticmethod
    def _convert_mode(name, value, mode):
        """Switching inheritance keeps a sensible value: relative starts at the neutral offset."""
        neutral = {"output_size": [0, 0], "pixel_size": [1.0, 1.0], "seed": 0}
        if mode == ABSOLUTE:
            v = value["value"]
            if name == "output_size" and value["mode"] != ABSOLUTE:
                v = [11 + v[0], 11 + v[1]]
            return {"mode": mode, "value": v}
        return {"mode": mode, "value": neutral.get(name, value["value"])}

    # --- graph --------------------------------------------------------------------------------

    def _graph_properties(self, g):
        """The graph's Properties: Base parameters, Input parameters (with Preview and
        Presets), Inputs, Outputs, Attributes."""
        self.title_changed.emit(f"{g.id} - Properties")
        self._base_rollout(g, None)
        self._input_parameters(g)
        self._io_list(g, "Inputs", lambda n: n.type in INPUT_TYPES)
        self._io_list(g, "Outputs", lambda n: n.type == "output")
        self._graph_attributes(g)

    def _input_parameters(self, g):
        key = (id(self.package), g.id)
        preview = self.doc.preview.get(key)
        r = Rollout("Input parameters")
        bar = QHBoxLayout()
        prev = QPushButton("Preview")
        prev.setCheckable(True)
        prev.setChecked(preview is not None)
        prev.setToolTip("Try values without changing the defaults (Apply makes them the defaults)")
        prev.toggled.connect(lambda on: self.doc.set_preview(
            self.package, g.id, {pid: p.param.default for pid, p in g.parameters.items()} if on else None))
        bar.addWidget(prev)
        presets = QComboBox()
        presets.setToolTip("Embedded presets")
        presets.addItem("Presets…")
        for pr in g.presets:
            presets.addItem(pr["name"])
        presets.activated.connect(lambda i: self._apply_graph_preset(g, i - 1) if i > 0 else None)
        bar.addWidget(presets, 1)
        add_p = tool_button("add", "New preset from the current values")
        add_p.clicked.connect(lambda: self._new_preset(g))
        bar.addWidget(add_p)
        gear = tool_button("menu", "Presets and inputs")
        m = QMenu(gear)
        for i, pr in enumerate(g.presets):
            sub = m.addMenu(pr["name"])
            sub.addAction("Rename…", lambda i=i: self._rename_preset(g, i))
            sub.addAction("Delete", lambda i=i: self._edit("Delete preset", lambda: g.presets.pop(i), True, "preset"))
        if g.presets:
            m.addSeparator()
        m.addAction("Save preset file…", lambda: self._save_preset_file(
            {pid: (preview or {}).get(pid, p.param.default) for pid, p in g.parameters.items()}))
        m.addAction("Load preset file…", lambda: self._load_preset_into_graph(g))
        m.addSeparator()
        m.addAction("Copy Inputs", lambda: self._copy_inputs(g))
        m.addAction("Paste Inputs", lambda: self._paste_inputs(g))
        m.addAction("Clean Inputs…", lambda: self._clean_inputs(g))
        gear.setMenu(m)
        gear.setPopupMode(QToolButton.InstantPopup)
        bar.addWidget(gear)
        holder = QWidget()
        holder.setLayout(bar)
        r.form.addWidget(holder)
        if preview is not None:
            apply = QPushButton("Apply  (preview values become the defaults)")
            apply.clicked.connect(lambda: self._apply_preview(g))
            r.form.addWidget(apply)
        groups = {}
        values = {pid: (preview or {}).get(pid, p.param.default) for pid, p in g.parameters.items()}
        for path, pid in grouped(list(g.parameters), lambda pid: g.parameters[pid].param.group):
            target = self._group(r, path, groups)
            self._input_row(g, target, pid, values, preview)
        add = QPushButton("Add graph input…")
        add.clicked.connect(lambda: self._add_input(g))
        r.form.addWidget(add)
        self.col.addWidget(r)

    def _group(self, rollout, path, cache):
        """The (nested) collapsible group for a group path inside a rollout."""
        parent = rollout
        for depth in range(1, len(path) + 1):
            sub = path[:depth]
            if sub not in cache:
                g = Rollout(sub[-1])
                g.header.setStyleSheet(g.header.styleSheet() + "font-weight: normal; padding-left: %dpx;" % (8 + 10 * depth))
                parent.form.addWidget(g)
                cache[sub] = g
            parent = cache[sub]
        return parent

    def _input_row(self, g, rollout, pid, values, preview):
        """One graph input: its value editor (the default, or the preview value) and, expanded, every
        property of the input parameter."""
        from trafilatura3d.visibility import check
        p = g.parameters[pid].param
        expanded = pid in self._expanded
        tools = []
        for ic, tip, delta in (("undo", "Move up", -1), ("redo", "Move down", 1)):
            b = tool_button(ic, tip)
            b.clicked.connect(lambda _=False, d=delta: self._edit("Reorder inputs", lambda: g.move_parameter(pid, d), True, "order"))
            tools.append(b)
        more = tool_button("properties", "Edit exposed graph input")
        more.setCheckable(True)
        more.setChecked(expanded)
        more.toggled.connect(lambda on: (self._expanded.add(pid) if on else self._expanded.discard(pid), self.rebuild()))
        tools.append(more)
        copy = tool_button("copy", "Copy this input")
        copy.clicked.connect(lambda: self._copy_inputs(g, [pid]))
        tools.append(copy)
        rm = tool_button("close", "Remove this graph input")
        rm.clicked.connect(lambda: self._edit("Remove input", lambda: g.unexpose(pid), True, "unexpose"))
        tools.append(rm)
        if preview is not None:
            editor = make_editor(p, values[pid], lambda v, final: self.doc.set_preview(
                self.package, g.id, {**self.doc.preview.get((id(self.package), g.id), {}), pid: v}))
        else:
            editor = make_editor(p, p.default, lambda v, final: self._edit(
                "Edit default", lambda: g.expose(pid, dataclasses.replace(p, default=v)), final, f"exposed.{pid}"))
        warning = check(p.visible_if, values)
        rollout.add(f"{p.label}  ({pid})" + ("  ⚠" if warning else ""), editor, tools)
        if warning:
            w = QLabel(warning)
            w.setStyleSheet(f"color: {theme.WARNING};")
            rollout.form.addWidget(w)
        if p.description:
            rollout.form.itemAt(rollout.form.count() - 1).widget().setToolTip(p.description)
        if expanded:
            self._input_properties(g, rollout, pid)

    def _input_properties(self, g, rollout, pid):
        p = g.parameters[pid].param
        box = QFrame()
        box.setStyleSheet(f"QFrame {{ background: {theme.BG_ROLLOUT}; }}")
        form = QVBoxLayout(box)
        form.setContentsMargins(8, 4, 8, 6)

        def row(label, w):
            line = QHBoxLayout()
            lab = QLabel(label)
            lab.setFixedWidth(84)
            lab.setObjectName("dim")
            line.addWidget(lab)
            line.addWidget(w, 1)
            form.addLayout(line)

        def setp(**changes):
            self._edit("Edit graph input", lambda: g.expose(pid, dataclasses.replace(p, **changes)), True, f"inprop.{pid}")

        ident = QLineEdit(pid)
        ident.editingFinished.connect(lambda: self._rename_input(g, pid, ident.text()))
        row("Identifier", ident)
        for label, field in (("Label", "label"), ("Group", "group"), ("Description", "description"),
                             ("User Data", "user_data"), ("Visible If", "visible_if")):
            e = QLineEdit(str(getattr(p, field)))
            e.editingFinished.connect(lambda e=e, field=field: setp(**{field: e.text()}) if e.text() != getattr(p, field) else None)
            if field == "visible_if":
                e.setPlaceholderText('e.g. input.mode == 2 && input.enabled')
            row(label, e)
        kinds = [k for k in ParamKind if k not in (ParamKind.GRADIENT, ParamKind.CURVE, ParamKind.FUNCTION, ParamKind.FXMAP)]
        kind = QComboBox()
        kind.addItems([k.value for k in kinds])
        kind.setCurrentText(p.kind.value)
        kind.currentTextChanged.connect(lambda t: self._retype_input(g, pid, ParamKind(t)))
        row("Type", kind)
        ed = QComboBox()
        names = EDITORS.get(p.kind, ())
        ed.addItems(list(names))
        ed.setCurrentText(p.editor_name())
        ed.currentTextChanged.connect(lambda t: setp(editor=t))
        row("Editor", ed)
        if p.kind in (ParamKind.FLOAT, ParamKind.INT, ParamKind.FLOAT2, ParamKind.FLOAT3, ParamKind.FLOAT4, ParamKind.INT2) \
                and p.editor_name() not in ("Drop down list", "Size pow 2"):
            for label, field in (("Min", "min"), ("Max", "max"), ("Step", "step")):
                f = NumberField(p.kind in (ParamKind.INT, ParamKind.INT2) and field != "step")
                f.set_value(getattr(p, field) if getattr(p, field) is not None else 0.0)
                f.committed.connect(lambda v, field=field: setp(**{field: v}))
                row(label, f)
            clamp = BoolToggle(p.clamp)
            clamp.edited.connect(lambda v, final: setp(clamp=v))
            row("Clamp", clamp)
        if p.kind is ParamKind.INT and p.editor_name() == "Drop down list":
            form.addWidget(self._dropdown_items(g, pid))
        if p.kind is ParamKind.ENUM:
            ch = QLineEdit(", ".join(p.choices))
            ch.editingFinished.connect(lambda: self._set_choices(g, pid, ch.text()))
            row("Values", ch)
        rollout.form.addWidget(box)

    def _dropdown_items(self, g, pid):
        """The Drop down list: items (integer value, label), one of them the default."""
        p = g.parameters[pid].param
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(84, 0, 0, 0)
        items = [list(i) for i in p.items] or [[1, "Item 1"]]

        def commit(new_items, default=None):
            d = default if default is not None else p.default
            vals = [int(v) for v, _ in new_items]
            if d not in vals and vals:
                d = vals[0]
            self._edit("Edit drop down list", lambda: g.expose(pid, dataclasses.replace(
                p, items=tuple((int(v), str(lbl)) for v, lbl in new_items), default=d)), True, f"items.{pid}")
        for i, (v, lbl) in enumerate(items):
            line = QHBoxLayout()
            val = NumberField(True)
            val.set_value(v)
            val.setFixedWidth(46)
            val.committed.connect(lambda x, i=i: commit([[int(x), it[1]] if j == i else it for j, it in enumerate(items)]))
            name = QLineEdit(lbl)
            name.editingFinished.connect(lambda name=name, i=i: commit([[it[0], name.text()] if j == i else it
                                                                         for j, it in enumerate(items)]))
            default = BoolToggle(v == p.default)
            default.edited.connect(lambda on, final, v=v: commit(items, v) if on else None)
            rm = tool_button("close", "Remove item")
            rm.clicked.connect(lambda _=False, i=i: commit([it for j, it in enumerate(items) if j != i]))
            line.addWidget(val)
            line.addWidget(name, 1)
            line.addWidget(default)
            line.addWidget(rm)
            lay.addLayout(line)
        add = QPushButton("+  Add item")
        add.clicked.connect(lambda: commit(items + [[max([int(i[0]) for i in items] + [0]) + 1, f"Item {len(items) + 1}"]]))
        lay.addWidget(add)
        return w

    def _set_choices(self, g, pid, text):
        p = g.parameters[pid].param
        choices = tuple(c.strip() for c in text.split(",") if c.strip())
        if not choices:
            return
        self._edit("Edit values", lambda: g.expose(pid, dataclasses.replace(
            p, choices=choices, default=p.default if p.default in choices else choices[0])), True, f"choices.{pid}")

    def _retype_input(self, g, pid, kind):
        p = g.parameters[pid].param
        self._edit("Change input type", lambda: g.expose(pid, Param(
            kind, DEFAULT_VALUES[kind], p.label, group=p.group, description=p.description, user_data=p.user_data,
            visible_if=p.visible_if, choices=("a", "b") if kind is ParamKind.ENUM else ())), True, f"type.{pid}")

    def _rename_input(self, g, pid, new):
        if new != pid:
            self._expanded.discard(pid)
            self._expanded.add(new)
            self._edit("Rename input", lambda: g.rename_parameter(pid, new), True, "rename_input")

    def _apply_preview(self, g):
        values = self.doc.preview.get((id(self.package), g.id), {})

        def do():
            for pid, v in values.items():
                if pid in g.parameters:
                    g.expose(pid, dataclasses.replace(g.parameters[pid].param, default=v))
        self._edit("Apply preview", do, True, "apply_preview")
        self.doc.set_preview(self.package, g.id, None)

    def _new_preset(self, g):
        name, ok = QInputDialog.getText(self, "New preset", "Name", text=f"Preset {len(g.presets) + 1}")
        if not ok or not name:
            return
        values = self.doc.preview.get((id(self.package), g.id)) or {pid: p.param.default for pid, p in g.parameters.items()}
        self._edit("New preset", lambda: g.presets.append({"name": name, "values": dict(values)}), True, "preset")

    def _rename_preset(self, g, i):
        name, ok = QInputDialog.getText(self, "Rename preset", "Name", text=g.presets[i]["name"])
        if ok and name:
            self._edit("Rename preset", lambda: g.presets[i].__setitem__("name", name), True, "preset")

    def _apply_graph_preset(self, g, i):
        """Load an embedded preset into the Preview (presets carry values, not the inputs themselves)."""
        values = {pid: v for pid, v in g.presets[i]["values"].items() if pid in g.parameters}
        self.doc.set_preview(self.package, g.id, {**{pid: p.param.default for pid, p in g.parameters.items()}, **values})

    def _save_preset_file(self, values):
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "preset" + PRESET_SUFFIX, f"Presets (*{PRESET_SUFFIX})")
        if path:
            open(path, "w", encoding="utf-8").write(json.dumps({"trafilatura3d_preset": 1, "values": values}, indent=1))

    def _read_preset_file(self):
        import json
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", f"Presets (*{PRESET_SUFFIX})")
        if not path:
            return None
        try:
            data = json.loads(open(path, encoding="utf-8").read())
            return dict(data["values"]) if data.get("trafilatura3d_preset") == 1 else None
        except (OSError, ValueError, KeyError) as e:
            self.doc.error.emit(f"Preset: {e}")
            return None

    def _load_preset_into_graph(self, g):
        values = self._read_preset_file()
        if values is not None:  # parameters the graph doesn't have are ignored
            self.doc.set_preview(self.package, g.id, {**{pid: p.param.default for pid, p in g.parameters.items()},
                                                       **{k: v for k, v in values.items() if k in g.parameters}})

    def _copy_inputs(self, g, ids=None):
        import json
        from PySide6.QtWidgets import QApplication
        ids = ids or list(g.parameters)
        QApplication.clipboard().setText(json.dumps({"trafilatura3d_inputs": [g.parameters[i].to_dict() for i in ids]}))

    def _paste_inputs(self, g):
        import json
        from PySide6.QtWidgets import QApplication
        from trafilatura3d.package import ExposedParam
        try:
            data = json.loads(QApplication.clipboard().text())["trafilatura3d_inputs"]
        except (ValueError, KeyError, TypeError):
            self.doc.error.emit("Paste Inputs: the clipboard holds no graph inputs")
            return

        def do():
            for d in data:
                ep = ExposedParam.from_dict(d)
                pid, n = ep.id, 1
                while pid in g.parameters:
                    n += 1
                    pid = f"{ep.id}{n}"
                g.expose(pid, ep.param)
        self._edit("Paste inputs", do, True, "paste_inputs")

    def _clean_inputs(self, g):
        from PySide6.QtWidgets import QMessageBox
        unused = g.unused_parameters()
        if not unused:
            QMessageBox.information(self, "Clean Inputs", "Every input is used.")
            return
        if QMessageBox.question(self, "Clean Inputs", "Remove the unused inputs?\n\n" + "\n".join(unused)) \
                == QMessageBox.Yes:
            self._edit("Clean inputs", lambda: [g.unexpose(pid) for pid in unused], True, "clean_inputs")

    def _io_list(self, g, title, which):
        """Inputs / Outputs: the graph's Input or Output nodes in their connector order on instances."""
        nodes = [n for n in g.nodes.values() if which(n)]
        if not nodes:
            return
        r = Rollout(title)
        for n in nodes:
            v = g.definition(n.id).values(n)
            tools = []
            for ic, tip, delta in (("undo", "Move up", -1), ("redo", "Move down", 1)):
                b = tool_button(ic, tip)
                b.clicked.connect(lambda _=False, nid=n.id, d=delta: self._edit(
                    "Reorder", lambda: g.move_node(nid, d, which), True, "order"))
                tools.append(b)
            text = QLabel(f"{v.get('label') or v['identifier']}  ({v['identifier']})")
            text.setObjectName("dim")
            r.add(n.type.replace("_", " ").capitalize(), text, tools)
        self.col.addWidget(r)

    def _graph_attributes(self, g):
        from trafilatura3d.package import GRAPH_ATTRIBUTES
        a = Rollout("Attributes")
        ident = QLineEdit(g.id)
        ident.editingFinished.connect(lambda: self._rename_graph(g, ident.text()))
        a.add("Identifier", ident)
        label = QLineEdit(g.label)
        label.editingFinished.connect(lambda: self._edit("Label", lambda: setattr(g, "label", label.text()), True, "glabel"))
        a.add("Label", label)
        desc = QPlainTextEdit(g.description)
        desc.setFixedHeight(60)
        desc.textChanged.connect(lambda: self._edit("Description", lambda: setattr(g, "description", desc.toPlainText()),
                                                    False, "gdesc"))
        a.add("Description", desc)
        labels = {"category": "Category", "author": "Author", "author_url": "Author URL", "tags": "Tags",
                  "group": "Group", "user_data": "User data"}
        for key, text in labels.items():
            e = QLineEdit(str(g.attributes.get(key, GRAPH_ATTRIBUTES[key])))
            e.editingFinished.connect(lambda e=e, key=key: self._edit(
                text, lambda: g.attributes.__setitem__(key, e.text()), True, f"gattr.{key}"))
            a.add(text, e)
        ps = VectorField(g.attributes.get("physical_size", GRAPH_ATTRIBUTES["physical_size"]), 0.0, 1000.0)
        ps.edited.connect(lambda v, final: self._edit("Physical size", lambda: g.attributes.__setitem__(
            "physical_size", list(v)), final, "gattr.physical_size"))
        a.add("Physical size", ps)
        self.col.addWidget(a)

    def _add_input(self, g):
        kinds = [k.value for k in ParamKind if k not in (ParamKind.GRADIENT, ParamKind.CURVE)]
        kind, ok = QInputDialog.getItem(self, "Add graph input", "Type", kinds, 0, False)
        if not ok:
            return
        pid, ok = QInputDialog.getText(self, "Add graph input", "Identifier")
        if not ok or not pid:
            return
        k = ParamKind(kind)
        default = DEFAULT_VALUES[k]
        choices = ("a", "b") if k is ParamKind.ENUM else ()
        self._edit("Add input", lambda: g.expose(pid, Param(k, default, pid, choices=choices)), True, "expose")

    def _rename_graph(self, g, new_id):
        if new_id == g.id:
            return
        self.rename_graph.emit(self.package, g.id, new_id)  # MainWindow.rename_graph: the views follow
