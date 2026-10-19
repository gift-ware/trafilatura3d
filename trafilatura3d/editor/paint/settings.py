"""Edit > Settings of the Paint workspace: General preferences, Shortcuts, Libraries configuration. Preferences
live in the application's QSettings under "paint_preferences/"; changes apply at once (no OK/Apply: ours).

General preferences offered are the ones the workspace supports: not offered Language (no translations),
Viewport Scaling, Save preprocessed scene files, Enable live preview baking process, Enable GPU Raytracing (always
on), the cache directories and the thumbnails' cache budget, Material preview shader, Hardware support acceleration,
Iray hardware, usage statistics. Ours: Engine > Cache memory budget, the computed textures' cache (the Graph
workspace Preferences' "Image cache memory budget": one setting for the evaluator both workspaces share)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QKeySequenceEdit, QLabel,
                               QListWidget, QMenu, QMessageBox, QPushButton, QRadioButton, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from trafilatura3d.editor import theme
from trafilatura3d.editor.paint import shortcuts
from trafilatura3d.editor.paint.widgets import PSlider, Swatch, header
from trafilatura3d.paint.materials import EXAMPLES
from trafilatura3d.paint.smart import SHIPPED

# key -> default; the documentation gives none of these defaults but the off "Only display the selected material"
# (it recommends keeping it off): the others are ours
DEFAULTS = {
    "keyboard_helper": True,
    "world_axes": True,
    "background_top": [0.3, 0.3, 0.3],
    "background_bottom": [0.137, 0.137, 0.137],
    "selected_material_only": False,
    "uv_scale": 1.0,
    "simplified_thumbnails": False,
    "fast_painting": False,
    "rotation_speed": 1.0,
    "zoom_speed": 1.0,
    "zoom_inverse": False,
    "wheel_speed": 1.0,
    "wheel_inverse": False,
}


def get(key: str):
    default = DEFAULTS[key]
    v = QSettings().value(f"paint_preferences/{key}", default)
    if isinstance(default, bool):
        return v in (True, "true", "True", 1, "1")
    if isinstance(default, float):
        return float(v)
    if isinstance(default, list):
        return [float(x) for x in v]
    return v


def put(key: str, value) -> None:
    QSettings().setValue(f"paint_preferences/{key}", value)


# --- Libraries configuration --------------------------------------------------------------------------------
# A library is a folder of resources with sub-folders per kind (ours, the kinds the Assets read: `materials`
# packages, `smart_materials`). The two default paths of the page: your_assets (the user's folder, where new
# content goes) and starter_assets (the installation: the examples' materials and the shipped smart materials).
# Stored in QSettings "paint_libraries": {"paths": [{"name", "path"}], "default": name, "disabled": [names]}.
STARTER, YOURS = "starter_assets", "your_assets"
KINDS = ("materials", "smart_materials")
LIBRARIES_KEY = "paint_libraries"


def _libraries_state() -> dict:
    try:
        d = json.loads(QSettings().value(LIBRARIES_KEY, "{}"))
    except (TypeError, ValueError):
        d = {}
    return {"paths": list(d.get("paths", [])), "default": d.get("default", YOURS),
            "disabled": list(d.get("disabled", []))}


def _save_libraries(state: dict) -> None:
    QSettings().setValue(LIBRARIES_KEY, json.dumps(state))


def libraries() -> list[dict]:
    """Every library: {"name", "path", "builtin", "enabled", "default"}, the default paths first."""
    from trafilatura3d.project import user_dir
    s = _libraries_state()
    rows = [{"name": YOURS, "path": str(user_dir())}, {"name": STARTER, "path": str(Path(EXAMPLES).parent)}]
    rows += [{"name": p["name"], "path": p["path"]} for p in s["paths"]]
    for r in rows:
        r["builtin"] = r["name"] in (YOURS, STARTER)
        r["enabled"] = r["name"] not in s["disabled"]
        r["default"] = r["name"] == s["default"]
    return rows


def library_folder(lib: dict, kind: str) -> Path:
    if lib["name"] == STARTER:  # the installation's layout
        return {"materials": Path(EXAMPLES), "smart_materials": SHIPPED}[kind]
    return Path(lib["path"]) / kind


def folders(kind: str) -> list[Path]:
    """A kind's folder in every enabled library."""
    return [library_folder(lib, kind) for lib in libraries() if lib["enabled"]]


def material_folders() -> list[Path]:
    """Where the Assets look for materials: the libraries, then the library folders of the project configuration
    (shared with the Graph workspace)."""
    from trafilatura3d.project import configuration
    return folders("materials") + [Path(p) for p in configuration().library_paths()]


def default_folder(kind: str) -> Path:
    """Where new content of a kind is saved: the Default library."""
    lib = next((x for x in libraries() if x["default"]), libraries()[0])
    return library_folder(lib, kind)


def add_library(name: str, path: str) -> None:
    """A new path, with its sub-folder structure (Plus button)."""
    if not name or any(x["name"] == name for x in libraries()):
        raise ValueError(f"a library named '{name}' exists already" if name else "a library needs a name")
    for kind in KINDS:
        (Path(path) / kind).mkdir(parents=True, exist_ok=True)
    s = _libraries_state()
    s["paths"].append({"name": name, "path": str(path)})
    _save_libraries(s)


def remove_library(name: str) -> None:
    """Minus button: a user path is removed; a default path cannot be, it is disabled (pressed again: enabled)."""
    s = _libraries_state()
    if name in (YOURS, STARTER):
        s["disabled"] = [n for n in s["disabled"] if n != name] if name in s["disabled"] else s["disabled"] + [name]
    else:
        s["paths"] = [p for p in s["paths"] if p["name"] != name]
        s["disabled"] = [n for n in s["disabled"] if n != name]
        if s["default"] == name:
            s["default"] = YOURS
    _save_libraries(s)


def set_default_library(name: str) -> None:
    if name == STARTER:
        raise ValueError("starter_assets is the installation: new content cannot go there")
    s = _libraries_state()
    s["default"] = name
    _save_libraries(s)


class GeneralPreferences(QWidget):
    """The General preferences page: `changed(key)` after every edit."""
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        theme.section_lines(self)
        lay = QVBoxLayout(self)
        self.widgets = {}

        def check(key, text):
            w = QCheckBox(text)
            w.setChecked(get(key))
            w.toggled.connect(lambda on: self._set(key, on))
            lay.addWidget(w)
            self.widgets[key] = w

        def slider(key, text, lo, hi):
            w = PSlider(text, get(key), lo, hi)
            w.edited.connect(lambda v, _final: self._set(key, v))
            lay.addWidget(w)
            self.widgets[key] = w

        lay.addWidget(header("Interface"))
        check("keyboard_helper", "Show keyboard helper")
        check("world_axes", "Show world axes")
        row = QHBoxLayout()
        row.addWidget(QLabel("Background color"), 1)
        for key in ("background_top", "background_bottom"):
            w = Swatch(get(key), "Background")
            w.setFixedWidth(72)
            w.edited.connect(lambda rgb, _final, key=key: self._set(key, rgb))
            row.addWidget(w)
            self.widgets[key] = w
        lay.addLayout(row)
        check("selected_material_only", "Only display the selected material when painting")
        slider("uv_scale", "Default UV scale for materials", 0.01, 16.0)
        check("simplified_thumbnails", "Use simplified thumbnails")
        # ours (as ZBrush lowering its subdivision while rotating): while painting only the painted channels' slots
        # update, the others at the release; it also turns on by itself when a stroke would cost too much
        check("fast_painting", "Fast painting: only the painted channels update while painting")
        lay.addWidget(header("Camera options"))
        slider("rotation_speed", "Rotation speed", 0.1, 5.0)
        slider("zoom_speed", "Zoom speed", 0.1, 5.0)
        check("zoom_inverse", "Inverse direction")
        slider("wheel_speed", "Wheel speed", 0.1, 5.0)
        check("wheel_inverse", "Inverse direction")
        lay.addWidget(header("Engine"))  # ours: the Graph Preferences' setting, the same evaluator
        from trafilatura3d.editor import preferences as editor_prefs
        row = QHBoxLayout()
        row.addWidget(QLabel("Cache memory budget"), 1)
        mode = QComboBox()
        mode.addItems(editor_prefs.CACHE_MODES)
        mode.setCurrentText(editor_prefs.get("cache_mode"))
        mb = QSpinBox()
        mb.setRange(128, 65536)
        mb.setSuffix(" MB")
        mb.setValue(editor_prefs.get("cache_mb"))
        mb.setEnabled(mode.currentText() == "Custom")
        mode.setToolTip(editor_prefs.AUTOMATIC_CACHE)

        def cache(key, value):
            editor_prefs.put(key, value)
            mb.setEnabled(mode.currentText() == "Custom")
            self.changed.emit(key)
        mode.currentTextChanged.connect(lambda t: cache("cache_mode", t))
        mb.valueChanged.connect(lambda v: cache("cache_mb", v))
        row.addWidget(mode)
        row.addWidget(mb)
        lay.addLayout(row)
        self.widgets["cache_mode"], self.widgets["cache_mb"] = mode, mb
        lay.addStretch(1)

    def _set(self, key, value):
        put(key, value)
        self.changed.emit(key)


class MouseCapture(QLabel):
    """Editing a mouse binding: the next click with its modifiers (and a letter key held) is the binding."""
    captured = Signal(str)

    def __init__(self):
        super().__init__("Hold modifiers or a key, then click here")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #f0a030; padding: 2px;")
        self.setFocusPolicy(Qt.StrongFocus)
        self.held = None

    def keyPressEvent(self, e):
        t = e.text().upper()
        self.held = t if len(t) == 1 and t.isalpha() else self.held

    def keyReleaseEvent(self, e):
        if e.text().upper() == self.held:
            self.held = None

    def mousePressEvent(self, e):
        button = {Qt.LeftButton: 1, Qt.RightButton: 2, Qt.MiddleButton: 3}.get(e.button())
        if button is None:
            return
        m = e.modifiers()
        mods = {name for flag, name in ((Qt.ControlModifier, "Control"), (Qt.MetaModifier, "Meta"),
                                        (Qt.AltModifier, "Alt"), (Qt.ShiftModifier, "Shift")) if m & flag}
        self.captured.emit(shortcuts.mouse_text(mods, self.held, button))


class ShortcutsPage(QWidget):
    """The Shortcuts page: every editable shortcut with its binding; the pen edits it (a key sequence, or a click
    for the mouse bindings; the last key pressed ends the edit), right click resets it to the default.
    `changed()` after every edit."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut", ""])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(2, 28)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.table)
        self.names = list(shortcuts.KEYBOARD) + list(shortcuts.MOUSE)
        self.table.setRowCount(len(self.names))
        for row, name in enumerate(self.names):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            pen = QPushButton("✎")
            pen.setFlat(True)
            pen.setToolTip("Edit the shortcut")
            pen.clicked.connect(lambda _=False, row=row: self.edit(row))
            self.table.setCellWidget(row, 2, pen)
            self._show(row)

    def _show(self, row):
        self.table.removeCellWidget(row, 1)
        name = self.names[row]
        self.table.setItem(row, 1, QTableWidgetItem(shortcuts.label(shortcuts.binding(name))))

    def edit(self, row):
        name = self.names[row]
        if name in shortcuts.MOUSE:
            w = MouseCapture()
            w.captured.connect(lambda text: self._set(row, text))
        else:
            w = QKeySequenceEdit()
            w.editingFinished.connect(lambda: self._set(row, w.keySequence().toString()))
        self.table.setItem(row, 1, None)
        self.table.setCellWidget(row, 1, w)
        w.setFocus()

    def _set(self, row, text):
        shortcuts.set_binding(self.names[row], text)
        self._show(row)
        self.changed.emit()

    def _menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        m = QMenu(self)
        m.addAction("Reset to default", lambda: (shortcuts.set_binding(self.names[row], None), self._show(row),
                                                 self.changed.emit()))
        m.exec(self.table.viewport().mapToGlobal(pos))


class LibrariesPage(QWidget):
    """The Libraries configuration page: Default (where new content is saved), Name and Path of every library;
    Plus adds a path (and its sub-folders), Minus removes one (a default path is disabled instead). Paths can be
    changed only with no project open (documentation). `changed()` after every edit."""
    changed = Signal()

    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        lay = QVBoxLayout(self)
        self.note = QLabel("Close the project to change the paths.")
        lay.addWidget(self.note)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Default", "Name", "Path"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        row.addStretch(1)
        self.plus, self.minus = QPushButton("+"), QPushButton("−")
        for b in (self.plus, self.minus):
            b.setFixedWidth(32)
            row.addWidget(b)
        self.plus.setToolTip("Add a path")
        self.minus.setToolTip("Remove the selected path (a default path is disabled, or enabled again)")
        self.plus.clicked.connect(self.add)
        self.minus.clicked.connect(self.remove)
        lay.addLayout(row)
        self.rebuild()

    def rebuild(self):
        editable = self.ws.doc is None
        self.note.setVisible(not editable)
        self.rows = libraries()
        self.table.setRowCount(len(self.rows))
        for i, lib in enumerate(self.rows):
            radio = QRadioButton()
            radio.setChecked(lib["default"])
            radio.setEnabled(editable and lib["name"] != STARTER and lib["enabled"])
            radio.toggled.connect(lambda on, name=lib["name"]: on and self._default(name))
            self.table.setCellWidget(i, 0, radio)
            for col, text in ((1, lib["name"]), (2, lib["path"])):
                it = QTableWidgetItem(text + ("" if lib["enabled"] else "  (disabled)") if col == 1 else text)
                if not lib["enabled"]:
                    it.setForeground(Qt.gray)
                self.table.setItem(i, col, it)
        self.plus.setEnabled(editable)
        self.minus.setEnabled(editable)

    def _default(self, name):
        set_default_library(name)
        self.changed.emit()

    def add(self, path=None, name=None):
        path = path or QFileDialog.getExistingDirectory(self, "Library path")
        if not path:
            return
        try:
            add_library(name or Path(path).name, path)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Libraries configuration", str(e))
            return
        self.rebuild()
        self.changed.emit()

    def remove(self):
        i = self.table.currentRow()
        if i < 0:
            return
        remove_library(self.rows[i]["name"])
        self.rebuild()
        self.changed.emit()


class SettingsDialog(QDialog):
    """The Settings window: the sections on the left, the page on the right."""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.setWindowTitle("Settings")
        self.resize(760, 560)
        lay = QHBoxLayout(self)
        self.sections = QListWidget()
        self.sections.setFixedWidth(180)
        self.pages = QStackedWidget()
        lay.addWidget(self.sections)
        lay.addWidget(self.pages, 1)
        self.general = GeneralPreferences()
        self.general.changed.connect(workspace.apply_preferences)
        self._add("General preferences", self.general)
        self.shortcuts = ShortcutsPage()
        self.shortcuts.changed.connect(workspace.apply_shortcuts)
        self._add("Shortcuts", self.shortcuts)
        self.libraries = LibrariesPage(workspace)
        self.libraries.changed.connect(workspace.libraries_changed)
        self._add("Libraries configuration", self.libraries)
        self.sections.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.sections.setCurrentRow(0)

    def _add(self, name, page):
        self.sections.addItem(name)
        self.pages.addWidget(page)
