"""Preferences (Edit > Preferences), sections: General, Projects, Graph, 3D View, Bakers,
Library, Shortcuts. Application-wide settings live in QSettings; the Projects section edits the
project stack (trafilatura3d.project): Default (read only), User, and the custom projects of the
configuration list."""

from __future__ import annotations

import json
import os
import sys

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QKeySequenceEdit, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from trafilatura3d import project as projects

# key -> default
DEFAULTS = {
    "recent_count": 10, "history_size": 200, "invert_zoom": False, "save_path": "Last path",
    "cache_mode": "Automatic", "cache_mb": 2048, "auto_backup_minutes": 5, "auto_backup_versions": 6,
    "cooking_size_limit": 8192, "disable_high_dpi": False,
    "tab_node_menu": True, "drag_connector_menu": True, "outputs_3d_on_open": True, "auto_thumbnails": True,
    "output_2d_on_open": True, "display_new_node": True, "auto_conversion": False,
    "environment_hidden": True,
    "shortcuts": "{}",  # JSON: node type -> key sequence
}
SAVE_PATHS = ("Last path", "Package path")
CACHE_MODES = ("Automatic", "Custom")
SIZE_LIMITS = (1024, 2048, 4096, 8192, 16384)


def settings() -> QSettings:
    """The application's settings (organization and name set by app.py; tests set their own)."""
    return QSettings()


def get(key: str):
    default = DEFAULTS[key]
    v = settings().value(f"preferences/{key}", default)
    if isinstance(default, bool):
        return v in (True, "true", "True", 1, "1")
    if isinstance(default, int):
        return int(v)
    return v


def put(key: str, value) -> None:
    settings().setValue(f"preferences/{key}", value)


def zoom_sign() -> int:
    """-1 when Preferences > Invert zoom in views is on."""
    return -1 if get("invert_zoom") else 1


def suggested_dir(package_path=None) -> str:
    """Preferences > Save/Export path: the last path used, or the package's folder."""
    from pathlib import Path
    if get("save_path") == "Package path" and package_path:
        return str(Path(package_path).parent)
    return str(settings().value("last_dir", str(Path.home())))


def shortcuts() -> dict[str, str]:
    try:
        return dict(json.loads(get("shortcuts")))
    except ValueError:
        return {}


AUTOMATIC_CACHE = "Automatic: an eighth of the memory, at least 2 GB"


def cache_bytes() -> int:
    """Image cache budget: Automatic or Custom MB. Automatic is ours: an eighth
    of the physical memory, at least 2 GB. A Paint project at 2048 with 5 UV tiles and 63 layers of painted masks:
    with 2 GB a layer low in the stack computed it all again at every edit (3 s), with 3 GB an edit takes 1 s as with
    4.8 GB and only the first after a selection is slower (2.7 s against 1.4), 1.5 GB less memory."""
    if get("cache_mode") != "Automatic":
        return get("cache_mb") << 20
    return max(2 << 30, physical_memory() // 8)


def physical_memory() -> int:
    """The machine's RAM in bytes (on Apple silicon also the GPU's)."""
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                        ("page_avail", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                        ("virtual_avail", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
        st = Status()
        st.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.total
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


class PreferencesDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Preferences")
        self.resize(860, 560)
        self.config = projects.configuration()
        lay = QHBoxLayout(self)
        self.sections = QListWidget()
        self.sections.setFixedWidth(150)
        self.pages = QStackedWidget()
        for name, page in (("General", self._general()), ("Projects", self._projects()), ("Graph", self._graph()),
                           ("3D View", self._view3d()), ("Bakers", self._bakers()), ("Library", self._library()),
                           ("Shortcuts", self._shortcuts())):
            self.sections.addItem(name)
            self.pages.addWidget(page)
        self.sections.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.sections.setCurrentRow(0)
        right = QVBoxLayout()
        right.addWidget(self.pages, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        lay.addWidget(self.sections)
        lay.addLayout(right, 1)
        self.values = {}

    # --- widgets bound to preferences -------------------------------------------------------------

    def _check(self, key, label):
        w = QCheckBox(label)
        w.setChecked(get(key))
        w.toggled.connect(lambda on: self.values.__setitem__(key, on))
        return w

    def _spin(self, key, lo, hi, suffix=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(get(key))
        if suffix:
            w.setSuffix(suffix)
        w.valueChanged.connect(lambda v: self.values.__setitem__(key, v))
        return w

    def _combo(self, key, items, as_int=False):
        w = QComboBox()
        w.addItems([str(i) for i in items])
        w.setCurrentText(str(get(key)))
        w.currentTextChanged.connect(lambda t: self.values.__setitem__(key, int(t) if as_int else t))
        return w

    def _general(self):
        page = QWidget()
        f = QFormLayout(page)
        f.addRow("Recent documents list contains", self._spin("recent_count", 1, 50))
        f.addRow("History stack size", self._spin("history_size", 10, 5000))
        lang = QComboBox()
        lang.addItem("System")
        lang.setEnabled(False)
        lang.setToolTip("The interface is in English only for now")
        f.addRow("Choose the language of the application", lang)
        f.addRow("Views", self._check("invert_zoom", "Invert zoom in views"))
        f.addRow("Save/Export path", self._combo("save_path", SAVE_PATHS))
        cache = QHBoxLayout()
        cache.addWidget(self._combo("cache_mode", CACHE_MODES))
        cache.addWidget(self._spin("cache_mb", 128, 65536, " MB"))
        f.addRow("Image cache memory budget", cache)
        f.addRow("Disable High DPI", self._check("disable_high_dpi", "Keep UI scaling constant (after restart)"))
        backup = QHBoxLayout()
        backup.addWidget(QLabel("every"))
        backup.addWidget(self._spin("auto_backup_minutes", 0, 240, " min"))
        backup.addWidget(QLabel("keep up to"))
        backup.addWidget(self._spin("auto_backup_versions", 1, 100, " versions"))
        f.addRow("Auto-backup", backup)
        f.addRow("Cooking size limit", self._combo("cooking_size_limit", SIZE_LIMITS, as_int=True))
        return page

    def _graph(self):
        page = QWidget()
        f = QVBoxLayout(page)
        f.addWidget(QLabel("<b>Common</b>"))
        f.addWidget(self._check("tab_node_menu", "Tab key shows node menu"))
        f.addWidget(self._check("drag_connector_menu", "Enable creation of nodes by click-dragging connectors"))
        f.addWidget(self._check("outputs_3d_on_open", "View outputs in 3D view when opening a graph"))
        f.addWidget(QLabel("<b>Compositing graph</b>"))
        f.addWidget(self._check("auto_thumbnails", "Automatically compute all nodes thumbnails when opening a graph"))
        f.addWidget(self._check("output_2d_on_open", "View output in 2D view when opening a graph"))
        f.addWidget(self._check("display_new_node", "Automatically display newly created compositing node"))
        f.addWidget(self._check("auto_conversion", "Automatically insert color/greyscale conversion node"))
        f.addStretch(1)
        return page

    def _view3d(self):
        page = QWidget()
        f = QFormLayout(page)
        f.addRow("Misc", self._check("environment_hidden", "Environment hidden by default"))
        scaling = QComboBox()
        scaling.addItems(["Auto"])
        scaling.setEnabled(False)
        scaling.setToolTip("The 3D View always renders at the display's resolution")
        f.addRow("Viewport scaling", scaling)
        return page

    def _bakers(self):
        page = QWidget()
        f = QVBoxLayout(page)
        gpu = QCheckBox("GPU raytracing")
        gpu.setChecked(True)
        gpu.setEnabled(False)
        gpu.setToolTip("Trafilatura 3D's bakers always trace rays on the GPU (there is no CPU baker)")
        f.addWidget(gpu)
        f.addStretch(1)
        return page

    def _library(self):
        page = QWidget()
        f = QVBoxLayout(page)
        b = QPushButton("Rebuild thumbnails")
        b.setToolTip("Cook every Library node again and replace the thumbnails")
        b.clicked.connect(self._rebuild_thumbnails)
        f.addWidget(b)
        self.thumb_status = QLabel("")
        self.thumb_status.setObjectName("dim")
        f.addWidget(self.thumb_status)
        f.addStretch(1)
        return page

    def _rebuild_thumbnails(self):
        from trafilatura3d.editor.paint import shaderball
        from trafilatura3d.thumbnails import rebuild
        self.thumb_status.setText("Rebuilding…")
        self.repaint()
        made, failed = rebuild(self.window.doc.ev)
        n, f = shaderball.rebuild()  # the Paint filters and generators on the shader ball
        made, failed = made + n, failed + f
        self.thumb_status.setText(f"{made} thumbnails rebuilt" + (f", {len(failed)} failed" if failed else ""))
        self.window.library.refresh()

    def _shortcuts(self):
        from trafilatura3d.editor.graph.nodemenu import graph_nodes
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel("Keyboard shortcuts that create a node at the mouse in the Graph view."))
        self.sc_table = QTableWidget(0, 2)
        self.sc_table.setHorizontalHeaderLabels(["Node", "Shortcut"])
        self.sc_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        current = shortcuts()
        types = sorted((d.label, t) for t, d in graph_nodes().items() if not getattr(d, "passthrough", False))
        self.sc_table.setRowCount(len(types))
        self.sc_edits = {}
        for row, (label, t) in enumerate(types):
            it = QTableWidgetItem(label)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.sc_table.setItem(row, 0, it)
            edit = QKeySequenceEdit(QKeySequence(current.get(t, "")))
            edit.keySequenceChanged.connect(lambda _s: self._check_conflicts())
            self.sc_table.setCellWidget(row, 1, edit)
            self.sc_edits[t] = edit
        lay.addWidget(self.sc_table, 1)
        self.sc_warning = QLabel("")
        self.sc_warning.setStyleSheet("color: #e8a33a;")
        lay.addWidget(self.sc_warning)
        return page

    def _check_conflicts(self):
        seen, dups = {}, set()
        for t, e in self.sc_edits.items():
            k = e.keySequence().toString()
            if k:
                if k in seen:
                    dups.add(k)
                seen[k] = t
        self.sc_warning.setText(("⚠ Conflicting shortcuts (they do nothing until resolved): " + ", ".join(sorted(dups)))
                                if dups else "")

    # --- projects ----------------------------------------------------------------------------------

    def _projects(self):
        page = QWidget()
        lay = QHBoxLayout(page)
        left = QVBoxLayout()
        self.proj_list = QListWidget()
        left.addWidget(QLabel("Projects (later ones override earlier ones)"))
        left.addWidget(self.proj_list, 1)
        row = QHBoxLayout()
        add = QPushButton("Add…")
        add.clicked.connect(self._add_project)
        new = QPushButton("New…")
        new.clicked.connect(lambda: self._add_project(create=True))
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove_project)
        for b in (add, new, rm):
            row.addWidget(b)
        left.addLayout(row)
        cfg = QLabel(f"Configuration list: {self.config.config_path or '(none: set TRAFILATURA3D_CONFIG or add a project)'}")
        cfg.setWordWrap(True)
        cfg.setObjectName("dim")
        left.addWidget(cfg)
        lay.addLayout(left, 1)
        self.proj_editor = QStackedWidget()
        lay.addWidget(self.proj_editor, 2)
        self.proj_list.currentRowChanged.connect(self._show_project)
        self._fill_projects()
        return page

    def _fill_projects(self):
        self.proj_list.clear()
        while self.proj_editor.count():
            self.proj_editor.removeWidget(self.proj_editor.widget(0))
        for p in self.config.projects:
            it = QListWidgetItem(p.name + ("  (read only)" if p.read_only else ""))
            it.setToolTip(str(p.path) if p.path else "Built in")
            self.proj_list.addItem(it)
            self.proj_editor.addWidget(ProjectEditor(p))
        self.proj_list.setCurrentRow(1)

    def _show_project(self, row):
        if row >= 0:
            self.proj_editor.setCurrentIndex(row)

    def _add_project(self, create=False):
        if create:
            path, _ = QFileDialog.getSaveFileName(self, "New project", f"project{projects.PROJECT_SUFFIX}",
                                                  f"Trafilatura 3D projects (*{projects.PROJECT_SUFFIX})")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Add project", "", f"Trafilatura 3D projects (*{projects.PROJECT_SUFFIX})")
        if path:
            self.config.add_project(path)
            self._fill_projects()
            self.proj_list.setCurrentRow(self.proj_list.count() - 1)

    def _remove_project(self):
        row = self.proj_list.currentRow()
        if row >= 2:  # the Default and User projects cannot be removed
            self.config.remove_project(self.config.projects[row])
            self._fill_projects()

    # --- apply -------------------------------------------------------------------------------------

    def accept(self):
        if self.sc_warning.text():
            return  # resolve conflicting shortcuts first
        for k, v in self.values.items():
            put(k, v)
        put("shortcuts", json.dumps({t: e.keySequence().toString() for t, e in self.sc_edits.items()
                                     if e.keySequence().toString()}))
        for i in range(self.proj_editor.count()):
            self.proj_editor.widget(i).commit()
        from trafilatura3d import colormanagement
        colormanagement.reset_cache()
        self.window.apply_preferences()
        super().accept()


class ProjectEditor(QWidget):
    """The settings of one project (the Default project is shown read only)."""

    def __init__(self, project):
        super().__init__()
        self.p = project
        f = QFormLayout(self)
        ro = project.read_only
        self.fields = {}

        def line(section, key, label, browse=None):
            e = QLineEdit(str(project.get(section, key, projects.DEFAULTS[section][key])))
            e.setReadOnly(ro)
            if browse and not ro:
                row = QHBoxLayout()
                row.addWidget(e, 1)
                b = QPushButton("…")
                b.setFixedWidth(28)
                b.clicked.connect(lambda: (path := QFileDialog.getOpenFileName(self, label, "", browse)[0]) and e.setText(path))
                row.addWidget(b)
                f.addRow(label, row)
            else:
                f.addRow(label, e)
            self.fields[(section, key)] = e

        def combo(section, key, label, items):
            c = QComboBox()
            c.addItems(items)
            c.setCurrentText(project.get(section, key, projects.DEFAULTS[section][key]))
            c.setEnabled(not ro)
            f.addRow(label, c)
            self.fields[(section, key)] = c

        f.addRow(QLabel("<b>General</b>"))
        combo("general", "normal_format", "Default normal format", list(projects.NORMAL_FORMATS))
        combo("general", "image_format", "Default image format", list(projects.IMAGE_FORMATS))
        f.addRow(QLabel("<b>Aliases</b>  (name = path, one per line; use as name://rest)"))
        self.aliases = QTableWidget(0, 2)
        self.aliases.setHorizontalHeaderLabels(["Alias", "Path"])
        self.aliases.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for name, path in project.get("aliases").items():
            r = self.aliases.rowCount()
            self.aliases.insertRow(r)
            self.aliases.setItem(r, 0, QTableWidgetItem(name))
            self.aliases.setItem(r, 1, QTableWidgetItem(path))
        self.aliases.setEnabled(not ro)
        self.aliases.setFixedHeight(110)
        f.addRow(self.aliases)
        if not ro:
            add = QPushButton("+ Alias")
            add.clicked.connect(lambda: self.aliases.insertRow(self.aliases.rowCount()))
            f.addRow(add)
        f.addRow(QLabel("<b>Library</b>"))
        self.library = QLineEdit("; ".join(project.get("library", "paths", [])))
        self.library.setReadOnly(ro)
        self.library.setPlaceholderText("watched folders, separated by ;")
        f.addRow("Watched paths", self.library)
        f.addRow(QLabel("<b>Color management</b>"))
        combo("color_management", "mode", "Mode", list(projects.COLOR_MODES))
        line("color_management", "ocio_config", "OCIO config (empty: built-in studio config)", "OCIO (*.ocio)")
        for key, label in (("working_space", "Working space"), ("texture_space", "Texture space"),
                           ("data_space", "Data space"), ("display", "Display"), ("view", "View")):
            line("color_management", key, label)
        f.addRow(QLabel("<b>Baking</b>"))
        line("baking", "high_suffix", "High poly suffix")
        line("baking", "low_suffix", "Low poly suffix")

    def commit(self):
        if self.p.read_only:
            return
        for (section, key), w in self.fields.items():
            v = w.currentText() if isinstance(w, QComboBox) else w.text()
            if v != str(self.p.get(section, key, projects.DEFAULTS[section][key])):
                self.p.set(section, key, v)
        aliases = {}
        for r in range(self.aliases.rowCount()):
            n, v = self.aliases.item(r, 0), self.aliases.item(r, 1)
            if n and v and n.text().strip():
                aliases[n.text().strip()] = v.text().strip()
        self.p.data["aliases"] = aliases
        self.p.set("library", "paths", [x.strip() for x in self.library.text().split(";") if x.strip()])
        self.p.save()
