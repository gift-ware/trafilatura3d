"""The Baking window (Explorer > 3D scene resource > Bake model information): the mesh
setup (common parameters) and a list of bakers, each with its own parameters; Start Render writes the
maps and links them to the package as bitmap resources. The configuration is saved in the resource as
it is edited (one undo step per editing burst), so the 3D View's Baking display (cage, rear distance,
high poly outside them) follows live; the window is not modal and there is one per resource."""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSplitter, QVBoxLayout, QWidget)

from trafilatura3d import bake
from trafilatura3d.editor.bake_form import baker_rows, editor_widget
from trafilatura3d.package import Resource

SIZES = [str(s) for s in bake.SIZES]
MESH_SETUP = "Mesh setup"
COMMON = [  # (Settings field, label, editor, condition)
    ("antialiasing", "Anti-Aliasing", list(bake.ANTIALIASING), None),
    ("dilation", "Dilation (px)", (0, 256), None),
    ("apply_diffusion", "Apply Diffusion", "bool", None),
    ("average_normals", "Average Normals", "bool", None),
    ("high_meshes", "High Definition Meshes", "files", ("use_low_as_high", False)),
    ("use_low_as_high", "Use Low as High Definition", "bool", None),
    ("use_cage", "Use Cage", "bool", None),
    ("cage_file", "Cage File", "file", ("use_cage", True)),
    ("frontal", "Max Frontal Distance", (0.0, 100.0), ("use_cage", False)),  # the implicit cage's inflate
    ("rear", "Max Rear Distance", (0.0, 100.0), ("use_cage", False)),
    ("relative_to_bounding_box", "Relative to Bounding Box", "bool", None),
    ("match", "Match", list(bake.MATCH), None),
    ("low_suffix", "Low Poly Mesh Suffix", "text", ("match", "By Mesh Name")),
    ("high_suffix", "High Poly Mesh Suffix", "text", ("match", "By Mesh Name")),
    ("ignore_backface_suffix", "Ignore Backfaces Suffix", "text", None),
    ("skew_correction", "Use Skew Correction", "bool", None),
    ("skew_map", "Skew Map", "file", ("skew_correction", True)),
    ("invert_skew", "Invert Skew Correction", "bool", ("skew_correction", True)),
]
CONDITIONS = {c[0] for *_, c in COMMON if c}  # common settings other rows depend on


TIPS = {"frontal": "How far in front of the low surface rays start: the implicit cage is the low pushed out by "
                   "this distance (Baking > Show cage in the 3D View)",
        "rear": "How far behind the low surface rays reach (Baking > Show rear distance in the 3D View)",
        "relative_to_bounding_box": "Frontal and rear distances as a fraction of the low's bounding box diagonal; "
                                    "off: in scene units",
        "use_cage": "Rays go from the cage file's surface towards the low instead of the implicit cage",
        "average_normals": "Ray directions averaged where the low has hard edges or UV seams (no gaps in the cage)"}
_OPEN: dict = {}  # (id(package), resource id) -> the open window


def settings_from(data: dict | None) -> bake.Settings:
    fields = {f.name for f in dataclasses.fields(bake.Settings)}
    s = bake.Settings(**{k: v for k, v in (data or {}).items() if k in fields})
    s.size = tuple(s.size)
    return s


def open_baker(document, package, resource_id: str, parent=None) -> "BakerDialog":
    """The resource's Baking window, raised if already open."""
    key = (id(package), resource_id)
    dlg = _OPEN.get(key)
    if dlg is None:
        dlg = _OPEN[key] = BakerDialog(document, package, resource_id, parent)
        dlg.finished.connect(lambda _r: _OPEN.pop(key, None))
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


class BakerDialog(QDialog):
    def __init__(self, document, package, resource_id: str, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.doc, self.pkg, self.rid = document, package, resource_id
        self._live = QTimer(self)  # edits saved to the resource after a short pause: the 3D View follows
        self._live.setSingleShot(True)
        self._live.setInterval(150)
        self._live.timeout.connect(self._save)
        self.mesh = str((package.path.parent / package.resources[resource_id].path).resolve())
        saved = package.resources[resource_id].baking or {}
        self.settings = settings_from(saved.get("settings"))
        self.bakers: list[dict] = [dict(b) for b in saved.get("bakers", [])]  # {"id", "enabled", "params"}
        self.setWindowTitle(f"Baking - {resource_id}")
        self.resize(900, 600)
        lay = QVBoxLayout(self)
        split = QSplitter()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("Add bakers")
        menu = QMenu(add)
        for bid, name in sorted(bake.BAKERS.items(), key=lambda kv: kv[1]):
            menu.addAction(name, lambda b=bid: self._add(b))
        add.setMenu(menu)
        ll.addWidget(add)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        self.list.itemChanged.connect(self._toggled)
        ll.addWidget(self.list, 1)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove)
        ll.addWidget(remove)
        split.addWidget(left)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        split.addWidget(self.scroll)
        split.setSizes([260, 640])
        lay.addWidget(split, 1)

        out = QFormLayout()
        row = QHBoxLayout()
        self.folder = QLineEdit(saved.get("folder") or str(Path(self.mesh).parent))
        browse = QPushButton("…")
        browse.clicked.connect(self._browse_folder)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        out.addRow("Output Folder", row)
        self.name = QLineEdit(saved.get("name", bake.NAME_TEMPLATE))
        out.addRow("Output Name", self.name)
        self.format = QComboBox()
        self.format.addItems(bake.FORMATS)
        self.format.setCurrentText(self.settings.format)
        out.addRow("Output Format", self.format)
        sizes = QHBoxLayout()
        self.width_, self.height_ = QComboBox(), QComboBox()
        for c, v in ((self.width_, self.settings.size[0]), (self.height_, self.settings.size[1])):
            c.addItems(SIZES)
            c.setCurrentText(str(v))
            sizes.addWidget(c)
        out.addRow("Output Size", sizes)
        self.link = QCheckBox("Add the maps to the package as linked bitmaps")
        self.link.setChecked(saved.get("link", True))
        out.addRow("", self.link)
        lay.addLayout(out)
        for w in (self.folder, self.name):
            w.textChanged.connect(self._changed)
        for w in (self.format, self.width_, self.height_):
            w.currentTextChanged.connect(self._changed)
        self.link.toggled.connect(self._changed)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.start = QPushButton("Start Render")
        self.start.setDefault(True)
        self.start.clicked.connect(self.render)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.start)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self._cancel = False
        self._rebuild_list(0)

    # --- the list ---------------------------------------------------------------------------------------

    def _rebuild_list(self, row):
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItem(QListWidgetItem(MESH_SETUP))
        for b in self.bakers:
            it = QListWidgetItem(bake.BAKERS[b["id"]])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if b.get("enabled", True) else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.list.setCurrentRow(min(row, self.list.count() - 1))
        self._show(self.list.currentRow())

    def _add(self, bid):
        self.bakers.append({"id": bid, "enabled": True, "params": bake.baker_defaults(bid)})
        self._rebuild_list(len(self.bakers))
        self._changed()

    def _remove(self):
        row = self.list.currentRow()
        if row >= 1:
            del self.bakers[row - 1]
            self._rebuild_list(row - 1)
            self._changed()

    def _toggled(self, item):
        row = self.list.row(item)
        if row >= 1:
            self.bakers[row - 1]["enabled"] = item.checkState() == Qt.Checked
            self._changed()

    def _changed(self, *_):
        self._live.start()

    # --- parameters ---------------------------------------------------------------------------------------

    def _show(self, row):
        page = QWidget()
        form = QFormLayout(page)
        if row <= 0:
            for name, label, editor, cond in COMMON:
                if cond and getattr(self.settings, cond[0]) != cond[1]:
                    continue  # applies only when another setting says so (as the page describes)

                def store(v, n=name):
                    setattr(self.settings, n, v)
                    if n in CONDITIONS:
                        QTimer.singleShot(0, lambda: self._show(self.list.currentRow()))
                w = editor_widget(getattr(self.settings, name), editor, lambda v, s=store: (s(v), self._changed()),
                                  self._pick)
                if name in TIPS:
                    w.setToolTip(TIPS[name])
                form.addRow(label, w)
                if name in TIPS:
                    form.labelForField(w).setToolTip(TIPS[name])
        else:
            b = self.bakers[row - 1]
            spec = bake.BAKER_PARAMS.get(b["id"], {})
            b["params"] = bake.resolved(b["id"], b["params"])

            def store(n, v, b=b):
                b["params"][n] = v
                self._changed()
            baker_rows(form, b["id"], b["params"], list(spec), store,
                       lambda: self._show(self.list.currentRow()), self._pick)
            if not spec:
                form.addRow("This baker has no specific parameters.", QWidget())
        self.scroll.setWidget(page)

    def _pick(self, line, many):
        from trafilatura3d.meshio import MESH_FILTER
        start = str(Path(self.mesh).parent)
        if many:
            names, _ = QFileDialog.getOpenFileNames(self, "High Definition Meshes", start, MESH_FILTER)
            if names:
                line.setText("; ".join(names))
        else:
            name, _ = QFileDialog.getOpenFileName(self, "File", start)
            if name:
                line.setText(name)

    def _browse_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Output Folder", self.folder.text())
        if name:
            self.folder.setText(name)

    # --- render ---------------------------------------------------------------------------------------------

    def configuration(self) -> dict:
        self.settings.format = self.format.currentText()
        self.settings.size = (int(self.width_.currentText()), int(self.height_.currentText()))
        s = dataclasses.asdict(self.settings)
        s["size"] = list(s["size"])
        return {"settings": s, "bakers": copy.deepcopy(self.bakers), "folder": self.folder.text(),
                "name": self.name.text(), "link": self.link.isChecked()}

    def closeEvent(self, event):
        self._live.stop()
        self._save()
        super().closeEvent(event)

    def _save(self):
        """The configuration into the resource: consecutive edits merge into one undo step."""
        config = self.configuration()
        if config != (self.pkg.resources[self.rid].baking or {}):
            self.doc.edit(self.pkg, "Baking settings", lambda: setattr(self.pkg.resources[self.rid], "baking", config),
                          merge=f"baking:{self.rid}")

    def render(self) -> list[Path]:
        from PySide6.QtWidgets import QApplication
        config = self.configuration()
        bakers = {b["id"]: b["params"] for b in self.bakers if b.get("enabled", True)}
        if not bakers:
            QMessageBox.information(self, "Baking", "Add a baker first.")
            return []
        self._save()
        self.start.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(bakers))

        def progress(done, total, baker):
            self.progress.setValue(done)
            self.progress.setFormat(f"{bake.BAKERS[baker]}  %v/%m" if baker else "%v/%m")
            QApplication.processEvents()
            return not self._cancel

        try:
            paths = bake.Baker().bake(self.mesh, bakers, self.settings, config["folder"], config["name"], progress)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Baking", str(e))
            return []
        finally:
            self.start.setEnabled(True)
        if config["link"] and self.pkg.path is not None:
            def link():
                for p in paths:
                    if p.suffix != ".svg":
                        # Both resolved: a folder reached through a symlink (/var on macOS) stays short.
                        rel = os.path.relpath(p.resolve(), self.pkg.path.resolve().parent).replace(os.sep, "/")
                        self.pkg.resources[p.stem] = Resource("bitmap", rel)
            self.doc.edit(self.pkg, "Bake model information", link)
        self.doc.changed.emit(self.pkg)  # new files under the same names: the 3D View reloads the maps
        return paths
