"""The New Project window (Project creation page): the template (shaders.TEMPLATES, ours: a shader with its blending
mode, whose channels the Texture Sets get), the mesh file, the resolution, the normal map format, the UV
Tile Settings (UDIMs) and the baked maps to import. Not offered (not supported yet): Compute Tangent
Space per Fragment (the viewport always computes the bitangent per fragment), the legacy UV Tile mode (UV Tiles
into Texture Sets), Import Cameras, Auto-unwrap, the USD options and Physical size."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from trafilatura3d import shaders
from trafilatura3d.meshio import MESH_FILTER
from trafilatura3d.paint.model import NORMAL_FORMATS, SIZES, UV_TILE_MODES
from trafilatura3d.paint.session import IMAGE_FILTER


class NewProjectDialog(QDialog):
    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.resize(560, 420)
        self.start_dir = start_dir
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.template = QComboBox()
        self.template.addItems(list(shaders.TEMPLATES))
        self.template.setCurrentText(shaders.DEFAULT_TEMPLATE)
        form.addRow("Template", self.template)
        row = QHBoxLayout()
        self.file = QLineEdit()
        self.file.setPlaceholderText("Select a mesh file")
        pick = QPushButton("Select...")
        pick.clicked.connect(self._pick)
        row.addWidget(self.file, 1)
        row.addWidget(pick)
        form.addRow("File", row)
        self.resolution = QComboBox()
        for s in SIZES:
            if s >= 128:
                self.resolution.addItem(str(s), s)
        self.resolution.setCurrentText("2048")
        form.addRow("Resolution", self.resolution)
        self.normal = QComboBox()
        self.normal.addItems(NORMAL_FORMATS)
        self.normal.setCurrentText("OpenGL (X+, Y+, Z+)")
        form.addRow("Normal Map Format", self.normal)
        lay.addLayout(form)
        self.uv_tiles = UVTileSettings()
        lay.addWidget(self.uv_tiles)
        lay.addWidget(QLabel("Import baked maps"))
        maps_row = QHBoxLayout()
        self.maps = QListWidget()
        maps_row.addWidget(self.maps, 1)
        side = QVBoxLayout()
        add, remove = QPushButton("Add"), QPushButton("Remove")
        add.clicked.connect(self._add_maps)
        remove.clicked.connect(lambda: [self.maps.takeItem(self.maps.row(i)) for i in self.maps.selectedItems()])
        side.addWidget(add)
        side.addWidget(remove)
        side.addStretch(1)
        maps_row.addLayout(side)
        lay.addLayout(maps_row, 1)
        hint = QLabel("Baked map files: <texture set>_<map>, for example DefaultMaterial_ambient_occlusion.png.")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("OK")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        self.ok.setEnabled(False)
        self.file.textChanged.connect(lambda t: self.ok.setEnabled(Path(t).is_file()))
        lay.addWidget(buttons)

    def _pick(self):
        name, _ = QFileDialog.getOpenFileName(self, "Select a mesh", self.start_dir, MESH_FILTER)
        if name:
            self.file.setText(name)

    def _add_maps(self):
        names, _ = QFileDialog.getOpenFileNames(self, "Import baked maps", self.start_dir, IMAGE_FILTER)
        for n in names:
            self.maps.addItem(n)

    def _accept(self):
        if Path(self.file.text()).is_file():
            self.accept()

    def values(self) -> dict:
        return {"template": self.template.currentText(), "mesh": self.file.text(),
                "resolution": self.resolution.currentData(),
                "normal_format": self.normal.currentText(), "uv_tiles": self.uv_tiles.mode(),
                "baked_maps": [self.maps.item(i).text() for i in range(self.maps.count())]}


class UVTileSettings(QWidget):
    """UV Tile Settings (UDIMs): Use UV Tile workflow and the UV Tile mode, set at the project's creation (New
    Project), shown without editing afterwards (Project configuration > UV Tiles settings)."""

    def __init__(self, mode: str | None = None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("UV Tile Settings (UDIMs)")
        title.setObjectName("dim")
        lay.addWidget(title)
        self.use = QCheckBox("Use UV Tile workflow")
        self.use.setToolTip("Paint across several UV tiles (UDIM) beyond the 0-1 range. "
                            "Fixed once the project is created.")
        lay.addWidget(self.use)
        self.group = QButtonGroup(self)
        self.radios = []
        for i, text in enumerate(UV_TILE_MODES):
            r = QRadioButton(text)
            self.group.addButton(r, i)
            self.radios.append(r)
            lay.addWidget(r)
        self.radios[0].setChecked(True)
        self.radios[1].setToolTip("Not offered yet")
        self.use.toggled.connect(self._sync)
        if mode is not None:  # an existing project's settings: view only
            self.use.setChecked(bool(mode))
            if mode:
                self.radios[UV_TILE_MODES.index(mode)].setChecked(True)
            self.use.setEnabled(False)
        self._sync()

    def _sync(self):
        editable = self.use.isEnabled()
        self.radios[0].setEnabled(editable and self.use.isChecked())
        self.radios[1].setEnabled(False)

    def mode(self) -> str:
        return UV_TILE_MODES[self.group.checkedId()] if self.use.isChecked() else ""


class ProjectConfiguration(QDialog):
    """Edit > Project configuration (Project configuration page): the 3D mesh (Select… checks Reimport mesh when the
    file differs), Reimport mesh, Normal Map Format. Not offered: Compute Tangent Space Per Fragment, the USD
    options, UV Tiles settings, Import Cameras, Preserve strokes positions on mesh (painting stays in texture
    space), Auto-Unwrap, unit scale, Physical size switch, color management. UV Tiles settings are shown, not
    editable (set at the project's creation)."""

    def __init__(self, workspace):
        super().__init__(workspace.main)
        self.ws = workspace
        s = workspace.doc.session
        self.setWindowTitle("Project configuration")
        self.resize(560, 300)
        form = QFormLayout(self)
        row = QHBoxLayout()
        self.file = QLineEdit(s.source)
        pick = QPushButton("Select...")
        pick.clicked.connect(self._pick)
        row.addWidget(self.file, 1)
        row.addWidget(pick)
        form.addRow("3D mesh", row)
        self.reimport = QCheckBox("Reimport mesh")
        form.addRow("", self.reimport)
        self.normal = QComboBox()
        self.normal.addItems(NORMAL_FORMATS)
        self.normal.setCurrentText(workspace.doc.project.normal_format)
        form.addRow("Normal Map Format", self.normal)
        form.addRow("UV Tiles settings", UVTileSettings(workspace.doc.project.uv_tiles))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _pick(self):
        name, _ = QFileDialog.getOpenFileName(self, "Select a mesh", str(Path(self.file.text()).parent), MESH_FILTER)
        if name:
            self.reimport.setChecked(name != self.ws.doc.session.source)
            self.file.setText(name)

    def apply(self) -> bool:
        ws, doc = self.ws, self.ws.doc
        if self.normal.currentText() != doc.project.normal_format:
            def do():
                doc.project.normal_format = self.normal.currentText()
            doc.edit("Normal Map Format", do)
            ws.log("Normal Map Format changed: bake the mesh maps again (Baking mode).")
        if self.reimport.isChecked():
            from trafilatura3d.paint.model import ProjectError
            try:
                added = doc.session.reimport(self.file.text())
            except (ProjectError, OSError, ValueError) as e:
                QMessageBox.warning(self, "Project configuration", f"Reimport mesh: {e}")
                return False
            doc.undo.clear()  # the history's snapshots belong to the old mesh
            doc.dirty = True
            ws.mesh_changed()
            ws.log(f"Mesh reimported from {self.file.text()}" + (f"; new Texture Sets: {', '.join(added)}" if added else ""))
        self.accept()
        return True
