"""The Export textures window (File > Export textures, Ctrl+Shift+E): Export settings, Output templates and
List of exports tabs; buttons Reset all, Cancel, Save settings, Export (Export window page).

Output templates: the predefined templates (not editable) and the user's (saved in the user folder,
export_templates.json): a template's output maps have a name with keywords, a type (Gray, RGB, R+G+B, RGB+A,
R+G+B+A), a format and bit depth, and per slot an input (a channel, a mesh map or a converted map) and its
component. The application assigns inputs by drag and drop; here each slot has a menu.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from trafilatura3d.paint.channels import CHANNELS
from trafilatura3d.paint.export import (COMPONENTS, CONVERTED, FILE_TYPES, MAP_TYPES, PADDING, PREDEFINED, SIZES,
                                    ExportSettings, Exporter, OutputMap, Template, library_templates, load_templates,
                                    save_templates)
from trafilatura3d.paint.model import MESH_MAPS


def templates_path() -> Path:
    from trafilatura3d.project import user_dir
    return user_dir() / "export_templates.json"


def inputs() -> list[tuple[str, str]]:
    out = [(f"channel:{c}", f"Channel: {CHANNELS[c].label}") for c in CHANNELS]
    out += [(f"mesh:{slot}", f"Mesh map: {label}") for slot, label, _ in MESH_MAPS]
    out += [(f"converted:{c}", f"Converted: {c}") for c in CONVERTED]
    return out


class ExportDialog(QDialog):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.ws = workspace
        self.setWindowTitle("Export textures")
        self.resize(900, 600)
        doc = workspace.doc
        saved = doc.project.export or {}
        self.settings = ExportSettings(**{k: v for k, v in saved.get("settings", {}).items()
                                          if k in ExportSettings.__dataclass_fields__})
        if not self.settings.output_dir:
            base = doc.session.path.parent if doc.session.path else Path.home()
            self.settings.output_dir = str(base / "export")
        self.templates = PREDEFINED + library_templates() + load_templates(templates_path())
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._settings_tab(), "Export settings")
        self.tabs.addTab(self._templates_tab(), "Output templates")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "List of exports")
        lay.addWidget(self.tabs, 1)
        buttons = QHBoxLayout()
        reset = QPushButton("Reset all")
        reset.clicked.connect(self._reset)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        cancel, save, export = QPushButton("Cancel"), QPushButton("Save settings"), QPushButton("Export")
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self._save_settings)
        export.clicked.connect(self._export)
        export.setDefault(True)
        for b in (cancel, save, export):
            buttons.addWidget(b)
        lay.addLayout(buttons)
        self.tabs.currentChanged.connect(lambda i: reset.setVisible(i == 0))

    # --- Export settings ------------------------------------------------------------------------------

    def _settings_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        self.sets = QTreeWidget()
        self.sets.setHeaderLabels(["Texture Sets"])
        for ts in self.ws.doc.project.texture_sets:
            it = QTreeWidgetItem([ts.name])
            it.setCheckState(0, Qt.Checked if not self.settings.texture_sets or ts.name in self.settings.texture_sets
                             else Qt.Unchecked)
            self.sets.addTopLevelItem(it)
        left = QVBoxLayout()
        left.addWidget(self.sets, 1)
        row = QHBoxLayout()
        for label, state in (("Check all", True), ("Uncheck all", False), ("Invert all", None)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=state: self._check(s))
            row.addWidget(b)
        left.addLayout(row)
        lay.addLayout(left, 1)
        form = QFormLayout()
        dir_row = QHBoxLayout()
        self.out_dir = QLineEdit(self.settings.output_dir)
        pick = QPushButton("…")
        pick.clicked.connect(self._pick_dir)
        dir_row.addWidget(self.out_dir, 1)
        dir_row.addWidget(pick)
        form.addRow("Output directory", dir_row)
        self.template = QComboBox()
        self._fill_templates()
        form.addRow("Output template", self.template)
        self.file_type = QComboBox()
        self.file_type.addItems(["Based on output template", *FILE_TYPES])
        self.file_type.setCurrentText(self.settings.file_type)
        self.depth = QComboBox()
        self.file_type.currentTextChanged.connect(self._fill_depths)
        self._fill_depths(self.file_type.currentText())
        if self.depth.findText(self.settings.depth) >= 0:
            self.depth.setCurrentText(self.settings.depth)
        type_row = QHBoxLayout()
        type_row.addWidget(self.file_type, 1)
        type_row.addWidget(self.depth)
        form.addRow("File type", type_row)
        self.size = QComboBox()
        self.size.addItems(SIZES)
        self.size.setCurrentText(self.settings.size)
        form.addRow("Size", self.size)
        self.padding = QComboBox()
        self.padding.addItems(PADDING)
        self.padding.setCurrentText(self.settings.padding)
        form.addRow("Padding", self.padding)
        self.dilation = QSpinBox()
        self.dilation.setRange(0, 256)
        self.dilation.setValue(self.settings.dilation)
        self.dilation.setSuffix(" px")
        form.addRow("Dilation distance", self.dilation)
        self.padding.currentTextChanged.connect(lambda t: self.dilation.setEnabled(t.startswith("Dilation +")))
        self.dilation.setEnabled(self.settings.padding.startswith("Dilation +"))
        right = QWidget()
        right.setLayout(form)
        lay.addWidget(right, 2)
        return w

    def _fill_templates(self):
        current = self.template.currentText() or self.settings.template
        self.template.clear()
        for t in self.templates:
            self.template.addItem(t.name)
        if self.template.findText(current) >= 0:
            self.template.setCurrentText(current)

    def _fill_depths(self, ftype):
        self.depth.clear()
        if ftype in FILE_TYPES:
            self.depth.addItems(FILE_TYPES[ftype][1])
        self.depth.setEnabled(ftype in FILE_TYPES)

    def _check(self, state):
        for i in range(self.sets.topLevelItemCount()):
            it = self.sets.topLevelItem(i)
            on = (it.checkState(0) != Qt.Checked) if state is None else state
            it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output directory", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _collect(self) -> ExportSettings:
        checked = [self.sets.topLevelItem(i).text(0) for i in range(self.sets.topLevelItemCount())
                   if self.sets.topLevelItem(i).checkState(0) == Qt.Checked]
        return ExportSettings(output_dir=self.out_dir.text(), template=self.template.currentText(),
                              file_type=self.file_type.currentText(), depth=self.depth.currentText() or "8",
                              size=self.size.currentText(), padding=self.padding.currentText(),
                              dilation=self.dilation.value(), texture_sets=checked)

    def _reset(self):
        self.settings = ExportSettings(output_dir=self.out_dir.text())
        self.file_type.setCurrentText(self.settings.file_type)
        self.depth.setCurrentText(self.settings.depth)
        self.size.setCurrentText(self.settings.size)
        self.padding.setCurrentText(self.settings.padding)
        self.dilation.setValue(self.settings.dilation)
        self.template.setCurrentText(self.settings.template)
        self._check(True)

    def _save_settings(self):
        s = self._collect()
        self.ws.doc.edit("Export settings", lambda: self.ws.doc.project.export.__setitem__("settings", asdict(s)),
                         structure=False)
        save_templates(templates_path(), self.templates)

    def _export(self):
        s = self._collect()
        template = next(t for t in self.templates if t.name == s.template)
        if not s.texture_sets:
            self.log.appendPlainText("No Texture Set checked.")
            self.tabs.setCurrentWidget(self.log)
            return
        self.tabs.setCurrentWidget(self.log)
        save_templates(templates_path(), self.templates)
        self.log.appendPlainText(f"Export to {s.output_dir}")

        def log(text, error):
            self.log.appendPlainText(("ERROR  " if error else "  ") + text)
            self.log.repaint()
        try:
            written = Exporter(self.ws.doc.session).run(template, s, log)
        except Exception as e:  # noqa: BLE001 - reported in the list
            log(str(e), True)
            return
        self.log.appendPlainText(f"{len(written)} files exported.")
        self.ws.log(f"Exported {len(written)} textures to {s.output_dir}")

    # --- Output templates -----------------------------------------------------------------------------

    def _templates_tab(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        left = QVBoxLayout()
        self.tlist = QListWidget()
        self.tlist.currentRowChanged.connect(self._template_selected)
        self.tlist.itemDoubleClicked.connect(self._rename_template)
        left.addWidget(self.tlist, 1)
        row = QHBoxLayout()
        for label, slot in (("Add", self._add_template), ("Duplicate", self._duplicate_template),
                            ("Remove", self._remove_template)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        left.addLayout(row)
        lay.addLayout(left, 1)
        mid = QVBoxLayout()
        mid.addWidget(QLabel("Output maps"))
        self.maps = QListWidget()
        self.maps.currentRowChanged.connect(self._map_selected)
        mid.addWidget(self.maps, 1)
        mrow = QHBoxLayout()
        for t in MAP_TYPES:
            b = QPushButton(t)
            b.clicked.connect(lambda _=False, t=t: self._add_map(t))
            mrow.addWidget(b)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove_map)
        mrow.addWidget(rm)
        mid.addLayout(mrow)
        lay.addLayout(mid, 2)
        self.map_box = QWidget()  # the selected output map's fields
        self.map_box_layout = QVBoxLayout(self.map_box)
        self.map_box_layout.setContentsMargins(0, 0, 0, 0)
        self.map_form = None
        lay.addWidget(self.map_box, 2)
        self._fill_template_list()
        return w

    def _fill_template_list(self, select=0):
        self.tlist.blockSignals(True)
        self.tlist.clear()
        for t in self.templates:
            it = QListWidgetItem(t.name + ("  (predefined)" if t.predefined else ""))
            self.tlist.addItem(it)
        self.tlist.blockSignals(False)
        self.tlist.setCurrentRow(min(select, len(self.templates) - 1))
        if hasattr(self, "template"):
            self._fill_templates()

    def _current_template(self) -> Template | None:
        i = self.tlist.currentRow()
        return self.templates[i] if 0 <= i < len(self.templates) else None

    def _template_selected(self, _i):
        t = self._current_template()
        self.maps.clear()
        if t is None:
            return
        if t.predefined:
            self.maps.addItem("(generated per Texture Set: not editable, duplicate it to edit)")
        else:
            for m in t.maps:
                self.maps.addItem(f"{m.name}   [{m.type}]")
        self.maps.setCurrentRow(0)

    def _add_template(self):
        name, ok = QInputDialog.getText(self, "New template", "Name")
        if ok and name.strip() and all(t.name != name.strip() for t in self.templates):
            self.templates.append(Template(name.strip()))
            self._fill_template_list(len(self.templates) - 1)

    def _duplicate_template(self):
        t = self._current_template()
        if t is None:
            return
        dup = Template(f"{t.name} copy", copy.deepcopy(t.maps))
        if t.predefined:  # a predefined template becomes its maps for the current Texture Set
            from trafilatura3d.paint.export import expand
            dup.maps = expand(t, self.ws.doc.ts)
            for m in dup.maps:
                m.name = m.name.replace(self.ws.doc.ts.name, "$textureSet")
        self.templates.append(dup)
        self._fill_template_list(len(self.templates) - 1)

    def _remove_template(self):
        t = self._current_template()
        if t is not None and not t.predefined:
            self.templates.remove(t)
            self._fill_template_list()

    def _rename_template(self, _it):
        t = self._current_template()
        if t is None or t.predefined:
            return
        name, ok = QInputDialog.getText(self, "Rename", "Name", text=t.name)
        if ok and name.strip():
            t.name = name.strip()
            self._fill_template_list(self.tlist.currentRow())

    def _add_map(self, kind):
        t = self._current_template()
        if t is None or t.predefined:
            return
        t.maps.append(OutputMap("$mesh_$textureSet_map", kind))
        self._template_selected(0)
        self.maps.setCurrentRow(len(t.maps) - 1)

    def _remove_map(self):
        t = self._current_template()
        i = self.maps.currentRow()
        if t is not None and not t.predefined and 0 <= i < len(t.maps):
            t.maps.pop(i)
            self._template_selected(0)

    def _map_selected(self, i):
        t = self._current_template()
        if self.map_form is not None:
            self.map_form.deleteLater()
        self.map_form = QWidget()
        self.map_box_layout.addWidget(self.map_form)
        if t is None or t.predefined or not 0 <= i < len(t.maps):
            return
        m = t.maps[i]
        form = QFormLayout(self.map_form)
        name = QLineEdit(m.name)
        name.setToolTip("Keywords: $project $mesh $textureSet $udim $colorSpace; / makes folders")
        name.editingFinished.connect(lambda: (setattr(m, "name", name.text()), self._refresh_map_labels()))
        form.addRow("Name", name)
        ftype = QComboBox()
        ftype.addItems(list(FILE_TYPES))
        ftype.setCurrentText(m.file_type)
        depth = QComboBox()

        def fill_depth(t, keep=None):
            depth.clear()
            depth.addItems(FILE_TYPES[t][1])
            if keep and depth.findText(keep) >= 0:
                depth.setCurrentText(keep)
        fill_depth(m.file_type, m.depth)
        ftype.currentTextChanged.connect(lambda t: (setattr(m, "file_type", t), fill_depth(t)))
        depth.currentTextChanged.connect(lambda d: d and setattr(m, "depth", d))
        form.addRow("Format", ftype)
        form.addRow("Bit depth", depth)
        for slot in MAP_TYPES[m.type]:
            src = QComboBox()
            src.addItem("(empty)", "")
            for key, label in inputs():
                src.addItem(label, key)
            spec = m.slots.setdefault(slot, {"input": "", "component": "RGB" if slot == "RGB" else "R"})
            src.setCurrentIndex(max(0, src.findData(spec.get("input", ""))))
            comp = QComboBox()
            comp.addItems(COMPONENTS)
            comp.setCurrentText(spec.get("component", "RGB"))
            src.currentIndexChanged.connect(lambda _i, s=spec, c=src: s.__setitem__("input", c.currentData()))
            comp.currentTextChanged.connect(lambda v, s=spec: s.__setitem__("component", v))
            row = QHBoxLayout()
            row.addWidget(src, 2)
            row.addWidget(comp, 1)
            form.addRow({"L": "Gray"}.get(slot, slot), row)

    def _refresh_map_labels(self):
        t = self._current_template()
        if t is None or t.predefined:
            return
        for i, m in enumerate(t.maps):
            if i < self.maps.count():
                self.maps.item(i).setText(f"{m.name}   [{m.type}]")
