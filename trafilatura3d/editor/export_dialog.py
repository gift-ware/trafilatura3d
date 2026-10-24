"""The Export outputs dialog: Destination, Format (one for all the
files), Pattern of the file names with a sample, the Outputs and Output Groups to export (with buttons that
toggle them all), Automatic Export (export again after each change of the graph), Export; the Batch tab
overrides the Output size. Its settings are kept per graph in the user's settings. Not offered: the Batch
tab's UV Tiles (graphs have no UV tile variation yet)."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

import sys

import trafilatura3d.export  # noqa: F401 - the module; `trafilatura3d.export` as an attribute is the package's function
from trafilatura3d.package import DocumentError

exporting = sys.modules["trafilatura3d.export"]

SIZES = [(f"{1 << k}", k) for k in range(5, 14)]


def settings_key(package, graph_id) -> str:
    return f"export/{package.path or id(package)}#{graph_id}"


def load_settings(package, graph_id) -> dict:
    from trafilatura3d.editor.preferences import settings, suggested_dir
    raw = settings().value(settings_key(package, graph_id))
    data = json.loads(raw) if raw else {}
    return {"destination": data.get("destination") or str(suggested_dir(package.path)),
            "format": data.get("format") or exporting.default_format(),
            "pattern": data.get("pattern") or exporting.PATTERN,
            "disabled": data.get("disabled", []), "automatic": data.get("automatic", False),
            "batch": data.get("batch", False), "size": data.get("size", list(package.graphs[graph_id].parent_size))}


def save_settings(package, graph_id, data: dict) -> None:
    from trafilatura3d.editor.preferences import settings
    settings().setValue(settings_key(package, graph_id), json.dumps(data))


def run_export(document, package, graph_id, data: dict) -> list:
    """Export with a dialog's settings (also what Automatic Export does)."""
    outputs = {a["identifier"] for a in exporting.output_attributes(package, graph_id)} - set(data["disabled"])
    return exporting.export(package, graph_id, data["destination"], document.ev, format=data["format"],
                            pattern=data["pattern"], outputs=outputs,
                            size=tuple(data["size"]) if data["batch"] else None)


class ExportDialog(QDialog):
    def __init__(self, document, package, graph_id, parent=None, on_automatic=None):
        super().__init__(parent)
        self.doc, self.pkg, self.gid = document, package, graph_id
        self.on_automatic = on_automatic  # callable(package, graph id, settings or None)
        self.data = load_settings(package, graph_id)
        self.setWindowTitle(f"Export outputs - {graph_id}")
        self.resize(560, 560)
        lay = QVBoxLayout(self)
        form = QFormLayout()  # Destination, Format, Pattern: the same for both tabs
        lay.addLayout(form)
        row = QHBoxLayout()
        self.destination = QLineEdit(self.data["destination"])
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.destination, 1)
        row.addWidget(browse)
        form.addRow("Destination", row)
        self.format = QComboBox()
        self.format.addItems(exporting.FORMATS)
        self.format.setCurrentText(self.data["format"])
        form.addRow("Format", self.format)
        self.pattern = QLineEdit(self.data["pattern"])
        self.pattern.setToolTip("Keywords: " + " ".join(f"$({v})" for v in exporting.VARIABLES))
        form.addRow("Pattern", self.pattern)
        self.sample = QLabel()
        self.sample.setObjectName("dim")
        form.addRow("", self.sample)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        graph_page = QWidget()  # From Graph: the Outputs and Output Groups to export
        glay = QVBoxLayout(graph_page)
        glay.addWidget(QLabel("Outputs"))
        self.outputs = QTreeWidget()
        self.outputs.setHeaderHidden(True)
        glay.addWidget(self.outputs, 1)
        toggles = QHBoxLayout()
        for text, on in (("All on", True), ("All off", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, on=on: self._toggle_all(on))
            toggles.addWidget(b)
        toggles.addStretch(1)
        glay.addLayout(toggles)
        self.tabs.addTab(graph_page, "From Graph")

        batch_page = QWidget()
        bform = QFormLayout(batch_page)
        sizes = QHBoxLayout()
        self.width_, self.height_ = QComboBox(), QComboBox()
        for combo, value in ((self.width_, self.data["size"][0]), (self.height_, self.data["size"][1])):
            for label, k in SIZES:
                combo.addItem(label, k)
            combo.setCurrentIndex(max(0, [k for _, k in SIZES].index(value) if value in [k for _, k in SIZES] else 0))
            sizes.addWidget(combo)
        bform.addRow("Output size", sizes)
        note = QLabel("Exports the outputs chosen in From Graph at this size instead of the working (parent) "
                      "resolution.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        bform.addRow("", note)
        self.tabs.addTab(batch_page, "Batch")
        self.tabs.setCurrentIndex(1 if self.data["batch"] else 0)

        self.automatic = QCheckBox("Automatic Export")
        self.automatic.setToolTip("Export this graph's outputs again after each change (this graph only; can be "
                                  "slow with large outputs)")
        self.automatic.setChecked(self.data["automatic"])
        lay.addWidget(self.automatic)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.export_button = QPushButton("Export")
        self.export_button.setDefault(True)
        self.export_button.clicked.connect(self.export)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.export_button)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self._fill_outputs()
        self.pattern.textChanged.connect(self._update_sample)
        self.format.currentTextChanged.connect(self._update_sample)
        self._update_sample()

    def _fill_outputs(self):
        """Outputs grouped by their Group attribute; a group's check toggles its outputs."""
        self.outputs.clear()
        groups = {}
        for a in exporting.output_attributes(self.pkg, self.gid):
            parent = self.outputs.invisibleRootItem()
            if a["group"]:
                if a["group"] not in groups:
                    g = QTreeWidgetItem([a["group"]])
                    g.setFlags(g.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
                    self.outputs.addTopLevelItem(g)
                    groups[a["group"]] = g
                parent = groups[a["group"]]
            it = QTreeWidgetItem([a["label"] or a["identifier"]])
            it.setData(0, Qt.UserRole, a["identifier"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Unchecked if a["identifier"] in self.data["disabled"] else Qt.Checked)
            parent.addChild(it)
        self.outputs.expandAll()

    def _items(self):
        out = []
        def walk(item):
            for i in range(item.childCount()):
                child = item.child(i)
                if child.data(0, Qt.UserRole):
                    out.append(child)
                walk(child)
        walk(self.outputs.invisibleRootItem())
        return out

    def _toggle_all(self, on):
        for it in self._items():
            it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)

    def _update_sample(self):
        first = next(iter(exporting.output_attributes(self.pkg, self.gid)), None)
        if first is None:
            self.sample.setText("The graph has no outputs")
            return
        self.sample.setText(f"{exporting.file_name(self.pattern.text(), self.gid, first)}.{self.format.currentText()}")

    def _browse(self):
        name = QFileDialog.getExistingDirectory(self, "Destination", self.destination.text())
        if name:
            self.destination.setText(name)

    def settings(self) -> dict:
        return {"destination": self.destination.text(), "format": self.format.currentText(),
                "pattern": self.pattern.text(),
                "disabled": [it.data(0, Qt.UserRole) for it in self._items() if it.checkState(0) != Qt.Checked],
                "automatic": self.automatic.isChecked(), "batch": self.tabs.currentIndex() == 1,
                "size": [self.width_.currentData(), self.height_.currentData()]}

    def _store(self):
        data = self.settings()
        save_settings(self.pkg, self.gid, data)
        if self.on_automatic is not None:
            self.on_automatic(self.pkg, self.gid, data if data["automatic"] else None)
        return data

    def closeEvent(self, event):
        self._store()
        super().closeEvent(event)

    def export(self) -> list:
        data = self._store()
        try:
            written = run_export(self.doc, self.pkg, self.gid, data)
        except (DocumentError, OSError, ValueError) as e:
            QMessageBox.warning(self, "Export outputs", str(e))
            return []
        self.written = written
        QTimer.singleShot(0, self, self.accept)
        return written
