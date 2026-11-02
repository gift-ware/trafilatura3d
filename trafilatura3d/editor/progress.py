"""The progress window of the long steps that keep the editor waiting: a bake (the maps tile by tile, the one being
baked shown on the model in the 3D View) and an export (compiling, cooking node by node, writing file by file).
Modal: the editor waits, its event loop runs between the steps; Cancel stops at the next one. The Baking window, the
Export outputs dialog and the plugins (EditorAccess.bake / export) go through here."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog

from trafilatura3d import bake

PREVIEW_INTERVAL = 0.25  # seconds between two maps shown in the 3D View: an upload a tile would slow the bake down


def _window(parent, title: str, text: str, maximum: int) -> QProgressDialog:
    dlg = QProgressDialog(text, "Cancel", 0, maximum, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.ApplicationModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoReset(False)  # a step's bar ending full keeps the window to the end
    dlg.setAutoClose(False)
    dlg.setValue(0)
    QApplication.processEvents()
    return dlg


def _closed(dlg: QProgressDialog) -> bool:
    """Whether it was canceled; then hidden (not closed: its closeEvent cancels)."""
    canceled = dlg.wasCanceled()
    dlg.hide()
    dlg.deleteLater()
    return canceled


def bake_maps(parent, document, package, resource_id: str, low_path, bakers: dict, settings: bake.Settings, out_dir,
              template: str = bake.NAME_TEMPLATE) -> tuple[list[Path], bool]:
    """`bake.Baker().bake` in the progress window, the map being baked shown in the 3D View when it shows the package's
    scene resource `resource_id` (Document.bake_progress): (the maps written, whether it was canceled)."""
    dlg = _window(parent, "Baking", "Preparing the meshes…", 1000)
    order, last = list(bakers), [0.0]

    def progress(done, total, baker):
        dlg.setValue(int(done / total * 1000))
        if baker:
            dlg.setLabelText(bake.BAKERS[baker])
        QApplication.processEvents()
        return not dlg.wasCanceled()

    def each_tile(baker, udim, done, total, image):
        dlg.setValue(int((order.index(baker) + done / total) / len(order) * 1000))
        where = f"  UDIM {bake.udim(udim)}" if tuple(udim) != (0, 0) else ""
        dlg.setLabelText(f"{bake.BAKERS[baker]}{where}  tile {done}/{total}")
        now = time.monotonic()
        if now - last[0] > PREVIEW_INTERVAL or done == total:
            last[0] = now
            document.bake_progress.emit(package, resource_id, (baker, udim, image))
        QApplication.processEvents()
        return not dlg.wasCanceled()

    try:
        paths = bake.Baker().bake(str(low_path), bakers, settings, out_dir, template, progress, each_tile)
    finally:
        canceled = _closed(dlg)
        document.bake_progress.emit(package, resource_id, None)
    return paths, canceled


def export_outputs(parent, run) -> list[Path] | None:
    """`run(progress)` (an `export.export` given that progress) in the progress window: its files, None when canceled."""
    dlg = _window(parent, "Export outputs", "Compiling the graph", 1)

    def progress(text, done, total):
        dlg.setMaximum(total)
        dlg.setValue(done)
        dlg.setLabelText(text)
        QApplication.processEvents()
        return not dlg.wasCanceled()

    try:
        written = run(progress)
    finally:
        canceled = _closed(dlg)
    return None if canceled else written
