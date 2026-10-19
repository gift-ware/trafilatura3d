"""The Export outputs dialog: settings kept per graph, outputs toggled, the
Batch tab's Output size, Automatic Export after a change."""

import os
import tempfile
import unittest
from pathlib import Path

from tests.scenes import material_package


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class ExportDialogTest(unittest.TestCase):
    def test_dialog(self):
        from PySide6.QtCore import QCoreApplication, Qt
        from PySide6.QtWidgets import QApplication
        import OpenImageIO as oiio
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        try:
            from trafilatura3d.editor.export_dialog import ExportDialog, load_settings
            from trafilatura3d.editor.mainwindow import MainWindow
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])
            p = w.doc.packages[0]
            gid = next(iter(p.graphs))
            with tempfile.TemporaryDirectory() as out:
                d = ExportDialog(w.doc, p, gid, w, on_automatic=w._set_auto_export)
                d.destination.setText(out)
                d.format.setCurrentText("tif")
                d.pattern.setText("$(identifier)")
                items = d._items()
                items[0].setCheckState(0, Qt.Unchecked)
                d.tabs.setCurrentIndex(1)  # Batch
                d.width_.setCurrentText("128")
                d.height_.setCurrentText("128")
                d.automatic.setChecked(True)
                written = d.export()
                names = {x.name for x in written}
                self.assertNotIn(f"{items[0].data(0, Qt.UserRole)}.tif", names)
                self.assertEqual(len(names), len(items) - 1)
                spec = oiio.ImageInput.open(str(written[0])).spec()
                self.assertEqual((spec.width, spec.height), (128, 128))
                kept = load_settings(p, gid)  # kept for the graph
                self.assertEqual((kept["format"], kept["batch"], kept["size"]), ("tif", True, [7, 7]))
                self.assertIn((id(p), gid), w.auto_exports)
                for f in written:
                    f.unlink()
                w._run_auto_exports()  # what a change of the graph triggers (debounced)
                self.assertEqual(len(list(Path(out).iterdir())), len(items) - 1)
            w._set_auto_export(p, gid, None)
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            app.processEvents()
        finally:
            rq.QRenderWidget.__init__ = init


if __name__ == "__main__":
    unittest.main()
