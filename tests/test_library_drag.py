"""Dragging from the libraries: the Graph workspace's Library list stays draggable whatever its display (Qt turns
dragging off in setMovement(Static) and setViewMode), and what it drags drops into the graph as nodes."""

import os
import tempfile
import unittest
from pathlib import Path

from tests.scenes import material_package


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class LibraryDragTest(unittest.TestCase):
    def test_graph_library(self):
        from PySide6.QtCore import QByteArray, QCoreApplication, QMimeData, QPointF
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        try:
            from trafilatura3d.editor.library import MIME
            from trafilatura3d.editor.mainwindow import MainWindow
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])
            lib = w.library
            for mode in ("List", "Small Icons", "Large Icons"):
                lib._display(mode)
                self.assertTrue(lib.list.dragEnabled(), mode)
            p = w.doc.packages[0]
            gid = next(iter(p.graphs))
            w.open_graph(p, gid)
            n = len(p.graphs[gid].nodes)
            mime = QMimeData()
            mime.setData(MIME, QByteArray(b"blend\ninvert"))
            self.assertTrue(w.graph._drop(mime, QPointF(10, 10)))
            self.assertEqual(len(w.doc.packages[0].graphs[gid].nodes), n + 2)
            w.doc.dirty.clear()
            w.close()
        finally:
            rq.QRenderWidget.__init__ = init


if __name__ == "__main__":
    unittest.main()
