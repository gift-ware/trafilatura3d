"""Editor shortcuts: Undo / Redo from anywhere in the window, the graph's Ctrl+D and Tab where the graph has
the keyboard, macOS' physical Control as well as Command, text fields keeping their own keys."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from tests.scenes import material_package


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class ShortcutTest(unittest.TestCase):
    def test_editing_shortcuts(self):
        from PySide6.QtCore import QCoreApplication, Qt
        from PySide6.QtWidgets import QApplication, QLineEdit
        import rendercanvas.qt as rq
        from PySide6.QtTest import QTest
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        try:
            from trafilatura3d.editor.mainwindow import MainWindow
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])
            w.show()
            p = w.doc.packages[0]
            gid = next(iter(p.graphs))
            w.open_graph(p, gid)
            app.processEvents()
            view = w.graph.view
            view.setFocus()
            app.processEvents()
            count = lambda: len(p.graphs[gid].nodes)  # undo rebuilds the graphs: look them up again
            n = count()
            mods = [Qt.ControlModifier] + ([Qt.MetaModifier] if sys.platform == "darwin" else [])
            for mod in mods:  # Command (Qt's Ctrl on macOS) and the physical Control key
                w.graph.scene.clearSelection()
                next(iter(w.graph.scene.nodes.values())).setSelected(True)
                QTest.keyClick(view.viewport(), Qt.Key_D, mod)
                app.processEvents()
                self.assertEqual(count(), n + 1)  # duplicate
                QTest.keyClick(view.viewport(), Qt.Key_Z, mod)
                app.processEvents()
                self.assertEqual(count(), n)  # undo
                QTest.keyClick(view.viewport(), Qt.Key_Z, mod | Qt.ShiftModifier)
                app.processEvents()
                self.assertEqual(count(), n + 1)  # redo
                QTest.keyClick(view.viewport(), Qt.Key_Z, mod)
                app.processEvents()
            opened = []
            w.graph.open_node_menu = lambda *a: opened.append(1)
            QTest.keyClick(view.viewport(), Qt.Key_Tab)
            app.processEvents()
            self.assertEqual(opened, [1])  # Tab: the node search
            field = QLineEdit(w)
            field.show()
            field.setFocus()
            app.processEvents()
            copied = []
            w.graph.copy = lambda: copied.append(1)
            QTest.keyClick(field, Qt.Key_C, Qt.ControlModifier)
            app.processEvents()
            self.assertEqual(copied, [])  # a text field's Ctrl+C is its own
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
        finally:
            rq.QRenderWidget.__init__ = init


if __name__ == "__main__":
    unittest.main()
