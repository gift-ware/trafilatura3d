"""Graph view interaction: a link dropped on a Dot's disc connects it, a drag inside a frame draws a selection box,
Shift adds to the selection, dragging nodes near the border scrolls the view, a linked resource can be removed."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class GraphInteractionTest(unittest.TestCase):
    def setUp(self):
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        from trafilatura3d.package import Package
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        self.app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pkg = Package()
        g = pkg.new_graph("G", 7)
        self.noise = g.add_node("gaussian_noise", (0.0, 0.0))
        self.dot = g.add_node("dot", (300.0, 30.0))
        self.blur = g.add_node("blur", (0.0, 400.0))
        self.levels = g.add_node("levels", (200.0, 400.0))
        self.frame = g.add_item("frame", (-100.0, 330.0), size=[600.0, 300.0])  # the blur and the levels in it
        path = Path(tmp.name) / "interaction.trafilatura3d"
        pkg.save(path)
        self.w = w = MainWindow([str(path)])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.deleteLater()  # not close(): an edited package would ask whether to save it
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        w.resize(1200, 800)
        w.show()
        self.app.processEvents()
        self.scene, self.view = w.graph.scene, w.graph.view
        self.view.resetTransform()
        self.view.centerOn(self.scene.nodes[self.blur])

    def graph(self):
        return self.w.graph.graph  # an undo restores the package: never keep its objects

    def at(self, item_or_point):
        """A scene point (or an item's centre) in viewport pixels."""
        from PySide6.QtCore import QPointF
        p = item_or_point if isinstance(item_or_point, QPointF) else item_or_point.sceneBoundingRect().center()
        return self.view.mapFromScene(p)

    def test_link_dropped_on_a_dot_connects_its_input(self):
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtWidgets import QGraphicsSceneMouseEvent
        scene = self.scene
        for kind, pos in ((QEvent.GraphicsSceneMousePress, scene.nodes[self.noise].outputs["output"].scenePos()),
                          (QEvent.GraphicsSceneMouseRelease, scene.nodes[self.dot].sceneBoundingRect().center())):
            e = QGraphicsSceneMouseEvent(kind)
            e.setScenePos(pos)
            e.setButton(Qt.LeftButton)
            (scene.mousePressEvent if kind == QEvent.GraphicsSceneMousePress else scene.mouseReleaseEvent)(e)
        self.assertEqual(self.graph().nodes[self.dot].inputs, {"input": [self.noise, "output"]})

    def test_drag_inside_a_frame_selects_and_shift_adds(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        frame = next(gi for gi in self.scene.graph_items if gi.item_id == self.frame)
        before = frame.pos()
        vp = self.view.viewport()
        a, b = self.at(QPointF(-60.0, 380.0)), self.at(QPointF(150.0, 560.0))  # inside the frame, around the blur
        QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, a)
        QTest.mouseMove(vp, b)
        QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, b)
        self.assertEqual(frame.pos(), before)  # the frame stays
        self.assertEqual(self.scene.selected_ids(), [self.blur])
        QTest.mouseClick(vp, Qt.LeftButton, Qt.ShiftModifier, self.at(self.scene.nodes[self.levels]))
        self.assertEqual(sorted(self.scene.selected_ids()), sorted([self.blur, self.levels]))
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, self.at(self.scene.nodes[self.levels]))
        self.assertEqual(self.scene.selected_ids(), [self.levels])

    def test_frame_resizes_from_every_side_and_corner(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        vp = self.view.viewport()

        def drag(a, b):
            QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, self.at(a))
            QTest.mouseMove(vp, self.at(b))
            QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, self.at(b))

        def frame():
            return self.graph().item(self.frame)
        drag(QPointF(-98.0, 480.0), QPointF(-148.0, 480.0))  # the left side, out by 50
        self.assertEqual((frame()["position"], frame()["size"]), ([-150.0, 330.0], [650.0, 300.0]))
        drag(QPointF(-148.0, 332.0), QPointF(-128.0, 352.0))  # the top left corner, in by 20 each way
        self.assertEqual((frame()["position"], frame()["size"]), ([-130.0, 350.0], [630.0, 280.0]))
        drag(QPointF(498.0, 480.0), QPointF(-400.0, 480.0))  # the right side past the left one: the smallest width
        self.assertEqual((frame()["position"], frame()["size"]), ([-130.0, 350.0], [80.0, 280.0]))

    def test_node_dragged_past_the_border_scrolls(self):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtTest import QTest
        from trafilatura3d.editor.graph import view as view_module
        view, item = self.view, self.scene.nodes[self.blur]
        vp = view.viewport()
        QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, self.at(item))
        QTest.mouseMove(vp, self.at(item) + QPointF(5, 0).toPoint())
        self.assertIs(self.scene.mouseGrabberItem(), item)
        held = view_module.QApplication.mouseButtons
        view_module.QApplication.mouseButtons = staticmethod(lambda: Qt.LeftButton)  # the button is down
        self.addCleanup(setattr, view_module.QApplication, "mouseButtons", held)
        r = vp.rect()
        scroll, x = view.horizontalScrollBar().value(), item.pos().x()
        view._edge_pos = QPointF(r.right() + 60, r.center().y())  # past the right border
        view._edge_scroll()
        self.assertGreater(view.horizontalScrollBar().value(), scroll)
        self.assertGreater(item.pos().x(), x)  # the node follows the mouse into the scrolled part
        view_module.QApplication.mouseButtons = held
        QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, self.at(item))

    def test_a_drag_previews_only_what_it_changes(self):
        import time
        w, live = self.w, self.w.doc.live
        pkg, g = w.graph.package, self.graph()
        w.graph.edit("Size", lambda: setattr(g, "parent_size", [11, 11]))  # 2048: above the preview size
        w.graph.connect(self.noise, "output", self.blur, "input")
        w.view2d.show_node(pkg, "G", self.blur)

        def settle(dragging=False):  # dragging: the refresh's steps only (a still value goes back to full size)
            t = time.perf_counter()
            while (live.busy or live.render_timer.isActive() if dragging else live.refreshing()) \
                    and time.perf_counter() - t < 60:
                self.app.processEvents()

        def drag(node, value):
            param = "scale" if node == self.noise else "level_in_mid"
            w.doc.edit(pkg, "Drag", lambda: self.graph().set_param(node, param, value), merge=node, drag=True)
            settle(dragging=True)
            self.assertTrue(live.previewing)
            return w.view2d.source()[0].size
        settle()
        self.assertEqual(drag(self.levels, [0.4] * 4), (2048, 2048))  # the Levels feeds nothing shown: full size
        w.doc.edit(pkg, "Release", lambda: None)
        settle()
        self.assertEqual(drag(self.noise, 7), (1024, 1024))  # upstream of the Blur shown: the preview

    def test_a_linked_resource_can_be_removed(self):
        from trafilatura3d.package import Resource
        pkg = self.w.graph.package
        pkg.resources["mesh"] = Resource("scene", "mesh.fbx")
        pkg.resources["used"] = Resource("bitmap", "used.png")
        bitmap = self.graph().add_node("bitmap")
        self.graph().set_param(bitmap, "pkg_resource_path", "used.png")
        explorer = self.w.explorer
        explorer.remove_resource(pkg, "mesh")
        self.assertNotIn("mesh", pkg.resources)
        self.w.doc.undo.undo()
        self.assertIn("mesh", self.w.graph.package.resources)
        with mock.patch("trafilatura3d.editor.explorer.QMessageBox.warning") as warned:
            explorer.remove_resource(self.w.graph.package, "used")  # a Bitmap node names its file: kept
        warned.assert_called_once()
        self.assertIn("used", self.w.graph.package.resources)


if __name__ == "__main__":
    unittest.main()
