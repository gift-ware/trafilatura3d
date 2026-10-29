"""Graph view: a link dragged near or past the view's border scrolls the view, its end kept under the mouse."""

import os
import tempfile
import unittest
from pathlib import Path

from tests.scenes import material_package


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class EdgeScrollTest(unittest.TestCase):
    def test_link_drag_scrolls_at_the_border(self):
        from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
        from PySide6.QtWidgets import QApplication, QGraphicsSceneMouseEvent
        import rendercanvas.qt as rq
        from trafilatura3d.editor.graph import view as view_module
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.close()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        w.resize(1200, 800)
        w.show()
        app.processEvents()
        scene, view = w.graph.scene, w.graph.view
        port = next(p for item in scene.nodes.values() for p in item.outputs.values())
        press = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
        press.setScenePos(port.scenePos())
        press.setButton(Qt.LeftButton)
        scene.mousePressEvent(press)
        self.assertTrue(scene.dragging_link())
        held = view_module.QApplication.mouseButtons
        view_module.QApplication.mouseButtons = staticmethod(lambda: Qt.LeftButton)  # the button is down
        self.addCleanup(setattr, view_module.QApplication, "mouseButtons", held)
        r = view.viewport().rect()
        before = view.horizontalScrollBar().value()
        view._edge_pos = QPointF(r.right() + 60, r.center().y())  # past the right border
        view._edge_scroll()
        self.assertGreater(view.horizontalScrollBar().value(), before)
        line = scene._drag[1]
        end = line.path().pointAtPercent(1.0)
        self.assertAlmostEqual(end.x(), view.mapToScene(view._edge_pos.toPoint()).x(), delta=1.0)
        middle = view.horizontalScrollBar().value()
        view._edge_pos = QPointF(r.center())  # back inside: no scrolling
        view._edge_scroll()
        self.assertEqual(view.horizontalScrollBar().value(), middle)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class ClearTest(unittest.TestCase):
    def test_mouse_moves_after_the_scene_is_cleared(self):
        """Clearing the scene (another graph shown) forgets the items it deleted, the Dot preview too: the next mouse
        moves, with or without Alt, still reach the scene."""
        from PySide6.QtCore import QCoreApplication, QPointF
        from PySide6.QtWidgets import QApplication
        from trafilatura3d.editor.graph.scene import GraphScene
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        QApplication.instance() or QApplication([])
        scene = GraphScene(None)
        scene._show_dot_preview(QPointF(10, 10))
        scene.clear()
        self.assertEqual((scene.nodes, scene.links, scene.graph_items, scene._dot_preview), ({}, [], [], None))
        scene._show_dot_preview(None)  # a move without Alt: nothing to hide, no error
        scene._show_dot_preview(QPointF(20, 20))  # with Alt over a link: a new preview
        self.assertIn(scene._dot_preview, scene.items())


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class FarItemsTest(unittest.TestCase):
    def test_the_view_reaches_far_items(self):
        """Imported graphs place nodes far from the origin (86 000 in one): the scrollable area grows to them."""
        from PySide6.QtCore import QCoreApplication, QPointF
        from PySide6.QtWidgets import QApplication, QGraphicsRectItem
        from trafilatura3d.editor.graph.scene import GraphScene
        from trafilatura3d.editor.graph.view import GraphView
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        QApplication.instance() or QApplication([])
        scene = GraphScene(None)
        scene.addItem(QGraphicsRectItem(90000, -70000, 100, 100))
        scene.fit_rect()
        view = GraphView(scene)
        view.resize(800, 600)
        view.centerOn(90050, -69950)
        center = view.mapToScene(view.viewport().rect().center())
        self.assertLess((center - QPointF(90050, -69950)).manhattanLength(), 4)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class BaseModeTest(unittest.TestCase):
    def test_absolute_enables_the_dropdown(self):
        """Choosing Absolute in a base parameter's inheritance menu shows the row again: its dropdown enabled (and
        its choice edits the node), one mode checked."""
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication, QComboBox, QToolButton
        import rendercanvas.qt as rq
        from trafilatura3d.compiler import DEPTHS
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        pkg, g = w.graph.package, w.graph.graph
        normal = next(n for n, node in g.nodes.items() if node.type == "normal")
        w.properties.show_target(pkg, g.id, normal)

        def format_row():
            fmt = next(c for c in w.properties.findChildren(QComboBox) if c.itemText(0) == DEPTHS["8"])
            menu = next(b for b in fmt.parentWidget().findChildren(QToolButton)
                        if b.toolTip().startswith("Inheritance")).menu()
            return fmt, menu
        fmt, menu = format_row()
        self.assertFalse(fmt.isEnabled())  # relative to the input
        next(a for a in menu.actions() if a.text() == "Absolute").trigger()
        app.processEvents()
        fmt, menu = format_row()
        self.assertTrue(fmt.isEnabled())
        self.assertEqual([a.text() for a in menu.actions() if a.isChecked()], ["Absolute"])
        fmt.setCurrentIndex(1)
        self.assertEqual(g.nodes[normal].base["output_format"], {"mode": "absolute", "value": "16"})

        noise = next(n for n, node in g.nodes.items() if node.type == "perlin_noise")
        w.doc.edit(pkg, "Graph format", lambda: g.set_base(None, "output_format", {"mode": "absolute", "value": "16"}))
        w.properties.show_target(pkg, g.id, noise)
        fmt, _ = format_row()
        self.assertFalse(fmt.isEnabled())  # still relative...
        self.assertEqual(fmt.currentText(), DEPTHS["16"])  # ...showing what it inherits, not its stored value


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class WarningsTest(unittest.TestCase):
    def test_warnings_close_until_they_change(self):
        """The ✕ closes a graph's warnings; the same ones stay closed, others show again."""
        from unittest import mock
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        from trafilatura3d.diagnostics import GraphWarning
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        panel = w.graph
        one, two = [GraphWarning("first", None, "")], [GraphWarning("first", None, ""), GraphWarning("second", None, "")]
        found = w.doc.warnings.setdefault(id(panel.package), {})  # as `Document.check` found them
        found[panel.graph_id] = one
        panel._show_warnings(panel.graph)
        self.assertFalse(panel.warnings_label.isHidden())
        panel._warning_link("#dismiss")
        self.assertTrue(panel.warnings_label.isHidden())
        panel._show_warnings(panel.graph)  # the same warnings: still closed
        self.assertTrue(panel.warnings_label.isHidden())
        found[panel.graph_id] = two
        panel._show_warnings(panel.graph)  # a new one: shown again
        self.assertFalse(panel.warnings_label.isHidden())

    def test_warnings_found_at_open_then_for_the_nodes_worked_on(self):
        """A package's warnings are found when it opens; an edit does not look again (Tools > Live warnings does), a
        node selected or connected has its own checked again: a link that now fits loses its warning."""
        from unittest import mock
        import rendercanvas.qt as rq
        from PySide6.QtCore import QCoreApplication, QEvent
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.diagnostics import GraphWarning
        from trafilatura3d.editor.mainwindow import MainWindow
        from trafilatura3d.package import Package
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pkg = Package()
        g = pkg.new_graph("G", 7)
        noise = g.add_node("gaussian_noise")
        blur = g.add_node("blur")
        path = Path(tmp.name) / "warn.trafilatura3d"
        pkg.save(path)
        w = MainWindow([str(path)])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        panel = w.graph
        pkg, g = panel.package, panel.graph

        def messages():
            return [w.message for w in w.doc.graph_warnings(pkg, "G")]
        self.assertIn("No output node defined", messages())  # found at the opening
        with mock.patch("trafilatura3d.diagnostics.package_warnings") as full:
            panel.edit("Add", lambda: g.add_node("output"))
            full.assert_not_called()  # an edit never checks the whole package
        self.assertIn("No output node defined", messages())  # kept as found (a graph's own check: at the opening)
        panel.connect(noise, "output", blur, "input")
        self.assertEqual([x.node for x in w.doc.graph_warnings(pkg, "G") if x.node], [])
        w.doc.warnings[id(pkg)]["G"].append(GraphWarning("stale", blur))
        panel.scene.nodes[blur].setSelected(True)  # selected: its own warnings checked again
        self.assertNotIn("stale", messages())


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class DropTest(unittest.TestCase):
    def test_dropped_files_open(self):
        """A package dropped on the window opens as File > Open does; other files are left to the panels (the 3D
        View takes meshes, not packages)."""
        from PySide6.QtCore import QCoreApplication, QEvent, QMimeData, QPointF, Qt, QUrl
        from PySide6.QtGui import QDropEvent
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        package = material_package(Path(tmp.name) / "scene.trafilatura3d")
        other = Path(tmp.name) / "notes.txt"
        other.write_text("x", encoding="utf-8")
        w = MainWindow([])

        def close():
            w.close()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        w.show()
        app.processEvents()

        def drop(path):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(path))])
            e = QDropEvent(QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            w.dropEvent(e)
            app.processEvents()
            return e.isAccepted(), mime
        accepted, _ = drop(other)
        self.assertFalse(accepted)
        self.assertEqual(w.doc.packages, [])
        accepted, mime = drop(package)
        self.assertTrue(accepted)
        self.assertEqual([Path(p.path).resolve() for p in w.doc.packages], [Path(package).resolve()])
        self.assertIsNone(w.view3d.view._dropped(mime))  # a package is not the 3D View's
        w.doc.undo.indexChanged.disconnect()
        w.graph.scene.selectionChanged.disconnect()


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class MoveTest(unittest.TestCase):
    def test_moves_recook_nothing(self):
        """Moving nodes and frames is a layout edit: no new revision, no `changed` (views, Properties and the cook stay
        as they are), the package is marked unsaved; undo and redo put the positions back on the same graph and node
        objects (Properties holds them) and the scene follows."""
        from PySide6.QtCore import QCoreApplication, QEvent, QPointF
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.mainwindow import MainWindow
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d"))])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        w.show()
        app.processEvents()
        panel, doc = w.graph, w.doc
        pkg, g = panel.package, panel.graph
        w.graph.add_item("frame", QPointF(0, 0))
        frame = g.items[0]["id"]
        revision, signals = doc._revision.get(id(pkg)), []
        doc.changed.connect(lambda p: signals.append("changed"))
        doc.layout_changed.connect(lambda p: signals.append("layout"))
        doc.dirty.discard(id(pkg))
        nid = next(iter(g.nodes))
        node = g.nodes[nid]
        start = list(node.position)
        panel.move_nodes({nid: (start[0] + 64, start[1] + 32)})
        panel.update_item(frame, (100, 50), None, {}, {})
        app.processEvents()
        self.assertEqual(signals, ["layout", "layout"])
        self.assertEqual(doc._revision.get(id(pkg)), revision)
        self.assertEqual(node.position, [start[0] + 64, start[1] + 32])
        self.assertIn(id(pkg), doc.dirty)
        self.assertTrue(w.explorer.tree.topLevelItem(0).text(0).endswith("*"))
        doc.undo.undo()
        doc.undo.undo()
        app.processEvents()
        self.assertIs(pkg.graphs[g.id], g)
        self.assertIs(g.nodes[nid], node)
        self.assertEqual(node.position, start)
        self.assertEqual(g.item(frame)["position"], [0, 0])
        self.assertEqual((panel.scene.nodes[nid].pos().x(), panel.scene.nodes[nid].pos().y()), tuple(start))
        doc.undo.redo()
        app.processEvents()
        self.assertEqual(node.position, [start[0] + 64, start[1] + 32])
        self.assertEqual(panel.scene.nodes[nid].pos().x(), start[0] + 64)
        self.assertNotIn("changed", signals)
        self.assertEqual(doc._revision.get(id(pkg)), revision)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class LiveRefreshTest(unittest.TestCase):
    def test_a_dragged_value_shows_at_the_preview_resolution(self):
        """An edit returns before cooking: the views follow a step per turn of the event loop. A dragged value shows
        at most PREVIEW_SIZE on a side and leaves the thumbnails as they are; on release the views and thumbnails
        come at full resolution. The selected node's inputs are held (what its next edit restarts from)."""
        import time

        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        self.addCleanup(setattr, rq.QRenderWidget, "__init__", init)
        from trafilatura3d.editor.live import PREVIEW_SIZE
        from trafilatura3d.editor.mainwindow import MainWindow
        from trafilatura3d.evaluator import PREVIEW
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MainWindow([str(material_package(Path(tmp.name) / "scene.trafilatura3d", 11))])

        def close():
            w.doc.undo.indexChanged.disconnect()
            w.graph.scene.selectionChanged.disconnect()
            w.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(close)
        live, panel = w.doc.live, w.graph
        pkg, g = panel.package, panel.graph

        def settle():
            t0 = time.perf_counter()
            while (live.busy or live.render_timer.isActive() or panel.thumb_timer.isActive()) \
                    and time.perf_counter() - t0 < 30:
                app.processEvents()
        normal = next(n for n, node in g.nodes.items() if node.type == "normal")
        noise = next(n for n, node in g.nodes.items() if node.type == "perlin_noise")
        w.view2d.show_node(pkg, g.id, normal)
        w.view3d.show_graph(pkg, g.id)
        panel.scene.nodes[normal].setSelected(True)
        panel.refresh_thumbnails()
        settle()
        thumbs = {n: item.thumbnail_key for n, item in panel.scene.nodes.items()}

        w.properties.show_target(pkg, g.id, normal)
        w.properties._edit("Intensity", lambda: g.set_param(normal, "intensity", 3.0), final=False, key="i")
        self.assertTrue(w.doc.dragging and live.previewing)
        settle()
        job, _ = w.view2d.source()
        self.assertEqual((job.size, job.partition), ((PREVIEW_SIZE, PREVIEW_SIZE), PREVIEW))
        self.assertEqual(w.view3d.view.renderer.materials[0]["spec"]["normal"][0].width, PREVIEW_SIZE)
        self.assertEqual({n: item.thumbnail_key for n, item in panel.scene.nodes.items()}, thumbs)
        held = w.doc.ev._held["graph selection"]
        self.assertIn(w.doc.compiled(pkg, g.id).jobs[noise].key, held)  # at full resolution and as shown
        self.assertIn(w.doc.shown(pkg, g.id).jobs[noise].key, held)

        w.properties._edit("Intensity", lambda: g.set_param(normal, "intensity", 3.0), final=True, key="i")
        self.assertFalse(w.doc.dragging or live.previewing)  # the release, the value unchanged
        settle()
        job, _ = w.view2d.source()
        self.assertEqual((job.size, job.partition), ((2048, 2048), ""))
        self.assertEqual(w.view3d.view.renderer.materials[0]["spec"]["normal"][0].width, 2048)
        self.assertEqual(panel.scene.nodes[normal].thumbnail_key, (job.key, "output"))

        # The 3D view takes each channel as soon as it is cooked, not those that did not change, and is not made
        # again once they are all there.
        r = w.view3d.view.renderer
        events = []
        update, rebuild = r.update_slots, r.set_materials
        r.update_slots = lambda i, slots, rect=None: (events.append(sorted(slots)), update(i, slots, rect))[1]
        r.set_materials = lambda specs: (events.append("all"), rebuild(specs))[1]
        w.doc.edit(pkg, "Seed", lambda: g.set_base(noise, "seed", {"mode": "absolute", "value": 5}))
        settle()
        self.assertEqual(events, [["baseColor"], ["normal"]])

        # A node not cooked yet shown in the 2D view: computed by the refresh while the view keeps being drawn (each
        # draw asking for it again restarted the refresh before it was done: heavy nodes never showed).
        w.doc.edit(pkg, "Seed", lambda: g.set_base(noise, "seed", {"mode": "absolute", "value": 9}))
        w.view2d.show_node(pkg, g.id, normal)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 30:
            w.view2d._texture()  # as every draw does
            app.processEvents()
            if not (live.busy or live.render_timer.isActive()):
                break
        job, out = w.view2d.source()
        self.assertIn(job.key, w.doc.ev.cache)
        self.assertIs(w.view2d._texture()[0], w.doc.ev.texture(job, out))


if __name__ == "__main__":
    unittest.main()
