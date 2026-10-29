"""The Paint workspace in the editor: a project opens in its own workspace with its own menus, layers are
created / masked / moved / undone, strokes paint the selected layer and undo, display modes, export and the
project file round trip through the window."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

from tests.test_paint import read, write_obj


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class PaintWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        cls.app = QApplication.instance() or QApplication([])
        cls._init = rq.QRenderWidget.__init__
        init = cls._init
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})

    @classmethod
    def tearDownClass(cls):
        import rendercanvas.qt as rq
        rq.QRenderWidget.__init__ = cls._init

    def setUp(self):
        from trafilatura3d.editor.mainwindow import MainWindow
        from trafilatura3d.editor.paint.document import PaintDocument
        from trafilatura3d.paint.session import Session
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.w = MainWindow([])
        self.addCleanup(self._close_window)  # registered first: runs after the tests' own cleanups
        self.w.resize(1400, 900)
        self.w.show()
        self.ws = self.w.paint
        self.ws.set_document(PaintDocument(Session.new(write_obj(self.dir), 128, evaluator=self.w.doc.ev)))
        self.ws.live.previewing = False  # the opening's preview and full resolution steps done at once
        self.ws.refresh_materials()
        self.app.processEvents()

    def _close_window(self):
        self.ws.doc.dirty = False
        self.ws.close_project()
        for p in list(self.w.doc.packages):
            self.w.doc.close(p)
        self.w.close()
        self.w.deleteLater()
        self.app.processEvents()
        from PySide6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # processEvents alone never deletes it
        self.tmp.cleanup()

    def test_open_shows_the_preview_first(self):
        """Opening a project returns before cooking: the view comes at the preview resolution a tile per turn of the
        event loop, then at full resolution by itself."""
        import time

        from trafilatura3d.editor.paint.document import PaintDocument
        from trafilatura3d.paint.session import Session
        ws = self.ws
        ws.set_document(PaintDocument(Session.new(write_obj(self.dir), 128, evaluator=self.w.doc.ev)))
        live = ws.live
        self.assertTrue(live.previewing)
        self.assertTrue(live.pending)
        t0 = time.perf_counter()
        while (live.previewing or live.pending or live.render_timer.isActive()) and time.perf_counter() - t0 < 20:
            self.app.processEvents()
        self.assertFalse(live.previewing)
        self.assertFalse(live.pending)
        self.assertTrue(ws._shown)

    def test_workspace_and_menus(self):
        self.assertIs(self.w.stack.currentWidget(), self.ws)
        for key, d in self.ws.docks.items():  # every visible dock has its place (none loose at the top left)
            if d.isVisible() and not d.isFloating():
                self.assertNotEqual(self.ws.dockWidgetArea(d), Qt.NoDockWidgetArea, key)
        titles = [a.text() for a in self.w.menuBar().actions()]
        self.assertEqual(titles, ["File", "Edit", "Mode", "Window", "Viewport", "Plugins", "Help"])
        self.assertFalse(self.w.graph_bar.isVisible())
        self.assertEqual(self.ws.texture_sets.tree.topLevelItemCount(), 2)
        self.w.show_home()
        self.w._close_home()
        self.assertIs(self.w.stack.currentWidget(), self.ws)

    def test_generator_parameter_groups(self):
        """A generator's parameters: the ungrouped ones, then each group under its collapsible title (collapsed it
        stays so when the panel is built again); the resources (Micro Normal…) after the parameters."""
        from PySide6.QtWidgets import QLabel, QPushButton
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        lp.add_effect("generator")
        ws.properties._set_procedural(doc.effect().uid, "node:paint_ambient_occlusion")
        def titles():
            body = ws.properties.widget()  # a rebuild makes another
            groups = {b.text()[3:]: b for b in body.findChildren(QPushButton) if b.text()[:1] in ("▾", "▸")}
            sections = [w.text() for w in body.findChildren(QLabel) if w.property("section")]
            return groups, sections
        groups, sections = titles()
        self.assertEqual(list(groups), ["Ambient Occlusion", "Texture", "Micro Details"])
        self.assertLess(sections.index("Parameters"), sections.index("Resources"))
        lay = ws.properties.lay
        order = [lay.itemAt(i).widget() for i in range(lay.count())]
        self.assertLess(order.index(groups["Micro Details"]), order.index(next(
            w for w in order if isinstance(w, QLabel) and w.text() == "Resources")))
        groups["Texture"].setChecked(False)
        ws.properties.rebuild()
        groups, _ = titles()
        self.assertFalse(groups["Texture"].isChecked())
        self.assertTrue(groups["Micro Details"].isChecked())

    def test_viewport_wireframe_and_grid_colors(self):
        """Display settings > Viewport: the wireframe's and the grid's color and opacity reach the viewport."""
        from trafilatura3d.editor.paint.widgets import PSlider, Swatch
        ws = self.ws
        panel, look = ws.display_settings, ws.view3d.look
        swatches = {s.title: s for s in panel.widget().findChildren(Swatch)}
        sliders = {s.label.text(): s for s in panel.widget().findChildren(PSlider)}
        ws.frame()

        def colored(rgb):  # pixels close to the color
            img = ws.view3d.render_image(128, 128)[..., :3].astype(np.float32)
            img = img / 255.0 if img.max() > 1.0 else img
            return int((np.abs(img - np.array(rgb)).max(axis=-1) < 0.1).sum())

        for attr, name, rgb in (("wireframe", "Wireframe", (1.0, 0.0, 0.0)), ("grid", "Grid", (0.0, 1.0, 0.0))):
            setattr(look, attr, True)
            swatches[name]._picked(list(rgb), True)
            self.assertEqual(tuple(getattr(look, f"{attr}_color")), rgb)
            sliders[f"{name} opacity"].edited.emit(1.0, True)
            self.assertGreater(colored(rgb), 20, name)
            sliders[f"{name} opacity"].edited.emit(0.0, True)
            self.assertEqual(getattr(look, f"{attr}_opacity"), 0.0)
            self.assertEqual(colored(rgb), 0, name)
            setattr(look, attr, False)

    def test_home(self):
        """The Home: a tile per kind of document (click creates), an Open link below each, the recent files with
        their kind's icon, folder and date; double click opens."""
        from PySide6.QtCore import QPoint, QSettings
        from PySide6.QtTest import QTest
        from trafilatura3d.editor.home import HomeScreen, RecentRow
        home = HomeScreen()  # not the window's: its tiles would open the modal New dialogs
        self.addCleanup(home.deleteLater)
        fired = []
        for sig in ("new_graph", "new_project", "open_package", "open_project", "open_recent"):
            getattr(home, sig).connect(lambda *a, sig=sig: fired.append(sig))
        for tile in home.tiles:
            QTest.mouseClick(tile, Qt.LeftButton, pos=QPoint(50, 50))
        self.assertEqual(fired[:2], ["new_graph", "new_project"])
        paint = self.dir / "a.trafilatura3dpaint"
        graph = self.dir / "b.trafilatura3d"
        paint.write_text("x", encoding="utf-8")
        graph.write_text("x", encoding="utf-8")
        saved = QSettings().value("recent", [], type=list)
        self.addCleanup(QSettings().setValue, "recent", saved)
        QSettings().setValue("recent", [str(paint), str(graph), str(self.dir / "gone.trafilatura3d")])
        home.refresh()
        self.assertEqual(home.recent.count(), 2)  # missing files are left out
        rows = [home.recent.itemWidget(home.recent.item(i)) for i in range(2)]
        self.assertIsInstance(rows[0], RecentRow)
        tips = [r.icon.toolTip() for r in rows]
        self.assertEqual(tips, ["Trafilatura 3D Paint project", "Trafilatura 3D Graph package"])
        home.recent.itemDoubleClicked.emit(home.recent.item(1))
        self.assertEqual(fired[-1], "open_recent")

    def test_selection_keeps_the_rows(self):
        """Clicking a row selects it without rebuilding the tree (a rebuild under the press ended the drag that
        reorders effects and made the selection jump); only showing the mask's effects instead of the content's
        rebuilds. Effects reorder by drag target, Alt+Up / Down and the menu."""
        from PySide6.QtTest import QTest
        from shiboken6 import isValid
        ws = self.ws
        lp, doc = ws.layers, ws.doc
        lp.add_layer("fill")
        layer = doc.selected
        for k in ("levels", "fill", "paint"):
            doc.select(layer)
            lp.add_effect(k)
        fx = [f.uid for f in doc.ts.find(layer).effects]
        doc.select(fx[2])
        items = dict(lp.items)
        target = lp.items[fx[0]]
        lp.tree.scrollToItem(target)
        rect = lp.tree.visualItemRect(target)
        from PySide6.QtCore import QPoint
        # beside the eye, on the row's icon and name (not its blending controls)
        QTest.mouseClick(lp.tree.viewport(), Qt.LeftButton, pos=QPoint(rect.left() + 40, rect.center().y()))
        self.assertEqual(doc.selected, fx[0])
        self.assertTrue(all(isValid(it) for it in items.values()))  # the same rows, alive
        self.assertIs(lp.items[fx[0]], target)
        self.assertIs(lp.tree.currentItem(), target)
        lp.move_effect(1)  # Alt+Up
        self.assertEqual([f.uid for f in doc.ts.find(layer).effects], [fx[1], fx[0], fx[2]])
        lp.drop_effect(fx[0], layer, fx[2])  # dropped between the layer's row and the top effect
        self.assertEqual([f.uid for f in doc.ts.find(layer).effects], [fx[1], fx[2], fx[0]])
        doc.select(layer)
        lp.add_mask(1.0)  # selects the mask: its effects are shown, the tree is rebuilt
        self.assertTrue(doc.mask_selected)
        self.assertNotIn(fx[0], lp.items)

    def test_mask_effects_reorder_in_the_mask(self):
        """Two effects under a mask, the bottom one dragged to the top: it goes above the other one, in the mask,
        whether the drop line is reported above the top effect or below the layer's row; it never lands in the
        layer's content. A mask's effect onto a layer without a mask, or between another layer's effects, is
        refused."""
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        layer = doc.selected
        lp.add_effect("levels")  # a content effect, never touched below
        content = [f.uid for f in doc.ts.find(layer).effects]
        doc.select(layer)
        lp.add_mask(1.0)
        for k in ("levels", "fill"):
            doc.select(layer, mask=True)
            lp.add_effect(k)
        mask_fx = lambda: [f.uid for f in doc.ts.find(layer).mask.effects]  # noqa: E731 - bottom first
        first, second = mask_fx()  # rows: second (top), first (bottom)
        tree = lp.tree
        for where, target in (("above", second), ("below", layer)):
            doc.select(first)
            self.assertTrue(doc.mask_selected)
            tree.drop_effect_at(first, lp.items[target], where)
            self.assertEqual(mask_fx(), [second, first], where)  # now on top, still in the mask
            self.assertEqual([f.uid for f in doc.ts.find(layer).effects], content)
            doc.undo.undo()
            self.assertEqual(mask_fx(), [first, second])
        # the bottom row dropped below the last row: back at the bottom (no move)
        doc.select(second)
        tree.drop_effect_at(second, lp.items[first], "below")
        self.assertEqual(mask_fx(), [second, first])
        doc.undo.undo()
        # another layer without a mask: refused; its content effects: refused
        doc.select(layer)
        lp.add_layer("fill")
        other = doc.selected
        lp.add_effect("levels")
        other_fx = doc.effect().uid
        n = doc.undo.index()
        lp.drop_effect(first, into=other)
        lp.drop_effect(first, other, other_fx)
        self.assertEqual(doc.undo.index(), n)
        self.assertEqual(mask_fx(), [first, second])

    def test_effect_drop_between_effects_through_qt(self):
        """Three effects under a mask, the bottom one dragged between the other two through Qt's own drag events:
        Qt accepts the drop line there (the layer row takes drops) and the effect lands in the middle, in the mask.
        A layer dragged among effects is not moved."""
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QDragMoveEvent, QDropEvent
        from PySide6.QtWidgets import QAbstractItemView
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        layer = doc.selected
        lp.add_mask(1.0)
        for k in ("levels", "fill", "paint"):
            doc.select(layer, mask=True)
            lp.add_effect(k)
        a, b, c = [f.uid for f in doc.ts.find(layer).mask.effects]  # rows top first: c, b, a
        tree = lp.tree
        tree.setDragDropMode(QAbstractItemView.DragDrop)  # synthetic events have no source (InternalMove checks it)
        self.addCleanup(tree.setDragDropMode, QAbstractItemView.InternalMove)

        def drag(uid, onto_uid, lower=True):
            doc.select(uid)
            item, onto = lp.items[uid], lp.items[onto_uid]
            tree.setCurrentItem(item)
            tree.scrollToItem(onto)
            r = tree.visualItemRect(onto)
            pos = QPoint(r.left() + 40, r.bottom() - 3 if lower else r.top() + 3)
            mime = tree.mimeData([item])
            move = QDragMoveEvent(pos, Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
            tree.dragMoveEvent(move)
            accepted = move.isAccepted()
            tree.dropEvent(QDropEvent(pos, Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier))
            return accepted
        self.assertTrue(drag(a, c))  # the lower half of the top row: between c and b
        self.assertEqual([f.uid for f in doc.ts.find(layer).mask.effects], [b, a, c])
        self.assertEqual(doc.ts.find(layer).effects, [])
        self.assertTrue(drag(c, b, lower=False))  # the upper half of b's row (now the bottom): between a and b
        self.assertEqual([f.uid for f in doc.ts.find(layer).mask.effects], [b, c, a])
        doc.select(layer)
        lp.add_layer("paint")
        other = doc.selected
        lp.add_effect("levels")
        other_fx = doc.effect().uid
        n = doc.undo.index()
        drag(layer, other_fx)  # a layer among effects: nothing moves
        self.assertEqual(doc.undo.index(), n)

    def test_mask_then_layer_click(self):
        """Clicking the mask thumbnail selects the mask, clicking the layer's thumbnail (or name) goes back to the
        layer, every time (the row was current already, so the click did nothing)."""
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("paint")
        layer = doc.selected
        lp.add_mask(1.0)

        def click(widget_of_row):
            row = lp.rows[layer]
            w = widget_of_row(row)
            lp.tree.scrollToItem(lp.items[layer])
            self.app.processEvents()
            QTest.mouseClick(w, Qt.LeftButton, pos=QPoint(w.width() // 2, w.height() // 2))
            self.app.processEvents()
        for _ in range(2):
            click(lambda r: r.mask)
            self.assertEqual((doc.selected, doc.mask_selected), (layer, True))
            click(lambda r: r.thumb)
            self.assertEqual((doc.selected, doc.mask_selected), (layer, False))
        click(lambda r: r.mask)
        click(lambda r: r.name)
        self.assertFalse(doc.mask_selected)

    def test_mac_control_key_on_mouse_bindings(self):
        """macOS: Cmd + right drag changes the brush size, and so does the Control key (as before the bindings
        table, and as the keyboard shortcuts); a press that is exactly another binding keeps it."""
        from unittest import mock
        from trafilatura3d.editor.paint import shortcuts
        with mock.patch.object(shortcuts, "MAC", True):
            self.assertEqual(shortcuts.match_mouse(2, {"Control"}, set()), ["Change tool Size / Hardness"])
            self.assertEqual(shortcuts.match_mouse(2, {"Meta"}, set()), ["Change tool Size / Hardness"])
            self.assertEqual(shortcuts.match_mouse(1, {"Meta"}, set()), ["Change tool Flow / Rotation"])
            self.assertEqual(shortcuts.match_mouse(1, {"Meta", "Shift"}, set()), ["Draw straight lines with snapping"])
            ws, v = self.ws, self.ws.view3d
            ws.layers.add_layer("paint")
            ws.set_tool("paint")
            size = ws.tools.brush.size
            for kind, x in (("pointer_down", 50), ("pointer_move", 90), ("pointer_up", 90)):
                v._on_event({"event_type": kind, "x": x, "y": 50, "button": 2, "modifiers": ("Meta",)})
            self.assertGreater(ws.tools.brush.size, size)

    def test_pen_pressure_reaches_the_brush(self):
        """A tablet's pen: Qt's tablet events carry the pressure, the mouse events Qt makes of them paint with it,
        in the 3D and the 2D view; the events still queued when the pen is lifted keep its pressure (a fast stroke
        left a last stamp at full intensity); a mouse press goes back to full pressure."""
        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QInputDevice, QPointingDevice, QTabletEvent
        from PySide6.QtTest import QTest
        ws = self.ws

        def mouse_press(view):  # a real mouse's press (not the pen's), handled by the view as Qt sends it
            QTest.mousePress(view, Qt.LeftButton)
            QTest.mouseRelease(view, Qt.LeftButton)
        pen = QPointingDevice("test pen", 77, QInputDevice.DeviceType.Stylus, QPointingDevice.PointerType.Pen,
                              QInputDevice.Capability.Position | QInputDevice.Capability.Pressure, 1, 2)

        def tablet(view, kind, pressure):
            e = QTabletEvent(kind, pen, QPointF(20, 20), QPointF(view.mapToGlobal(QPointF(20, 20))), pressure, 0, 0,
                             0.0, 0.0, 0.0, Qt.NoModifier, Qt.LeftButton, Qt.LeftButton)
            self.app.sendEvent(view, e)
            return e.isAccepted()
        got = []
        ws.stroke_begin = lambda x, y, p, **k: got.append(p) or False
        ws.stroke2d_begin = lambda x, y, p, r: got.append(p) or False
        v3, v2 = ws.view3d, ws.view2d
        self.assertFalse(tablet(v3, QEvent.TabletPress, 0.3))  # left to Qt, which makes the mouse events
        v3._on_event({"event_type": "pointer_down", "x": 20, "y": 20, "button": 1, "modifiers": ()})
        v3._on_event({"event_type": "pointer_up", "x": 20, "y": 20, "button": 1, "modifiers": ()})
        tablet(v3, QEvent.TabletRelease, 0.0)
        self.assertEqual(v3.pen_pressure(), 0.3)  # the pen's moves still queued paint with it
        mouse_press(v3)
        self.assertEqual(v3.pen_pressure(), 1.0)
        self.assertEqual(got[0], 0.3)
        got.clear()
        self._show_2d()
        c = v2.width() / 2, v2.height() / 2  # inside the 0-1 range of the 2D view
        click = [{"event_type": kind, "x": c[0], "y": c[1], "button": 1, "modifiers": ()}
                 for kind in ("pointer_down", "pointer_up")]
        tablet(v2, QEvent.TabletPress, 0.6)
        for e in click:
            v2._on_event(e)
        tablet(v2, QEvent.TabletRelease, 0.0)
        self.assertAlmostEqual(v2.pen_pressure(), 0.6, places=5)
        mouse_press(v2)
        got.clear()
        for e in click:  # the mouse
            v2._on_event(e)
        self.assertEqual(got, [1.0])

    def test_layer_row_ux(self):
        """A layer row: the opacity has a slider under its field (each follows the other, the layer's opacity
        follows both); the row's labels and icons let the selection through; the Home button in the panel bar."""
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("paint")
        uid = doc.selected
        row = lp.rows[uid]
        row.opacity_slider.setValue(40)
        self.assertEqual(row.opacity.value(), 40)
        self.assertAlmostEqual(doc.ts.find(uid).blend_of(doc.viewmode).opacity, 0.4)
        row = lp.rows[uid]  # the stack may have been rebuilt by the edit
        row.opacity.setValue(70)
        self.assertEqual(row.opacity_slider.value(), 70)
        self.assertEqual(row.objectName(), "stackRow")
        from trafilatura3d.editor import theme
        self.assertIn("#stackRow, #stackRow QLabel", theme.STYLESHEET)  # the rows' parts let the selection through
        homes = [a for a in ws.docks_bar.actions() if a.text() == "Home screen"]
        self.assertEqual(len(homes), 1)
        homes[0].trigger()
        self.assertIs(self.w.stack.currentWidget(), self.w.home)

    def test_assets_names_and_thumbnail_size(self):
        """Assets: the tooltip gives the full name then the kind; Thumbnail size (Small, Medium, Large, List)
        changes the grid, keeps the drag, and is remembered."""
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QListWidget
        from trafilatura3d.editor.paint.assets import THUMB_SIZES
        assets = self.ws.assets
        self.addCleanup(QSettings().remove, "paint/assets_size")
        assets.refresh()
        it = assets.grid.item(0)
        self.assertEqual(it.toolTip().split("\n")[0], it.text())
        self.assertEqual(len(it.toolTip().split("\n")), 2)
        for mode, size in THUMB_SIZES.items():
            assets.set_thumbnail_size(mode)
            self.assertEqual(assets.grid.iconSize().width(), size)
            self.assertEqual(assets.grid.viewMode(), QListWidget.IconMode)
            self.assertTrue(assets.grid.dragEnabled())
        assets.set_thumbnail_size("List")
        self.assertEqual(assets.grid.viewMode(), QListWidget.ListMode)
        self.assertTrue(assets.grid.dragEnabled())
        self.assertEqual(QSettings().value("paint/assets_size"), "List")
        self.assertEqual([a.text() for a in assets.size_button.menu().actions()], ["Small", "Medium", "Large", "List"])

    def test_shaderball_thumbnail(self):
        """A Paint node's thumbnail on the shader ball: 96 px, the ball opaque, the background transparent; made
        in a hidden window of its own (the open project and the eyedropper stay as they were)."""
        import OpenImageIO as oiio
        from trafilatura3d.editor.paint import color_picker, shaderball
        doc, sampler = self.ws.doc, color_picker.SAMPLER
        made, failed = shaderball.rebuild(only=["paint_curvature", "blur_hq"], out=self.dir, log=lambda _m: None)
        self.assertEqual((made, failed), (1, []))  # the Graph filter blur_hq: no ball (its initials, thumbnails.cooked)
        self.assertFalse((self.dir / "blur_hq.png").exists())
        img = oiio.ImageBuf(str(self.dir / "paint_curvature.png")).get_pixels(oiio.UINT8)
        self.assertEqual(img.shape, (96, 96, 4))
        self.assertEqual(int(img[0, 0, 3]), 0)  # the corner: no environment
        self.assertEqual(int(img[48, 48, 3]), 255)  # the ball
        self.assertIs(self.ws.doc, doc)
        self.assertIs(color_picker.SAMPLER, sampler)

    def test_pressure_toggles(self):
        """Size and Flow have the pen pressure toggle in the contextual toolbar and in the Properties (Pen pressure
        page), Stroke opacity and Spacing do not; the two follow each other and the brush."""
        from trafilatura3d.editor.paint.widgets import PressureButton
        ws = self.ws
        ws.layers.add_layer("paint")
        ws.set_tool("paint")
        self.assertEqual(sorted(ws.context_pressure), ["flow", "size"])
        b = ws.tools.brush
        top = ws.context_pressure["size"]
        top.setChecked(not b.size_pressure)
        self.assertEqual(b.size_pressure, top.isChecked())
        props = [w for w in ws.properties.findChildren(PressureButton)]
        self.assertEqual(len(props), 2)
        self.assertEqual(props[0].isChecked(), b.size_pressure)  # Size first, as in the panel
        props[1].setChecked(not b.flow_pressure)
        self.assertEqual(ws.context_pressure["flow"].isChecked(), b.flow_pressure)
        self.assertFalse(top.icon().isNull())
        for button in (ws.lazy_button, ws.sym_button):  # the text is never under the menu arrow
            text = button.fontMetrics().horizontalAdvance(button.text())
            self.assertGreaterEqual(button.sizeHint().width(), text + 20)

    def test_wheel_needs_focus(self):
        """The wheel over a slider, spin box or dropdown without the focus does not change it and is left for the
        panel to scroll; with the focus it edits the value."""
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtWidgets import QComboBox, QSlider, QSpinBox, QVBoxLayout, QWidget
        box = QWidget()
        self.addCleanup(box.deleteLater)
        lay = QVBoxLayout(box)
        slider, spin, combo = QSlider(Qt.Horizontal), QSpinBox(), QComboBox()
        slider.setRange(0, 100)
        spin.setRange(0, 100)
        combo.addItems(["a", "b", "c"])
        for w in (slider, spin, combo):
            lay.addWidget(w)
        box.show()
        self.app.processEvents()

        def wheel(w):
            e = QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, -120),
                            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
            self.app.sendEvent(w, e)
            return e.isAccepted()
        values = lambda: (slider.value(), spin.value(), combo.currentIndex())  # noqa: E731
        before = values()
        self.assertEqual([wheel(w) for w in (slider, spin, combo)], [False] * 3)
        self.assertEqual(values(), before)
        for w in (slider, spin, combo):
            w.setFocus()
            self.app.processEvents()
            wheel(w)
        self.assertNotEqual(values(), before)
        self.assertNotEqual(slider.focusPolicy(), Qt.WheelFocus)
        # a panel still scrolls: the scroll area hands the wheel to its scroll bar (a slider without the focus)
        from PySide6.QtWidgets import QScrollArea
        area = QScrollArea()
        self.addCleanup(area.deleteLater)
        tall = QWidget()
        tall.setMinimumHeight(3000)
        area.setWidget(tall)
        area.resize(200, 200)
        area.show()
        self.app.processEvents()
        start = area.verticalScrollBar().value()
        wheel(area.viewport())
        self.assertGreater(area.verticalScrollBar().value(), start)

    def test_layers_and_undo(self):
        doc, lp = self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        lp.add_layer("paint")
        lp.add_layer("folder")
        ts = doc.ts
        self.assertEqual([x.kind for x in ts.layers], ["fill", "paint", "folder"])
        paint = ts.layers[1].uid
        folder = ts.layers[2].uid
        lp.move(paint, folder, "into")
        self.assertEqual(doc.ts.find(folder).children[0].uid, paint)
        doc.select(paint)
        lp.add_mask(0.0)
        self.assertIsNotNone(doc.ts.find(paint).mask)
        self.assertTrue(doc.mask_selected)
        doc.undo.undo()
        self.assertIsNone(doc.ts.find(paint).mask)
        doc.undo.undo()  # the move
        self.assertEqual(len(doc.ts.layers), 3)
        doc.undo.redo()
        self.assertEqual(len(doc.ts.layers), 2)
        lp.set_blend(folder, mode="multiply")
        self.assertEqual(doc.ts.find(folder).blend_of(doc.viewmode).mode, "multiply")
        self.app.processEvents()
        self.assertEqual(len(lp.rows), 3)

    def test_rebuild_keeps_collapsed_folders(self):
        """A folder the user collapsed stays collapsed when an edit rebuilds the tree (reopening it moved every row
        below it and lost the scroll)."""
        doc, lp = self.ws.doc, self.ws.layers
        lp.add_layer("paint")
        lp.add_layer("folder")
        ts = doc.ts
        paint, folder = ts.layers[-2].uid, ts.layers[-1].uid
        lp.move(paint, folder, "into")
        self.app.processEvents()
        lp.items[folder].setExpanded(False)
        lp.set_blend(folder, mode="multiply")  # an edit of the structure: the tree is made again
        self.app.processEvents()
        self.assertFalse(lp.items[folder].isExpanded())
        lp.items[folder].setExpanded(True)
        lp.set_blend(folder, mode="screen")
        self.app.processEvents()
        self.assertTrue(lp.items[folder].isExpanded())

    def test_folder_mask_effects_shown(self):
        """A folder's mask takes effects, shown under the folder (above its layers) while the mask is selected."""
        doc, lp = self.ws.doc, self.ws.layers
        lp.add_layer("paint")
        lp.add_layer("folder")
        ts = doc.ts
        paint, folder = ts.layers[-2].uid, ts.layers[-1].uid
        lp.move(paint, folder, "into")
        lp.add_mask(1.0)
        doc.select(folder, mask=True)
        lp.add_effect("levels")
        self.app.processEvents()
        fx = ts.find(folder).mask.effects[0].uid
        self.assertEqual(doc.selected, fx)
        it = lp.items[folder]
        self.assertEqual([it.child(i).data(0, Qt.UserRole) for i in range(it.childCount())], [fx, paint])
        doc.select(paint)
        self.app.processEvents()
        self.assertNotIn(fx, lp.items)  # the mask not selected: only the layers

    def stroke(self):
        v = self.ws.view3d
        w, h = v.physical_size()
        self.ws.view3d.yaw, self.ws.view3d.pitch = 0.0, 1.5  # from above
        self.ws.frame()
        ok = self.ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        for k in range(1, 20):
            self.ws.stroke_add(w * (0.3 + 0.4 * k / 19), h * 0.5, 1.0)
        self.ws.stroke_end()
        return ok

    def test_every_stroke_updates_while_painting(self):
        """Each frame of a stroke updates the stack over the brush's rectangle (PaintWorkspace._freeze_stroke): the
        second stroke too, whose slots are cached already (its outside inputs were missing and every frame failed,
        the stroke showing only at the release); the frozen channel is what a whole cook gives."""
        doc, lp = self.ws.doc, self.ws.layers
        for name in [ts.name for ts in doc.project.texture_sets]:
            doc.select_texture_set(name)
            if name == "Left":
                break
        lp.add_layer("paint")
        v = self.ws.view3d
        w, h = v.physical_size()
        v.yaw, v.pitch = 0.0, 1.5
        self.ws.frame()
        self.ws.set_tool("paint")
        for n in range(2):
            self.assertTrue(self.ws.stroke_begin(w * 0.2, h * (0.4 + 0.1 * n), 1.0))
            for k in range(1, 8):
                self.ws.stroke_add(w * (0.2 + 0.3 * k / 7), h * (0.4 + 0.1 * n), 1.0)
                self.ws._stroke_timer.stop()
                self.ws._refresh_stroke()  # raised when an input of the chain was missing
            udim = sorted(self.ws._frozen)[0]
            jobs = self.ws._frozen[udim][0]
            fresh = self.ws._slot_jobs(doc.ts, udim, padded=False, only={"baseColor"})["baseColor"]
            self.ws.ev.cook([fresh])
            np.testing.assert_array_equal(self.ws.ev.read(jobs["baseColor"], "output"),
                                          self.ws.ev.read(fresh, "output"))
            self.ws.stroke_end()
            self.app.processEvents()

    def test_fast_painting_turns_on_by_itself(self):
        """Frames slower than FAST_PAINTING_MS for FAST_FRAMES frames: Fast painting turns on for the selected layer
        (the painted channels only while painting), the Log says it once, another layer turns it off."""
        import trafilatura3d.editor.paint.workspace as wsm
        doc, lp, ws = self.ws.doc, self.ws.layers, self.ws
        lp.add_layer("paint")
        v = ws.view3d
        w, h = v.physical_size()
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        ws.set_tool("paint")
        saved, wsm.FAST_PAINTING_MS = wsm.FAST_PAINTING_MS, 0.0
        try:
            self.assertTrue(ws.stroke_begin(w * 0.3, h * 0.5, 1.0))
            self.assertFalse(ws._fast)
            for k in range(1, 8):
                ws.stroke_add(w * (0.3 + 0.4 * k / 7), h * 0.5, 1.0)
                ws._stroke_timer.stop()
                ws._refresh_stroke()
            self.assertTrue(ws._fast)
            ws.stroke_end()
        finally:
            wsm.FAST_PAINTING_MS = saved
        self.assertEqual(ws.log_view.toPlainText().count("Fast painting turned on"), 1)
        uid = doc.selected
        lp.add_layer("paint")  # another layer selected
        self.assertIsNone(ws._fast_layer)
        doc.select(uid)
        self.assertTrue(ws.stroke_begin(w * 0.3, h * 0.5, 1.0))
        self.assertFalse(ws._fast)
        ws.stroke_end()

    def test_paint_and_undo(self):
        doc, lp = self.ws.doc, self.ws.layers
        self.assertFalse(self.stroke())  # no paint layer selected: nothing happens
        lp.add_layer("paint")
        uid = doc.selected
        painted = 0
        for name in [ts.name for ts in doc.project.texture_sets]:  # one of the two quads is under the stroke
            doc.select_texture_set(name)
            if doc.ts.layers:
                doc.select(doc.ts.layers[-1].uid)
            else:
                lp.add_layer("paint")
            self.assertTrue(self.stroke())
            a = doc.session.store.read(doc.selected, "basecolor", 1001)
            painted += int((a[..., 3] > 0.5).sum()) if a is not None else 0
        self.assertGreater(painted, 50)
        steps = doc.undo.count()
        doc.undo.undo()
        doc.undo.undo()
        doc.undo.redo()
        doc.undo.redo()
        self.assertEqual(doc.undo.count(), steps)
        self.ws.set_display("channel:roughness")
        self.assertEqual(self.ws.display3d.currentData(), "channel:roughness")
        self.ws.step_display("channel", 1)
        self.ws.set_display("material")
        self.app.processEvents()

    def test_effects(self):
        doc, lp = self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        layer = doc.ts.layers[-1].uid
        lp.add_effect("levels")
        fx = doc.effect()
        self.assertEqual(fx.kind, "levels")
        self.assertEqual(self.ws.properties.widget().findChildren(type(self.ws.properties))[:0], [])
        self.assertIn("Levels", self.ws.docks["properties"].titleBarWidget().title.text())
        lp.add_effect("paint")
        self.assertEqual([e.kind for e in doc.ts.find(layer).effects], ["levels", "paint"])
        lp.move_effect(-1)
        self.assertEqual([e.kind for e in doc.ts.find(layer).effects], ["paint", "levels"])
        doc.select(layer)
        lp.add_mask(1.0)
        self.assertTrue(doc.mask_selected)
        lp.add_effect("fill")  # on the mask
        self.assertEqual(len(doc.ts.find(layer).mask.effects), 1)
        self.assertTrue(doc.mask_selected)
        self.assertIn("Fill", self.ws.docks["properties"].titleBarWidget().title.text())
        lp.add_effect("generator")
        gen = doc.effect()
        self.assertEqual(gen.kind, "generator")
        self.ws.properties._set_procedural(gen.uid, "node:paint_metal_edge_wear")
        self.assertEqual(doc.effect().procedural, "node:paint_metal_edge_wear")
        self.ws.refresh_materials()  # missing mesh maps: logged, not an error
        self.assertIn("mesh map", self.ws.log_view.toPlainText())
        lp.delete()
        doc.select(doc.ts.find(layer).mask.effects[0].uid)
        lp.delete()
        self.assertEqual(len(doc.ts.find(layer).mask.effects), 0)
        doc.undo.undo()
        self.assertEqual(len(doc.ts.find(layer).mask.effects), 1)
        self.app.processEvents()

    def _shaderball(self):
        """The light scene: a project on the shipped shader ball at 128 px, and a small material package written in
        the test's folder; returns (package path, graph id)."""
        from tests.scenes import MATERIAL_GRAPH, material_package, shaderball_session
        from trafilatura3d.editor.paint.document import PaintDocument
        self.ws.set_document(PaintDocument(shaderball_session(self.w.doc.ev)))
        self.ws.live.previewing = False
        self.ws.refresh_materials()
        self.app.processEvents()
        return str(material_package(self.dir / "material.trafilatura3d")), MATERIAL_GRAPH

    def test_material_mode(self):
        """A material graph added as a fill layer fills the channels of its outputs' usages; its parameters edit."""
        ws = self.ws
        path, gid = self._shaderball()
        doc = ws.doc
        ws.layers.add_material_layer(path, gid, "Material")
        layer = doc.layer()
        self.assertEqual(layer.resource["graph"], gid)
        ws.refresh_materials()
        a = doc.session.ev.read(doc.session.compiler(1001).channel(doc.ts, "basecolor"), "output")
        self.assertGreater(a[..., :3].astype(float).std(), 1.0)
        ws.properties._set_resource(layer.uid, {**layer.resource, "values": {}})
        self.assertEqual(doc.layer().resource["values"], {})

    def test_assets_drag_and_drop(self):
        """The Assets grid drags; dropped on the layer stack a material, a smart material and a filter make
        layers; a procedural dropped on a channel slot becomes its source; the types are all listed."""
        import json

        from PySide6.QtCore import QByteArray, QMimeData, QPointF, Qt
        from PySide6.QtGui import QDropEvent
        from trafilatura3d.editor.paint.assets import MIME, TYPES
        from trafilatura3d.paint import graph_nodes as dn
        from trafilatura3d.paint import smart
        path, gid = self._shaderball()
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        self.assertTrue(ws.assets.grid.dragEnabled())
        for kind in TYPES:
            ws.assets.set_kind(kind)  # every type lists (Textures and Materials may be empty)
        ws.assets.set_kind("Procedurals")
        self.assertGreater(ws.assets.grid.count(), 50)
        ws.assets.set_kind("Filters")
        self.assertGreater(ws.assets.grid.count(), 50)
        ws.assets.set_kind("Smart materials")
        self.assertGreaterEqual(ws.assets.grid.count(), 4)
        n = len(doc.ts.layers)
        lp.drop_asset({"kind": "material", "package": path, "graph": gid, "label": "Material"})
        self.assertEqual(len(doc.ts.layers), n + 1)
        lp.drop_asset({"kind": "smart", "path": str(smart.library([smart.SHIPPED])[0])})
        self.assertEqual(doc.ts.layers[-1].kind, "folder")
        lp.drop_asset({"kind": "filter", "id": "node:paint_invert", "label": "Invert"})
        top = doc.ts.layers[-1]
        self.assertEqual(top.blend_of("basecolor").mode, "passthrough")
        self.assertEqual(top.effects[0].procedural, "node:paint_invert")
        # a real drop event on the tree, from the MIME the grid makes
        mime = QMimeData()
        mime.setData(MIME, QByteArray(json.dumps({"kind": "generator", "id": "node:paint_light", "label": "Light"}).encode()))
        n = len(doc.ts.layers)
        lp.tree.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(len(doc.ts.layers), n + 1)
        # a procedural on a fill channel
        doc.select(doc.ts.layers[0].uid)
        fill = doc.layer()
        proc = dn.PREFIX + dn.procedurals()[0].type
        ws.properties._set_image(fill.uid, "roughness", proc)
        self.assertEqual(doc.ts.find(fill.uid).material["roughness"].procedural, proc)
        ws.refresh_materials()
        self.app.processEvents()
        # double click: a generator on the selected layer
        ws.use_asset({"kind": "generator", "id": dn.PREFIX + dn.generators()[0].type, "label": "G"})
        self.assertEqual(doc.effect().kind, "generator")
        ws.properties.rebuild()

    def test_resource_slots_and_picker(self):
        """A fill channel's slot shows its resource's thumbnail and name; its picker lists the textures and
        procedurals with their thumbnails, searches, takes a click; the effect and alpha choosers are slots too."""
        from PySide6.QtCore import Qt
        from trafilatura3d.editor.paint.assets import ResourceSlot
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        doc.session.resources.add("bricks", np.full((16, 16, 3), 0.5, np.float32))
        lp.add_layer("fill")
        uid = doc.layer().uid
        ws.properties.rebuild()
        slots = ws.properties.findChildren(ResourceSlot)
        base = next(s for s in slots if s.text() == "uniform color")
        self.assertTrue(base.icon().isNull())
        picker = base.open_picker()
        items = [picker.grid.item(i) for i in range(picker.grid.count())]
        self.assertEqual(items[0].text(), "uniform color")  # the slot empties first
        self.assertIn("bricks", [it.text() for it in items])
        self.assertGreater(len(items), 50)  # and the procedurals
        self.assertTrue(all(not it.icon().isNull() for it in items[1:]))
        self.assertTrue(picker.type_box.isVisibleTo(picker))  # Textures and Procedurals: the type
        picker.search.setText("brick")
        texts = [picker.grid.item(i).text() for i in range(picker.grid.count())]
        self.assertIn("bricks", texts)
        self.assertTrue(all("brick" in t.lower() for t in texts))
        picker.type_box.setCurrentText("Textures")
        picker.search.returnPressed.emit()  # Enter: the first match
        self.assertEqual(doc.ts.find(uid).material["basecolor"].image, "bricks")
        ws.properties.rebuild()
        base = next(s for s in ws.properties.findChildren(ResourceSlot) if s.text() == "bricks")
        self.assertFalse(base.icon().isNull())  # the texture's thumbnail
        picker = base.open_picker()
        picker._take(picker.grid.item(0).data(Qt.UserRole))  # uniform color again
        self.assertEqual(doc.ts.find(uid).material["basecolor"].image, "")
        lp.add_effect("filter", "node:paint_invert", "Invert")  # the filter chooser
        ws.properties.rebuild()
        chooser = next(s for s in ws.properties.findChildren(ResourceSlot) if s.text() == "Invert")
        self.assertFalse(chooser.icon().isNull())
        picker = chooser.open_picker()
        self.assertFalse(picker.type_box.isVisibleTo(picker))
        picker.close()

    def test_anchor_points(self):
        """Add anchor point from the Layers; a slot of a layer above offers it in its picker (one below does not),
        takes it with the Referenced channel and Go to anchor point; the anchor's Properties list the reference."""
        from PySide6.QtWidgets import QComboBox, QPushButton
        from trafilatura3d.editor.paint.assets import ResourceSlot
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        holder = doc.layer().uid
        lp.add_effect("anchor")
        anchor = doc.effect()
        self.assertEqual((anchor.kind, anchor.name), ("anchor", "Anchor Point 1"))
        lp.add_layer("fill")
        user = doc.layer().uid
        ws.properties.rebuild()
        base = next(s for s in ws.properties.findChildren(ResourceSlot) if s.text() == "uniform color")
        picker = base.open_picker()
        picker.type_box.setCurrentText("Anchor points")
        self.assertEqual([picker.grid.item(i).text() for i in range(picker.grid.count())],
                         ["uniform color", "Anchor Point 1"])
        picker._take(picker.grid.item(1).data(0x0100))
        src = doc.ts.find(user).material["basecolor"]
        self.assertEqual((src.anchor, src.anchor_channel), (anchor.uid, "basecolor"))
        ws.properties.rebuild()
        self.assertTrue(any(b.text() == "Go to anchor point" for b in ws.properties.findChildren(QPushButton)))
        self.assertTrue(ws.properties.findChildren(QComboBox))  # Referenced channel
        doc.select(holder)  # the layer under: the anchor of its own content is not offered
        ws.properties.rebuild()
        base = next(s for s in ws.properties.findChildren(ResourceSlot) if s.text() == "uniform color")
        picker = base.open_picker()
        self.assertNotIn("Anchor Point 1", [picker.grid.item(i).text() for i in range(picker.grid.count())])
        picker.close()
        doc.select(anchor.uid)
        ws.properties.rebuild()
        self.assertTrue(any(b.text().endswith("(Base color)") for b in ws.properties.findChildren(QPushButton)))
        ws.refresh_materials()
        self.app.processEvents()

    def test_duplicate_renews_effect_uids(self):
        """Duplicate: the copy's effects have their own uids (a paint effect its own pixels, an anchor point its
        own identity), as a paste."""
        from trafilatura3d.paint.paint import storage_for
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        layer = doc.layer().uid
        lp.add_effect("paint")
        pe = doc.effect().uid
        doc.session.store.write(pe, "roughness", 1001, np.ones((128, 128, 4), np.float32), storage_for("L8"))
        doc.select(layer)
        lp.add_effect("anchor")
        doc.select(layer)
        lp.duplicate()
        copy = doc.layer()
        self.assertNotEqual(copy.uid, layer)
        new = [e.uid for e in copy.effects]
        old = [e.uid for e in doc.ts.find(layer).effects]
        self.assertFalse(set(new) & set(old))
        self.assertIsNotNone(doc.session.store.get(new[0], "roughness", 1001))
        self.assertIsNot(doc.session.store.get(new[0], "roughness", 1001), doc.session.store.get(pe, "roughness", 1001))
        self.assertEqual(len(doc.ts.anchors()), 2)

    def test_viewport_reads_the_project_normal_format(self):
        """The Paint viewport shows the stack's normal channel in the project's Normal Map Format, whatever
        the Graph workspace's project setting (the graphs' 3D view) says; changing the project's format follows."""
        from trafilatura3d.paint.model import NORMAL_FORMATS
        v, project = self.ws.view3d, self.ws.doc.project
        for fmt in NORMAL_FORMATS:
            project.normal_format = fmt
            self.assertEqual(v.project_look().opengl_normals, not project.directx, fmt)

    def test_levels_editor(self):
        """The Levels effect's properties: the levels editor over the histogram of what the effect receives (the
        layer's content up to it; the mask in a mask), handles and fields edit its values (one History step per
        drag), Reset / Auto / Invert, Colors chooses the histogram's channel."""
        from PySide6.QtWidgets import QComboBox, QPushButton
        from tests.test_levels_editor import drag
        from trafilatura3d.editor.widgets import LevelsEditor
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        layer = doc.layer().uid
        ws.properties._set_image(layer, "basecolor", "node:perlin_noise")
        lp.add_effect("levels")
        uid = doc.effect().uid

        def levels():  # read again each time: the History restores the project
            return doc.ts.owner_of(uid)[1].levels

        def settle():  # the histogram asked for computed (a side cook, a step per turn of the event loop)
            import time
            t0 = time.perf_counter()
            self.app.processEvents()
            while (ws.live.busy or ws.live.side_busy) and time.perf_counter() - t0 < 30:
                self.app.processEvents()
        settle()
        ed = ws.properties.findChild(LevelsEditor)
        self.assertEqual(len(ed.series), 3)  # Base color, RGB
        self.assertEqual(ed.series[0][0].sum(), 128 * 128)
        steps = doc.undo.count()
        drag(ed.input, 1.0, 0.8)
        self.assertAlmostEqual(levels().in_high, 0.8, delta=0.01)
        drag(ed.output, 0.0, 0.1)
        self.assertAlmostEqual(levels().out_low, 0.1, delta=0.01)
        self.assertEqual(doc.undo.count(), steps + 2)
        ed.fields[1].setText("0.4")
        ed.fields[1].editingFinished.emit()
        self.assertEqual(levels().in_mid, 0.4)
        buttons = {b.text(): b for b in ws.properties.findChildren(QPushButton)}
        buttons["Invert"].click()
        self.assertEqual((levels().out_low, levels().out_high), (1.0, 0.1))
        self.assertEqual(ed.values[3:], [1.0, 0.1])
        buttons["Auto"].click()
        settle()
        got = []
        ws.properties._levels_histogram(doc.ts.owner_of(uid)[1], got.append)  # the darkest and brightest of RGB
        settle()
        _, lo, hi = got[0]
        self.assertEqual((levels().in_low, levels().in_high), (float(lo[:3].min()), float(hi[:3].max())))
        self.assertLess(levels().in_low, levels().in_high)
        buttons["Reset"].click()
        self.assertEqual([levels().in_low, levels().in_mid, levels().in_high, levels().out_low, levels().out_high],
                         [0.0, 0.5, 1.0, 0.0, 1.0])
        doc.undo.undo()
        self.assertEqual(levels().in_mid, 0.4)
        colors = next(c for c in ws.properties.findChildren(QComboBox) if c.toolTip() == "Colors")
        colors.setCurrentText("Red")
        settle()
        self.assertEqual(len(ws.properties.findChild(LevelsEditor).series), 1)
        doc.select(layer)
        lp.add_mask(1.0)  # a Levels in the mask: the mask's histogram, gray
        lp.add_effect("levels")
        settle()
        self.assertTrue(doc.ts.owner_of(doc.effect().uid)[2])
        ed = ws.properties.findChild(LevelsEditor)
        self.assertEqual(len(ed.series), 1)
        self.assertEqual(ed.series[0][0][:-1].sum(), 0)  # white

    def test_effect_drag_and_mask_menu(self):
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        lp.add_layer("fill")
        a = doc.layer().uid
        lp.add_effect("levels")
        lv = doc.effect().uid
        doc.select(a)
        lp.add_effect("filter", "node:paint_invert", "Invert")
        inv = doc.effect().uid
        self.assertEqual([e.uid for e in doc.ts.find(a).effects], [lv, inv])
        lp.drop_effect(inv, lv, None)  # rows top first: under the Levels row = under it in the stack
        self.assertEqual([e.uid for e in doc.ts.find(a).effects], [inv, lv])
        lp.add_layer("fill")
        b = doc.layer().uid
        lp.drop_effect(lv, into=b)  # onto another layer
        self.assertEqual([e.uid for e in doc.ts.find(b).effects], [lv])
        doc.undo.undo()
        self.assertEqual([e.uid for e in doc.ts.find(a).effects], [inv, lv])
        # the mask: menu actions, copy / paste with its painting, Alt+click isolation
        doc.select(a)
        lp.add_mask(0.0)
        import numpy as np
        from trafilatura3d.paint.paint import storage_for
        img = np.zeros((128, 128, 4), np.float32)
        img[:20] = 1
        doc.session.store.write(a, "mask", 1001, img, storage_for(None))
        lp.copy_mask(a)
        doc.select(b)
        lp.add_mask(1.0)
        lp.paste_mask(b)
        self.assertEqual(doc.ts.find(b).mask.background, 0.0)
        self.assertGreater(doc.session.store.read(b, "mask", 1001)[..., 3].sum(), 100)
        ws.isolate_mask(b)
        self.assertEqual(ws.display, f"mask:{b}")
        ws.refresh_materials()
        ws.isolate_mask(b)
        self.assertEqual(ws.display, "material")
        ws.isolate_mask(b)
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest
        QTest.mouseClick(lp.rows[a].thumb, Qt.LeftButton, Qt.NoModifier, QPoint(5, 5))
        self.assertEqual(ws.display, "material")  # a click on a layer's thumbnail: back to the material

    def test_multi_selection_copy_paste(self):
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        for _ in range(3):
            lp.add_layer("fill")
        a, b, c = [x.uid for x in doc.ts.layers]
        doc.select(c, multi=[a, c])
        lp.group()  # the two into a folder, the middle one stays
        kinds = [x.kind for x in doc.ts.layers]
        self.assertEqual(sorted(kinds), ["fill", "folder"])
        folder = next(x for x in doc.ts.layers if x.kind == "folder")
        self.assertEqual([x.uid for x in folder.children], [a, c])
        doc.undo.undo()
        doc.select(b, multi=[a, b])
        lp.copy()
        lp.paste()
        self.assertEqual(len(doc.ts.layers), 5)
        self.assertEqual(len(doc.multi), 2)
        lp.delete()  # both pasted ones
        self.assertEqual(len(doc.ts.layers), 3)
        doc.select(c)
        lp.add_effect("levels")
        lp.duplicate()  # an effect duplicates on its layer
        self.assertEqual(len(doc.ts.find(c).effects), 2)
        lp.move_many([a, b], c, "above")
        self.assertEqual([x.uid for x in doc.ts.layers], [c, a, b])
        # pressure toggles and a texture as alpha
        import numpy as np
        doc.session.resources.add("stamp", np.random.default_rng(0).random((32, 32, 3)).astype(np.float32))
        ws.tools.alphas["paint"] = "texture:stamp"
        self.assertTrue(ws._alpha_name().startswith("texture:stamp:"))
        lp.add_layer("paint")
        ws.properties.rebuild()

    def test_mesh_map_settings_conditional_rows(self):
        """Mesh map settings show a row only when the others say it applies (the bakers' pages: "if X is ..."):
        Curvature's Tonemapping Min/Max with Automatic Tonemapping off, and the panel follows the change."""
        from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel
        from trafilatura3d.paint import baking
        ws = self.ws
        ws.set_mode("baking")
        b = ws.baking
        b.list.setCurrentRow([s for s, *_ in baking.BAKERS].index("curvature"))

        def labels():
            form = b.settings.widget().layout()
            out = []
            for i in range(form.rowCount()):
                it = form.itemAt(i, QFormLayout.LabelRole)
                if it is not None and isinstance(it.widget(), QLabel):
                    out.append(it.widget().text())
            return out
        self.assertIn("Automatic Tonemapping bounds", labels())
        self.assertNotIn("Tonemapping Min", labels())
        form = b.settings.widget().layout()
        auto = next(form.itemAt(i, QFormLayout.FieldRole).widget() for i in range(form.rowCount())
                    if form.itemAt(i, QFormLayout.LabelRole) is not None
                    and form.itemAt(i, QFormLayout.LabelRole).widget().text() == "Automatic Tonemapping bounds")
        self.assertIsInstance(auto, QCheckBox)
        auto.setChecked(False)
        self.app.processEvents()
        self.assertIn("Tonemapping Min", labels())
        self.assertIn("Tonemapping Max", labels())
        self.assertNotIn("Normal Orientation", labels())  # needs an input normal map (not offered)
        # Common settings: Ignore backface, saved in the Texture Set's configuration
        common = b.common.widget().layout()
        box = next(common.itemAt(i, QFormLayout.FieldRole).widget() for i in range(common.rowCount())
                   if common.itemAt(i, QFormLayout.LabelRole) is not None
                   and common.itemAt(i, QFormLayout.LabelRole).widget().text() == "Ignore backface")
        self.assertFalse(baking.config(ws.doc.ts)["common"]["ignore_backface"])
        box.setChecked(True)
        self.assertTrue(baking.config(ws.doc.ts)["common"]["ignore_backface"])
        self.assertTrue(baking.settings_of(baking.config(ws.doc.ts)["common"], 1).ignore_backface)
        ws.set_mode("painting")

    def test_baking_mode(self):
        ws = self.ws
        ws.set_mode("baking")
        self.assertTrue(ws.bake_bar.isVisibleTo(ws))
        self.assertFalse(ws.docks["layers"].isVisibleTo(ws))
        self.assertTrue(ws.docks["bakers"].isVisibleTo(ws))
        self.app.processEvents()
        for key in ("bakers", "map_settings"):  # wide enough for the settings' rows (Secondary Rays…)
            self.assertGreaterEqual(ws.docks[key].width(), 280, key)
        self.assertFalse(ws.stroke_begin(10, 10, 1.0))  # no painting while baking
        b = ws.baking
        b._check(lambda on: False)
        it = b.list.item(0)  # Normal
        from PySide6.QtCore import Qt
        it.setCheckState(Qt.Checked)
        ws.texture_sets.unchecked = {"Right"}
        b.log.appendPlainText("an older bake")
        ws.bake()
        self.assertIn("normal", ws.doc.project.texture_set("Left").mesh_maps)
        self.assertNotIn("normal", ws.doc.project.texture_set("Right").mesh_maps)
        text = b.log.toPlainText()  # this bake only, map by map, and its outcome (also in the Log)
        self.assertNotIn("an older bake", text)
        self.assertIn("✓ Normal", text)
        self.assertTrue(text.endswith(" maps baked") and "\nBake finished, " in text)
        self.assertIn("Bake finished, ", ws.log_view.toPlainText())
        # the neutral material: the baked normal on the project mesh, its Roughness a constant
        ws.doc.select_texture_set("Left")
        index = ws._slot(ws.doc.ts, 1001)
        ws.bake_vis.roughness.set_value(0.8)
        ws.refresh_materials()
        spec = ws.view3d.renderer.materials[index]["spec"]
        self.assertIn("normal", spec)
        self.assertNotIn("baseColor", spec)
        self.assertAlmostEqual(spec["constants"]["roughness"][0], 0.8, 3)
        # a map's visualize icon: that map in the viewport, again: the neutral material; a map not baked: nothing
        b._visualize("normal")
        self.assertEqual(ws.display, "mesh:normal")
        self.assertIn("baseColor", ws.view3d.renderer.materials[index]["spec"])
        b._visualize("normal")
        self.assertEqual(ws.display, "material")
        b._visualize("bent_normals")
        self.assertEqual(ws.display, "material")
        ws.doc.undo.undo()
        self.assertNotIn("normal", ws.doc.project.texture_set("Left").mesh_maps)
        r = ws.view3d.renderer
        self.assertTrue(ws.bake_vis.isVisibleTo(ws))
        self.assertTrue(r.bake_cage and r.bake_high)  # the implicit cage and the low as high poly
        self.assertTrue(ws.view3d.look.show_cage)
        img = ws.view3d.render_image(64, 64)
        self.assertEqual(img.shape, (64, 64, 4))
        # a bake that fails says so, map by map
        b._save("High poly", lambda cfg: cfg["common"].update(use_low_as_high=False,
                                                                high_meshes=[str(self.dir / "missing.obj")]))
        ws.bake()
        text = b.log.toPlainText()
        self.assertIn("✗ Normal", text)
        self.assertIn("Bake failed: Left", text)
        self.assertIn("ERROR: Bake failed: Left", ws.log_view.toPlainText())
        self.assertFalse(r.bake_high)  # the missing high poly: nothing to show
        ws.doc.undo.undo()  # the low as high poly again: shown again (rebuilt on the next turn of the event loop)
        self.app.processEvents()
        self.assertTrue(r.bake_cage and r.bake_high)
        ws.set_mode("painting")
        self.assertTrue(ws.docks["layers"].isVisibleTo(ws))
        self.assertFalse(ws.bake_bar.isVisibleTo(ws))
        self.assertFalse(r.bake_cage)
        self.assertFalse(ws.view3d.look.show_cage)

    def test_straight_line_lazy_symmetry(self):
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        v = ws.view3d
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        w, h = v.physical_size()
        for name in [ts.name for ts in doc.project.texture_sets]:  # a paint layer in both quads
            doc.select_texture_set(name)
            lp.add_layer("paint")
        hit = None
        for name in [ts.name for ts in doc.project.texture_sets]:  # the quad under the left of the view
            doc.select_texture_set(name)
            doc.select(doc.ts.layers[-1].uid)
            if ws.stroke_begin(w * 0.3, h * 0.45, 1.0):
                ws.stroke_end()
                a = doc.session.store.read(doc.selected, "basecolor", 1001)
                if a is not None and (a[..., 3] > 0.5).any():
                    hit = name
                    break
        self.assertIsNotNone(hit)
        before = (doc.session.store.read(doc.selected, "basecolor", 1001)[..., 3] > 0.5).sum()
        ws.stroke_begin(w * 0.45, h * 0.45, 1.0, shift=True)  # a line from the last stroke's end
        ws.stroke_add(w * 0.2, h * 0.9, 1.0)  # moving while the line is drawn does nothing
        ws.stroke_end()
        after = (doc.session.store.read(doc.selected, "basecolor", 1001)[..., 3] > 0.5).sum()
        self.assertGreater(after, before * 2)
        ws.tools.lazy, ws.tools.lazy_distance = True, 1000.0  # the paint never catches up with the cursor
        n0 = doc.undo.count()
        ws.stroke_begin(w * 0.3, h * 0.55, 1.0)
        ws.stroke_add(w * 0.35, h * 0.55, 1.0)
        ws.stroke_end()
        ws.tools.lazy = False
        ws.sym_button.setChecked(True)
        self.assertTrue(ws.tools.symmetry)
        ws.hover((w * 0.3, h * 0.45))
        self.assertGreater(len(v.renderer.extra_lines), 100)  # two cursors
        self.assertGreaterEqual(doc.undo.count(), n0)

    def paint_layer_under(self, x, y):
        """A paint layer in both quads, the one under the view point selected; the view from above."""
        ws, doc, lp = self.ws, self.ws.doc, self.ws.layers
        v = ws.view3d
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        w, h = v.physical_size()
        for name in [ts.name for ts in doc.project.texture_sets]:
            doc.select_texture_set(name)
            lp.add_layer("paint")
        view, cam = ws._camera()
        hit = cam.hit(w * x, h * y)
        mat = int(round(hit[0][3])) - 1
        doc.select_texture_set(doc.session.materials[mat])
        doc.select(doc.ts.layers[-1].uid)
        return w, h

    def test_projection_tool(self):
        """A ramp projected from the viewport: the painted base color follows the screen's X; the preview
        overlay is set; S + drag moves the image."""
        ws, doc = self.ws, self.ws.doc
        ramp = np.zeros((16, 16, 4), np.float32)
        ramp[..., :3] = np.linspace(0, 1, 16)[None, :, None]
        ramp[..., 3] = 1
        doc.session.resources.add("ramp", ramp)
        w, h = self.paint_layer_under(0.3, 0.5)
        ws.set_tool("projection")
        st = ws.tools
        st.proj_source("basecolor").image = "ramp"
        st.projection.tiling = "No Tiling"
        st.projection.scale = 2.0  # the image twice the view's height: the whole view inside it
        st.brushes["projection"].size = 100.0
        ws.update_projection_preview()
        self.assertIsNotNone(ws.view3d.renderer.screen_overlay)
        self.assertTrue(ws.stroke_begin(w * 0.3, h * 0.5, 1.0))
        ws.stroke_add(w * 0.31, h * 0.5, 1.0)
        ws.stroke_end()
        a = doc.session.store.read(doc.selected, "basecolor", 1001)
        covered = a[..., 3] > 0.5
        self.assertTrue(covered.any())
        rows = np.where(covered.any(1))[0]
        row = a[rows[len(rows) // 2]]
        vals = row[row[:, 3] > 0.5, 0]
        self.assertGreater(float(vals.max() - vals.min()), 0.05)  # a ramp, not a uniform value
        img = ws.view3d.render_image(64, 64)
        ws.drag_projection(10, 0, 3, False)  # middle: move
        self.assertNotEqual(st.projection.offset[0], 0.0)
        self.assertFalse(np.array_equal(img, ws.view3d.render_image(64, 64)))
        ws.set_tool("paint")
        self.assertIsNone(ws.view3d.renderer.screen_overlay)

    def test_projection_falloff_and_image_controls(self):
        """Fade by angle: with a fade ending at 1 degree nothing faces the view enough and nothing is laid; with the
        defaults the stroke lands; the falloff overlay follows the tool; a dropped image goes on the Base color
        without asking; flips and rotation change the preview."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        uid = doc.layer().uid
        ws.drop_on_viewport({"kind": "alpha", "name": "Round hard"})
        self.assertEqual(ws.tools.proj_source("basecolor").image, "alpha Round hard")
        self.assertIsNotNone(ws.view3d.renderer.falloff_overlay)
        pr = ws.tools.projection
        pr.fade_start, pr.fade_end = 0.0, 1.0
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        self.assertFalse((doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.05).any())
        pr.fade_start, pr.fade_end = 60.0, 85.0
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        self.assertTrue((doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.5).any())
        before = ws.view3d.renderer.screen_overlay["tiling"]
        pr.flip_h = True
        ws.update_projection_preview()
        self.assertNotEqual(ws.view3d.renderer.screen_overlay["tiling"], before)
        pr.show_falloff = False
        ws.update_projection_preview()
        self.assertIsNone(ws.view3d.renderer.falloff_overlay)
        ws.set_tool("paint")
        self.assertIsNone(ws.view3d.renderer.falloff_overlay)
        self.assertIsNotNone(ws.view3d.render_image(64, 64))

    def test_projection_drop_on_viewport(self):
        """An alpha dragged from the Assets onto the 3D viewport (a real drop event): the Projection tool is
        equipped with it on the chosen channel and its image shows over the view."""
        import json
        from PySide6.QtCore import QByteArray, QMimeData, QPointF
        from PySide6.QtGui import QDropEvent
        from trafilatura3d.editor.paint.assets import MIME
        ws = self.ws
        ws.set_tool("paint")
        self.assertIsNone(ws.view3d.renderer.screen_overlay)
        ws.drop_on_viewport({"kind": "alpha", "name": "Cross"}, "roughness")
        self.assertEqual(ws.tools.tool, "projection")
        self.assertEqual(ws.tools.proj_source("roughness").image, "alpha Cross")
        self.assertIsNotNone(ws.view3d.renderer.screen_overlay)
        ws.set_tool("paint")
        mime = QMimeData()
        mime.setData(MIME, QByteArray(json.dumps({"kind": "texture", "name": "alpha Cross"}).encode()))
        filt = [o for o in ws.view3d.children() if type(o).__name__ == "_DropFilter"][0]
        filt.on_drop = lambda a: ws.drop_on_viewport(a, "basecolor")  # the channel menu skipped in a test
        ev = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        self.assertTrue(filt.eventFilter(ws.view3d, ev))  # the viewport's drop handler (no real drag offscreen)
        self.assertEqual(ws.tools.tool, "projection")
        self.assertEqual(ws.tools.proj_source("basecolor").image, "alpha Cross")

    def test_geometry_mask(self):
        """Two meshes on one material: clicking one in the Geometry mask edition excludes it (the fill lays nothing
        there, the row shows the count), Hide excluded geometry removes it from the view and the brush, Escape ends
        the edition."""
        from trafilatura3d.editor.paint.document import PaintDocument
        from trafilatura3d.paint.session import Session
        (self.dir / "m.mtl").write_text("newmtl M\nKd 1 1 1\n", encoding="utf-8")
        lines = ["mtllib m.mtl"]
        for i, (name, x0, u0) in enumerate((("A", -1.0, 0.0), ("B", 0.0, 0.5))):
            lines += [f"o {name}", f"v {x0} 0 -0.5", f"v {x0 + 1} 0 -0.5", f"v {x0 + 1} 0 0.5", f"v {x0} 0 0.5",
                      f"vt {u0} 1", f"vt {u0 + 0.5} 1", f"vt {u0 + 0.5} 0", f"vt {u0} 0", "vn 0 1 0", "usemtl M"]
            b = 4 * i + 1
            lines += [f"f {b}/{b}/{i + 1} {b + 3}/{b + 3}/{i + 1} {b + 2}/{b + 2}/{i + 1}",
                      f"f {b}/{b}/{i + 1} {b + 2}/{b + 2}/{i + 1} {b + 1}/{b + 1}/{i + 1}"]
        (self.dir / "ab.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
        ws = self.ws
        ws.doc.dirty = False
        ws.set_document(PaintDocument(Session.new(self.dir / "ab.obj", 128, evaluator=self.w.doc.ev)))
        doc = ws.doc
        v = ws.view3d
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        w, h = v.physical_size()
        ws.layers.add_layer("fill")
        uid = doc.layer().uid
        view, cam = ws._camera()
        tris = doc.session.triangles(doc.ts)
        hit = cam.hit(w * 0.3, h * 0.5)
        name = doc.session.objects[int(tris.object[tris.pick(hit[0][:3])])].name
        ws.edit_geometry_mask(uid)
        self.assertFalse(ws.stroke_begin(w * 0.3, h * 0.5, 1.0))
        self.assertEqual(doc.ts.find(uid).geometry_excluded, [name])
        self.assertIsNotNone(v.renderer.extra_lines)  # the excluded mesh outlined
        ws.key("Escape", set())
        self.assertIsNone(ws.geometry_edit)
        ws.refresh_materials()
        n_before = sum(len(m.submeshes) for m in v.renderer.meshes)
        ws.set_hide_excluded(True)
        self.assertLess(sum(len(m.submeshes) for m in v.renderer.meshes), n_before)
        self.assertLess(sum(len(s) for _b, _i, s in ws._meshes()), n_before)
        ws.set_hide_excluded(False)
        ws.layers.set_geometry_excluded(uid, [])
        self.assertEqual(doc.ts.find(uid).geometry_excluded, [])

    def test_polygon_fill(self):
        """Triangle Fill fills one triangle of the quad, Polygon Fill both (the quad), on the mask; the hover
        outlines the selection; one History step per click."""
        from trafilatura3d.paint.paint import MASK
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        ws.layers.add_mask(0.0)
        ws.set_tool("polygon_fill")
        uid = doc.layer().uid
        ws.tools.polygon_mode = "Triangle Fill"
        n0 = doc.undo.count()
        self.assertTrue(ws.stroke_begin(w * 0.3, h * 0.5, 1.0))
        ws.stroke_end()
        self.assertEqual(doc.undo.count(), n0 + 1)
        tri = (doc.session.store.read(uid, MASK, 1001)[..., 3] > 0.5).mean()
        self.assertTrue(0.3 < tri < 0.7, tri)  # half the quad's UV square
        ws.tools.polygon_mode = "Polygon Fill"
        ws.hover((w * 0.3, h * 0.5))
        self.assertEqual(len(ws.view3d.renderer.extra_lines), 12)  # two triangles' edges
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        self.assertGreater((doc.session.store.read(uid, MASK, 1001)[..., 3] > 0.5).mean(), 0.95)

    def test_polygon_fill_rectangle(self):
        """A rectangle dragged: Intersection takes the triangles it touches, Contained center those whose center it
        holds; nothing filled until the release; the Material's channels and the Blending (Multiply, opacity)."""
        from trafilatura3d.paint.paint import MASK
        ws, doc = self.ws, self.ws.doc
        self.paint_layer_under(0.3, 0.5)
        view = ws.view3d.brush_view()

        def screen(p):
            c = np.asarray(view.view_proj, np.float64) @ np.array([*p, 1.0])
            return (c[0] / c[3] * 0.5 + 0.5) * view.width, (0.5 - c[1] / c[3] * 0.5) * view.height
        cx, cy = screen((-0.5, 0.0, 0.0))  # the left quad's center, on its diagonal
        ax, ay = screen((-0.98, 0.0, -0.48))
        bx, by = screen((-0.02, 0.0, 0.48))
        doc.select_texture_set("Left")
        doc.select(doc.ts.layers[-1].uid)
        uid = doc.layer().uid
        ws.layers.add_mask(0.0)
        ws.set_tool("polygon_fill")
        t = ws.tools
        t.polygon_mode, t.polygon_hit = "Triangle Fill", "Contained center"

        def drag(x0, y0, x1, y1):
            self.assertTrue(ws.stroke_begin(x0, y0, 1.0))
            ws.stroke_add(x1, y1, 1.0)
            self.assertIsNotNone(ws.view3d.renderer.extra_lines)  # the rectangle (and what it takes) shown
            ws.stroke_end()

        def covered(ch=MASK):
            a = doc.session.store.read(uid, ch, 1001)
            return 0.0 if a is None else float((a[..., 3] > 0.5).mean())
        drag(cx - 6, cy - 6, cx + 6, cy + 6)  # no triangle's center in it
        self.assertEqual(covered(), 0.0)
        t.polygon_hit = "Intersection"
        self.assertTrue(ws.stroke_begin(cx - 6, cy - 6, 1.0))
        ws.stroke_add(cx + 6, cy + 6, 1.0)
        self.assertEqual(covered(), 0.0)  # filled on the release only
        ws.stroke_end()
        self.assertGreater(covered(), 0.95)  # both triangles touch it
        doc.undo.undo()
        # Visible polygons only: a point of the surface is seen, one under it is not; nothing seen, nothing taken
        cam_view, cam = ws._camera()
        m = doc.session.material_index(doc.ts)
        self.assertEqual(list(cam.sees(cam_view, [(-0.5, 0.0, 0.0), (-0.5, -0.3, 0.0)], m, 1e-3)), [True, False])
        tris = doc.session.triangles(doc.ts)
        rect = (ax, ay, bx, by)
        self.assertEqual(len(tris.in_rect(cam_view.view_proj, (cam_view.width, cam_view.height), rect,
                                          "Intersection", lambda p: np.zeros(len(p), bool))), 0)
        self.assertEqual(len(tris.in_rect(cam_view.view_proj, (cam_view.width, cam_view.height), rect,
                                          "Intersection", lambda p: cam.sees(cam_view, p, m, 1e-3))), 2)
        t.polygon_hit = "Contained center"
        drag(ax, ay, bx, by)
        self.assertGreater(covered(), 0.95)
        # on the layer: its own channels, blended
        doc.select(uid)
        t.values["basecolor"] = [1.0, 1.0, 1.0]
        t.values["roughness"] = [0.25] * 3
        drag(ax, ay, bx, by)
        self.assertAlmostEqual(float(doc.session.store.read(uid, "roughness", 1001)[64, 64, 0]), 0.25, delta=0.01)
        t.polygon_disabled = {"roughness"}
        t.polygon_blend, t.values["basecolor"] = "multiply", [0.5, 0.5, 0.5]
        drag(ax, ay, bx, by)
        drag(ax, ay, bx, by)  # 0.5 x 0.5 in linear
        c = doc.session.store.read(uid, "basecolor", 1001)[64, 64]
        self.assertAlmostEqual(float(c[0]), 0.235, delta=0.02)
        self.assertAlmostEqual(float(doc.session.store.read(uid, "roughness", 1001)[64, 64, 0]), 0.25, delta=0.01)
        t.polygon_blend, t.polygon_opacity, t.values["basecolor"] = "normal", 0.5, [1.0, 1.0, 1.0]
        drag(ax, ay, bx, by)
        self.assertAlmostEqual(float(doc.session.store.read(uid, "basecolor", 1001)[64, 64, 0]), 0.6175, delta=0.02)

    def test_material_picker(self):
        """P equips the picker; a click takes the material under the cursor (a fill's roughness) into the paint
        tool's values and the paint tool comes back."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        ts = doc.ts
        fill = next(l for l in ts.layers if l.kind == "fill") if any(l.kind == "fill" for l in ts.layers) else None
        if fill is None:
            ws.layers.add_layer("fill")
            fill = doc.layer()
        fill.material["roughness"].color = [0.25] * 3
        ws.set_tool("paint")
        ws.set_tool("material_picker")
        self.assertFalse(ws.stroke_begin(w * 0.3, h * 0.5, 1.0))
        self.assertEqual(ws.tools.tool, "paint")
        self.assertAlmostEqual(ws.tools.values["roughness"][0], 0.25, delta=2 / 255)

    def test_quick_mask(self):
        """U: painting goes to the quick mask (no History step); U again: strokes skip the protected texels; I
        inverts; Y clears."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        uid = doc.layer().uid
        ws.tools.polygon_mode = "UV Chunk Fill"
        ws.set_tool("polygon_fill")
        ws.toggle_quick_mask()
        n0 = doc.undo.count()
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)  # the whole quad protected
        ws.stroke_end()
        self.assertEqual(doc.undo.count(), n0)
        self.assertIn("baseColor", ws._slot_jobs(doc.ts, 1001))
        ws.toggle_quick_mask()
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        self.assertFalse((doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.01).any())  # nothing landed
        ws.toggle_quick_mask()
        ws.invert_quick_mask()
        ws.toggle_quick_mask()
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        self.assertGreater((doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.5).mean(), 0.95)
        ws.clear_quick_mask()
        self.assertFalse(ws.quick_edit)
        self.assertEqual(ws._quick_masks, {})

    def test_clone_and_smudge(self):
        """Clone copies the painting from the V source's offset; Smudge drags it along the stroke; both are one
        History step; without a source Clone does nothing."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        uid = doc.layer().uid
        ws.tools.values["basecolor"] = [1.0, 0.0, 0.0]
        ws.tools.brushes["paint"].size = 8.0
        ws.set_tool("paint")
        ws.stroke_begin(w * 0.25, h * 0.5, 1.0)  # a red dot on the left
        ws.stroke_end()
        red0 = (doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.5).sum()
        ws.set_clone(True)
        ws.tools.brushes["clone"].size = 8.0
        self.assertFalse(ws.stroke_begin(w * 0.35, h * 0.5, 1.0))  # no source yet
        self.assertTrue(ws.set_clone_source(w * 0.25, h * 0.5))
        n0 = doc.undo.count()
        self.assertTrue(ws.stroke_begin(w * 0.35, h * 0.5, 1.0))  # the dot cloned to the right
        ws.stroke_end()
        self.assertEqual(doc.undo.count(), n0 + 1)
        a = doc.session.store.read(uid, "basecolor", 1001)
        self.assertGreater((a[..., 3] > 0.5).sum(), red0 * 1.5)
        ws.set_tool("smudge")
        ws.tools.brushes["smudge"].size = 8.0
        before = (a[..., 3] > 0.05).sum()
        ws.stroke_begin(w * 0.25, h * 0.5, 1.0)  # from the dot outwards: color dragged along
        for k in range(1, 20):
            ws.stroke_add(w * (0.25 - k * 0.004), h * 0.5, 1.0)
        ws.stroke_end()
        after = (doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.05).sum()
        self.assertGreater(after, before)
        doc.undo.undo()
        np.testing.assert_array_equal(doc.session.store.read(uid, "basecolor", 1001), a)

    def test_path_tool(self):
        """Clicks make a path (one point per click, Enter ends), it paints along itself; a click on its line
        inserts a point, dragging moves it, Delete removes it, double click makes a corner; the Paths panel lists
        it, hiding it removes its pixels; one History step per edit, undo restores."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        ws.set_tool("path")
        self.assertTrue(ws.paths_panel.isVisibleTo(ws))
        ws.path_tool.template.brush = {}
        ws.tools.brushes["path"].size = 10.0
        n0 = doc.undo.count()
        self.assertTrue(ws.stroke_begin(w * 0.22, h * 0.5, 1.0))
        ws.stroke_end()
        self.assertTrue(ws.stroke_begin(w * 0.36, h * 0.5, 1.0))  # the same quad
        ws.stroke_end()
        ws.key("Enter", set())
        self.assertFalse(ws.path_tool.editing)
        self.assertEqual(len(doc.layer().paths), 1)
        self.assertIsNotNone(ws.path_tool.overlay())
        p = doc.layer().paths[0]
        self.assertEqual(len(p.points), 2)
        self.assertEqual(doc.undo.count(), n0 + 2)
        cov = read(doc.session.compiler(1001), doc.session.compiler(1001).channel(doc.ts, "basecolor"))[..., 3]
        self.assertGreater((cov > 0.5).sum(), 50)
        self.assertEqual(ws.paths_panel.list.count(), 1)
        ws.stroke_begin(w * 0.29, h * 0.5, 1.0)  # on the line: a point inserted, then dragged
        ws.stroke_add(w * 0.29, h * 0.45, 1.0)
        ws.stroke_end()
        self.assertEqual(len(doc.layer().paths[0].points), 3)
        ws.double_click(w * 0.29, h * 0.45)
        self.assertEqual(doc.layer().paths[0].points[1][7], 1.0)
        ws.key("Delete", set())
        self.assertEqual(len(doc.layer().paths[0].points), 2)
        ws.path_tool.toggle(p.uid, "visible")
        cov2 = read(doc.session.compiler(1001), doc.session.compiler(1001).channel(doc.ts, "basecolor"))[..., 3]
        self.assertLess((cov2 > 0.5).sum(), 5)
        self.assertEqual(doc.undo.count(), n0 + 7)  # new, add, insert, move (one), corner, delete, hide
        doc.undo.setIndex(n0)
        self.assertEqual(len(doc.layer().paths), 0)

    def test_manipulator_and_warp(self):
        """A fill with a Planar projection selected: the contextual toolbar shows the manipulator, its lines are
        drawn; dragging the X axis moves the projection; Rotate turns it; Surface puts it on the mesh; mirror and
        reset. Warp: Edit vertices drags a grid point, split doubles the cells, reset restores the grid."""
        ws, doc = self.ws, self.ws.doc
        v = ws.view3d
        v.yaw, v.pitch = 0.4, 0.9
        ws.frame()
        w, h = v.physical_size()
        ws.layers.add_layer("fill")
        uid = doc.layer().uid
        pr = lambda: doc.ts.find(uid).projection  # noqa: E731

        def set_mode(mode):
            def do():
                doc.ts.find(uid).projection.mode = mode
            doc.edit("p", do)
            ws._selection_changed()
        set_mode("Planar projection")
        self.assertTrue(ws.manip_bar.isVisibleTo(ws))
        self.assertFalse(ws.brush_bar.isVisibleTo(ws))
        self.assertIsNotNone(v.renderer.extra_lines)
        m = ws.manip
        origin, axes, length = m._gizmo_axes(pr())
        a, b = m._screen(origin), m._screen(origin + axes[:, 0] * length)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        self.assertTrue(ws.stroke_begin(*mid, 1.0))
        ws.stroke_add(mid[0] + (b[0] - a[0]), mid[1] + (b[1] - a[1]), 1.0)
        ws.stroke_end()
        self.assertGreater(pr().offset3d[0], 0.01)
        self.assertAlmostEqual(pr().offset3d[1], 0.0, places=5)
        ws.manip_bar.set_mode("Rotate")
        origin, axes, length = m._gizmo_axes(pr())
        ring = m._screen(origin + axes[:, 1] * length)  # on the X axis' circle (turning about X)
        ws.stroke_begin(*ring, 1.0)
        c = m._screen(origin)
        ws.stroke_add(c[0] + (ring[1] - c[1]), c[1] - (ring[0] - c[0]), 1.0)  # a quarter turn around the center
        ws.stroke_end()
        self.assertGreater(max(abs(x) for x in pr().rotation3d), 5.0)
        ws.manip_bar.set_mode("Surface")
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        from trafilatura3d.editor.paint.manipulator import euler_matrix
        from trafilatura3d.paint.warp import points
        self.assertGreater(abs(euler_matrix(pr().rotation3d)[1, 2]), 0.99)  # Z along the quad's normal (Y)
        m.mirror(0)
        self.assertLess(pr().scale3d[0], 0)
        m.reset("transform")
        self.assertEqual(pr().offset3d, [0.0, 0.0, 0.0])
        set_mode("Warp projection")
        ws.manip_bar.warp_box.setCurrentText("Edit vertices")
        self.assertTrue(m.warp_edit)
        pts = points(pr()).reshape(-1, 3)
        s = m._screen(m._local_to_world(pr(), pts[5]))
        ws.stroke_begin(*s, 1.0)
        ws.stroke_add(s[0] + 15, s[1] + 5, 1.0)
        ws.stroke_end()
        self.assertTrue(pr().warp_points)
        self.assertEqual(m.selected_points, {5})
        m.split(rows=True, cols=True)
        self.assertEqual((pr().warp_rows, pr().warp_cols), (6, 6))
        self.assertEqual(len(pr().warp_points), 49)
        m.reset("all")
        self.assertEqual(pr().warp_points, [])

    def test_stencil(self):
        """A black stencil over the view masks the Paint tool's stroke; N held ignores it; the preview shows it."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        uid = doc.layer().uid
        black = np.zeros((8, 8, 4), np.float32)
        black[..., 3] = 1
        doc.session.resources.add("black", black)
        ws.set_tool("paint")
        from trafilatura3d.paint.model import ChannelSource
        ws.tools.stencils["paint"] = ChannelSource([1, 1, 1], image="black")
        ws.tools.stencil_settings().tiling = "H and V Tiling"
        ws.update_projection_preview()
        self.assertIsNotNone(ws.view3d.renderer.screen_overlay)
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        a = doc.session.store.read(uid, "basecolor", 1001)
        self.assertFalse((a[..., 3] > 0.01).any())  # masked everywhere
        ws.view3d.held.add("n")
        ws.stroke_begin(w * 0.3, h * 0.5, 1.0)
        ws.stroke_end()
        ws.view3d.held.discard("n")
        self.assertTrue((doc.session.store.read(uid, "basecolor", 1001)[..., 3] > 0.5).any())

    def test_window_fits_a_laptop_screen(self):
        """The bars over the views do not set the window's minimum width: it fits 1280 px, Paint and Graph."""
        self.assertLessEqual(self.w.minimumSizeHint().width(), 1280)
        self.assertLessEqual(self.w.workspace.minimumSizeHint().width(), 1280)

    def test_path_alpha_stitches(self):
        """The Path's Properties choose the alpha (the next path's, then the selected path's): a Stitch alpha with
        a wide Spacing along the path gives separate stitches (gaps along the line), unlike the round default."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        uid = doc.layer().uid
        ws.set_tool("path")
        from PySide6.QtWidgets import QLabel
        sections = [lb.text() for lb in ws.properties.findChildren(QLabel) if lb.property("section")]
        self.assertIn("Alpha", sections)  # the section is there
        ws.tools.alphas["path"] = "Stitch"
        b = ws.tools.brushes["path"]
        b.size, b.spacing, b.follow_path = 10.0, 250.0, True
        for x in (0.15, 0.45):
            self.assertTrue(ws.stroke_begin(w * x, h * 0.5, 1.0))
            ws.stroke_end()
        ws.key("Enter", set())
        p = doc.ts.find(uid).paths[0]
        self.assertEqual(p.alpha, "Stitch")
        cov = read(doc.session.compiler(1001), doc.session.compiler(1001).channel(doc.ts, "basecolor"))[..., 3]
        row = cov[cov.sum(1).argmax()]  # the row along the path
        on = row > 0.3
        self.assertGreater(on.sum(), 5)
        runs = np.diff(on.astype(int))
        self.assertGreaterEqual((runs == 1).sum(), 2)  # several separate stitches
        ws.path_tool.set_attr("alpha", "Round hard")
        self.assertEqual(doc.ts.find(uid).paths[0].alpha, "Round hard")

    def test_paths_on_masks_and_paint_effects(self):
        """A path on a layer's mask paints the mask (any layer kind); a path on a paint effect paints the effect;
        a fill layer's content (no painting) takes none."""
        ws, doc = self.ws, self.ws.doc
        w, h = self.paint_layer_under(0.3, 0.5)
        ws.layers.add_layer("fill")
        fill = doc.layer()
        ws.set_tool("path")
        self.assertIsNone(ws.path_tool.owner())  # the fill layer's content
        ws.layers.add_mask(0.0)
        self.assertTrue(doc.mask_selected)
        owner = ws.path_tool.owner()
        self.assertTrue(owner[2])
        for x in (0.22, 0.36):
            self.assertTrue(ws.stroke_begin(w * x, h * 0.5, 1.0))
            ws.stroke_end()
        ws.key("Enter", set())
        self.assertEqual(len(doc.ts.find(fill.uid).mask.paths), 1)
        m = read(doc.session.compiler(1001), doc.session.compiler(1001).mask(doc.ts, doc.ts.find(fill.uid)))
        self.assertGreater((m[..., 0] > 0.5).sum(), 20)  # the black mask painted white along the path
        doc.select(fill.uid)
        ws.layers.add_effect("paint")
        fx = doc.effect()
        self.assertEqual(ws.path_tool.owner()[1], fx.uid)
        for x in (0.22, 0.36):
            ws.stroke_begin(w * x, h * 0.45, 1.0)
            ws.stroke_end()
        ws.key("Enter", set())
        self.assertEqual(len(doc.ts.owner_of(fx.uid)[1].paths), 1)
        d = doc.project.to_dict()
        from trafilatura3d.paint.model import Project
        self.assertEqual(len(Project.from_dict(d).texture_set(doc.ts.name).find(fill.uid).mask.paths), 1)

    def _show_2d(self):
        """The 2D view shown and drawn once (its scale comes from the texture it shows)."""
        self.ws.set_viewport_mode("3D/2D")
        self.ws._update_view2d()
        self.app.processEvents()
        self.ws.view2d.force_draw()

    def test_paint_in_2d_view(self):
        ws, doc = self.ws, self.ws.doc
        ws.layers.add_layer("paint")
        v2 = ws.view2d
        self._show_2d()
        w, y = v2.width(), v2.height() / 2  # across the 0-1 range, fitted at 90% of the short side
        side = min(v2.width(), v2.height()) * 0.9
        a, b = w / 2 - side * 0.3, w / 2 + side * 0.3
        v2._on_event({"event_type": "pointer_down", "x": a, "y": y, "button": 1, "modifiers": ()})
        for k in range(1, 11):
            v2._on_event({"event_type": "pointer_move", "x": a + (b - a) * k / 10, "y": y, "button": 1,
                          "modifiers": ()})
        v2._on_event({"event_type": "pointer_up", "x": b, "y": y, "button": 1, "modifiers": ()})
        img = doc.session.store.read(doc.selected, "basecolor", 1001)
        self.assertIsNotNone(img)
        self.assertGreater((img[..., 3] > 0.5).mean(), 0.02)
        self.assertEqual(doc.undo.undoText(), "Paint stroke")

    def test_uv_tiles_crossing_warning(self):
        """A UV Tile project on a mesh with triangles across a tile border: the Log warns, naming the mesh."""
        from tests.test_paint import write_cross_obj
        from trafilatura3d.editor.paint.document import PaintDocument
        from trafilatura3d.paint.model import UV_TILE_MODES
        from trafilatura3d.paint.session import Session
        self.ws.doc.dirty = False
        s = Session.new(write_cross_obj(self.dir), 128, evaluator=self.w.doc.ev, uv_tiles=UV_TILE_MODES[0])
        self.ws.set_document(PaintDocument(s))
        self.ws.refresh_materials()
        log = self.ws.log_view.toPlainText()
        self.assertIn("Warning: Mesh Cross: 2 triangle(s) span several UV tiles", log)
        self.assertNotIn("ERROR", log)

    def test_uv_tiles_paint_across(self):
        """A UV Tile project: a renderer material per (mesh material, tile); a stroke over the two quads of Body
        (tiles 1001 and 1002) paints both tiles, in the 3D view and in the 2D view where the tiles lie side by side."""
        from tests.test_paint import write_udim_obj
        from trafilatura3d.editor.paint.document import PaintDocument
        from trafilatura3d.paint.model import UV_TILE_MODES
        from trafilatura3d.paint.session import Session
        ws = self.ws
        ws.doc.dirty = False
        s = Session.new(write_udim_obj(self.dir), 128, evaluator=self.w.doc.ev, uv_tiles=UV_TILE_MODES[0])
        ws.set_document(PaintDocument(s))
        doc = ws.doc
        self.assertEqual(ws._slots, [(m, t) for m in range(2) for t in (1001, 1002, 1011)])
        self.assertTrue(ws.tile_info_button.isVisibleTo(ws))
        doc.select_texture_set("Body")
        ws.layers.add_layer("paint")
        uid = doc.layer().uid
        v = ws.view3d
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        view, _cam = ws._camera()
        tris = s.triangles(doc.ts)
        ends = []
        for k in (0, 1):  # the centers of the two quads on screen
            c = tris.pos[np.floor(tris.uv.mean(1)[:, 0]) == k].reshape(-1, 3).mean(0)  # the quad in tile 1001 + k
            clip = np.asarray(view.view_proj, np.float64) @ np.array([*c, 1.0])
            ends.append(((clip[0] / clip[3] * 0.5 + 0.5) * view.width, (0.5 - clip[1] / clip[3] * 0.5) * view.height))
        ws.set_tool("paint")
        ws.stroke_begin(*ends[0], 1.0)
        for i in range(1, 21):
            ws.stroke_add(ends[0][0] + (ends[1][0] - ends[0][0]) * i / 20, ends[0][1], 1.0)
        ws.stroke_end()
        for udim in (1001, 1002):
            self.assertGreater((s.store.read(uid, "basecolor", udim)[..., 3] > 0.5).mean(), 0.01, udim)
        doc.undo.undo()  # one History step for both tiles
        for udim in (1001, 1002):
            self.assertFalse((s.store.read(uid, "basecolor", udim)[..., 3] > 0.5).any(), udim)
        ws.layers.add_layer("paint")
        uid = doc.layer().uid
        ws.set_viewport_mode("3D/2D")
        ws._update_view2d()
        self.app.processEvents()
        v2 = ws.view2d
        v2.force_draw()
        self.assertEqual([(o, k) for _t, _g, o, k in v2.shown()], [((0, 0), 1.0), ((128, 0), 1.0)])
        self.assertTrue(ws.stroke2d_begin(100.0, 64.0, 1.0, 6.0))  # from 1001 into 1002 (texels of the grid)
        ws.stroke2d_add(156.0, 64.0, 1.0, 6.0)
        ws.stroke_end()
        a, b = s.store.read(uid, "basecolor", 1001), s.store.read(uid, "basecolor", 1002)
        self.assertGreater(a[64, 110, 3], 0.5)
        self.assertGreater(b[64, 20, 3], 0.5)

    def test_uv_tile_panels(self):
        """UV Tile project panels: a bake check box per tile (Baking mode), Fill (match per UV Tile) among the
        projections, a resolution per tile (Texture Set settings), Hide excluded geometry hiding excluded tiles."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QComboBox, QLabel
        from tests.test_paint import write_udim_obj
        from trafilatura3d.editor.paint.document import PaintDocument
        from trafilatura3d.paint.model import MATCH_PER_TILE, UV_TILE_MODES
        from trafilatura3d.paint.session import Session
        ws = self.ws
        ws.doc.dirty = False
        s = Session.new(write_udim_obj(self.dir), 128, evaluator=self.w.doc.ev, uv_tiles=UV_TILE_MODES[0])
        ws.set_document(PaintDocument(s))
        doc = ws.doc
        doc.select_texture_set("Body")
        ws.set_mode("baking")
        tree = ws.texture_sets.tree
        body = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                    if tree.topLevelItem(i).data(1, Qt.UserRole) == "Body")
        self.assertEqual([body.child(i).text(1) for i in range(body.childCount())], ["1001", "1002"])
        body.child(1).setCheckState(1, Qt.Unchecked)
        self.assertEqual(ws.texture_sets.checked(), [("Head", [1011]), ("Body", [1001])])
        ws.set_mode("painting")
        ws.layers.add_layer("paint")
        paint = doc.selected
        ws.layers.add_layer("fill")
        self.app.processEvents()
        ws.layers.update_thumbnails()  # fill layers computed (the same on every tile), the rest icons
        ws.live.refresh_now()  # the refresh's last step computes them
        from trafilatura3d.editor.paint.layers import THUMB
        self.assertEqual(ws.layers.rows[doc.selected].thumb.pixmap().width(), THUMB - 2)
        self.assertLess(ws.layers.rows[paint].thumb.pixmap().width(), THUMB - 2)  # the icon
        fill = doc.selected
        ws.layers.add_layer("folder")
        self.app.processEvents()
        ws.layers.update_thumbnails()
        self.assertTrue(ws.layers.rows[doc.selected].thumb.pixmap().isNull())  # only the row's folder icon
        doc.select(fill)
        self.app.processEvents()
        combos =[c for c in ws.properties.findChildren(QComboBox) if c.findText(MATCH_PER_TILE) >= 0]
        self.assertEqual(len(combos), 1)
        ws.ts_settings.rebuild()
        own = next(c for c in ws.ts_settings.findChildren(QComboBox) if c.findText("Texture Set size") >= 0)
        own.setCurrentIndex(own.findData(256))
        self.assertEqual(doc.project.texture_set("Body").tile_sizes, {"1001": [256, 256]})
        layer = doc.layer()
        ws.layers.set_geometry_type(layer.uid, "UV Tiles")
        ws.layers.set_geometry_tiles(layer.uid, [1002])
        ws.set_hide_excluded(True)
        body_m = s.material_index(doc.ts)
        self.assertTrue(ws._hidden(0, body_m, 1002))
        self.assertFalse(ws._hidden(0, body_m, 1001))
        self.assertTrue(any(lbl.text() == "UV Tile" for lbl in ws.ts_settings.findChildren(QLabel)))

    def test_uv_tile_settings(self):
        """New Project: Use UV Tile workflow enables the UV Tile mode (the legacy one is not offered); Project
        configuration shows the project's settings without editing."""
        from trafilatura3d.editor.paint.new_project import NewProjectDialog, ProjectConfiguration, UVTileSettings
        from trafilatura3d.paint.model import UV_TILE_MODES
        dlg = NewProjectDialog(self.w)
        u = dlg.uv_tiles
        self.assertEqual(dlg.values()["uv_tiles"], "")
        self.assertFalse(u.radios[0].isEnabled())
        u.use.setChecked(True)
        self.assertTrue(u.radios[0].isEnabled())
        self.assertFalse(u.radios[1].isEnabled())
        self.assertEqual(dlg.values()["uv_tiles"], UV_TILE_MODES[0])
        dlg.deleteLater()
        shown = ProjectConfiguration(self.ws).findChild(UVTileSettings)
        self.assertFalse(shown.use.isEnabled() or shown.use.isChecked() or shown.radios[0].isEnabled())

    def test_save_open_export(self):
        from trafilatura3d.editor.paint.export_dialog import ExportDialog
        lp = self.ws.layers
        lp.add_layer("fill")
        path = self.dir / "proj.trafilatura3dpaint"
        self.assertTrue(self.ws.doc.save(path) is None)
        self.ws.doc.dirty = False
        self.assertTrue(self.ws.close_project())
        self.assertTrue(self.ws.open_path(path))
        self.assertEqual(len(self.ws.doc.ts.layers), 1)
        dlg = ExportDialog(self.ws)
        dlg.out_dir.setText(str(self.dir / "tex"))
        dlg._export()
        files = list((self.dir / "tex").glob("*.png"))
        self.assertGreater(len(files), 4)
        dlg.close()

    def test_project_configuration_and_export_mesh(self):
        """Reimport keeps the layers and adds a Texture Set for a new material; a missing material refuses it; the
        normal format is an undoable edit; Export mesh writes the file's coordinates back."""
        from trafilatura3d import meshio
        from trafilatura3d.editor.paint.new_project import ProjectConfiguration
        ws = self.ws
        ws.layers.add_layer("fill")
        painted = ws.doc.ts.name
        obj = self.dir / "two.obj"
        text = obj.read_text(encoding="utf-8")
        three = self.dir / "three.obj"
        three.write_text(text + "o Extra\nv 0 1 0\nv 1 1 0\nv 1 1 1\nvt 0 0\nvt 1 0\nvt 1 1\nvn 0 1 0\n"
                         "usemtl Extra\nf 9/9/3 10/10/3 11/11/3\n", encoding="utf-8")
        (self.dir / "two.mtl").write_text((self.dir / "two.mtl").read_text(encoding="utf-8") + "newmtl Extra\nKd 1 1 1\n", encoding="utf-8")
        dlg = ProjectConfiguration(ws)
        dlg.file.setText(str(three))
        dlg.reimport.setChecked(True)
        dlg.normal.setCurrentText("DirectX (X+, Y-, Z+)")
        self.assertTrue(dlg.apply())
        self.assertTrue(ws.doc.project.directx)
        self.assertEqual([t.name for t in ws.doc.project.texture_sets][-1], "Extra")
        self.assertEqual({t.name for t in ws.doc.project.texture_sets}, {"Left", "Right", "Extra"})
        self.assertEqual(len(ws.doc.project.texture_set(painted).layers), 1)
        one = self.dir / "one.obj"
        one.write_text(text.split("o Right")[0], encoding="utf-8")
        self.assertRaises(Exception, ws.doc.session.reimport, one)  # Right and Extra would be gone
        out = self.dir / "mesh.obj"
        self.assertTrue(ws.export_mesh(out))
        objects, _m, _c = meshio.load(out)
        lo = np.min([o.pos.min(0) for o in objects], 0)
        np.testing.assert_allclose(lo, [-1.0, 0.0, -0.5], atol=1e-5)

    def test_shader_settings(self):
        """Displacement feeds the height slot and moves the surface; tessellation subdivides the viewport meshes;
        settings are saved in the project and undoable."""
        ws = self.ws
        panel = ws.shader_settings
        before = ws.view3d.render_image(128, 128)
        panel._edit("scale_amount", 0.2)
        panel._edit("subdivision_count", 2)
        self.assertTrue(ws.view3d.look.enable_displacement)
        self.assertIn("height", ws._slot_jobs(ws.doc.ts, 1001))
        self.assertEqual(ws._tessellation, 2)
        self.assertEqual(ws.doc.project.to_dict()["shader"]["subdivision_count"], 2)
        ws.doc.undo.undo()
        ws.doc.undo.undo()
        ws.refresh_materials()
        self.assertEqual(ws.doc.project.shader.scale_amount, 0.0)
        self.assertEqual(ws._tessellation, 0)
        self.assertFalse(ws.view3d.look.enable_displacement)
        np.testing.assert_array_equal(before, ws.view3d.render_image(128, 128))
        panel.restore_defaults()

    def test_shader(self):
        """Shader settings > Shader: every template (shaders.TEMPLATES), Metallic roughness by default; choosing one
        renders the Texture Set with that model, saved in the project and undoable."""
        from trafilatura3d import shaders
        ws = self.ws
        self.assertEqual(ws.doc.project.shader.shader, "Metallic roughness")
        self.assertEqual(ws.shader_spec()["shader"], shaders.index("Metallic roughness"))
        ws.shader_settings._edit("shader", "Adobe Standard Material")
        self.assertEqual(ws.shader_spec()["shader"], shaders.index("Adobe Standard Material"))
        self.assertEqual(ws.shader_spec()["constants"]["specularWeight"][0], 0.5)  # specular level: the IOR's
        self.assertEqual(ws.doc.project.to_dict()["shader"]["shader"], "Adobe Standard Material")
        self.assertGreater(ws.view3d.render_image(64, 64)[..., :3].mean(), 10)
        ws.doc.undo.undo()
        self.assertEqual(ws.doc.project.shader.shader, "Metallic roughness")

    def test_shader_reads_its_channels(self):
        """The viewport takes the channels the project's shader reads (paint.channels.render_slots): with Specular
        glossiness the Diffuse channel feeds the color, the Base color no longer does; the Add channel menu lists
        Diffuse among the Supported channels."""
        from trafilatura3d.paint.channels import CHANNELS, supported
        from trafilatura3d.paint.model import Channel, ChannelSource
        ws, doc = self.ws, self.ws.doc
        doc.edit("Add channel", lambda: doc.ts.channels.append(Channel("diffuse", CHANNELS["diffuse"].format)))
        ws.layers.add_layer("fill")
        layer = doc.layer()

        def fill():
            layer.material = {"diffuse": ChannelSource(color=[1.0, 0.0, 0.0]),
                              "basecolor": ChannelSource(color=[0.0, 0.0, 1.0])}
        doc.edit("Colors", fill)
        ws.shader_settings._edit("shader", "Metallic roughness")
        blue = ws.view3d.render_image(64, 64)[..., :3].astype(int)
        self.assertGreater(blue[..., 2].mean(), blue[..., 0].mean())
        ws.shader_settings._edit("shader", "Specular glossiness")
        self.assertIn("diffuse", supported("Specular glossiness"))
        self.assertNotIn("basecolor", supported("Specular glossiness"))
        red = ws.view3d.render_image(64, 64)[..., :3].astype(int)
        self.assertGreater(red[..., 0].mean(), red[..., 2].mean())

    def test_alpha_shaders(self):
        """The opacity channel reaches the viewport only with a shader that sees through: opaque ignores it
        (the view unchanged), the alpha blending variant blends it, the alpha test one cuts it at Shader settings'
        Alpha threshold (lib-alpha 0.33)."""
        from trafilatura3d import shaders
        from trafilatura3d.paint.channels import CHANNELS
        from trafilatura3d.paint.model import Channel, ChannelSource
        ws, doc = self.ws, self.ws.doc
        before = ws.view3d.render_image(96, 96)
        doc.edit("Add channel", lambda: doc.ts.channels.append(Channel("opacity", CHANNELS["opacity"].format)))
        ws.layers.add_layer("fill")
        layer = doc.layer()

        def fill():
            layer.material = {"opacity": ChannelSource(color=[0.5, 0.5, 0.5])}
        doc.edit("Opacity", fill)
        ws.refresh_materials()
        self.assertNotIn("opacity", ws._slot_jobs(doc.ts, 1001))
        np.testing.assert_array_equal(before, ws.view3d.render_image(96, 96))
        slot = ws._slot(doc.ts, 1001)
        for file, mode in (("Metallic roughness - Alpha blend", shaders.ALPHA_BLEND),
                           ("Metallic roughness - Alpha test", shaders.ALPHA_TEST)):
            ws.shader_settings._edit("shader", file)
            self.assertIn("opacity", ws._slot_jobs(doc.ts, 1001))
            self.assertEqual(ws.view3d.renderer.materials[slot]["blending"], mode)
        kept = ws.view3d.render_image(96, 96)  # 0.5 above the threshold: the opaque surface
        np.testing.assert_array_equal(before, kept)
        ws.shader_settings._edit("alpha_threshold", 0.7)
        self.assertGreater(np.abs(ws.view3d.render_image(96, 96).astype(int) - before).sum(), 1000)
        self.assertEqual(ws.doc.project.to_dict()["shader"]["alpha_threshold"], 0.7)
        ws.shader_settings._edit("shader", "Metallic roughness - Alpha blend")
        blended = ws.view3d.render_image(96, 96).astype(int)
        self.assertGreater(np.abs(blended - before).sum(), 1000)
        ws.shader_settings.restore_defaults()
        self.assertEqual(ws.doc.project.shader.alpha_threshold, 0.33)

    def test_free_rotation(self):
        """Camera Rotation Mode: Free keeps the view when switched on, turns past the poles and rolls (the up leaves
        the plane of the world Y and the view direction); Shift snaps to an axis-aligned view; back to Constrained the
        view direction stays and the roll goes."""
        v = self.ws.view3d
        v.yaw, v.pitch = 0.6, 0.3
        eye = v.eye().copy()
        self.ws.rotation.setCurrentIndex(1)
        self.assertTrue(v.free_rotation)
        np.testing.assert_allclose(v.eye(), eye, atol=1e-5)
        np.testing.assert_allclose(v.up(), [0, 1, 0], atol=0.3)
        v._rotate_free(0.0, 200.0, False)  # 2 radians up: over the pole, Constrained stops at 1.55
        self.assertLess(v.up()[1], 0.0)
        v._rotate_free(80.0, 0.0, False)
        v._rotate_free(0.0, -200.0, False)
        back = (v.eye() - v.target) / v.distance
        self.assertGreater(abs(float(np.dot(v.up(), np.cross(back, [0, 1, 0])))), 0.1)  # rolled
        np.testing.assert_allclose(v.orient.T @ v.orient, np.eye(3), atol=1e-6)
        v._rotate_free(3.0, 2.0, True)
        self.assertEqual(sorted(np.abs(v.orient).flat), [0.0] * 6 + [1.0] * 3)
        v._rotate_free(30.0, 20.0, False)
        back = (v.eye() - v.target) / v.distance
        self.ws.rotation.setCurrentIndex(0)
        self.assertFalse(v.free_rotation)
        np.testing.assert_allclose((v.eye() - v.target) / v.distance, back, atol=1e-5)
        self.assertEqual(v.up(), (0, 1, 0))

    def test_orbit_around_cursor(self):
        """Alt + left drag started on the mesh orbits around the point under the cursor: it stays on the same
        pixel (Constrained and Free); started off the mesh the target stays."""
        ws, v = self.ws, self.ws.view3d
        v.yaw, v.pitch = 0.6, 0.3
        ws.frame()
        r = v.get_pixel_ratio()
        w, h = v.physical_size()

        def screen(p):
            c = v.view_proj(w, h) @ np.array([*p, 1.0])
            return np.array([(c[0] / c[3] * 0.5 + 0.5) * w, (0.5 - c[1] / c[3] * 0.5) * h])

        def orbit(x, y):
            ev = lambda kind, px, py: v._on_event({"event_type": kind, "x": px / r, "y": py / r, "button": 1,  # noqa
                                                   "modifiers": ("Alt",)})
            ev("pointer_down", x, y)
            ev("pointer_move", x + 40, y + 25)
            ev("pointer_up", x + 40, y + 25)

        for free in (False, True):
            v.free_rotation = free
            x, y = w * 0.42, h * 0.45
            p = ws.surface_point(x, y)
            self.assertIsNotNone(p)
            target = v.target.copy()
            orbit(x, y)
            self.assertGreater(np.linalg.norm(v.target - target), 1e-3)
            np.testing.assert_allclose(screen(p), (x, y), atol=1.5)
        v.free_rotation = False
        target = v.target.copy()
        self.assertIsNone(ws.surface_point(2, 2))
        orbit(2, 2)
        np.testing.assert_allclose(v.target, target, atol=1e-6)

    def test_quick_properties(self):
        """Mouse right in the viewport opens the Properties in a popup at the cursor; closing it gives the panel back
        to its dock; a right drag or a modifier does not open it."""
        ws, v = self.ws, self.ws.view3d

        def click(x1, mods=()):
            for kind, x in (("pointer_down", 50), ("pointer_move", x1), ("pointer_up", x1)):
                v._on_event({"event_type": kind, "x": x, "y": 60, "button": 2, "modifiers": mods})

        click(50)
        popup = ws._quick
        self.assertTrue(popup.isVisible())
        self.assertIs(ws.properties.parent(), popup)
        popup.hide()
        self.assertIs(ws.docks["properties"].widget(), ws.properties)
        self.assertIsNone(ws._quick)
        click(90)
        click(50, ("Shift",))
        self.assertIsNone(ws._quick)

    def _clean_settings(self):
        from PySide6.QtCore import QSettings
        for key in ("paint_preferences", "paint_shortcuts", "paint_libraries"):
            QSettings().remove(key)
        self.ws.apply_preferences()
        self.ws.apply_shortcuts()

    def test_settings_general(self):
        """Edit > Settings > General preferences: world axes and background colors in the 3D view, keyboard
        helper while a modifier is held, Only display the selected material, simplified thumbnails, default UV
        scale of materials, inverse wheel."""
        from trafilatura3d.editor.paint import settings as prefs
        from trafilatura3d.editor.paint.settings import SettingsDialog
        self._clean_settings()
        self.addCleanup(self._clean_settings)
        ws, v = self.ws, self.ws.view3d
        dlg = SettingsDialog(ws)
        self.assertEqual([dlg.sections.item(i).text() for i in range(dlg.sections.count())],
                         ["General preferences", "Shortcuts", "Libraries configuration"])
        g = dlg.general
        g.widgets["world_axes"].setChecked(False)
        plain = v.render_image(200, 200).astype(int)
        g.widgets["world_axes"].setChecked(True)
        axes = v.render_image(200, 200).astype(int)
        diff = np.abs(axes - plain).sum(-1) > 30
        ys, xs = np.nonzero(diff)
        self.assertGreater(len(ys), 20)
        self.assertGreater(xs.min(), 140)  # bottom right only
        self.assertGreater(ys.min(), 140)
        v.look.env_opacity = 0.0  # the background colors alone
        g._set("background_top", [1.0, 0.0, 0.0])
        g._set("background_bottom", [0.0, 0.0, 1.0])
        img = v.render_image(200, 200)
        self.assertGreater(img[2, 2, 0], 200)
        self.assertGreater(img[197, 2, 2], 200)
        v.look.env_opacity = 1.0
        v._on_event({"event_type": "key_down", "key": "Alt", "modifiers": ()})
        self.assertIsNotNone(v.renderer.hud)
        v._on_event({"event_type": "key_up", "key": "Alt", "modifiers": ("Alt",)})
        self.assertIsNone(v.renderer.hud)
        g.widgets["keyboard_helper"].setChecked(False)
        v._on_event({"event_type": "key_down", "key": "Alt", "modifiers": ()})
        self.assertIsNone(v.renderer.hud)
        self.assertEqual(len(ws._visible_sets()), 2)
        g.widgets["selected_material_only"].setChecked(True)
        self.assertEqual(list(ws._visible_sets().values()), [ws.doc.ts])
        g.widgets["simplified_thumbnails"].setChecked(True)
        ws.layers.add_layer("fill")
        ws.layers.update_thumbnails()
        self.assertFalse(ws.layers.rows[ws.doc.selected].thumb.pixmap().isNull())
        prefs.put("uv_scale", 3.0)
        uid = ws.doc.selected
        ws.properties._set_resource(uid, {"package": "x.trafilatura3d", "graph": "g", "values": {}})
        self.assertEqual(ws.doc.ts.find(uid).projection.tiling, 3.0)
        d0 = v.distance
        v._on_event({"event_type": "wheel", "dy": 100, "x": 5, "y": 5, "modifiers": ()})
        g.widgets["wheel_inverse"].setChecked(True)
        d1 = v.distance
        v._on_event({"event_type": "wheel", "dy": 100, "x": 5, "y": 5, "modifiers": ()})
        self.assertAlmostEqual(v.distance, d0, places=5)
        self.assertNotAlmostEqual(d1, d0)

    def test_settings_shortcuts(self):
        """Shortcuts page: a new key on an action, reset by right click; a mouse binding changed drives the
        viewport; Texture set selection picks the Texture Set under the cursor."""
        from trafilatura3d.editor.paint import shortcuts
        from trafilatura3d.editor.paint.settings import SettingsDialog
        self._clean_settings()
        self.addCleanup(self._clean_settings)
        ws, v = self.ws, self.ws.view3d
        page = SettingsDialog(ws).shortcuts
        row = page.names.index("Center the whole mesh")
        self.assertEqual(ws.named_actions["Center the whole mesh"].shortcut().toString(), "F")
        page._set(row, "G")
        self.assertEqual(ws.named_actions["Center the whole mesh"].shortcut().toString(), "G")
        self.assertEqual(page.table.item(row, 1).text(), "G")
        shortcuts.set_binding("Center the whole mesh", None)
        ws.apply_shortcuts()
        self.assertEqual(ws.named_actions["Center the whole mesh"].shortcut().toString(), "F")
        self.assertIn("(1)", ws.tool_actions["paint"].toolTip())
        # Camera rotate on Shift + middle: that press now orbits, Alt + left no longer does
        page._set(page.names.index("Camera rotate"), "Shift+Mouse middle")
        ws.frame()
        w, h = v.physical_size()
        r = v.get_pixel_ratio()

        def drag(button, mods):
            yaw = v.yaw
            for kind, x in (("pointer_down", 2), ("pointer_move", 40), ("pointer_up", 40)):
                v._on_event({"event_type": kind, "x": x, "y": 2, "button": button, "modifiers": mods})
            return v.yaw != yaw
        self.assertTrue(drag(3, ("Shift",)))
        self.assertFalse(drag(1, ("Alt",)))
        # Texture set selection: Ctrl + Alt + right click on the other quad
        other = next(t for t in ws.doc.project.texture_sets if t.name != ws.doc.ts_name)
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        for x in (w * 0.3, w * 0.7):
            p = ws.surface_point(x, h * 0.5)
            if p is not None and (p[0] > 0) == (other.material == "Right"):
                break
        for kind in ("pointer_down", "pointer_up"):
            v._on_event({"event_type": kind, "x": x / r, "y": h * 0.5 / r, "button": 2,
                         "modifiers": ("Control", "Alt")})
        self.assertEqual(ws.doc.ts_name, other.name)

    def test_settings_libraries(self):
        """Libraries configuration: a new path gets its sub-folders and its materials in the Assets; as Default
        it receives new smart materials; Minus removes it, and disables (then enables) a default path; paths are
        locked while a project is open."""
        from tests.scenes import material_package
        from trafilatura3d.editor.paint import settings as prefs
        from trafilatura3d.editor.paint.settings import SettingsDialog
        from trafilatura3d.paint.materials import EXAMPLES
        self._shaderball()
        self._clean_settings()
        self.addCleanup(self._clean_settings)
        ws = self.ws
        page = SettingsDialog(ws).libraries
        self.assertFalse(page.plus.isEnabled())  # a project is open
        self.assertEqual([x["name"] for x in prefs.libraries()], ["your_assets", "starter_assets"])
        lib = self.dir / "mylib"
        page.ws = type("NoProject", (), {"doc": None})()
        page.rebuild()
        self.assertTrue(page.plus.isEnabled())
        page.add(str(lib), "mine")
        self.assertTrue((lib / "materials").is_dir() and (lib / "smart_materials").is_dir())
        material_package(lib / "materials" / "mine.trafilatura3d")
        ws.libraries_changed()
        self.assertTrue(any(p.endswith("mine.trafilatura3d") for p, _g, _l in ws.assets.materials()))
        prefs.set_default_library("mine")
        ws.layers.add_layer("folder")
        ws.layers.create_smart_material(ws.doc.selected)
        self.assertEqual(len(list((lib / "smart_materials").glob("*.trafilatura3dsmart"))), 1)
        page.rebuild()
        page.table.setCurrentCell(2, 1)
        page.remove()
        self.assertEqual([x["name"] for x in prefs.libraries()], ["your_assets", "starter_assets"])
        self.assertTrue(prefs.libraries()[0]["default"])
        page.table.setCurrentCell(1, 1)
        page.remove()
        self.assertFalse(prefs.libraries()[1]["enabled"])
        self.assertNotIn(EXAMPLES, prefs.folders("materials"))
        page.remove()
        self.assertTrue(prefs.libraries()[1]["enabled"])

    def test_color_picker(self):
        """The color picker: a color field opens it titled after the field; hex, spectrum and sliders edit
        the field; the previous color restores; swatches add once, persist, delete; HSV fields in degrees and
        percent; the context closes it; the eyedropper with Shift reads the channel on the mesh; node color
        parameters keep their alpha."""
        import os
        from trafilatura3d.editor.paint import color_picker
        from trafilatura3d.editor.paint.widgets import ColorField, Swatch
        from trafilatura3d.editor.params_ui import make_editor
        from trafilatura3d.nodes.base import Param, ParamKind
        os.environ["TRAFILATURA3D_USER_DIR"] = str(self.dir / "user")
        self.addCleanup(os.environ.pop, "TRAFILATURA3D_USER_DIR", None)
        ws = self.ws
        ws.layers.add_layer("paint")
        ws.set_tool("paint")
        field = next(s for s in ws.properties.findChildren(Swatch) if s.channel == "basecolor")
        field.button.click()
        p = color_picker.picker()
        self.assertTrue(p.isVisible())
        self.assertEqual(p.windowTitle(), "Base color color")
        p.hex.setText("FF0000")
        p.hex.editingFinished.emit()
        np.testing.assert_allclose(ws.tools.values["basecolor"], [1, 0, 0], atol=1e-6)
        self.assertEqual(field.rgb, [1.0, 0.0, 0.0])
        p.spectrum.moved.emit(1.0, 0.5, True)  # half value
        np.testing.assert_allclose(ws.tools.values["basecolor"], [0.5, 0, 0], atol=1e-6)
        p._slid(1, 1.0, True)  # green slider (RGB)
        np.testing.assert_allclose(ws.tools.values["basecolor"], [0.5, 1, 0], atol=1e-6)
        self.assertEqual(p.fields[1].text(), "255")
        p.mode_box.setCurrentText("HSV")
        self.assertEqual(p.fields[0].text(), str(round(p.hsv[0] * 360)))
        self.assertEqual(p.fields[2].text(), "100")
        p.mode_box.setCurrentText("RGB")
        p.pair.restore.emit()
        np.testing.assert_allclose(ws.tools.values["basecolor"], p.previous, atol=1e-6)
        p._set_swatches([])
        p.add_swatch()
        p.add_swatch()
        self.assertEqual(len(color_picker.load_swatches()), 1)
        p._set_swatches([])
        self.assertEqual(color_picker.load_swatches(), [])
        ws.set_tool("eraser")
        self.assertFalse(p.isVisible())
        # Shift eyedropper over the mesh: the Base color channel of a uniform red fill
        ws.layers.add_layer("fill")
        uid = ws.doc.selected
        ws.doc.edit("c", lambda: ws.doc.ts.find(uid).material["basecolor"].__setattr__("color", [1.0, 0.0, 0.0]))
        v = ws.view3d
        v.yaw, v.pitch = 0.0, 1.5
        ws.frame()
        w, h = v.physical_size()
        r = v.get_pixel_ratio()
        x = next(x for x in (w * 0.3, w * 0.7) if ws.channel_values_at(x, h * 0.5) and
                 ws.channel_values_at(x, h * 0.5)[0] is ws.doc.ts)
        from PySide6.QtCore import QPoint
        gp = v.mapToGlobal(QPoint(int(x / r), int(h * 0.5 / r)))
        got = ws.eyedrop_at(gp, True, "basecolor")
        np.testing.assert_allclose(got, [1, 0, 0], atol=0.01)
        self.assertIsNone(ws.eyedrop_at(gp, False, "basecolor"))  # without Shift: the screen's pixel
        # the eyedropper session: an overlay per screen showing its capture, preview while moving, click picks
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QImage
        picked = []
        s = color_picker.start_eyedropper(picked.append, "basecolor")
        self.assertEqual(len(s.overlays), len(self.app.screens()))
        o = s.overlays[0]
        o.image = QImage(o.width(), o.height(), QImage.Format_RGB32)
        o.image.fill(QColor(0, 255, 0))
        s.moved(o, QPointF(30, 30))
        self.assertIsNotNone(o.pointer)
        o.grab()  # the magnifier paints
        s.pick(o, QPointF(30, 30), QPoint(-10000, -10000), False)
        self.assertEqual(picked, [[0.0, 1.0, 0.0]])
        self.assertEqual(s.overlays, [])
        s = color_picker.start_eyedropper(picked.append)
        s.finish(None)  # Esc / right click
        self.assertEqual(len(picked), 1)
        s = color_picker.start_eyedropper(picked.append, "basecolor")
        s.pick(s.overlays[0], QPointF(0, 0), gp, True)  # Shift over the 3D view: the channel
        np.testing.assert_allclose(picked[-1], [1, 0, 0], atol=0.01)
        w2 = make_editor(Param(ParamKind.COLOR, [0.1, 0.2, 0.3, 0.4]), [0.1, 0.2, 0.3, 0.4],
                         lambda val, final: seen.append(val), lambda val: ColorField(val, "Tint"))
        seen = []
        self.assertIsInstance(w2, ColorField)
        w2._picked([1.0, 1.0, 1.0], True)
        self.assertEqual(seen, [[1.0, 1.0, 1.0, 0.4]])

    def test_autosave(self):
        """Backups in the .autosave folder next to the saved project (the Graph's convention: <name>_<date>_<time>),
        the project's own path untouched, the last `files` kept (another project's left alone), no backup when nothing
        changed."""
        from unittest import mock
        from trafilatura3d.editor import backups
        from trafilatura3d.editor.paint import autosave
        auto = self.ws.autosave
        autosave.set_setting("files", 2)
        self.addCleanup(lambda: [autosave.set_setting(k, v) for k, v in autosave.DEFAULTS.items()])
        path = self.dir / "proj.trafilatura3dpaint"
        self.ws.doc.save(path)
        folder = self.dir / ".autosave"
        folder.mkdir()
        (folder / "proj_x_20260101_000000.trafilatura3dpaint").write_text("{}")  # another project's: stays
        stamps, strftime = iter(["20261030_100000", "20261030_100100", "20261030_100200"]), backups.time.strftime
        with mock.patch.object(backups.time, "strftime",  # the autosaves' stamps only (the Log has its own)
                               lambda fmt, *a: next(stamps) if fmt == "%Y%m%d_%H%M%S" else strftime(fmt, *a)):
            self.ws.layers.add_layer("fill")
            first = auto.save_now()
            self.assertEqual(first, folder / "proj_20261030_100000.trafilatura3dpaint")
            self.assertEqual(self.ws.doc.session.path, path)
            self.assertFalse(auto._changed(self.ws.doc))
            self.ws.layers.add_layer("fill")
            self.assertTrue(auto._changed(self.ws.doc))
            auto.save_now()
            auto.save_now()  # a third: the oldest goes
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ["proj_20261030_100100.trafilatura3dpaint", "proj_20261030_100200.trafilatura3dpaint",
                          "proj_x_20260101_000000.trafilatura3dpaint"])
        auto.due = 0.0  # due and nothing changed: no save, the timer restarts
        auto._tick()
        self.assertGreater(auto.due, 0.0)
        self.ws.autosave_disable.setChecked(True)
        self.assertFalse(auto.tick.isActive())


if __name__ == "__main__":
    unittest.main()
