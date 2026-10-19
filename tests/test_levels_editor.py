"""The levels editor of the Graph Properties (Levels node, Levels Histogram): the histogram of the image entering the
node, handles and fields that edit one channel of the five FLOAT4 values, a drag as one undo step, the quick actions
(Invert, Auto level, Switch interfaces)."""

import os
import unittest


def drag(bar, x0: float, x1: float, steps: int = 5):
    """Drag a levels bar's handle from `x0` to `x1` (0-1 along the bar) with the mouse."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    y = 16 if bar.out else 5

    def send(kind, t):
        x = bar._x(t)
        buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        QApplication.sendEvent(bar, QMouseEvent(kind, QPointF(x, y), bar.mapToGlobal(QPointF(x, y)), Qt.LeftButton,
                                                buttons, Qt.NoModifier))
    send(QEvent.MouseButtonPress, x0)
    for i in range(1, steps + 1):
        send(QEvent.MouseMove, x0 + (x1 - x0) * i / steps)
    send(QEvent.MouseButtonRelease, x1)
    QApplication.processEvents()


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class LevelsHistogramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from trafilatura3d.editor.document import Document
        from trafilatura3d.editor.properties import PropertiesPanel
        from trafilatura3d.nodes.atomic.gradient_map import DEFAULT
        self.doc = Document()
        self.pkg, self.gid = self.doc.new_package(), "t"
        self.pkg.new_graph(self.gid, 7)
        g = self.pkg.graphs[self.gid]
        noise = g.add_node("gaussian_noise")
        self.gray = g.add_node("levels")
        g.connect(noise, "output", self.gray, "input")
        tint = g.add_node("gradient_map")
        g.set_param(tint, "gradient", {**DEFAULT, "keys": [{"pos": 0.0, "color": [0.1, 0.3, 0.6, 1.0]},
                                                          {"pos": 1.0, "color": [1.0, 0.7, 0.2, 1.0]}]})
        g.connect(noise, "output", tint, "input")
        self.color = g.add_node("levels")
        g.connect(tint, "output", self.color, "input")
        self.panel = PropertiesPanel(self.doc)
        self.panel.resize(380, 900)
        self.panel.show()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.doc.undo.clear()
        self.app.processEvents()

    def editor(self, node):
        from trafilatura3d.editor.widgets import LevelsEditor
        if node is not None:
            self.panel.show_target(self.pkg, self.gid, node)
        self.app.processEvents()
        return self.panel.findChild(LevelsEditor)

    def values(self, node):
        g = self.pkg.graphs[self.gid]  # an undo restores the package: never keep its objects
        return g.definition(node).values(g.nodes[node])

    def tool(self, tip):
        from PySide6.QtWidgets import QToolButton
        b = next(b for b in self.panel.findChildren(QToolButton) if b.toolTip() == tip)
        b.click()
        self.app.processEvents()
        self.app.processEvents()

    def test_grayscale_drags_fields_and_undo(self):
        ed = self.editor(self.gray)
        self.assertEqual([ed.combo.itemText(i) for i in range(ed.combo.count())], ["Luminance Channel"])
        self.assertEqual(len(ed.series), 1)
        self.assertEqual(ed.series[0][0].sum(), 128 * 128)  # every texel of the input
        steps = self.doc.undo.count()
        drag(ed.input, 0.0, 0.2)
        self.assertAlmostEqual(self.values(self.gray)["level_in_low"][0], 0.2, delta=0.01)
        self.assertEqual(self.doc.undo.count(), steps + 1)  # the drag is one step
        drag(ed.output, 1.0, 0.7)
        self.assertAlmostEqual(self.values(self.gray)["level_out_high"][0], 0.7, delta=0.01)
        drag(ed.input, 0.6, 0.7)  # the gray handle: halfway between 0.2 and 1, moved 0.1
        self.assertAlmostEqual(self.values(self.gray)["level_in_mid"][0], 0.625, delta=0.01)
        ed.fields[2].setText("0.9")
        ed.fields[2].editingFinished.emit()
        v = self.values(self.gray)
        self.assertEqual(v["level_in_high"], [0.9, 0.9, 0.9, 1.0])  # the three colour components, not alpha
        self.assertEqual(self.doc.undo.count(), steps + 4)
        self.doc.undo.undo()
        self.assertEqual(self.values(self.gray)["level_in_high"], [1.0] * 4)
        self.assertEqual(self.editor(None).values[2], 1.0)  # the panel shows the undone value

    def test_quick_actions(self):
        self.editor(self.gray)
        self.tool("Auto level")
        v = self.values(self.gray)
        self.assertGreater(v["level_in_low"][0], 0.0)
        self.assertEqual(self.editor(None).values[0], v["level_in_low"][0])
        self.tool("Invert")
        v = self.values(self.gray)
        self.assertEqual((v["level_out_low"], v["level_out_high"]), ([1.0] * 4, [0.0] * 4))
        self.tool("Switch interfaces")  # the sliders, one row per value
        self.assertIsNone(self.editor(None))
        self.tool("Switch interfaces")
        self.assertIsNotNone(self.editor(None))

    def test_color_channels(self):
        ed = self.editor(self.color)
        self.assertEqual(ed.combo.count(), 5)
        self.assertEqual(len(ed.series), 3)  # R, G and B over each other
        drag(ed.input, 0.0, 0.3)
        low = self.values(self.color)["level_in_low"]
        self.assertEqual(low[:3], [low[0]] * 3)
        self.assertEqual(low[3], 0.0)
        ed.combo.setCurrentIndex(2)  # Green Channel
        self.app.processEvents()
        ed = self.editor(None)
        self.assertEqual(ed.combo.currentText(), "Green Channel")
        self.assertEqual(len(ed.series), 1)
        drag(ed.output, 0.0, 0.4)
        out = self.values(self.color)["level_out_low"]
        self.assertAlmostEqual(out[1], 0.4, delta=0.01)
        self.assertEqual([out[0], out[2], out[3]], [0.0, 0.0, 0.0])

    def test_linked_values_keep_their_rows(self):
        g = self.pkg.graphs[self.gid]
        from trafilatura3d.nodes.atomic.levels import LEVELS_PARAMS
        from trafilatura3d.nodes.base import Param, ParamKind
        self.doc.edit(self.pkg, "Expose", lambda: (g.expose("low", Param(ParamKind.FLOAT4, [0.0] * 4, "Low")),
                                                   g.link(self.gray, LEVELS_PARAMS[0], "low")))
        self.assertIsNone(self.editor(self.gray))  # a linked value: the sliders interface


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class CurveEditorTest(unittest.TestCase):
    def test_handles_drag_alone_and_follow_their_point(self):
        from PySide6.QtCore import QCoreApplication, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication
        from trafilatura3d.editor.widgets import CurveEditor
        QCoreApplication.setOrganizationName("Trafilatura3DTest")
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        QApplication.instance() or QApplication([])
        w = CurveEditor([[0, 0], [0.5, 0.5, 0.3, 0.4, 0.7, 0.6], [1, 1]])
        w.resize(206, 150)
        got = []
        w.edited.connect(lambda v, final: got.append(v))

        def drag(a, b):
            for kind, pt in ((QEvent.MouseButtonPress, a), (QEvent.MouseMove, b), (QEvent.MouseButtonRelease, b)):
                pos = w._to_screen(pt)
                buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
                QApplication.sendEvent(w, QMouseEvent(kind, pos, w.mapToGlobal(pos), Qt.LeftButton, buttons,
                                                      Qt.NoModifier))
        drag((0.7, 0.6), (0.8, 0.5))  # the right handle alone
        mid = next(p for p in got[-1] if len(p) == 6)
        self.assertEqual([round(v, 2) for v in mid], [0.5, 0.5, 0.3, 0.4, 0.8, 0.5])
        drag((0.5, 0.5), (0.6, 0.5))  # the point, its handles moving with it
        mid = next(p for p in got[-1] if len(p) == 6)
        self.assertEqual([round(v, 2) for v in mid], [0.6, 0.5, 0.4, 0.4, 0.9, 0.5])


if __name__ == "__main__":
    unittest.main()
