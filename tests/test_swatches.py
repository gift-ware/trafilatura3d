"""Swatches: a color parameter that offers the colors of an image input (an ID map's), read from its cooked values;
the colors found (`distinct_colors`), the graph input property and its checks, the Properties picker."""

import os
import time
import unittest

import numpy as np

from trafilatura3d import DocumentError, Package
from trafilatura3d.evaluator import COLORS_MAX, distinct_colors
from trafilatura3d.nodes.base import Param, ParamKind

ORANGE, NAVY = [1.0, 137 / 255, 0.0], [13 / 255, 17 / 255, 26 / 255]


class DistinctColorsTest(unittest.TestCase):
    def test_most_covering_first_edges_left_out(self):
        img = np.zeros((100, 100, 4), np.uint8)
        img[:, :60, :3] = [255, 137, 0]
        img[:, 60:, :3] = [13, 17, 26]
        img[0, 0, :3] = [128, 77, 13]  # one blended edge texel: under the share kept
        colors = distinct_colors(img)
        self.assertEqual([c for c, _ in colors], [ORANGE, NAVY])
        self.assertAlmostEqual(colors[0][1], 0.5999)
        self.assertAlmostEqual(colors[1][1], 0.4)

    def test_edge_blends_left_out(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[:, :50] = [255, 137, 0]
        img[:, 50:] = [13, 17, 26]
        img[:, 49] = [134, 77, 13]  # where they meet: half of each, covering 1% (over the share kept)
        img[10:20, 10:20] = [102, 102, 102]  # a color of its own, small: kept
        img[30:40, 10:20] = [45, 45, 45]  # between two others (a gray of a gray ramp), but an area: kept
        img[30:40, 20:30] = [4, 4, 4]
        colors = [[round(x * 255) for x in c] for c, _ in distinct_colors(img)]
        self.assertEqual(colors[:2], [[13, 17, 26], [255, 137, 0]])
        self.assertEqual(sorted(colors[2:]), [[4, 4, 4], [45, 45, 45], [102, 102, 102]])  # as large: any order

    def test_float_and_gray_images(self):
        img = np.full((10, 10, 1), 0.4, np.float32)  # a one channel read: gray
        self.assertEqual(distinct_colors(img), [([102 / 255] * 3, 1.0)])
        many = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3) % 200
        self.assertLessEqual(len(distinct_colors(many)), COLORS_MAX)


class SwatchesInputTest(unittest.TestCase):
    def setUp(self):
        self.pkg = Package()
        self.g = self.pkg.new_graph("palette")
        self.id_map = self.g.add_node("input_color")
        self.g.set_param(self.id_map, "identifier", "id_map")
        self.mask = self.g.add_node("color_to_mask")

    def test_offered_from_an_image_input_of_the_graph(self):
        self.g.expose("slot", Param(ParamKind.FLOAT3, [0.5] * 3, "Slot", swatches="id_map"))
        self.assertEqual(Package.from_dict(self.pkg.to_dict()).graphs["palette"].parameters["slot"].param.swatches,
                         "id_map")
        with self.assertRaisesRegex(DocumentError, "not an image input"):
            self.g.edit_parameter("slot", swatches="nothing")
        with self.assertRaisesRegex(DocumentError, "not a color"):
            self.g.expose("count", Param(ParamKind.INT, 1, "Count", swatches="id_map"))
        self.g.edit_parameter("slot", swatches="")
        self.assertEqual(self.g.parameters["slot"].param.swatches, "")

    def test_a_node_parameter_exposed_leaves_its_own_swatches(self):
        self.assertEqual(self.g.definition(self.mask).params["color"].swatches, "input")
        self.assertEqual(self.g.expose_node_param(self.mask, "color", "slot").swatches, "")


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class SwatchesMenuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtWidgets import QApplication
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings
        QCoreApplication.setApplicationName("Trafilatura3DTest")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from trafilatura3d.editor.document import Document
        from trafilatura3d.editor.live import LiveCook
        from trafilatura3d.editor.properties import PropertiesPanel
        self.doc = Document()
        self.pkg, self.gid = self.doc.new_package(), "t"
        g = self.pkg.new_graph(self.gid, 7)
        a, b, shape = g.add_node("uniform_color"), g.add_node("uniform_color"), g.add_node("shape")
        g.set_param(a, "output_color", [*ORANGE, 1.0])
        g.set_param(b, "output_color", [*NAVY, 1.0])
        g.set_param(shape, "size", [0.5, 0.5])  # a square in the middle: navy around it
        two = g.add_node("blend")  # an ID map of two colors
        g.connect(a, "output", two, "foreground")
        g.connect(b, "output", two, "background")
        g.connect(shape, "output", two, "opacity")
        self.mask = g.add_node("color_to_mask")
        g.connect(two, "output", self.mask, "input")
        self.doc.live = LiveCook(None, self.doc.ev, lambda: None, lambda step: None)  # the colors' side cooks
        self.panel = PropertiesPanel(self.doc)
        self.panel.show()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self.doc.undo.clear()
        self.app.processEvents()

    def pickers(self, node):
        """The swatch pickers of a node's Properties, once the colors they offer are read (a side cook)."""
        from trafilatura3d.editor.widgets import SwatchPicker
        self.panel.show_target(self.pkg, self.gid, node)
        t0 = time.perf_counter()
        self.app.processEvents()
        while self.doc.live.side_busy and time.perf_counter() - t0 < 30:
            self.app.processEvents()
        self.app.processEvents()
        return self.panel.findChildren(SwatchPicker)

    @staticmethod
    def offered(picker):
        return {a.text().split()[0]: a for a in picker.menu().actions() if a.isEnabled()}

    def test_picker_offers_the_input_colors_and_sets_one(self):
        picker, = self.pickers(self.mask)  # Color to mask's Color: a color of its input
        actions = self.offered(picker)
        self.assertEqual(sorted(actions), ["#0d111a", "#ff8900"])
        steps = self.doc.undo.count()
        actions["#ff8900"].trigger()
        self.assertEqual(self.pkg.graphs[self.gid].nodes[self.mask].params["color"], ORANGE)
        self.assertEqual(self.doc.undo.count(), steps + 1)
        picker, = self.pickers(self.mask)
        self.assertTrue(self.offered(picker)["#ff8900"].isChecked())  # the value's color marked
        self.assertIn("#ff8900", picker.text())

    def test_every_color_of_an_instance_gets_its_picker(self):
        """Two inputs offering the same ID map's colors: each picker filled (a request never replaces another's)."""
        palette = self.pkg.new_graph("palette", 7)
        n = palette.add_node("input_color")
        palette.set_param(n, "identifier", "id_map")
        for slot in ("id_2", "id_3"):
            palette.expose(slot, Param(ParamKind.FLOAT3, [0.5] * 3, slot, swatches="id_map"))
        g = self.pkg.graphs[self.gid]
        inst = g.add_node("graph_instance", ref="palette")
        g.connect(g.nodes[self.mask].inputs["input"][0], "output", inst, "id_map")
        pickers = self.pickers(inst)
        self.assertEqual(len(pickers), 2)
        for p in pickers:
            self.assertEqual(sorted(self.offered(p)), ["#0d111a", "#ff8900"])
        g.disconnect(inst, "id_map")
        picker = self.pickers(inst)[0]
        self.assertEqual(self.offered(picker), {})
        self.assertEqual([a.text() for a in picker.menu().actions()], ["Nothing connected to id_map"])


if __name__ == "__main__":
    unittest.main()
