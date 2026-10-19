"""Color to mask (Legacy) (trafilatura3d/nodes/filters/legacy/color_to_mask_legacy.py): the three keying distances,
Mask range / 4 as the threshold, Mask softness as the fade, the reference kept at range 0."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES

EV = Evaluator()
REF = [0.2, 0.6, 0.3]


def mask(rgb, **params) -> float:
    pkg = Package()
    g = pkg.new_graph("t", 4)
    src = g.add_node("uniform_color")
    g.set_param(src, "output_color", [*rgb, 1.0])
    g.set_base(src, "output_format", {"mode": "absolute", "value": "32f"})
    n = g.add_node("color_to_mask_legacy")
    g.connect(src, "output", n, "input")
    g.set_base(n, "output_format", {"mode": "absolute", "value": "32f"})
    for k, v in {"color": REF, **params}.items():
        g.set_param(n, k, v)
    return float(EV.read(compile_graph(pkg, "t").jobs[n], "output")[0, 0, 0])


class ColorToMaskLegacy(unittest.TestCase):
    def test_node(self):
        d = NODES["color_to_mask_legacy"]
        self.assertEqual((d.label, d.category, d.catalog, list(d.inputs)),
                         ("Color to mask (Legacy)", "Filters/Legacy", "", ["input"]))

    def test_rgb_threshold_is_a_quarter_of_the_range(self):
        near = [0.25, 0.6, 0.3]  # channels' absolute differences sum to 0.05
        self.assertEqual(mask(near, mask_range=0.3), 1.0)  # 0.05 < 0.075
        self.assertEqual(mask(near, mask_range=0.1), 0.0)  # 0.05 > 0.025
        self.assertEqual(mask(REF, mask_range=0.0), 1.0)  # the reference itself even at range 0

    def test_softness_fades_from_one_minus_softness(self):
        # White below 0.075 x (1 - 0.6) = 0.03, black at 0.075: 0.05 is (0.075 - 0.05) / 0.045 of the way.
        self.assertAlmostEqual(mask([0.25, 0.6, 0.3], mask_range=0.3, mask_softness=0.6), 0.025 / 0.045, places=4)

    def test_chrominance_ignores_brightness_luminance_ignores_hue(self):
        brighter = [0.3, 0.7, 0.4]  # the same chroma, 0.1 brighter
        self.assertEqual(mask(brighter, mask_range=0.05, keying_type="Chrominance"), 1.0)
        self.assertEqual(mask(brighter, mask_range=0.05, keying_type="RGB"), 0.0)
        same_luma = [0.2 + 0.587 * 0.05 / 0.299, 0.55, 0.3]  # another hue, the Rec. 601 luma kept
        self.assertEqual(mask(same_luma, mask_range=0.01, keying_type="Luminance"), 1.0)
        self.assertEqual(mask(same_luma, mask_range=0.01, keying_type="Chrominance"), 0.0)


if __name__ == "__main__":
    unittest.main()
