"""Ambient occlusion and Ambient occlusion (Deprecated) (trafilatura3d/nodes/filters/legacy/ambient_occlusion.py):
properties of an AO on a height and the parameters' effects."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.filters.legacy import ambient_occlusion

EV = Evaluator()
MINE = sorted(t for t, d in NODES.items() if type(d).__module__ == ambient_occlusion.__name__)
LOG2 = 8


class Graph:
    """A graph at 2^LOG2 whose nodes are 16 bit (clamped)."""

    def __init__(self, height: str = "test_pattern", **height_params):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", LOG2)
        self.height = self.node(height, **height_params)

    def node(self, node_type, inputs=None, size=None, depth="16", **params):
        g = self.g
        n = g.add_node(node_type)
        for k, v in params.items():
            g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            g.connect(src, "output", n, port)
        if size is not None:
            g.set_base(n, "output_size", {"mode": "relative_parent", "value": [size, size]})
        g.set_base(n, "output_format", {"mode": "absolute", "value": depth})
        return n

    def read(self, n):
        return EV.read(compile_graph(self.pkg, "t").jobs[n], "output").astype(np.float64)[:, :, 0]


def cook(g: Graph, node_type, **params):
    return g.read(g.node(node_type, {"source": g.height}, depth="32f", **params))


class LegacyAmbientOcclusion(unittest.TestCase):
    def test_nodes(self):
        self.assertEqual(MINE, ["ambient_occlusion_2_legacy", "ambient_occlusion_legacy"])
        self.assertEqual(NODES["ambient_occlusion_2_legacy"].label, "Ambient occlusion (Deprecated)")
        self.assertEqual(NODES["ambient_occlusion_legacy"].label, "Ambient occlusion")
        for t in MINE:
            self.assertEqual((NODES[t].category, NODES[t].catalog, list(NODES[t].inputs)), ("Filters/Legacy", "", ["source"]))

    def test_ambient_occlusion_light_lights_one_side(self):
        # A Paraboloid bump: with Light intensity its two sides along the light differ, and turning the light half a
        # turn swaps them; without light the bump is symmetric.
        g = Graph("shape", pattern="Paraboloid", scale=0.5)
        t = "ambient_occlusion_legacy"
        side = lambda a: a[96:160, 64:112].mean() - a[96:160, 144:192].mean()  # noqa: E731 left minus right
        self.assertLess(abs(side(cook(g, t))), 0.01)
        east = side(cook(g, t, light_intensity=1.0, light_angle=0.0, light_distance=0.5))
        west = side(cook(g, t, light_intensity=1.0, light_angle=0.5, light_distance=0.5))
        self.assertGreater(abs(east), 0.03)
        self.assertLess(east * west, 0.0)

    def test_ambient_occlusion_spread_and_levels(self):
        g = Graph("shape", pattern="Paraboloid", scale=0.5)
        t = "ambient_occlusion_legacy"
        narrow, wide = (cook(g, t, ao_spread=s) for s in (0.1, 1.0))
        self.assertGreater((wide < 0.95).mean(), (narrow < 0.95).mean())
        cut = cook(g, t, ao_spread=1.0, ao_levels=[0.2, 0.8])
        np.testing.assert_allclose(cut, np.clip((wide - 0.2) / 0.6, 0.0, 1.0), atol=2e-3)

    def test_ambient_occlusion_2_equalizer(self):
        # The coarse band darkens with a positive Equalizer x and lightens with a negative one; z scales the fine
        # detail; at 0, 0, 0 a tiny Equalizer changes almost nothing.
        g = Graph("shape", pattern="Paraboloid", scale=0.5)
        t = "ambient_occlusion_2_legacy"
        ao = cook(g, t)
        np.testing.assert_allclose(cook(g, t, equalizer=[0.0, 0.0, 1e-4]), ao, atol=1e-3)
        self.assertLess(cook(g, t, equalizer=[0.5, 0.0, 0.0]).mean(), ao.mean() - 0.01)
        self.assertGreater(cook(g, t, equalizer=[-0.5, 0.0, 0.0]).mean(), ao.mean() + 0.01)
        rough = lambda a: np.abs(np.diff(a, axis=1)).mean()  # noqa: E731
        self.assertGreater(rough(cook(g, t, equalizer=[0.0, 0.0, 0.5])), rough(ao))

    def test_ambient_occlusion_2_spreading_widens_the_shadow(self):
        g = Graph("shape", pattern="Paraboloid", scale=0.5)
        narrow, wide = (cook(g, "ambient_occlusion_2_legacy", spreading=s) for s in (0.05, 0.3))
        self.assertGreater((wide < 0.95).mean(), (narrow < 0.95).mean())

    def test_flat_height_is_white(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(Graph("uniform_color", color_mode="Grayscale", output_color=[0.3, 0.3, 0.3, 1.0]), t)
                np.testing.assert_allclose(a, 1.0, atol=1e-3)

    def test_creases_darken(self):
        # A Paraboloid bump: its foot (a crease against the flat ground) is the darkest, its top and the ground white.
        for t in MINE:
            with self.subTest(node=t):
                g = Graph("shape", pattern="Paraboloid", scale=0.5)
                a, h = cook(g, t), g.read(g.height)
                self.assertGreater(a[h > 0.9].mean(), 0.95)
                foot = (h > 0.0) & (h < 0.15)
                self.assertLess(a[foot].mean(), a[h > 0.9].mean() - 0.05)
                self.assertGreater(a.mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
