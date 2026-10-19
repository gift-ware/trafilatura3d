"""The Legacy Clouds, BnW noise, Dirt and Fractal sum generators (trafilatura3d/nodes/generators/legacy/clouds.py):
determinism, seeds, instances, resolution, Disorder and properties of the finished images."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.generators.legacy import clouds

EV = Evaluator()
ABS = lambda v: {"mode": "absolute", "value": v}  # noqa: E731
MINE = sorted(t for t, d in NODES.items() if type(d).__module__ == clouds.__name__)


def cook(node_type, log2=8, seed=0, **params):
    """The node's gray output as float (H, W), 32-bit float."""
    pkg = Package()
    g = pkg.new_graph("t", log2)
    n = g.add_node(node_type)
    for k, v in params.items():
        g.set_param(n, k, v)
    g.set_base(n, "output_format", ABS("32f"))
    g.set_base(n, "seed", ABS(seed))
    return EV.read(compile_graph(pkg, "t").jobs[n], "output").astype(np.float64)[:, :, 0]


class LegacyClouds(unittest.TestCase):
    def test_thirteen_legacy_generators(self):
        self.assertEqual(len(MINE), 13)
        for t in MINE:
            d = NODES[t]
            self.assertEqual(d.category, "Texture generators/Legacy")
            self.assertTrue(d.label.endswith(" (Legacy)") and t.endswith("_legacy"), t)
            self.assertEqual(d.catalog, "")
            if "disorder" in d.params:  # the legacy nodes' Disorder: 0-100
                self.assertEqual((d.params["disorder"].label, d.params["disorder"].max), ("Disorder", 100.0))

    def test_deterministic_seeded_and_in_range(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(t, 7)
                EV.clear()
                np.testing.assert_array_equal(a, cook(t, 7))
                self.assertGreater(np.abs(cook(t, 7, seed=3) - a).mean(), 1e-3)
                self.assertTrue(0.0 <= a.min() and a.max() <= 1.0)

    def test_the_large_scale_holds_at_any_output(self):
        # The stamps noises are relative to the output: at 256 and 512 px out the large scale is the same image.
        for t in ("bnw_noise_soft_legacy", "clouds_creased_legacy", "dirt_gradient_legacy"):
            with self.subTest(node=t):
                a, b = cook(t, 8), cook(t, 9)
                small = b.reshape(256, 2, 256, 2).mean((1, 3))
                coarse = lambda x: x.reshape(16, 16, 16, 16).mean((1, 3))  # noqa: E731
                self.assertGreater(np.corrcoef(coarse(a).ravel(), coarse(small).ravel())[0, 1], 0.9)

    def test_disorder_moves_the_pattern_and_keeps_its_look(self):
        for t in ("clouds_creased_legacy", "bnw_noise_rough_legacy", "dirt_fine_legacy", "dirt_patchy_legacy"):
            with self.subTest(node=t):
                a, b, c = cook(t, 8), cook(t, 8, disorder=1.0), cook(t, 8, disorder=50.0)
                self.assertLess(np.abs(b - a).mean(), np.abs(c - a).mean())  # a small Disorder moves a little
                self.assertGreater(np.abs(c - a).mean(), 0.2 * a.std())
                self.assertAlmostEqual(c.mean(), a.mean(), delta=0.25 * a.std())  # coarse layers: a few stamps
                self.assertAlmostEqual(c.std(), a.std(), delta=0.15 * a.std())

    def test_dirt_2_is_dirt_1_through_levels(self):
        d1 = cook("dirt_patchy_legacy", 8, seed=6, disorder=20.0)
        d2 = cook("dirt_sparse_legacy", 8, seed=0, disorder=20.0)
        low = clouds.Dirt2Legacy.LOW
        np.testing.assert_allclose(d2, np.clip((d1 - low) / (1 - low), 0, 1), atol=1e-4)

    def test_dirt_patchy_has_dense_and_empty_patches(self):
        # The specks' density follows a smooth field: the local means at 1/16 of the image spread far more than
        # those of evenly spread specks (Dirt Fine).
        def spread(t):
            a = cook(t, 9)
            return a.reshape(16, 32, 16, 32).mean((1, 3)).std() / a.mean()
        self.assertGreater(spread("dirt_patchy_legacy"), 2.0 * spread("dirt_fine_legacy"))

    def test_dirt_gradient_rises_from_the_bottom_and_tiles_across(self):
        tops, bottoms = [], []
        for seed in range(4):
            a = cook("dirt_gradient_legacy", 8, seed)
            tops.append(a[:64].mean())
            bottoms.append(a[-64:].mean())
            jump = np.abs(a[:, 0] - a[:, -1]).mean()
            self.assertLess(jump, np.abs(np.diff(a, axis=1)).mean() * 3)
        self.assertGreater(np.mean(bottoms) - np.mean(tops), 0.05)

    def test_dirt_4_streaks_are_horizontal(self):
        a = cook("dirt_coarse_legacy", 9)
        a = a - a.mean()
        along = (a[:, :-1] * a[:, 1:]).mean() / a.var()  # correlation of neighbours, along the streaks and across
        across = (a[:-1] * a[1:]).mean() / a.var()
        self.assertGreater(along, across + 0.2)
        far = (a[:, :-4] * a[:, 4:]).mean() / a.var()
        self.assertLess(abs(far), 0.05)  # short dashes: no correlation 4 px apart

    def test_dirt_4_grain_is_the_pixel(self):
        # The grain stays one pixel at every size: a finer output is smoother and less contrasted.
        small, large = cook("dirt_coarse_legacy", 8), cook("dirt_coarse_legacy", 10)
        self.assertGreater(small.std(), large.std() + 0.03)
        self.assertAlmostEqual(small.mean(), large.mean(), delta=0.03)


if __name__ == "__main__":
    unittest.main()
