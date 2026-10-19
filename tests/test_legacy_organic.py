"""The Legacy Fur, Directional noise, Herb, Wood fibers, Scratches and Fluid generators (trafilatura3d/nodes/generators/legacy/
organic.py): their strokes, the directions of their streaks, their parameters' effects, properties of the images."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.fxmap import PATTERNS
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.generators.legacy import organic

EV = Evaluator()
ABS = lambda v: {"mode": "absolute", "value": v}  # noqa: E731
MINE = sorted(t for t, d in NODES.items() if type(d).__module__ == organic.__name__)
RNG = lambda: np.random.default_rng(0)  # noqa: E731


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


def along(a: np.ndarray) -> tuple[float, float]:
    """Mean jump between neighbours across x and across y: a streak along x jumps less across x."""
    return np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=0)).mean()


class LegacyOrganic(unittest.TestCase):
    def test_the_legacy_generators(self):
        self.assertEqual(MINE, ["directionnal_noise_legacy", "fluid_legacy", "fur_straight_legacy", "fur_wavy_legacy",
                                "herb_legacy", "scratches_curved_legacy", "scratches_generator_legacy",
                                "scratches_parallel_legacy", "scratches_scattered_legacy", "wood_fibers_dark_legacy",
                                "wood_fibers_light_legacy"])
        for t in MINE:
            d = NODES[t]
            self.assertEqual(d.category, "Texture generators/Legacy")
            self.assertTrue(d.label.endswith(" (Legacy)"), t)
            self.assertEqual(d.catalog, "")
        # The graphs' parameters, identifiers in snake case, labels and ranges of their widgets.
        s1 = NODES["scratches_parallel_legacy"].params
        self.assertEqual(list(s1), ["pattern_size_width", "pattern_size_height", "rotation_var", "disorder",
                                    "rotation"])
        self.assertEqual((s1["rotation_var"].label, s1["rotation"].min, s1["pattern_size_width"].max),
                         ("Rotation variation", -360.0, 1000.0))
        self.assertEqual(NODES["scratches_scattered_legacy"].params["pattern_size_width"].max, 100.0)
        self.assertEqual(NODES["fur_straight_legacy"].params["disorder"].max, 100.0)
        self.assertEqual([p.label for p in NODES["fur_wavy_legacy"].params.values()],
                         ["Disorder", "Waves scale", "Waves rotation"])

    def test_deterministic_seeded_and_in_range(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(t, 7)
                EV.clear()
                np.testing.assert_array_equal(a, cook(t, 7))
                self.assertGreater(np.abs(cook(t, 7, seed=3) - a).mean(), 1e-3)
                self.assertTrue(0.0 <= a.min() and a.max() <= 1.0)

    def test_fluid_falls_with_flow_and_widens(self):
        """Flow moves the drips down (1.22 cells of 24 per unit, each a little more or less); Pattern width widens
        them."""
        a, b = cook("fluid_legacy", 8), cook("fluid_legacy", 8, flow=1.0)
        fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
        c = np.real(np.fft.ifft2(np.conj(fa) * fb))
        dy, dx = np.unravel_index(np.argmax(c), c.shape)
        self.assertEqual(dx, 0)
        self.assertAlmostEqual(dy, 1.22 * 256 / 24, delta=2)  # down
        self.assertGreater(cook("fluid_legacy", 8, pattern_width=1.0).mean(),
                           cook("fluid_legacy", 8, pattern_width=0.1).mean() + 0.2)

    def test_scratches_generator_mask(self):
        """Every scratch without a mask; with one, those whose centre reads less than half Mask threshold are left
        out."""
        def scratches(mask=None, **params):
            pkg = Package()
            g = pkg.new_graph("t", 8)
            n = g.add_node("scratches_generator_legacy")
            for k, v in params.items():
                g.set_param(n, k, v)
            if mask is not None:
                m = g.add_node("uniform_color")
                g.set_param(m, "color_mode", "Grayscale")
                g.set_param(m, "output_color", [mask, mask, mask, 1.0])
                g.connect(m, "output", n, "mask")
            return EV.read(compile_graph(pkg, "t").jobs[n], "output").astype(np.float64)[:, :, 0].mean()
        self.assertGreater(scratches(), 0.05)
        self.assertEqual(scratches(0.0), 0.0)
        self.assertAlmostEqual(scratches(0.3, mask_threshold=0.5), scratches(), places=6)
        self.assertEqual(scratches(0.3, mask_threshold=0.8), 0.0)
        self.assertGreater(scratches(scratches_amount=4096), 1.5 * scratches())

    def test_fur_1_long_strands_and_short_ones_leaning_a_little(self):
        r = organic.Fur1Legacy().rows({"disorder": 0.0}, RNG())
        n = organic.Fur1Legacy.COUNT
        self.assertEqual(len(r), 5 * n)  # the long ones and four times as many short ones
        self.assertGreater(r[:n, 3].min(), r[n:, 3].max())
        self.assertTrue(np.all(np.abs(r[:, 4]) < 0.08))  # within 30 degrees of the vertical
        # Even spots: every eighth of the tile holds about the same number of long strands.
        counts = np.histogram2d(r[:n, 0] % 1, r[:n, 1] % 1, 8, [[0, 1], [0, 1]])[0]
        self.assertLess(counts.std() / counts.mean(), 0.2)

    def test_fur_2_turns_its_strands_by_position(self):
        flat = organic.Fur2Legacy().rows({"disorder": 0.0, "waves_scale": 0.0, "waves_rotation": 0.0}, RNG())
        steep = organic.Fur2Legacy().rows({"disorder": 0.0, "waves_scale": 0.0, "waves_rotation": 1.0}, RNG())
        self.assertGreater(np.abs(steep[:, 4] - 0.25).max(), 5 * np.abs(flat[:, 4] - 0.25).max())
        # The turn follows the strand's spot: strands close together lie alike.
        a = flat[:2000]
        near = np.linalg.norm(a[:, None, :2] - a[None, :, :2], axis=2) < 0.02
        np.fill_diagonal(near, False)
        i, j = np.nonzero(near)
        self.assertLess(np.abs(a[i, 4] - a[j, 4]).mean(), 0.2 * np.abs(a[:, 4] - a[:, 4].mean()).mean() + 0.01)

    def test_scratches_4_crescents_lightened_not_added(self):
        r = organic.Scratches4Legacy().rows({}, RNG())
        self.assertTrue(np.all(r[:, 14] == 1))  # Max
        self.assertTrue(np.all(r[:, 6] == PATTERNS.index("Crescent")))
        self.assertLessEqual(cook("scratches_curved_legacy", 8).max(), 1.0)

    def test_disorder_moves_wood_and_fur(self):
        for t in ("wood_fibers_dark_legacy", "fur_straight_legacy"):
            with self.subTest(node=t):
                a, b = cook(t, 8), cook(t, 8, disorder=100.0)
                self.assertGreater(np.abs(a - b).mean(), 0.02)
                self.assertLess(abs(a.mean() - b.mean()), 0.1 * a.mean())  # the same look elsewhere

    def test_herb_grain_is_the_pixel(self):
        # The blades are a few pixels long at every output size: the image at 512 is not the one at 256 enlarged.
        a, b = cook("herb_legacy", 8), cook("herb_legacy", 9)
        self.assertAlmostEqual(along(a)[0], along(b)[0], delta=0.2 * along(a)[0])

    def test_streak_directions(self):
        # Fur 1 and the wood fibres run down the image, Fur 2 and Directional noise across.
        for t, vertical in (("fur_straight_legacy", True), ("wood_fibers_dark_legacy", True), ("wood_fibers_light_legacy", True),
                            ("fur_wavy_legacy", False), ("directionnal_noise_legacy", False)):
            with self.subTest(node=t):
                dx, dy = along(cook(t, 8))
                self.assertLess(dy if vertical else dx, 0.7 * (dx if vertical else dy))

    def test_scratch_width_and_disorder(self):
        thin, wide = cook("scratches_parallel_legacy", 8), cook("scratches_parallel_legacy", 8, pattern_size_width=300.0)
        self.assertGreater(wide.mean(), 1.5 * thin.mean())
        self.assertGreater(np.abs(cook("scratches_scattered_legacy", 8, disorder=100.0) - cook("scratches_scattered_legacy", 8)).mean(),
                           0.05)

    def test_wood_fibers_large_scale_holds_at_any_output(self):
        # The octaves are fixed over the tile, the finer ones added down to the pixel: the large scale is the same
        # image at 256 and 512.
        a, b = cook("wood_fibers_light_legacy", 8), cook("wood_fibers_light_legacy", 9)
        small = b.reshape(256, 2, 256, 2).mean((1, 3))
        coarse = lambda x: x.reshape(16, 16, 16, 16).mean((1, 3))  # noqa: E731
        self.assertGreater(np.corrcoef(coarse(a).ravel(), coarse(small).ravel())[0, 1], 0.9)


if __name__ == "__main__":
    unittest.main()
