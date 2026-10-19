"""Legacy pattern generators (trafilatura3d/nodes/generators/legacy/patterns.py) follow their own methods."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes.generators.legacy.patterns import middle_widths, seeded_random

EV = Evaluator()
ABS = lambda v: {"mode": "absolute", "value": v}
MINE = ("tile_generator_legacy", "weave_twill_legacy", "waveform_legacy")


def cook(node_type, log2=8, seed=0, **params):
    """The node's output (gray, float), nothing connected."""
    pkg = Package()
    g = pkg.new_graph("t", log2)
    n = g.add_node(node_type)
    for k, v in params.items():
        g.set_param(n, k, v)
    g.set_base(n, "output_format", ABS("32f"))
    g.set_base(n, "seed", ABS(seed))
    return EV.read(compile_graph(pkg, "t").jobs[n], "output").astype(np.float64)[:, :, 0]


class LegacyPatterns(unittest.TestCase):
    def test_deterministic_in_range_and_seeded(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(t, 7)
                np.testing.assert_array_equal(a, cook(t, 7))
                self.assertGreaterEqual(a.min(), -1e-4)
                self.assertLessEqual(a.max(), 1.0 + 1e-4)
                self.assertGreater(a.std(), 0.05)
        params = {"pattern_type": "Square", "position_random": [1.0, 1.0]}
        self.assertFalse(np.array_equal(cook("tile_generator_legacy", 7, 0, **params),
                                        cook("tile_generator_legacy", 7, 1, **params)))

    def test_splatter_circular_ring(self):
        """Number patterns on a circle of Radius about the centre, the first on the right; Pattern pivot Min moves
        them half their width along their x axis; Random mask 1 leaves none."""
        a = cook("splatter_circular_legacy", 8)
        for x, y in ((0.75, 0.5), (0.5, 0.25), (0.25, 0.5), (0.5, 0.75)):
            self.assertEqual(a[int(y * 256), int(x * 256)], 1.0)
        self.assertEqual(a[128, 128], 0.0)
        moved = cook("splatter_circular_legacy", 8, pattern_size=[0.2, 0.05], pattern_pivot="Min")
        self.assertEqual((moved[128, int(0.84 * 256)], moved[128, int(0.66 * 256)]), (1.0, 0.0))
        self.assertEqual(cook("splatter_circular_legacy", 8, random_mask=1.0).max(), 0.0)

    def test_tiles(self):
        # (Weave 2's Rotate 45 degrees zooms by sqrt 2: with Disorder the weave no longer tiles.)
        for t, params in (("tile_generator_legacy", {"position_random": [3.0, 3.0], "free_rotation_random": 360.0}),
                          ("weave_twill_legacy", {"disorder": 20.0, "rotate_45_degrees": False})):
            with self.subTest(node=t):
                a = cook(t, 8, **params)
                inner_x = np.abs(np.diff(a, axis=1)).mean(0).max()
                inner_y = np.abs(np.diff(a, axis=0)).mean(1).max()
                self.assertLess(np.abs(a[:, 0] - a[:, -1]).mean(), 1.25 * inner_x)
                self.assertLess(np.abs(a[0] - a[-1]).mean(), 1.25 * inner_y)
        a = cook("weave_twill_legacy", 8, disorder=20.0, tiling=2, rotate_45_degrees=False)
        np.testing.assert_allclose(a[:, :128], a[:, 128:], atol=0.02)  # Tiling repeats the same weave

    def test_seeded_random_is_a_seeded_hash(self):
        r = seeded_random(np.arange(1000), 3)
        self.assertTrue(np.all((r >= 0) & (r < 1)))
        self.assertGreater(len(np.unique(r)), 900)
        np.testing.assert_array_equal(seeded_random([7, 7], 3), [r[7]] * 2)
        self.assertNotEqual(seeded_random(7, 4), r[7])  # another seed, another value

    def test_middle_size_packs_alternate_lines(self):
        for m, n in ((0.3, 4), (0.7, 5), (0.5, 3)):
            we, wo = middle_widths(m, n)
            self.assertAlmostEqual(we * ((n + 1) // 2) + wo * (n // 2), 1.0)
        self.assertAlmostEqual(middle_widths(0.5, 4)[0], 0.25)

    def test_interstice_and_scale_size_the_tiles(self):
        a = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=4, nb_y=4, interstice=[0.5, 0, 0.5, 0])
        self.assertAlmostEqual((a > 0.5).mean(), 0.25, delta=0.02)
        b = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=4, nb_y=4, scale=0.5)
        self.assertAlmostEqual((b > 0.5).mean(), 0.25, delta=0.02)
        # Scale variation shrinks each tile by the same GlobalRandom factor whatever the seed.
        c = [cook("tile_generator_legacy", 7, s, pattern_type="Square", nb_x=4, nb_y=4, scale_variation=1.0)
             for s in (0, 3)]
        np.testing.assert_array_equal(*c)

    def test_luminance_by_number_counts_rows_from_zero(self):
        a = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=2, nb_y=2, luminance_by_number=True)
        cell = lambda i, j: a[64 + 128 * j, 64 + 128 * i]
        self.assertAlmostEqual(cell(0, 0), 0.0, places=3)  # tile 0: black
        self.assertAlmostEqual(cell(1, 0), 0.25, places=2)  # tile 1: the second column of the first row
        self.assertAlmostEqual(cell(0, 1), 0.5, places=2)
        self.assertAlmostEqual(cell(1, 1), 0.75, places=2)

    def test_masks(self):
        a = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=4, nb_y=4, vertical_mask=True)
        cols = [a[:, 32 + 64 * i].mean() for i in range(4)]
        self.assertGreater(cols[0], 0.9)  # even columns kept, odd hidden
        self.assertLess(cols[1], 0.1)
        b = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=4, nb_y=4, checker_mask=True,
                 invert_mask=True)
        self.assertLess(b[32, 32], 0.1)
        self.assertGreater(b[32, 96], 0.9)

    def test_contrast_pushes_luminance_from_mid_gray(self):
        flat = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=8, nb_y=8,
                    luminance_variation_contrast=[1.0, 0.0])
        hard = cook("tile_generator_legacy", 8, pattern_type="Square", nb_x=8, nb_y=8,
                    luminance_variation_contrast=[1.0, 1.0])
        self.assertGreater(np.abs(hard - 0.5).mean(), np.abs(flat - 0.5).mean() + 0.05)

    def test_weave_rotates_by_default(self):
        self.assertFalse(np.allclose(cook("weave_twill_legacy", 7), cook("weave_twill_legacy", 7, rotate_45_degrees=False)))

    def test_waveform_follows_the_wave(self):
        a = cook("waveform_legacy", 8, samples=8, noise=0.0, wave_number=1)
        heights = [(a[:, 16 + 32 * k] > 0.5).sum() for k in range(8)]
        # W = 0.5 - 0.5 cos(2 pi (k + 1) / 8): highest at k = 3, zero at k = 7.
        self.assertEqual(int(np.argmax(heights)), 3)
        self.assertEqual(heights[7], 0)
        flat = cook("waveform_legacy", 8, samples=8, noise=0.0, wave_number=0)
        self.assertGreater((flat > 0.5).mean(), 0.3)

    def test_waveform_noise_is_seeded_and_size_max_scales(self):
        self.assertFalse(np.array_equal(cook("waveform_legacy", 7, 0), cook("waveform_legacy", 7, 1)))
        np.testing.assert_array_equal(cook("waveform_legacy", 7, 0, noise=0.0), cook("waveform_legacy", 7, 1, noise=0.0))
        square = dict(samples=8, noise=0.0, wave_number=0, pattern=2)
        full, half = cook("waveform_legacy", 8, **square), cook("waveform_legacy", 8, size_max=0.5, **square)
        self.assertAlmostEqual(full.mean(), 1.0, delta=0.02)
        self.assertAlmostEqual(half.mean(), 0.5, delta=0.02)

    def test_waveform_narrower_than_its_samples_averages_them(self):
        # 512 bars at 256 px: each pixel averages two bars; the image keeps its mean.
        wide = cook("waveform_legacy", 9, samples=512, noise=0.0, wave_number=3)
        narrow = cook("waveform_legacy", 8, samples=512, noise=0.0, wave_number=3)
        self.assertAlmostEqual(narrow.mean(), wide.mean(), delta=0.01)

    def test_weave_disorder_is_seeded(self):
        self.assertFalse(np.array_equal(cook("weave_twill_legacy", 7, 0, disorder=30.0),
                                        cook("weave_twill_legacy", 7, 1, disorder=30.0)))
        np.testing.assert_array_equal(cook("weave_twill_legacy", 7, 0), cook("weave_twill_legacy", 7, 1))
        calm, wild = cook("weave_twill_legacy", 8, rotate_45_degrees=False), cook("weave_twill_legacy", 8, disorder=50.0,
                                                                                rotate_45_degrees=False)
        self.assertGreater(wild.std(), calm.std() + 0.05)  # moved floats overlap and leave gaps


if __name__ == "__main__":
    unittest.main()
