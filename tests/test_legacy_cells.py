"""The Legacy cellular noises and their family (trafilatura3d/nodes/generators/legacy/cells.py): behaviour and
properties of the finished images."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.generators.legacy import cells

EV = Evaluator()
ABS = lambda v: {"mode": "absolute", "value": v}  # noqa: E731
MINE = sorted(t for t, d in NODES.items() if type(d).__module__ == cells.__name__)


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


def roughness(a):
    """Mean absolute difference between neighbours: grows with the frequency of a pattern."""
    return np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()


class LegacyCells(unittest.TestCase):
    def test_all_eleven_are_legacy_generators(self):
        self.assertEqual(len(MINE), 11)
        for t in MINE:
            d = NODES[t]
            self.assertEqual(d.category, "Texture generators/Legacy")
            self.assertTrue(d.label.endswith(" (Legacy)"), t)
            self.assertEqual(d.catalog, "")

    def test_deterministic_and_seeded(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(t, 7)
                EV.clear()
                np.testing.assert_array_equal(a, cook(t, 7))
                self.assertGreater(np.abs(cook(t, 7, seed=3) - a).mean(), 1e-3)

    def test_cells_share_the_domes_cells(self):
        # Cells Flat, Mask and Walls are cut along the borders of the same cells (the sites of Cells Domes): where the
        # flat gray changes the mask is black and the walls are darker than the cells' insides.
        flat, mask = cook("cells_flat_legacy", 9), cook("cells_mask_legacy", 9)
        walls = cook("cells_walls_legacy", 9)
        edge = np.zeros(flat.shape, bool)
        edge[:, 1:] = np.abs(np.diff(flat, axis=1)) > 0.02
        self.assertGreater(np.mean(flat[:, 1:] == flat[:, :-1]), 0.8)  # flat cells
        self.assertLess(mask[edge].mean(), 0.2)
        self.assertLess(walls[edge].mean(), walls[~edge].mean() - 0.01)

    def test_cells_flat_colors_and_distance(self):
        few, many = cook("cells_flat_legacy", 9, distance=6), cook("cells_flat_legacy", 9, distance=12)
        self.assertGreater(roughness(many), 1.5 * roughness(few))
        self.assertGreater(np.abs(cook("cells_flat_legacy", 9, colors=3) - few).max(), 0.05)
        self.assertLess(np.abs(cook("cells_flat_legacy", 9, colors=3) - few).max(), 1.0 + 1e-6)

    def test_crystal_facets_dark_creases_on_bright_facets(self):
        a = cook("crystal_facets_legacy", 9)
        self.assertLess(a.min(), 0.01)
        self.assertGreater(a.max(), 0.6)
        self.assertGreater(np.abs(cook("crystal_facets_legacy", 9, disorder=50.0) - a).mean(), 0.02)

    def test_small_spots_are_sparse_and_disorder_moves_them(self):
        a = cook("gaussian_spots_small_legacy", 9)
        self.assertGreater(np.mean(a < 0.1), 0.4)
        self.assertGreater(a.max(), 0.5)
        b = cook("gaussian_spots_small_legacy", 9, disorder=50.0)
        self.assertGreater(np.abs(a - b).mean(), 0.02)
        self.assertAlmostEqual(a.mean(), b.mean(), delta=0.01)

    def test_shards_and_creases_run_vertically(self):
        for t in ("crystal_shards_legacy", "creased_legacy"):
            with self.subTest(node=t):
                a = cook(t, 9)
                self.assertLess(np.abs(np.diff(a, axis=0)).mean(), 0.6 * np.abs(np.diff(a, axis=1)).mean())

    def test_warp_intensity_bends_shards_and_creases(self):
        for t in ("crystal_shards_legacy", "creased_legacy"):
            with self.subTest(node=t):
                self.assertGreater(np.abs(cook(t, 8) - cook(t, 8, warp_intensity=100.0)).mean(), 0.005)

    def test_moisture_pattern_size_stretches_one_axis(self):
        a = cook("moisture_noise_legacy", 8, pattern_size_width=50.0)
        across, down = np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=0)).mean()
        self.assertGreater(max(across, down) / min(across, down), 1.5)

    def test_cells_1_distance_sets_the_cell_size_not_the_tone(self):
        # (2 Distance + 1)^2 domes 5 steps wide: from 13 to 25 across the cells halve, the covered amount stays.
        few, many = cook("cells_domes_legacy", 9, distance=6), cook("cells_domes_legacy", 9, distance=12)
        self.assertAlmostEqual(roughness(many) / roughness(few), 25 / 13, delta=0.3)
        self.assertAlmostEqual(few.mean(), many.mean(), delta=0.03)

    def test_cells_1_levels_cut_black_gaps(self):
        a = cook("cells_domes_legacy")
        self.assertLess(a.min(), 0.1)
        self.assertGreater(np.mean(a > 0.9), 0.3)

    def test_moisture_is_centred_on_gray(self):
        for seed in range(3):
            self.assertAlmostEqual(cook("moisture_noise_legacy", seed=seed).mean(), 0.5, delta=0.08)

    def test_plasma_is_dark_where_the_noise_meets_its_mirror(self):
        a = cook("plasma_legacy", 8)
        self.assertLess(a.min(), 0.02)
        self.assertLess(np.mean(a < 0.05), 0.25)  # thin veins, not areas

    def test_crystal_1_warp_intensity_warps(self):
        a, b = cook("crystal_facets_legacy", 7), cook("crystal_facets_legacy", 7, warp_intensity=100.0)
        self.assertGreater(np.abs(a - b).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
