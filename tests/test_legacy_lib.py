"""The Legacy generators' shared machinery (trafilatura3d/nodes/generators/legacy/_lib.py), the noises of noises.py, the
Grunge maps of grunge.py, and the atomic Blur's Pixel Size the legacy nodes rely on."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.generators.legacy import _lib, noises

EV = Evaluator()
ABS = lambda v: {"mode": "absolute", "value": v}  # noqa: E731
LEGACY = sorted(t for t, d in NODES.items() if isinstance(d, _lib.LegacyNode) and not d.inputs)  # the generators


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


def correlation(a, dy, dx):
    b = a - a.mean()
    return (b * np.roll(b, (dy, dx), (0, 1))).mean() / b.var()


class Library(unittest.TestCase):
    def test_every_legacy_node_is_in_the_legacy_category(self):
        self.assertGreater(len(LEGACY), 40)
        for t in LEGACY:
            self.assertEqual(NODES[t].category, _lib.CATEGORY, t)

    def test_quadtree_as_the_graphs(self):
        rng = np.random.default_rng(0)
        # Every quadrant of a level draws; a partial children list prunes the levels below.
        r = _lib.quadtree([_lib.Level()] * 3 + [_lib.Level(pattern="Disc", children=(0, 1, 3)),
                                                _lib.Level(pattern="Disc")], rng)
        self.assertEqual(len(r), 64 + 64 * 3)
        np.testing.assert_allclose(r[:64, 2:4], 1 / 8)  # Pattern Size in cells of the level
        # A Branch Offset moves the descendants, a Frame Offset only its own pattern.
        moved = _lib.quadtree([_lib.Level(lambda g, k: np.full((k, 2), 0.25)),
                               _lib.Level(pattern="Disc", frame=lambda g, k: np.full((k, 2), 0.1))], rng)
        # Positions stay where the offsets took them: the engine wraps them along the tiled axes only (Dirt gradient's
        # branch moved off its untiled bottom side is not drawn at the top).
        np.testing.assert_allclose(np.sort(moved[:, 0]), [0.6, 0.6, 1.1, 1.1])

    def test_an_instance_draws_what_the_node_draws(self):
        # The common interface: a graph's instance of a node at the same values, seed and size is the node's image
        # (Dirt Sparse is Dirt Patchy at seed + 6 through a Levels: the instance read inside it).
        dirt_patchy = cook("dirt_patchy_legacy", 8, seed=6)
        dirt_sparse = cook("dirt_sparse_legacy", 8, seed=0)
        low = NODES["dirt_sparse_legacy"].LOW
        expected = np.clip((dirt_patchy - low) / (1 - low), 0, 1)
        np.testing.assert_allclose(dirt_sparse, expected, atol=1e-4)

    def test_absolute_sizes_are_declared(self):
        # The legacy nodes that draw some passes at absolute sizes say so (test_fxmap.Resolution exempts them); the
        # stamps noises of clouds.py are relative to the output.
        self.assertTrue(NODES["cells_mask_legacy"].absolute_sizes)
        for t in ("clouds_creased_legacy", "bnw_noise_rough_legacy", "bnw_noise_soft_legacy", "dirt_gradient_legacy"):
            self.assertFalse(NODES[t].absolute_sizes, t)


class BlurPixelSize(unittest.TestCase):
    def test_pixel_size_multiplies_the_radius(self):
        # The legacy nodes blur at Intensity 1 and Pixel Size 64 ... 1/16 (Fractal Sum Base): the radius is
        # Intensity x Pixel Size pixels at 256, per axis.
        def blurred(intensity, pixel_size):
            pkg = Package()
            g = pkg.new_graph("t", 8)
            src = g.add_node("test_pattern")
            n = g.add_node("blur")
            g.connect(src, "output", n, "input")
            g.set_param(n, "intensity", intensity)
            g.set_base(n, "pixel_size", ABS([pixel_size[0], pixel_size[1]]))
            g.set_base(n, "output_format", ABS("32f"))
            return EV.read(compile_graph(pkg, "t").jobs[n], "output").astype(np.float64)
        np.testing.assert_allclose(blurred(2.0, (4.0, 4.0)), blurred(8.0, (1.0, 1.0)), atol=1e-5)
        wide = blurred(4.0, (4.0, 0.25))  # 16 px across, 1 px down
        np.testing.assert_allclose(wide, blurred(16.0, (1.0, 0.0625)), atol=1e-5)
        self.assertFalse(np.allclose(wide, blurred(16.0, (1.0, 1.0)), atol=1e-3))


class Noises(unittest.TestCase):
    """The Legacy noises of noises.py."""

    def test_perlin_noise_zoom_distance_sets_the_lattice_not_the_tone(self):
        # One Gaussian 6 steps wide per point of a (2 Distance + 1)^2 lattice: the blobs shrink with Distance, the
        # covered amount (mean) stays.
        fine, coarse = cook("perlin_noise_zoom", 8, distance=14), cook("perlin_noise_zoom", 8, distance=4)
        self.assertGreater(roughness(fine), 2.5 * roughness(coarse))
        self.assertAlmostEqual(fine.mean(), coarse.mean(), delta=0.03)

    def test_perlin_noise_zoom_disorder_moves_blobs_only(self):
        a, b = cook("perlin_noise_zoom", 8, disorder=0.0), cook("perlin_noise_zoom", 8, disorder=50.0)
        self.assertGreater(np.abs(a - b).mean(), 0.05)
        self.assertAlmostEqual(a.mean(), b.mean(), delta=0.02)

    def test_perlin_noise_1_and_2_are_stretched_coarse_presets(self):
        zoom = roughness(cook("perlin_noise_zoom", 8))
        one, two = cook("perlin_noise_large_legacy", 8), cook("perlin_noise_huge_legacy", 8)
        self.assertLess(roughness(one), zoom)
        self.assertLess(roughness(two), roughness(one))  # 5 x 5 blobs against 9 x 9
        for t in ("perlin_noise_large_legacy", "perlin_noise_huge_legacy"):  # their Levels cut the dark end
            self.assertLess(min(cook(t, 8, seed=s).min() for s in range(4)), 0.02, t)

    def test_vertical_noise_streaks_are_vertical(self):
        a = cook("vertical_noise", 8)
        self.assertGreater(correlation(a, 16, 0), 0.4)
        self.assertLess(abs(correlation(a, 0, 16)), 0.1)
        self.assertGreater(np.abs(cook("vertical_noise", 8, disorder=50.0) - a).mean(), 0.05)

    def test_dirt_6_keeps_only_what_is_above_the_gray(self):
        # Levels in low 0.498 first: the stamps darker than the background are black.
        a = cook("dirt_faint_legacy", 8)
        self.assertGreater((a < 0.02).mean(), 0.02)
        self.assertGreater(a.max(), 0.5)

    def test_fur_4_is_mirrored_on_the_anti_diagonal(self):
        # Blend Max with itself through the matrix (0, -1, -1, 0): symmetric, (x, y) -> (1 - y, 1 - x).
        a = cook("fur_swirled_legacy", 8)
        np.testing.assert_allclose(a, a[::-1, ::-1].T, atol=1e-6)
        self.assertGreater(np.abs(a - cook("fur_swirled_legacy", 8, disorder=0.0)).mean(), 0.05)  # turns the strokes

    def test_the_moved_nodes_are_legacy_nodes(self):
        for t in ("perlin_noise_zoom", "perlin_noise_large_legacy", "perlin_noise_huge_legacy", "vertical_noise", "dirt_faint_legacy", "fur_swirled_legacy"):
            self.assertEqual(type(NODES[t]).__module__, noises.__name__)


class Cells2(unittest.TestCase):
    def test_flat_cells_at_every_resolution(self):
        # The Mosaic climbs Cells 1's slopes pixel by pixel: drawn in half floats the domes were quantised, the
        # facets broke into blotches at 2048 and the Levels 0.94-0.98 cut everything to black (mean 0.008).
        for log2 in (8, 10, 11):
            with self.subTest(size=1 << log2):
                self.assertGreater(cook("cells_mask_legacy", log2).mean(), 0.3)


class Grunge(unittest.TestCase):
    def test_deterministic_seeded_and_tiling(self):
        for t in ("grunge_stains_legacy", "grunge_blots_legacy"):
            with self.subTest(node=t):
                a = cook(t, 8)
                np.testing.assert_array_equal(a, cook(t, 8))
                self.assertGreater(np.abs(cook(t, 8, seed=3) - a).mean(), 0.02)
                self.assertGreater(a.std(), 0.15)
                # Tiles: the wrap-around step is like a step between two columns inside (in a contrasted grunge they
                # range up to ~5x their mean: compared with the sharpest, not with the mean).
                self.assertLess(np.abs(a[:, 0] - a[:, -1]).mean(), np.abs(np.diff(a, axis=1)).mean(0).max())

    def test_character(self):
        # Stains: mostly white with a few solid dark stains; Blots: a broad gray range from black to white.
        stains, blots = cook("grunge_stains_legacy", 9), cook("grunge_blots_legacy", 9)
        self.assertGreater(np.median(stains), 0.75)
        self.assertGreater((stains > 0.9).mean(), 0.3)
        self.assertGreater((stains < 0.01).mean(), 0.01)
        self.assertTrue(0.35 < blots.mean() < 0.55)
        self.assertGreater(np.percentile(blots, 90) - np.percentile(blots, 10), 0.5)

    def test_controls(self):
        for t in ("grunge_stains_legacy", "grunge_blots_legacy"):
            with self.subTest(node=t):
                a = cook(t, 8)
                self.assertLess(cook(t, 8, balance=0.2).mean(), a.mean())  # Histogram Scan: Balance
                self.assertLess(np.corrcoef(a.ravel(), cook(t, 8, invert=True).ravel())[0, 1], -0.5)
                # Brush pattern: a brick mask darkens the borders.
                self.assertLess(cook(t, 8, brush_pattern=0.5)[:8].mean(), a[:8].mean())


if __name__ == "__main__":
    unittest.main()
