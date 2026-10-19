"""Texture generators/Noises (second half, trafilatura3d/nodes/generators/noises_b) behave as described."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES

EV = Evaluator()
MINE = sorted(t for t, d in NODES.items() if type(d).__module__.startswith("trafilatura3d.nodes.generators.noises_b"))


def cook(node_type, log2=7, seed=0, size=None, **params):
    """One node at 2^log2 px (or `size` = [log2 w, log2 h]), 32-bit float, as a 2D array."""
    pkg = Package()
    g = pkg.new_graph("t", log2)
    n = g.add_node(node_type)
    for k, v in params.items():
        g.set_param(n, k, v)
    g.set_base(n, "output_format", {"mode": "absolute", "value": "32f"})
    g.set_base(n, "seed", {"mode": "absolute", "value": seed})
    if size:
        g.set_base(n, "output_size", {"mode": "absolute", "value": list(size)})
    a = EV.read(compile_graph(pkg, "t").jobs[n], "output").astype(np.float64)
    return a[..., 0] if a.ndim == 3 else a


def roughness(a):
    """Mean absolute difference between neighbours (wrapping): grows with the frequency."""
    return (np.abs(a - np.roll(a, 1, 0)).mean() + np.abs(a - np.roll(a, 1, 1)).mean()) / 2


class EveryNoise(unittest.TestCase):
    def test_all_registered(self):
        self.assertEqual(len(MINE), 43)

    def test_range_and_not_flat(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(t)
                self.assertGreaterEqual(a.min(), 0.0)
                self.assertLessEqual(a.max(), 1.0)
                self.assertGreater(a.std(), 0.02, "flat output")

    def test_seed_changes_pattern(self):
        for t in MINE:
            if t == "waveform" or (t.startswith("voronoi")):
                continue  # checked below with their own parameters
            with self.subTest(node=t):
                self.assertGreater(np.abs(cook(t, seed=0) - cook(t, seed=7)).mean(), 0.01)


class Scale(unittest.TestCase):
    # Scale values (several nodes read them on their own lattices: `internal`).
    CASES = {"perlin_noise": (4, 16), "gaussian_spots_small": (1, 3), "liquid": (16, 64), "plasma": (16, 64),
             "microscope_view": (4, 24), "moisture_noise": (2, 8), "messy_fibers_thin": (1, 2),
             "grunge_spots_dirty": (1, 4), "voronoi": (3.0, 12.0)}

    FINE = {"gaussian_spots_small", "grunge_spots_dirty"}  # Scale 1 is already fine: measured at 512 px

    def test_higher_scale_is_denser(self):
        for t, (lo, hi) in self.CASES.items():
            with self.subTest(node=t):
                log2 = 9 if t in self.FINE else 8
                self.assertGreater(roughness(cook(t, log2, scale=hi)), roughness(cook(t, log2, scale=lo)) * 1.3)

    def test_denser_thread_ribbons(self):
        """Messy fibers 1 and 2: the threads are about a pixel apart at any Scale, so the density shows in the
        ribbons: with more of them the 32 px block means even out."""
        blocks = lambda a: a.reshape(16, 32, 16, 32).mean((1, 3))
        for t in ("messy_fibers_bright", "messy_fibers_soft"):
            with self.subTest(node=t):
                self.assertLess(roughness(blocks(cook(t, 9, scale=2))),
                                roughness(blocks(cook(t, 9, scale=1))) / 1.2)

    def test_tiling_scales(self):
        """Lattice sizes of Scrach Tiling, Leak Scale, Dust Tiling... change the frequency too."""
        self.assertGreater(roughness(cook("grunge_leaky_paint", 8, leak_scale=3)),
                           roughness(cook("grunge_leaky_paint", 8, leak_scale=1)))
        self.assertGreater(roughness(cook("grunge_scratches_rough", 8, scratch_tiling=3)),
                           roughness(cook("grunge_scratches_rough", 8, scratch_tiling=1)))


class Disorder(unittest.TestCase):
    def test_disorder_moves_and_loops(self):
        """Disorder displaces the ingredients (animation); one full turn of Disorder x speed loops."""
        for t in ("perlin_noise", "white_noise", "gaussian_spots_small", "moisture_noise", "messy_fibers_bright"):
            with self.subTest(node=t):
                a = cook(t)
                self.assertGreater(np.abs(cook(t, disorder=0.3) - a).mean(), 0.005)
                np.testing.assert_allclose(cook(t, disorder=1.0), a, atol=2e-3)

    def test_phase_disorder(self):
        for t in ("liquid", "plasma", "microscope_view", "grunge_wavy"):
            with self.subTest(node=t):
                self.assertGreater(np.abs(cook(t, disorder=0.3) - cook(t)).mean(), 0.005)


class WhiteNoise(unittest.TestCase):
    def test_histograms(self):
        u = cook("white_noise", 8, noise_distribution="Uniform")
        g = cook("white_noise", 8, noise_distribution="Gaussian")
        t = cook("white_noise", 8, noise_distribution="Triangle")
        for a in (u, g, t):
            self.assertAlmostEqual(a.mean(), 0.5, delta=0.01)
        self.assertAlmostEqual(u.std(), 1 / np.sqrt(12), delta=0.01)  # flat
        self.assertAlmostEqual(t.std(), 1 / np.sqrt(24), delta=0.01)  # triangle
        self.assertAlmostEqual(g.std(), 1 / 6, delta=0.01)  # bell, sigma 1/6
        # Uniform: every tenth of the range equally filled.
        h, _ = np.histogram(u, bins=10, range=(0, 1))
        self.assertLess(h.max() / h.min(), 1.15)

    def test_fast_is_uniform_per_pixel(self):
        a = cook("white_noise_fast", 8)
        self.assertAlmostEqual(a.mean(), 0.5, delta=0.01)
        self.assertLess(abs(np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]), 0.05)


class NonSquare(unittest.TestCase):
    def test_expansion_keeps_tile_square(self):
        """2:1 image with Non-square expansion: the square tile repeats twice across."""
        a = cook("perlin_noise", size=[8, 7], non_square_expansion=True)
        np.testing.assert_allclose(a[:, :128], a[:, 128:], atol=2e-3)
        b = cook("perlin_noise", size=[8, 7], non_square_expansion=False)
        self.assertGreater(np.abs(b[:, :128] - b[:, 128:]).mean(), 0.02)


class Moisture(unittest.TestCase):
    def test_global_opacity(self):
        """At 0 the base gray stays flat, at 1 the spots add and subtract fully."""
        np.testing.assert_allclose(cook("moisture_noise", global_opacity=0.0), 0.5, atol=1e-6)
        self.assertGreater(cook("moisture_noise", global_opacity=1.0).std(),
                           cook("moisture_noise", global_opacity=0.3).std())


class Fibers(unittest.TestCase):
    def test_luminance_random_darkens_threads(self):
        self.assertLess(cook("messy_fibers_thin", luminance_random=1.0).mean(),
                        cook("messy_fibers_thin", luminance_random=0.0).mean())

    def test_angle_orients_threads(self):
        """Angle 0: horizontal threads vary more across rows than along them; 0.25 turn: the opposite."""
        h = cook("messy_fibers_bright", 8, angle=0.0, angle_random=0.0)
        v = cook("messy_fibers_bright", 8, angle=0.25, angle_random=0.0)
        along = lambda a: np.abs(np.diff(a, axis=1)).mean()
        across = lambda a: np.abs(np.diff(a, axis=0)).mean()
        self.assertGreater(across(h), along(h) * 1.5)
        self.assertGreater(along(v), across(v) * 1.5)


class Waveform(unittest.TestCase):
    def test_samples_are_steps(self):
        """A Square sample is one constant column (inside its antialiased sides)."""
        a = cook("waveform", 8, samples=8, pattern="Square", amplitude_min=0.5)  # no sample under a pixel tall
        for k in range(8):
            col = a[:, k * 32 + 2:(k + 1) * 32 - 2]
            np.testing.assert_allclose(col, np.repeat(col[:, :1], col.shape[1], 1), atol=1e-3)

    def test_position(self):
        c = cook("waveform", 8)
        np.testing.assert_allclose(c, c[::-1], atol=1e-3, err_msg="Centered is symmetric about the middle")
        b = cook("waveform", 8, position="Bottom", pattern="Square")
        self.assertGreater(b[-1].mean(), 0.5)
        self.assertEqual(b[0].mean(), 0.0)

    def test_amplitude(self):
        """Amplitude min. 1, max. 0.5 (height = max x (F x (1 - min) + min)): every Square covers half the height."""
        a = cook("waveform", 8, samples=16, amplitude_min=1.0, amplitude_max=0.5, pattern="Square")
        self.assertAlmostEqual(a.mean(), 0.5, delta=0.01)

    def test_roughness_levels(self):
        """Roughness 1: every octave weighs the same, so the waveform is rougher."""
        smooth = cook("waveform", 8, noise=0.0, pattern="Square", samples=256)
        rough = cook("waveform", 8, noise=0.0, pattern="Square", samples=256, roughness=1.0)
        height = lambda a: a.sum(0)
        self.assertGreater(np.abs(np.diff(height(rough))).mean(), np.abs(np.diff(height(smooth))).mean() * 2)

    def test_noise_subtracts(self):
        self.assertLess(cook("waveform", 8, noise=1.0).mean(), cook("waveform", 8).mean())


class Voronoi(unittest.TestCase):
    def test_invert(self):
        np.testing.assert_allclose(cook("voronoi", invert=True), 1 - cook("voronoi", invert=False), atol=1e-5)

    def test_random_color_is_flat_per_cell(self):
        a = cook("voronoi", 8, style="Random color")
        self.assertLess(len(np.unique(np.round(a, 5))), 200)
        b = cook("voronoi", 8, style="Random color", random_color_seed_mode="Manual Seed", random_color_seed=3)
        self.assertGreater(np.abs(a - b).mean(), 0.05)

    def test_edges_are_thin_lines(self):
        a = cook("voronoi", 8, style="Edge", edge_thickness=0.05, invert=False)
        self.assertLess(a.mean(), 0.35)
        self.assertGreater(a.max(), 0.99)

    def test_f1_le_f2(self):
        """Drawn as 1 - distance: F1 at least as bright as F2, bright at the points."""
        f1 = cook("voronoi", style="F1")
        self.assertTrue(np.all(f1 >= cook("voronoi", style="F2") - 1e-6))
        self.assertGreater(f1.max(), 0.95)

    def test_seed(self):
        self.assertGreater(np.abs(cook("voronoi", seed=0) - cook("voronoi", seed=5)).mean(), 0.01)

    def test_fractal_opacity_and_blend(self):
        a = cook("voronoi_fractal", invert=False)
        np.testing.assert_allclose(cook("voronoi_fractal", global_opacity=0.5, invert=False), a * 0.5, atol=1e-5)
        # Max of the levels weighted by Roughness (1: 1/5 each), Min of the unweighted levels.
        first = cook("voronoi_fractal", min_level=0, max_level=0)
        mx = cook("voronoi_fractal", blend_mode="Max", roughness=1.0)
        mn = cook("voronoi_fractal", blend_mode="Min")
        self.assertTrue(np.all(mx >= first / 5 - 1e-5) and np.all(mn <= first + 1e-5))

    def test_fractal_levels_add_detail(self):
        self.assertGreater(roughness(cook("voronoi_fractal", 8, min_level=0, max_level=5, roughness=0.8)),
                           roughness(cook("voronoi_fractal", 8, min_level=0, max_level=0)))


class Grunge(unittest.TestCase):
    GRUNGES = [t for t in MINE if t.startswith("grunge")]

    def test_invert(self):
        for t in self.GRUNGES:
            with self.subTest(node=t):
                np.testing.assert_allclose(cook(t, 6, invert=True), 1 - cook(t, 6), atol=1e-5)

    def test_balance_brightens(self):
        for t in self.GRUNGES:
            with self.subTest(node=t):
                self.assertGreater(cook(t, 6, balance=0.8).mean(), cook(t, 6, balance=0.5).mean())

    def test_contrast(self):
        """Contrast pushes values away from mid-gray (the std can drop on a nearly white map such as
        Grunge Map 012, where the light grays clip to white)."""
        for t in self.GRUNGES:
            with self.subTest(node=t):
                self.assertGreater(np.abs(cook(t, 6, contrast=0.6) - 0.5).mean(), np.abs(cook(t, 6, contrast=0.0) - 0.5).mean())

    def test_brush_pattern_masks_edges(self):
        """Brush Pattern: black around the edges, centre kept."""
        a = cook("grunge_blotchy", 7, brush_pattern=0.5)
        b = cook("grunge_blotchy", 7)
        self.assertEqual(a[0, 0], 0.0)
        np.testing.assert_allclose(a[60:68, 60:68], b[60:68, 60:68], atol=1e-5)

    def test_cracks(self):
        for t in ("grunge_pitted", "grunge_cracked"):
            with self.subTest(node=t):
                self.assertLess(cook(t, 8, cracks=1.0).mean(), cook(t, 8, cracks=0.0).mean() - 0.02)

    def test_layer_controls(self):
        """Each advanced control acts in the documented direction."""
        lt = self.assertLess
        lt(cook("grunge_concrete", invert_dirt=False).mean(), cook("grunge_concrete", invert_dirt=True).mean())
        # The default (True) gives the preview's scratches, lighter than inverted.
        lt(cook("grunge_rough_dirty", invert_scratches=True).mean(), cook("grunge_rough_dirty", invert_scratches=False).mean())
        lt(cook("grunge_scratches_fine", scratches_amount=0.1).mean(), cook("grunge_scratches_fine", scratches_amount=0.9).mean())
        lt(cook("grunge_scratches_dirty", scratches_intensity=0.0).mean(), cook("grunge_scratches_dirty", scratches_intensity=1.0).mean())
        lt(cook("grunge_spots", coverage=0.2).mean(), cook("grunge_spots", coverage=0.9).mean())
        lt(cook("grunge_spots_dirty", coverage=0.1).mean(), cook("grunge_spots_dirty", coverage=0.9).mean())
        lt(cook("grunge_leaky_paint", leak_intensity=0.9).mean(), cook("grunge_leaky_paint", leak_intensity=0.1).mean())
        lt(cook("grunge_leaks", shape_contrast=0.0).mean(), cook("grunge_leaks", shape_contrast=1.0).mean())
        lt(cook("grunge_shavings", scratch_spots_amount=0.1).mean(), cook("grunge_shavings", scratch_spots_amount=0.9).mean())
        lt(cook("grunge_scratches_rough", scratch_quantity=0.1).mean(), cook("grunge_scratches_rough", scratch_quantity=0.9).mean())
        lt(cook("grunge_galvanic_small", dirt=1.0).mean(), cook("grunge_galvanic_small", dirt=0.0).mean())
        lt(cook("grunge_galvanic_large", ridge_detail_opacity=0.0).mean(), cook("grunge_galvanic_large", ridge_detail_opacity=1.0).mean())
        self.assertGreater(roughness(cook("grunge_concrete", 8, sharpen=1.0)), roughness(cook("grunge_concrete", 8, sharpen=0.0)))

    def test_damas_divisions(self):
        self.assertGreater(roughness(cook("grunge_damas", 8, divisions=30)), roughness(cook("grunge_damas", 8, divisions=6)))

    def test_parameter_values_reach_the_shader(self):
        """Parameter values are converted to the shader's own units (`internal`): the default is the tuned look, and
        values outside our old ranges still act (Galvanic Large's warp 32-128)."""
        self.assertGreater(np.abs(cook("grunge_galvanic_large", 7, warp_intensity=128.0)
                                  - cook("grunge_galvanic_large", 7)).mean(), 0.01)
        values = {"warp_intensity": 64.0, "ridge_detail_opacity": 1.0, "sharpen_intensity": 0.8}
        self.assertAlmostEqual(NODES["grunge_galvanic_large"].shader_values(values)["warp_intensity"], 0.4)

    def test_grunge_maps_differ(self):
        maps = [cook(t, 6) for t, d in NODES.items() if type(d).__module__.endswith(".grunge_maps")]
        self.assertEqual(len(maps), 15)
        for i in range(15):
            for j in range(i + 1, 15):
                self.assertGreater(np.abs(maps[i] - maps[j]).mean(), 0.03, f"{i + 1} and {j + 1} look alike")


if __name__ == "__main__":
    unittest.main()
