"""Texture generators/Noises (first half, trafilatura3d/nodes/generators/noises_a) behave as described."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES

EV = Evaluator()
MINE = sorted(t for t, d in NODES.items() if type(d).__module__.startswith("trafilatura3d.nodes.generators.noises_a"))
ABS = lambda v: {"mode": "absolute", "value": v}


def cook(node_type, log2=7, seed=0, out_size=None, inputs=None, out="output", **params):
    """The node's output as float (H, W, C), 32-bit float, inputs: port -> (node type, params)."""
    pkg = Package()
    g = pkg.new_graph("t", log2)
    n = g.add_node(node_type)
    for k, v in params.items():
        g.set_param(n, k, v)
    g.set_base(n, "output_format", ABS("32f"))
    g.set_base(n, "seed", ABS(seed))
    if out_size:
        g.set_base(n, "output_size", ABS(list(out_size)))
    for port, (src_type, src_params) in (inputs or {}).items():
        s = g.add_node(src_type)
        for k, v in src_params.items():
            g.set_param(s, k, v)
        g.set_base(s, "output_format", ABS("32f"))
        g.connect(s, "output", n, port)
    return EV.read(compile_graph(pkg, "t").jobs[n], out).astype(np.float64)


def gray(a):
    return a[:, :, 0]


def roughness(a):
    """Mean absolute difference between neighbours: grows with the frequency of a pattern."""
    return np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()


# Noises with a grain one pixel wide across their (horizontal by default) streaks, whatever the Scale.
PIXEL_GRAIN = {"directional_noise_fibrous", "directional_noise_dark", "directional_noise_light", "directional_scratches"}
# Noises that are pixel grain (or one-pixel strands) at their default Scale 1, the least allowed.
GRAIN_DOMINATED = {"dirt_patchy", "dirt_sparse", "dirt_fine", "dirt_coarse", "dirt_clumpy", "fur_wavy", "fur_fanned"}


def streak(a):
    """Correlation 16 pixels along x: how far the horizontal streaks reach."""
    b = a - a.mean()
    return (b * np.roll(b, 16, 1)).mean() / b.var()


def has_param(node_type, name):
    return name in NODES[node_type].params


class EveryNoise(unittest.TestCase):
    def test_all_46_registered(self):  # the 6 Legacy noises live in generators/legacy
        self.assertEqual(len(MINE), 46)

    def test_range_and_not_flat(self):
        for t in MINE:
            with self.subTest(node=t):
                a = cook(t, 7)  # Directional scratches' Cones are 2 pixels wide at 512: mush below 128
                self.assertGreaterEqual(a.min(), -1e-4)
                self.assertLessEqual(a.max(), 1.0 + 1e-4)
                if t != "caustics":  # flat without a height map, by design
                    self.assertGreater(a[:, :, 0].std(), 0.02, "flat output")

    def test_seed_changes_pattern(self):
        for t in MINE:
            if t == "caustics":
                continue
            with self.subTest(node=t):
                self.assertGreater(np.abs(cook(t, 6, seed=0) - cook(t, 6, seed=7)).mean(), 0.01)

    def test_scale_raises_frequency(self):
        for t in MINE:
            if not has_param(t, "scale"):
                continue
            with self.subTest(node=t):
                # Around the default (Scale: 1 repeats the whole pattern once, the cells noises 32).
                prm = NODES[t].params["scale"]
                lo, hi = prm.min or 1, prm.max or 64
                s0 = max(lo, prm.default / 2)
                s1 = min(hi, s0 * 4)
                if NODES[t].params["scale"].kind.value == "int":
                    s0, s1 = int(s0), int(s1)
                else:
                    s0, s1 = float(s0), float(s1)
                if t in PIXEL_GRAIN:  # grain one pixel across: Scale shortens the streaks instead
                    self.assertLess(streak(gray(cook(t, 8, scale=s1))), streak(gray(cook(t, 8, scale=s0))) - 0.1)
                    continue
                a, b = gray(cook(t, 8, scale=s0)), gray(cook(t, 8, scale=s1))
                if t in GRAIN_DOMINATED:  # already pixel grain at Scale 1: the pattern changes, no coarser
                    self.assertGreater(np.abs(a - b).mean(), 0.03)
                    self.assertGreater(roughness(b), 0.95 * roughness(a))
                    continue
                # Finer: rougher, or (detail already down to the pixel) less correlated 8 pixels away.
                self.assertTrue(roughness(b) > 1.3 * roughness(a) or correlation(b, 8, 8) < correlation(a, 8, 8) - 0.1)

    def test_non_square_tiles(self):
        for t in MINE:
            if NODES[t].inputs or not getattr(NODES[t], "seamless", True):
                continue
            with self.subTest(node=t):
                a = gray(cook(t, out_size=(7, 6)))
                inner = np.abs(np.diff(a, axis=1)).mean() + 1e-3
                self.assertLess(np.abs(a[:, 0] - a[:, -1]).mean(), inner * 3)


class Resolution(unittest.TestCase):
    def test_same_look_at_every_resolution(self):
        """The output at defaults does not depend on the output size (pixel detail apart): mean and spread
        within 0.04 from 256 to 2048 (Cells 3 and the Furs drifted with a pixel-sized crease step and edge)."""
        for t in ("cells_domes", "cells_walls", "crystal_facets", "fluid", "fur_straight", "fur_wavy", "fur_fanned"):
            with self.subTest(node=t):
                stats = [(a.mean(), a.std()) for a in (gray(cook(t, log2)) for log2 in (8, 9, 10, 11))]
                for m, s in stats:
                    self.assertAlmostEqual(m, stats[1][0], delta=0.04)
                    self.assertAlmostEqual(s, stats[1][1], delta=0.04)


class Specific(unittest.TestCase):
    def test_cells_2_is_a_binary_mask_and_invert_swaps(self):
        a = gray(cook("cells_mask", 8))
        self.assertGreater(((a < 0.05) | (a > 0.95)).mean(), 0.8)
        b = gray(cook("cells_mask", 8, invert=True))
        np.testing.assert_allclose(a + b, 1.0, atol=1e-5)

    def test_cells_2_edge_width_thickens_walls(self):
        thin = (gray(cook("cells_mask", 8, edge_width=0.5)) < 0.5).mean()
        thick = (gray(cook("cells_mask", 8, edge_width=3.0)) < 0.5).mean()
        self.assertGreater(thick, thin * 2)

    def test_cells_4_image_input_gives_the_input_value(self):
        a = gray(cook("cells_flat", 6, color_source="Image input",
                      inputs={"input": ("uniform_color", {"color_mode": "Grayscale", "output_color": [0.3, 0.3, 0.3, 1]})}))
        np.testing.assert_allclose(a, 0.3, atol=1e-3)

    def test_cells_4_pseudorandom_seed_ignores_node_seed(self):
        a = cook("cells_flat", 6, seed=1, color_source="Pseudorandom (Manual seed)", pseudorandom_seed=5, disorder=0.0)
        b = cook("cells_flat", 6, seed=1, color_source="Pseudorandom (Manual seed)", pseudorandom_seed=9, disorder=0.0)
        self.assertGreater(np.abs(a - b).mean(), 0.05)

    def test_cells_1_luminance_random_darkens(self):
        self.assertLess(cook("cells_domes", 7, luminance_random=1.0).mean(), cook("cells_domes", 7, luminance_random=0.0).mean())

    def test_cells_1_sites_leave_dark_gaps(self):
        # Patterns up to 1.5 cells off their cell, so some places no pattern covers.
        a = gray(cook("cells_domes", 9))
        self.assertLess(a.min(), 0.1)
        self.assertGreater(a.mean(), 0.8)

    def test_anisotropic_strips_are_hard_across_and_soft_along(self):
        a = gray(cook("anisotropic_noise", 8, y_amount=256))
        along = np.abs(np.diff(a, axis=1)).mean()
        across = np.abs(np.diff(a, axis=0)).mean()
        self.assertGreater(across, along * 5)
        r = gray(cook("anisotropic_noise", 8, y_amount=256, rotate=True))
        self.assertGreater(np.abs(np.diff(r, axis=1)).mean(), np.abs(np.diff(r, axis=0)).mean() * 5)

    def test_anisotropic_y_amount_by_resolution(self):
        a = gray(cook("anisotropic_noise", 7, y_amount=4, y_amount_by_resolution=True))
        # One strip per pixel row: every row differs from the next.
        self.assertGreater((np.abs(np.diff(a, axis=0)).mean(axis=1) > 1e-4).mean(), 0.95)

    def test_directional_noise_follows_angle(self):
        for t in ("directional_noise_fibrous", "directional_noise_dark", "directional_noise_light", "directional_noise_smooth"):
            with self.subTest(node=t):
                h = gray(cook(t, 8, angle=0.0))
                v = gray(cook(t, 8, angle=0.25))
                self.assertLess(np.abs(np.diff(h, axis=1)).mean(), np.abs(np.diff(h, axis=0)).mean())
                self.assertLess(np.abs(np.diff(v, axis=0)).mean(), np.abs(np.diff(v, axis=1)).mean())

    def test_scratches_amount_and_angle(self):
        few = gray(cook("directional_scratches", 8, pattern_amount=0.25)).mean()
        many = gray(cook("directional_scratches", 8, pattern_amount=1.0)).mean()
        self.assertGreater(many, few + 0.1)  # Max of dense scratches: more of them, brighter
        h = gray(cook("directional_scratches", 8, angle_random=0.0))
        self.assertLess(np.abs(np.diff(h, axis=1)).mean(), np.abs(np.diff(h, axis=0)).mean())

    def test_dirt_gradient_falls_off_vertically(self):
        a = gray(cook("dirt_gradient", 8))
        self.assertGreater(a[-64:].mean(), a[:64].mean() + 0.3)  # dark top, dirt at the bottom

    def test_fractal_sum_base_opacity_burns_and_levels(self):
        a = gray(cook("fractal_sum_base", 8))
        b = gray(cook("fractal_sum_base", 8, global_opacity=2.0))
        self.assertGreater((b >= 1.0 - 1e-4).mean(), (a >= 1.0 - 1e-4).mean() + 0.1)
        lo = gray(cook("fractal_sum_base", 8, min_level=1, max_level=3))
        hi = gray(cook("fractal_sum_base", 8, min_level=5, max_level=7))
        self.assertGreater(roughness(hi), roughness(lo) * 2)
        smooth = gray(cook("fractal_sum_base", 8, roughness=0.2))
        rough = gray(cook("fractal_sum_base", 8, roughness=1.8))
        self.assertGreater(roughness(rough), roughness(smooth) * 1.5)

    def test_bnw_spots_are_rough_gray_spots(self):
        # Mid-gray fractal spots (no threshold), detail down to the pixel.
        for t in ("bnw_noise_rough", "bnw_noise_even", "bnw_noise_soft"):
            with self.subTest(node=t):
                a = gray(cook(t, 8))
                self.assertAlmostEqual(np.median(a), 0.43, delta=0.1)
                self.assertGreater(a.std(), 0.07)
                self.assertLess(((a < 0.02) | (a > 0.98)).mean(), 0.02)
        smooth = gray(cook("bnw_noise_rough", 8, roughness=0.4))
        rough = gray(cook("bnw_noise_rough", 8, roughness=1.6))
        self.assertGreater(roughness(rough), roughness(smooth) * 2)

    def test_gaussian_noise_is_dark_and_spots_are_sparse(self):
        # Summed positive bells: dark mean, longer bright tail.
        a = gray(cook("gaussian_noise", 8))
        self.assertAlmostEqual(a.mean(), 0.32, delta=0.05)
        self.assertGreater(a.mean(), np.median(a))
        s = gray(cook("gaussian_spots_large", 8))
        self.assertLess(np.median(s), 0.15)
        self.assertGreater((s < 0.01).mean(), 0.15)

    def test_disorder_moves_the_ingredients(self):
        for t in MINE:
            if not has_param(t, "disorder") or t.endswith("_3d"):
                continue
            with self.subTest(node=t):
                d = 0.3 if NODES[t].params["disorder"].max <= 1.0 else 30.0
                self.assertGreater(np.abs(cook(t, 7) - cook(t, 7, disorder=d)).mean(), 0.005)

    def test_blue_noise_has_no_low_frequencies(self):
        a = gray(cook("blue_noise_fast", 8))
        blocks = a.reshape(64, 4, 64, 4).mean(axis=(1, 3))
        # White noise: block std = pixel std / 4; blue noise: well below.
        self.assertLess(blocks.std(), a.std() / 6)

    def test_blue_noise_rotation_changes_it(self):
        self.assertGreater(np.abs(cook("blue_noise_fast", 7) - cook("blue_noise_fast", 7, rotation=0.3)).mean(), 0.05)


class ThreeD(unittest.TestCase):
    TYPES = ("perlin_noise_3d", "perlin_noise_fractal_3d", "ridged_noise_fractal_3d", "simplex_noise_3d",
             "voronoi_3d", "voronoi_fractal_3d", "worley_noise_3d")

    def test_unconnected_slice_tiles(self):
        for t in self.TYPES:
            with self.subTest(node=t):
                a = gray(cook(t, 7))
                inner = np.abs(np.diff(a, axis=1)).mean() + 1e-3
                self.assertLess(np.abs(a[:, 0] - a[:, -1]).mean(), inner * 3)
                self.assertLess(np.abs(a[0] - a[-1]).mean(), inner * 3)

    def test_position_map_drives_the_noise(self):
        for t in self.TYPES:
            with self.subTest(node=t):
                a = cook(t, 6)
                b = cook(t, 6, inputs={"position_map": ("test_pattern_color", {})})
                self.assertGreater(np.abs(a - b).mean(), 0.02)

    def test_invert(self):
        for t in ("perlin_noise_3d", "voronoi_3d", "worley_noise_3d"):
            with self.subTest(node=t):
                np.testing.assert_allclose(gray(cook(t, 6, invert=False)) + gray(cook(t, 6, invert=True)), 1.0, atol=1e-5)

    def test_offset_moves_and_size_stretches(self):
        a = gray(cook("perlin_noise_3d", 7))
        self.assertGreater(np.abs(a - gray(cook("perlin_noise_3d", 7, offset=[0.37, 0.0, 0.0]))).mean(), 0.02)
        s = gray(cook("perlin_noise_3d", 8, size=[4.0, 1.0, 1.0]))
        self.assertGreater(np.abs(np.diff(s, axis=1)).mean(), 2 * np.abs(np.diff(s, axis=0)).mean())

    def test_distortion_warps(self):
        a = cook("perlin_noise_3d", 7)
        self.assertGreater(np.abs(a - cook("perlin_noise_3d", 7, distortion_intensity=0.8)).mean(), 0.02)

    def test_absolute_folds_below_half(self):
        a = gray(cook("perlin_noise_3d", 7, absolute=True))
        self.assertLess(a.mean(), gray(cook("perlin_noise_3d", 7)).mean())

    def test_random_color_style_is_color(self):
        a = cook("voronoi_3d", 6, style="Random color")
        self.assertEqual(a.shape[2], 4)
        self.assertGreater(np.abs(a[:, :, 0] - a[:, :, 1]).mean(), 0.05)
        self.assertEqual(cook("voronoi_3d", 6).shape[2], 1)
        self.assertEqual(cook("worley_noise_3d", 6, style="Random Color").shape[2], 4)

    def test_voronoi_edge_style_is_thin_lines(self):
        a = gray(cook("voronoi_3d", 8, style="Edge", invert=False))
        self.assertLess((a > 0.5).mean(), 0.35)
        self.assertGreater((a > 0.5).mean(), 0.02)

    def test_voronoi_fractal_blend_modes(self):
        mx = gray(cook("voronoi_fractal_3d", 7, blend_mode="Max", invert=False)).mean()
        mn = gray(cook("voronoi_fractal_3d", 7, blend_mode="Min", invert=False)).mean()
        self.assertGreater(mx, mn + 0.1)

    def test_fractal_levels_add_detail(self):
        a = gray(cook("perlin_noise_fractal_3d", 8, max_level=0))
        b = gray(cook("perlin_noise_fractal_3d", 8, max_level=6))
        self.assertGreater(roughness(b), roughness(a) * 1.5)


class Caustics(unittest.TestCase):
    FLAT = {"height_map": ("uniform_color", {"color_mode": "Grayscale", "output_color": [0.5, 0.5, 0.5, 1]})}
    BUMPY = {"height_map": ("clouds_soft", {"scale": 5})}

    def test_flat_surface_is_uniform(self):
        a = gray(cook("caustics", 6, inputs=self.FLAT, jittering=0.0))
        np.testing.assert_allclose(a, 0.4, atol=0.01)
        j = gray(cook("caustics", 6, inputs=self.FLAT))  # jittered photons: grain around the same level
        self.assertAlmostEqual(j.mean(), 0.4, delta=0.01)
        self.assertLess(j.std(), 0.03)

    def test_relief_focuses_light(self):
        a = gray(cook("caustics", 7, inputs=self.BUMPY))
        self.assertGreater(a.max(), 0.8)
        self.assertLess(np.percentile(a, 10), 0.3)
        self.assertAlmostEqual(a.mean(), 0.4, delta=0.08)  # energy is moved, not created

    def test_srgb_and_color_mode(self):
        lin = gray(cook("caustics", 6, inputs=self.FLAT)).mean()
        srgb = gray(cook("caustics", 6, inputs=self.FLAT, output_color_space="sRGB")).mean()
        self.assertGreater(srgb, lin + 0.1)
        c = cook("caustics", 6, inputs=self.FLAT, color_mode="Color", background_color=[0.0, 0.0, 0.5, 1.0])
        self.assertEqual(c.shape[2], 4)
        self.assertGreater(c[:, :, 2].mean(), c[:, :, 0].mean() + 0.2)

    def test_dispersion_separates_channels(self):
        c = cook("caustics", 7, inputs=self.BUMPY, color_mode="Color", surface_ior=1.6, dispersion=0.01)
        self.assertGreater(np.abs(c[:, :, 0] - c[:, :, 2]).mean(), 0.02)


def correlation(a, dy, dx):
    b = a - a.mean()
    return (b * np.roll(b, (dy, dx), (0, 1))).mean() / b.var()


if __name__ == "__main__":
    unittest.main()
