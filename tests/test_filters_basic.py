"""Basic filters (Adjustments, Blending, Blurs, Channels, Tiling, Transforms) behave as
documented."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph

EV = Evaluator()
F32 = {"mode": "absolute", "value": "32f"}


class Graph:
    """A small graph read in 32-bit float at 2^log2 px."""

    def __init__(self, log2=5):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", log2)

    def node(self, type, inputs=None, **params):
        n = self.g.add_node(type)
        for k, v in params.items():
            self.g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            src, out = src if isinstance(src, tuple) else (src, "output")
            self.g.connect(src, out, n, port)
        self.g.set_base(n, "output_format", F32)
        return n

    def color(self, rgba):
        return self.node("uniform_color", output_color=list(rgba))

    def gray(self, v):
        return self.node("uniform_color", color_mode="Grayscale", output_color=[v, v, v, 1])

    def pattern(self, color=False):
        return self.node("test_pattern_color" if color else "test_pattern")

    def read(self, n, out="output"):
        a = EV.read(compile_graph(self.pkg, "t").jobs[n], out)
        return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)


def px(a):
    return a[3, 5]


class Adjustments(unittest.TestCase):
    def test_invert(self):
        t = Graph()
        self.assertAlmostEqual(px(t.read(t.node("invert", {"input": t.gray(0.2)})))[0], 0.8, places=5)
        np.testing.assert_allclose(px(t.read(t.node("invert", {"input": t.color((0.2, 0.4, 0.6, 0.5))})))[:4],
                                   [0.8, 0.6, 0.4, 0.5], atol=1e-5)
        self.assertAlmostEqual(px(t.read(t.node("invert", {"input": t.gray(0.2)}, invert=False)))[0], 0.2, places=5)

    def test_hcl_measured_responses(self):
        # Measured on the reference: luma kept while the hexagonal hue turns (red at 90 degrees clips to green),
        # chroma added along the hue (a gray turns red), percentages towards the limits, chroma below 0 flipping
        # the hue only without Clamp Chroma, alpha passed through.
        t = Graph()
        cases = [((1, 0, 0), dict(hue_degrees=90), (0.0, 0.392, 0.0)),
                 ((1, 0, 0), dict(hue_turns=0.25, hue_mode="Turns"), (0.0, 0.392, 0.0)),
                 ((1 / 3, 1 / 3, 1 / 3), dict(chrominance=0.2), (0.491, 0.291, 0.291)),
                 ((1, 0, 0), dict(hue_degrees=5.11, chrominance=-0.08, luminance=-0.14), (0.741, 0.0, 0.0)),
                 ((1 / 3, 1 / 3, 1 / 3), dict(proportional=True, proportional_luminance=20), (0.467, 0.467, 0.467)),
                 ((1, 0, 0), dict(proportional=True, proportional_chrominance=-50), (0.606, 0.106, 0.106)),
                 ((0.6, 0.4, 0.4), dict(chrominance=-0.5), (0.443, 0.443, 0.443)),
                 ((0.6, 0.4, 0.4), dict(chrominance=-0.5, clamp_chroma=False), (0.206, 0.506, 0.506))]
        for rgb, params, want in cases:
            with self.subTest(rgb=rgb, **params):
                out = px(t.read(t.node("hcl", {"input": t.color((*rgb, 0.5))}, **params)))
                np.testing.assert_allclose(out, [*want, 0.5], atol=0.004)

    def test_threshold_is_binary(self):
        t = Graph()
        a = t.read(t.node("threshold", {"input": t.pattern()}, threshold=0.4))
        self.assertTrue(set(np.unique(a)) <= {0.0, 1.0})
        self.assertEqual(len(np.unique(a)), 2)
        src = t.read(t.pattern())
        np.testing.assert_array_equal(a[..., 0], (src[..., 0] >= 0.4).astype(float))

    def test_clamp(self):
        t = Graph()
        a = t.read(t.node("clamp", {"input": t.pattern()}, min=0.3, max=0.6))
        self.assertAlmostEqual(a.min(), 0.3, places=5)
        self.assertAlmostEqual(a.max(), 0.6, places=5)
        v = px(t.read(t.node("clamp", {"input": t.color((0.1, 0.5, 0.9, 0.95))}, min=0.2, max=0.8)))
        np.testing.assert_allclose(v, [0.2, 0.5, 0.8, 0.8], atol=1e-5)  # default: alpha too
        v = px(t.read(t.node("clamp", {"input": t.color((0.1, 0.5, 0.9, 0.95))}, min=0.2, max=0.8,
                             apply_to_alpha=False)))
        np.testing.assert_allclose(v, [0.2, 0.5, 0.8, 0.95], atol=1e-5)

    def test_histogram_shift_wraps(self):
        t = Graph()
        self.assertAlmostEqual(px(t.read(t.node("histogram_shift", {"input": t.gray(0.7)}, position=0.5)))[0],
                               0.2, places=5)
        self.assertAlmostEqual(px(t.read(t.node("histogram_shift", {"input": t.gray(0.7)}, position=1.0)))[0],
                               0.7, places=5)

    def test_histogram_scan(self):
        t = Graph()
        p = t.pattern()
        src = t.read(p)
        np.testing.assert_allclose(t.read(t.node("histogram_scan", {"input": p}, position=0.5, contrast=0.0)), src,
                                   atol=1e-5)
        self.assertEqual(t.read(t.node("histogram_scan", {"input": p}, position=0.0)).max(), 0.0)
        hard = t.read(t.node("histogram_scan", {"input": p}, position=0.5, contrast=1.0))
        self.assertTrue(set(np.unique(hard)) <= {0.0, 1.0})
        # The window is the part of 0..1 around 1 - Position (0.6..1 at Position 0.2),
        # Contrast shrinks it towards its centre, Invert Position mirrors the position.
        g = t.gray(0.7)
        for pos, con, inv, want in ((0.2, 0.0, False, 0.25), (0.2, 0.5, False, 0.0), (0.4, 0.5, False, 0.75),
                                    (0.8, 0.0, True, 0.25)):
            with self.subTest(pos=pos, con=con, inv=inv):
                n = t.node("histogram_scan", {"input": g}, position=pos, contrast=con, invert_position=inv)
                self.assertAlmostEqual(px(t.read(n))[0], want, places=5)

    def test_histogram_range_and_select(self):
        t = Graph()
        g = t.gray(0.5)
        # Range: output range Position -/+ Range / 2, both sides narrowed to the room the nearer one has.
        for r, pos, v, want in ((0.5, 0.5, 0.5, 0.5), (0.5, 0.8, 0.0, 0.6), (1.0, 0.5, 0.2, 0.2), (0.0, 0.3, 0.9, 0.3),
                                (0.73, 0.14, 1.0, 0.28), (0.73, 0.14, 0.0, 0.0)):
            with self.subTest(range=r, position=pos, value=v):
                n = t.node("histogram_range", {"input": t.gray(v)}, range=r, position=pos)
                self.assertAlmostEqual(px(t.read(n))[0], want, places=5)
        # Select: a tent of half-width Range, Contrast through Levels from Contrast / 2 to 1 - Contrast / 2.
        for pos, r, con, want in ((0.6, 0.25, 0.0, 0.6), (0.6, 0.25, 0.5, 0.7), (0.75, 0.25, 0.0, 0.0)):
            with self.subTest(position=pos, range=r, contrast=con):
                n = t.node("histogram_select", {"input": g}, position=pos, range=r, contrast=con)
                self.assertAlmostEqual(px(t.read(n))[0], want, places=4)

    def test_auto_levels_spans_full_range(self):
        t = Graph()
        lv = t.node("levels", {"input": t.pattern()}, level_out_low=[0.3] * 4, level_out_high=[0.6] * 4)
        a = t.read(t.node("auto_levels", {"input": lv}))
        self.assertAlmostEqual(a.min(), 0.0, places=4)
        self.assertAlmostEqual(a.max(), 1.0, places=4)

    def test_min_max(self):
        t = Graph()
        p = t.pattern()
        src = t.read(p)
        n = t.node("min_max", {"input": p})
        self.assertAlmostEqual(t.read(n, "min")[0, 0, 0], src.min(), places=5)  # values: 1x1
        self.assertAlmostEqual(t.read(n, "max")[0, 0, 0], src.max(), places=5)

    def test_highpass_of_flat_is_mid_gray(self):
        t = Graph()
        np.testing.assert_allclose(t.read(t.node("highpass", {"input": t.gray(0.8)})), 0.5, atol=1e-4)
        # Input - box blur(input) + 0.5, the box Radius px at 256 wide on each side: 6 px
        # from the edge of a square the result is flat again (a gaussian of that sigma would still dip).
        t = Graph(8)
        square = t.node("shape", pattern="Square", scale=0.5)
        a = t.read(t.node("highpass", {"input": square}, radius=4.0))[128, :, 0]
        edge = int(np.argmax(t.read(square)[128, :, 0] > 0.5))
        self.assertAlmostEqual(a[edge - 7], 0.5, delta=2e-3)
        self.assertLess(a[edge - 3], 0.45)

    def test_convert_round_trip_and_known_value(self):
        t = Graph()
        lin = t.node("convert_to_linear", {"input": t.gray(0.5)})
        self.assertAlmostEqual(px(t.read(lin))[0], 0.21404, places=4)
        self.assertAlmostEqual(px(t.read(t.node("convert_to_srgb", {"input": lin})))[0], 0.5, places=4)

    def test_channel_mixer_swaps(self):
        t = Graph()
        n = t.node("channel_mixer", {"input": t.color((0.1, 0.2, 0.3, 1))}, red_channel=[0, 0, 100, 0],
                   green_channel=[0, 100, 0, 0], blue_channel=[100, 0, 0, 0])
        np.testing.assert_allclose(px(t.read(n))[:3], [0.3, 0.2, 0.1], atol=1e-5)
        # The fourth component is a constant, and the sums are clipped to 0..1.
        n = t.node("channel_mixer", {"input": t.color((0.1, 0.2, 0.3, 1))}, red_channel=[100, 0, 0, 25],
                   green_channel=[0, -100, 0, 0], blue_channel=[0, 0, 200, 50])
        np.testing.assert_allclose(px(t.read(n))[:3], [0.35, 0.0, 1.0], atol=1e-5)

    def test_grayscale_conversion_advanced(self):
        t = Graph()
        c = t.color((0.2, 0.6, 0.4, 1))
        for kind, expected in (("Desaturation", 0.4), ("Average", 0.4), ("Max", 0.6), ("Min", 0.2),
                               ("Luma", 0.2126 * 0.2 + 0.7152 * 0.6 + 0.0722 * 0.4)):
            with self.subTest(kind=kind):
                self.assertAlmostEqual(px(t.read(t.node("grayscale_conversion_advanced", {"input": c},
                                                        grayscale_type=kind)))[0], expected, places=4)

    def test_quantize_color_simple(self):
        t = Graph()
        a = t.read(t.node("quantize_color_simple", {"input": t.pattern()}, quantize_red_channel=4))
        np.testing.assert_allclose(np.unique(a) * 3, np.round(np.unique(a) * 3), atol=1e-5)

    def test_quantize_grayscale_is_the_simple_quantize_of_gray(self):
        # N evenly spaced levels including 0 and 1, round(v (N - 1)) / (N - 1).
        t = Graph()
        a = t.read(t.node("quantize_grayscale", {"input": t.pattern()}, quantize=5))
        self.assertLessEqual(set(np.round(np.unique(a[..., 0]) * 4, 5)), {0.0, 1.0, 2.0, 3.0, 4.0})
        b = t.read(t.node("quantize_color_simple", {"input": t.pattern()}, quantize_red_channel=5))
        np.testing.assert_allclose(a[..., 0], b[..., 0], atol=1e-6)
        self.assertEqual(len(np.unique(t.read(t.node("quantize_grayscale", {"input": t.pattern()}))[..., 0])), 3)

    def test_quantize_color_limits_colors(self):
        t = Graph(6)
        n = t.node("quantize_color", {"input": t.pattern(True)}, max_color_amount=5)
        a = t.read(n)
        colors = np.unique(np.round(a[..., :3].reshape(-1, 3), 4), axis=0)
        self.assertLessEqual(len(colors), 5)
        self.assertGreaterEqual(len(colors), 3)
        ids = np.unique(np.round(t.read(n, "id")[..., 0] * 4, 4))
        np.testing.assert_allclose(ids, np.round(ids))  # index / (N - 1)

    def test_color_to_mask(self):
        t = Graph()
        n = t.node("color_to_mask", {"input": t.color((1, 0, 0, 1))}, color=[1.0, 0.0, 0.0])
        self.assertAlmostEqual(px(t.read(n))[0], 1.0, places=5)
        n = t.node("color_to_mask", {"input": t.color((0, 0, 1, 1))}, color=[1.0, 0.0, 0.0])
        self.assertAlmostEqual(px(t.read(n))[0], 0.0, places=5)

    def test_replace_color_range_hits_target(self):
        t = Graph()
        n = t.node("replace_color_range", {"input": t.color((1, 0, 0, 1))}, source_color=[1, 0, 0, 1],
                   target_color=[0, 0.5, 0, 1])
        np.testing.assert_allclose(px(t.read(n))[:3], [0, 0.5, 0], atol=1e-4)
        # Nearer colours are replaced more; Source Range 0 replaces nothing and a larger one reaches further.
        def blue(red, **kw):
            n = t.node("replace_color_range", {"input": t.color((red, 0, 0, 1))}, source_color=[1, 0, 0, 1],
                       target_color=[0, 0, 1, 1], threshold=0.32, **kw)
            return px(t.read(n))[2]
        self.assertGreater(blue(0.8), blue(0.6))
        self.assertGreater(blue(0.6), 0.0)
        self.assertAlmostEqual(blue(0.6, source_range=0.0), 0.0, places=5)
        self.assertGreater(blue(0.6, source_range=0.8), blue(0.6))
        n = t.node("replace_color", {"input": t.color((0.5, 0.25, 0.25, 1))}, source_color=[0.5, 0.25, 0.25, 1],
                   target_color=[0.2, 0.2, 0.6, 1])
        np.testing.assert_allclose(px(t.read(n))[:3], [0.2, 0.2, 0.6], atol=1e-3)  # hue, saturation, lightness

    def test_color_match_smoothness_rounds_the_selection(self):
        t = Graph(7)
        pat = t.pattern(color=True)
        kw = dict(mask_hue_range=90.0, mask_chroma_range=0.6, mask_luma_range=0.4)
        raw = t.read(t.node("color_match", {"input": pat}, **kw), "mask")[..., 0]
        smooth = t.read(t.node("color_match", {"input": pat}, mask_smoothness=1.0, **kw), "mask")[..., 0]
        self.assertTrue(0.0 <= smooth.min() and smooth.max() <= 1.0)
        self.assertLess(abs(smooth.mean() - raw.mean()), 0.15)  # the same regions, shrunk or grown a little
        edges = lambda m: np.abs(np.diff(m, axis=0)).sum() + np.abs(np.diff(m, axis=1)).sum()
        self.assertLess(edges(smooth), edges(raw))  # shorter contour: specks and notches gone

    def test_formulas_on_uniform_inputs(self):
        # The formulas, on uniform inputs (blurs do nothing there).
        t = Graph()
        v = (0.6, 0.3, 0.2, 1)
        c = t.color(v)
        # Color Match: a uniform image is its own average, so it becomes the target colour.
        np.testing.assert_allclose(px(t.read(t.node("color_match", {"input": c}, target_color=[0.2, 0.4, 0.7, 1])))[:3],
                                   [0.2, 0.4, 0.7], atol=2e-3)
        # Lighting Cancel Low Frequencies: a flat image (no detail) gives its mean.
        np.testing.assert_allclose(px(t.read(t.node("lighting_cancel_low_frequencies", {"input": c})))[:3], v[:3],
                                   atol=1e-4)
        np.testing.assert_allclose(px(t.read(t.node("lighting_cancel_high_frequencies", {"input": c})))[:3], v[:3],
                                   atol=1e-4)
        np.testing.assert_allclose(px(t.read(t.node("luminance_highpass", {"input": c})))[:3], v[:3], atol=1e-4)
        # Histogram Scan Non-Uniform: a ramp; contrast 0.5 about the input as is, position 0 black, 1 white,
        # more contrast pushes away from mid gray.
        def hsnu(value, position, contrast):
            n = t.node("histogram_scan_non_uniform", {"input": t.gray(value)}, use_position_input=False,
                       use_contrast_input=False, position=position, contrast=contrast)
            return px(t.read(n))[0]
        self.assertAlmostEqual(hsnu(0.6, 0.5, 0.5), 0.6, delta=0.02)
        self.assertAlmostEqual(hsnu(0.6, 0.0, 0.5), 0.0, places=3)
        self.assertAlmostEqual(hsnu(0.6, 1.0, 0.5), 1.0, places=3)
        self.assertGreater(hsnu(0.6, 0.5, 0.7), hsnu(0.6, 0.5, 0.5))
        self.assertLess(hsnu(0.4, 0.5, 0.7), hsnu(0.4, 0.5, 0.5))
        # Height Map Frequencies Mapper: Displacement pulled to mid gray by Relief / 32, Relief = input - it + 0.5.
        h = t.node("height_map_frequencies_mapper", {"input": t.gray(0.8)}, relief=8.0)
        disp = 0.8 + (0.498039 - 0.8) * 0.25
        self.assertAlmostEqual(px(t.read(h, "displacement"))[0], disp, places=4)
        self.assertAlmostEqual(px(t.read(h, "relief"))[0], 0.8 - disp + 0.5, places=4)


class Blending(unittest.TestCase):
    def blend(self, type, f, b, **kw):
        t = Graph()
        return px(t.read(t.node(type, {"foreground": t.color(f), "background": t.color(b)}, **kw)))

    def test_formulas(self):
        f, b = (0.5, 0.8, 0.25, 1), (0.4, 0.6, 0.9, 1)
        fa, ba = np.array(f[:3]), np.array(b[:3])
        cases = {"color_burn": np.clip(1 - (1 - ba) / fa, 0, 1), "color_dodge": np.clip(ba / (1 - fa), 0, 1),
                 "linear_burn": np.maximum(fa + ba - 1, 0), "difference": np.abs(fa - ba)}
        for type, expected in cases.items():
            with self.subTest(type=type):
                np.testing.assert_allclose(self.blend(type, f, b)[:3], expected, atol=1e-4)

    def test_input_of_another_size_is_read_bilinearly(self):
        """An input smaller than the node (the foreground at 1/4 of the background) is read bilinearly at each
        output pixel's centre, wrapping on a tiled graph: no 4 x 4 steps."""
        t = Graph(6)
        small = t.pattern()
        t.g.set_base(small, "output_size", {"mode": "absolute", "value": [4, 4]})
        a = t.read(t.node("blend", {"foreground": small, "background": t.gray(0.0)}))[..., 0]
        s = t.read(small)[..., 0]
        x = (np.arange(64) + 0.5) / 4.0 - 0.5
        i, f = np.floor(x).astype(int), x - np.floor(x)
        rows = s[i % 16] * (1 - f)[:, None] + s[(i + 1) % 16] * f[:, None]
        expected = rows[:, i % 16] * (1 - f)[None, :] + rows[:, (i + 1) % 16] * f[None, :]
        np.testing.assert_allclose(a, expected, atol=1e-5)

    def test_deprecated_modes(self):
        f, b = (0.5, 0.8, 0.25, 1), (0.4, 0.6, 0.9, 1)
        fa, ba = np.array(f[:3]), np.array(b[:3])
        exclusion = np.array([0.65, 0.4, 0.71875])  # clamped sum less 1.25 x product here
        bright = self.blend("exclusion", (0.9, 1, 0.5, 1), (0.9, 1, 0, 1))[:3]
        np.testing.assert_allclose(bright, [0.095, 0, 0.5], atol=1e-4)  # bright pairs: half of sum - product
        hard = np.where(fa < 0.5, 2 * ba * fa, 1 - 2 * (1 - ba) * (1 - fa))
        np.testing.assert_allclose(self.blend("exclusion", f, b)[:3], exclusion, atol=1e-4)
        np.testing.assert_allclose(self.blend("hard_light", f, b)[:3], hard, atol=1e-4)
        for mode, expected in (("Multiply", fa * ba), ("Hard light", hard), ("Exclusion", exclusion),
                               ("Color burn", np.clip(1 - (1 - ba) / fa, 0, 1)), ("Screen", 1 - (1 - ba) * (1 - fa))):
            with self.subTest(mode=mode):
                np.testing.assert_allclose(self.blend("blend_deprecated", f, b, blending_mode=mode)[:3], expected,
                                           atol=1e-4)

    def test_opacity_and_mask(self):
        np.testing.assert_allclose(self.blend("difference", (1, 1, 1, 1), (0.2, 0.4, 0.6, 1), opacity=0.0)[:3],
                                   [0.2, 0.4, 0.6], atol=1e-5)
        half = self.blend("linear_burn", (1, 1, 1, 1), (0.2, 0.4, 0.6, 1), opacity=0.5)[:3]
        np.testing.assert_allclose(half, [0.2, 0.4, 0.6], atol=1e-5)  # f = 1: linear burn is identity

    def test_luminosity_and_color(self):
        f, b = (0.9, 0.2, 0.1, 1), (0.2, 0.3, 0.8, 1)
        lum = lambda c: 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
        self.assertAlmostEqual(lum(self.blend("color_blend", f, b)), lum(b), places=3)
        self.assertAlmostEqual(lum(self.blend("luminosity_blend", f, b)), lum(f), places=3)

    def test_dissolve_picks_either(self):
        t = Graph()
        a = t.read(t.node("dissolve", {"foreground": t.color((1, 1, 1, 1)), "background": t.color((0, 0, 0, 1))},
                          opacity=0.5))
        self.assertTrue(set(np.unique(a[..., 0])) <= {0.0, 1.0})
        self.assertAlmostEqual(a[..., 0].mean(), 0.5, delta=0.1)

    def test_switches(self):
        t = Graph()
        a, b = t.gray(0.25), t.gray(0.75)
        self.assertAlmostEqual(px(t.read(t.node("switch", {"input_1": a, "input_2": b})))[0], 0.25, places=5)
        self.assertAlmostEqual(px(t.read(t.node("switch", {"input_1": a, "input_2": b}, switch_value=False)))[0],
                               0.75, places=5)
        ms = {f"input_{i}": t.gray(i / 10) for i in range(1, 6)}
        self.assertAlmostEqual(px(t.read(t.node("multi_switch", ms, input_number=5, input_selection=4)))[0], 0.4,
                               places=5)


class Blurs(unittest.TestCase):
    def test_blurs_keep_flat_images_and_mean(self):
        t = Graph()
        p = t.pattern()
        mean = t.read(p).mean()
        for type in ("blur_hq", "anisotropic_blur", "non_uniform_blur", "radial_blur"):
            with self.subTest(type=type):
                np.testing.assert_allclose(t.read(t.node(type, {"input": t.gray(0.3)})), 0.3, atol=1e-4)
                b = t.read(t.node(type, {"input": p}))
                self.assertAlmostEqual(b.mean(), mean, delta=0.02)
                self.assertLess(np.abs(np.diff(b[..., 0], axis=1)).mean(), np.abs(np.diff(t.read(p)[..., 0], axis=1)).mean())

    def test_reduced_blur_looks_like_the_full_one(self):
        """A blur from sigma 4 texels on runs on a reduced copy (its cost then never above a narrower blur's), narrowed
        by what the reduction and the enlargement add: it matches the blur at full size, on white noise."""
        from trafilatura3d.nodes.filters.basic import _lib
        self.assertEqual(_lib.reduction((256, 256), [3.9], _lib.GAUSS), 1)
        self.assertEqual(_lib.reduction((256, 256), [4.0], _lib.GAUSS), 2)
        reduced = {}
        for sigma_limit in (_lib.REDUCED_SIGMA, 1e9):  # 1e9: never reduced
            old = _lib.REDUCED_SIGMA
            _lib.REDUCED_SIGMA = sigma_limit
            EV.clear()  # same jobs: cooked again
            try:
                t = Graph(8)
                reduced[sigma_limit] = [t.read(t.node("blur_hq", {"input": t.node("white_noise")}, intensity=i))
                                        for i in (7.0, 8.0, 14.0)]  # sigma 4.2, 4.9, 8.4 px at 256
            finally:
                _lib.REDUCED_SIGMA = old
        EV.clear()
        for a, b in zip(*reduced.values()):
            self.assertGreater(np.abs(a - b).max(), 0.0)  # one was reduced
            self.assertLess(np.abs(a - b).max(), 0.012)

    def test_enlarged_blur_has_no_lines(self):
        """A blur reduced by 4 and enlarged back has no lines along the reduced copy's rows and columns: its curvature
        is the same at every phase of the copy's texels (a bilinear enlargement is flat inside a texel and bends only
        at its edges: the slope jumps there, lines in any relief made from it)."""
        from trafilatura3d.nodes.filters.basic import _lib
        t = Graph(8)
        a = t.read(t.node("blur_hq", {"input": t.node("white_noise")}, intensity=16.0))[..., 0]  # sigma 9.7 px
        self.assertEqual(_lib.reduction((256, 256), [9.7], _lib.GAUSS), 4)
        for axis in (0, 1):
            bend = np.abs(np.roll(a, -1, axis) - 2 * a + np.roll(a, 1, axis)).mean(axis=1 - axis)
            phases = [bend[k::4].mean() for k in range(4)]
            self.assertLess(max(phases) / min(phases), 1.3)

    def test_non_uniform_blur_passes(self):
        # Chained passes of Blades warps scaled by the Blur Map; black map or Intensity 0
        # = the input, more passes (Samples) = more blur.
        t = Graph(6)
        p = t.pattern()
        src = t.read(p)
        for params, inputs in (({"intensity": 0.0}, {}), ({}, {"blur_map": t.gray(0.0)})):
            with self.subTest(params=params):
                np.testing.assert_allclose(t.read(t.node("non_uniform_blur", {"input": p, **inputs}, **params)), src,
                                           atol=1e-5)
        rough = [np.abs(np.diff(t.read(t.node("non_uniform_blur", {"input": p}, samples=s))[..., 0], axis=1)).mean()
                 for s in (1, 4)]
        self.assertLess(rough[1], rough[0])

    def test_slope_blur_modes_order(self):
        t = Graph()
        p = t.pattern()
        s = t.node("blur_hq", {"input": p}, intensity=3.0)
        lo = t.read(t.node("slope_blur", {"input": p, "slope": s}, mode="Min"))
        hi = t.read(t.node("slope_blur", {"input": p, "slope": s}, mode="Max"))
        avg = t.read(t.node("slope_blur", {"input": p, "slope": s}))
        self.assertTrue(np.all(lo <= avg + 1e-5) and np.all(avg <= hi + 1e-5))
        # Soft min / max over a path as long as Warp's: a small spread on this gentle slope
        self.assertGreater((hi - lo).max(), 0.01)
        np.testing.assert_allclose(t.read(t.node("slope_blur", {"input": p, "slope": s}, intensity=0.0)), t.read(p),
                                   atol=1e-6)  # Intensity 0: the input


class Channels(unittest.TestCase):
    def test_rgba_split_merge_round_trip(self):
        t = Graph()
        src = t.node("levels", {"input": t.pattern(True)}, level_out_high=[1.0, 1.0, 1.0, 0.5])
        s = t.node("rgba_split", {"input": src})
        m = t.node("rgba_merge", {k: (s, k) for k in "rgba"})
        np.testing.assert_allclose(t.read(m), t.read(src), atol=1e-6)

    def test_alpha_split_merge(self):
        t = Graph()
        c = t.color((0.2, 0.4, 0.6, 0.5))
        s = t.node("alpha_split", {"input": c})
        np.testing.assert_allclose(px(t.read(s, "rgb")), [0.2, 0.4, 0.6, 1.0], atol=1e-6)
        self.assertAlmostEqual(px(t.read(s, "a"))[0], 0.5, places=6)
        m = t.node("alpha_merge", {"rgb": (s, "rgb"), "a": (s, "a")})
        np.testing.assert_allclose(px(t.read(m)), [0.2, 0.4, 0.6, 0.5], atol=1e-6)

    def test_premultiply_round_trip(self):
        t = Graph()
        pm = t.node("straight_to_pre_multiplied", {"input": t.color((0.2, 0.4, 0.6, 0.5))})
        np.testing.assert_allclose(px(t.read(pm)), [0.1, 0.2, 0.3, 0.5], atol=1e-6)
        np.testing.assert_allclose(px(t.read(t.node("pre_multiplied_to_straight", {"input": pm}))),
                                   [0.2, 0.4, 0.6, 1.0], atol=1e-6)


class Tiling(unittest.TestCase):
    def seam(self, a):
        g = a[..., 0]
        inner = np.abs(np.diff(g, axis=1)).mean() + 1e-3
        return np.abs(g[:, 0] - g[:, -1]).mean() / inner, np.abs(g[0] - g[-1]).mean() / inner

    def test_outputs_tile(self):
        t = Graph(6)
        ramp = t.pattern()  # its gentle slope in x does not tile
        for type in ("make_tileable_patch", "make_tileable_photo"):
            with self.subTest(type=type):
                sx, sy = self.seam(t.read(t.node(type, {"input": ramp})))
                self.assertLess(sx, 3.0)
                self.assertLess(sy, 3.0)

    def test_photo_keeps_centre(self):
        t = Graph(6)
        p = t.pattern()
        a, src = t.read(t.node("make_tileable_photo", {"input": p}, mask_warping_h=0.0, mask_warping_v=0.0)), t.read(p)
        np.testing.assert_allclose(a[28:36, 28:36], src[28:36, 28:36], atol=1e-6)


class Transforms(unittest.TestCase):
    def test_mirror_symmetry(self):
        t = Graph()
        p = t.pattern()
        a = t.read(t.node("mirror", {"input": p}))
        np.testing.assert_allclose(a, a[:, ::-1], atol=1e-6)
        np.testing.assert_allclose(a[:, :16], t.read(p)[:, :16], atol=1e-6)  # left side is the source
        b = t.read(t.node("mirror", {"input": p}, mode="Mirror Axis Y", invert_axis_y=True))
        np.testing.assert_allclose(b, b[::-1], atol=1e-6)
        np.testing.assert_allclose(b[16:], t.read(p)[16:], atol=1e-6)
        c = t.read(t.node("mirror", {"input": p}, mode="Mirror Corner"))
        np.testing.assert_allclose(c, c[::-1, ::-1], atol=1e-6)

    def test_symmetry_modes_are_symmetric(self):
        t = Graph()
        p = t.pattern()
        src = t.read(p)
        # Each stage blends the image with a transformed copy (Mirror Y flips top-bottom,
        # Diagonal Left mirrors across the TR-BL diagonal); Copy is the copy itself, Max is symmetric.
        checks = {"Mirror Y": lambda a: a[::-1], "Mirror X": lambda a: a[:, ::-1],
                  "Diagonal Left": lambda a: a.transpose(1, 0, 2)[::-1, ::-1],
                  "Diagonal Right": lambda a: a.transpose(1, 0, 2), "Mirror X/Y": lambda a: a[::-1, ::-1]}
        for mode, f in checks.items():
            with self.subTest(mode=mode):
                a = t.read(t.node("symmetry", {"input": p}, symmetry_mode=mode, transfer_mode="Max"))
                np.testing.assert_allclose(a, f(a), atol=1e-6)
                a = t.read(t.node("symmetry", {"input": p}, symmetry_mode=mode, transfer_mode="Copy"))
                np.testing.assert_allclose(a, f(src), atol=1e-6)
        a = t.read(t.node("symmetry", {"input": p}, transfer_mode="Max"))  # 8: both diagonals, quarter turns
        for f in (lambda a: a.transpose(1, 0, 2), lambda a: np.rot90(a), lambda a: a[::-1]):
            np.testing.assert_allclose(a, f(a), atol=1e-6)

    def test_symmetry_slice_blend(self):
        t = Graph()
        p = t.pattern()
        src = t.read(p)
        # Default Corner, the top-right quadrant mirrored onto the others.
        a = t.read(t.node("symmetry_slice", {"input": p}))
        np.testing.assert_allclose(a, a[:, ::-1], atol=1e-6)
        np.testing.assert_allclose(a, a[::-1], atol=1e-6)
        np.testing.assert_allclose(a[:16, 16:], src[:16, 16:], atol=1e-6)
        v = t.read(t.node("symmetry_slice", {"input": p}, symmetry_mode="Vertical"))
        np.testing.assert_allclose(v, v[:, ::-1], atol=1e-6)
        np.testing.assert_allclose(v[:, :16], src[:, :16], atol=1e-6)
        # Blend widens the cut into a smooth transition across the image: the copy fades in from the left edge,
        # half way at the mirror line.
        b = t.read(t.node("symmetry_slice", {"input": p}, symmetry_mode="Vertical", blend=1.0))
        x = (np.arange(32) + 0.5) / 32 - 0.5
        ramp = np.clip((x + 0.5), 0, 1) ** 2 * (3 - 2 * np.clip((x + 0.5), 0, 1))
        np.testing.assert_allclose(b, src + (src[:, ::-1] - src) * ramp[None, :, None], atol=1e-4)

    def test_non_uniform_rotation_quarter_turn_clockwise(self):
        t = Graph()
        p = t.pattern()
        a = t.read(t.node("non_uniform_rotation", {"input": p}, rotation_angle_offset=0.25, filtering_mode="Nearest"))
        np.testing.assert_allclose(a, np.rot90(t.read(p), k=-1), atol=1e-6)

    def test_safe_transform_tiles_exactly(self):
        t = Graph(6)
        p = t.pattern()
        src = t.read(p)
        np.testing.assert_allclose(t.read(t.node("safe_transform", {"input": p})), src, atol=1e-6)
        a = t.read(t.node("safe_transform", {"input": p}, tile=2))
        np.testing.assert_allclose(a[:32, :32], a[32:, 32:], atol=1e-6)
        sh = t.read(t.node("safe_transform", {"input": p}, offset=[0.25, 0.0]))
        np.testing.assert_allclose(sh, np.roll(src, 16, axis=1), atol=1e-6)
        # An even Tile puts whole copies in the corners; Tile Safe Rotation goes down to a
        # multiple of 45 degrees.
        blocks = src.reshape(32, 2, 32, 2, -1).mean(axis=(1, 3))
        np.testing.assert_allclose(a[:32, :32], blocks, atol=1e-5)
        np.testing.assert_allclose(t.read(t.node("safe_transform", {"input": p}, rotation=0.1)), src, atol=1e-6)

    def test_identity_transforms(self):
        t = Graph()
        p = t.pattern()
        src = t.read(p)
        cases = {"skew": {"amount": 0.0}, "trapezoid_transform": {"top_stretch": 0.0},
                 "quad_transform": {"p00": [0.0, 0.0], "p10": [1.0, 0.0]}, "non_square_transform": {},
                 "texture_3d_offset": {}, "auto_crop": {"mode": "Crop auto"}}
        for type, params in cases.items():
            with self.subTest(type=type):
                np.testing.assert_allclose(t.read(t.node(type, {"input": p}, **params)), src, atol=1e-5)

    def test_skew_shifts_rows(self):
        t = Graph()
        p = t.pattern()
        a = t.read(t.node("skew", {"input": p}, amount=0.5, align="Top Left"))
        row = t.read(p)[16]  # row centre 16.5 / 32: shifted right by 8.25 px, bilinear
        np.testing.assert_allclose(a[16], 0.75 * np.roll(row, 8, axis=0) + 0.25 * np.roll(row, 9, axis=0), atol=1e-4)

    def test_polar_round_trip(self):
        t = Graph(7)
        p = t.pattern()
        back = t.read(t.node("cartesian_to_polar", {"input": t.node("polar_to_cartesian", {"input": p})}))
        src = t.read(p)
        r = np.hypot(*np.meshgrid(np.arange(128) - 63.5, np.arange(128) - 63.5))
        ring = (r > 16) & (r < 56)  # away from the centre (all angles squeezed) and the corners
        self.assertLess(np.abs(back - src)[ring].mean(), 0.03)

    def test_clone_copies_masked_area(self):
        t = Graph()
        p = t.pattern(True)
        src = t.read(p)
        n = t.node("clone", {"source": p, "mask": t.gray(1.0)}, offset=[0.25, 0.0])
        a = t.read(n)
        np.testing.assert_allclose(a[:, :20], src[:, 8:28], atol=1e-4)  # content moved left by 8 px
        np.testing.assert_allclose(t.read(t.node("clone", {"source": p})), src, atol=1e-6)  # black mask

    def test_3d_offset_moves_slices(self):
        t = Graph(6)  # 64 px = 4 x 4 slices of 16 px
        p = t.pattern()
        src = t.read(p)
        a = t.read(t.node("texture_3d_offset", {"input": p}, offset=[0.0, 0.0, 1.0 / 16.0]))
        np.testing.assert_allclose(a[:16, 16:32], src[:16, :16], atol=1e-5)  # slice 1 now shows slice 0

    def test_noise_upscale_3_copies(self):
        # Four 2x copies (turned or mirrored) picked by the Mask; a uniform mask shows
        # one copy, repeating twice across the image, and black and white show different copies.
        t = Graph(6)
        p = t.pattern()
        a = t.read(t.node("noise_upscale_quad", {"grayscale": p}))
        b = t.read(t.node("noise_upscale_quad", {"grayscale": p, "mask": t.gray(0.0)}))
        for img in (a, b):
            np.testing.assert_allclose(img, np.roll(img, (32, 32), axis=(0, 1)), atol=1e-5)
        self.assertGreater(np.abs(a - b).mean(), 0.05)

    def test_noise_upscale_tiles(self):
        t = Graph(6)
        n = t.node("noise_upscale_masked", {"input": t.node("make_tileable_patch", {"input": t.pattern()})})
        a = t.read(n)[..., 0]
        inner = np.abs(np.diff(a, axis=1)).mean()
        self.assertLess(np.abs(a[:, 0] - a[:, -1]).mean(), inner * 3)


if __name__ == "__main__":
    unittest.main()
