"""The Paint workspace's color filters and generators (trafilatura3d/nodes/paint/color/) behave as described."""

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


class GrayscaleConversion(unittest.TestCase):
    def test_channel_weights(self):  # "weighing the luminance of each color channel"
        t = Graph()
        red = t.color((1.0, 0.0, 0.0, 1.0))
        n = t.node("paint_grayscale_conversion", {"input": red}, mode="Channels Weights", red=0.3, green=0.59,
                   blue=0.11, alpha=0.0)
        self.assertAlmostEqual(px(t.read(n))[0], 0.3, places=5)
        n = t.node("paint_grayscale_conversion", {"input": red}, mode="Channels Weights", red=0.0, green=0.0,
                   blue=0.0, alpha=0.5)
        self.assertAlmostEqual(px(t.read(n))[0], 0.5, places=5)

    def test_modes_and_invert(self):
        t = Graph()
        c = t.color((0.2, 0.6, 0.4, 1.0))
        for mode, want in (("Desaturation", 0.4), ("Average", 0.4), ("Max", 0.6), ("Min", 0.2)):
            self.assertAlmostEqual(px(t.read(t.node("paint_grayscale_conversion", {"input": c}, mode=mode)))[0],
                                   want, places=5, msg=mode)
        inv = t.node("paint_grayscale_conversion", {"input": c}, mode="Max", invert=True)
        self.assertAlmostEqual(px(t.read(inv))[0], 0.4, places=5)

    def test_custom_input(self):  # Channel Input picks the custom texture
        t = Graph()
        n = t.node("paint_grayscale_conversion", {"input": t.color((0, 0, 0, 1)), "custom_input": t.color((1, 1, 1, 1))},
                   channel_input="Custom Input", mode="Max")
        self.assertAlmostEqual(px(t.read(n))[0], 1.0, places=5)

    def test_generator_types(self):  # Red channel: "Uses only the red channel"...
        t = Graph()
        c = t.color((0.2, 0.6, 0.4, 1.0))
        for kind, want in (("Red channel", 0.2), ("Green channel", 0.6), ("Blue channel", 0.4), ("Max", 0.6),
                           ("Min", 0.2), ("Desaturation", 0.4), ("Average", 0.4)):
            n = t.node("paint_grayscale_conversion_generator", {"source_color": c}, grayscale_type=kind)
            self.assertAlmostEqual(px(t.read(n))[0], want, places=4, msg=kind)
        luma = t.node("paint_grayscale_conversion_generator", {"source_color": t.color((0, 1, 0, 1))},
                      grayscale_type="Luma")
        self.assertGreater(px(t.read(luma))[0], 0.5)  # favoring green


class Tones(unittest.TestCase):
    def test_balance_defaults_unchanged(self):
        t = Graph()
        a = t.read(t.node("paint_color_balance", {"input": t.pattern(True)}))
        np.testing.assert_allclose(a, t.read(t.pattern(True)), atol=1e-5)

    def test_balance_shifts_by_range(self):
        t = Graph()
        bright, dark = t.color((0.9, 0.9, 0.9, 1)), t.color((0.1, 0.1, 0.1, 1))
        n = t.node("paint_color_balance", {"input": bright}, highlights_cyan_red=1.0)
        r, g, b = px(t.read(n))[:3]
        self.assertGreater(r, 0.95)  # toward red
        self.assertAlmostEqual(g, 0.9, places=5)
        n = t.node("paint_color_balance", {"input": dark}, highlights_cyan_red=1.0)
        self.assertLess(px(t.read(n))[0], 0.12)  # the highlights barely touch the shadows
        n = t.node("paint_color_balance", {"input": dark}, shadows_yellow_blue=-1.0)
        self.assertLess(px(t.read(n))[2], 0.1)  # toward yellow: less blue

    def test_correct(self):
        t = Graph()
        src = t.pattern(True)
        np.testing.assert_allclose(t.read(t.node("paint_color_correct", {"input": src})), t.read(src), atol=1e-5)
        flat = {f"{r}_saturation": -1.0 for r in ("shadows", "midtones", "highlights")}
        a = t.read(t.node("paint_color_correct", {"input": src}, **flat))
        np.testing.assert_allclose(a[..., 0], a[..., 1], atol=1e-5)
        np.testing.assert_allclose(a[..., 1], a[..., 2], atol=1e-5)
        mid = t.color((0.5, 0.5, 0.5, 1))
        self.assertGreater(px(t.read(t.node("paint_color_correct", {"input": mid}, midtones_luminosity=0.5)))[0], 0.6)


class Gradients(unittest.TestCase):
    KEYS = {"color_1": [1.0, 0.0, 0.0], "color_1_position": 0.0, "color_2": [0.0, 0.0, 1.0], "color_2_position": 1.0,
            "color_3": [0.0, 1.0, 0.0], "color_3_position": 0.5}

    def test_gradient_remaps_gray(self):
        t = Graph()
        mid = t.gray(0.5)
        two = t.node("paint_gradient", {"input": mid}, color_quantity=2, **self.KEYS)
        np.testing.assert_allclose(px(t.read(two))[:3], [0.5, 0.0, 0.5], atol=1e-5)
        three = t.node("paint_gradient", {"input": mid}, color_quantity=3, **self.KEYS)  # Color Quantity
        np.testing.assert_allclose(px(t.read(three))[:3], [0.0, 1.0, 0.0], atol=1e-5)
        low = t.node("paint_gradient", {"input": t.gray(0.0)}, color_quantity=2, **self.KEYS)
        np.testing.assert_allclose(px(t.read(low))[:3], [1.0, 0.0, 0.0], atol=1e-5)
        inv = t.node("paint_gradient", {"input": t.gray(0.0)}, color_quantity=2, invert=True, **self.KEYS)
        np.testing.assert_allclose(px(t.read(inv))[:3], [0.0, 0.0, 1.0], atol=1e-5)

    def test_gradient_debug_shows_the_gradient(self):
        t = Graph()
        a = t.read(t.node("paint_gradient", {"input": t.gray(0.0)}, display_gradient_debug=True, **self.KEYS))
        self.assertGreater(a[0, 0, 0], 0.9)  # left = Color 1
        self.assertGreater(a[0, -1, 2], 0.9)  # right = Color 2

    def test_gradient_curve(self):
        t = Graph()
        g = t.gray(0.3)
        self.assertAlmostEqual(px(t.read(t.node("paint_gradient_curve", {"input": g}, preset="Linear")))[0], 0.3, places=5)
        inv = t.node("paint_gradient_curve", {"input": g}, preset="Linear", invert=True)
        self.assertAlmostEqual(px(t.read(inv))[0], 0.7, places=5)
        bright = t.node("paint_gradient_curve", {"input": g}, preset="Linear", balance=0.6)
        self.assertGreater(px(t.read(bright))[0], 0.3)  # "similar to a brightness adjustment"

    def test_gradient_dynamic_samples_the_source(self):
        t = Graph()
        src = t.color((0.1, 0.8, 0.3, 1))
        n = t.node("paint_gradient_dynamic", {"input": t.gray(0.4), "gradient_source": src})
        np.testing.assert_allclose(px(t.read(n))[:3], [0.1, 0.8, 0.3], atol=1e-5)
        ramp = t.node("paint_gradient_dynamic", {"input": t.gray(0.25), "gradient_source": t.pattern(True)},
                      gradient_orientation="Vertical", gradient_input_position=0.9)
        ramp_h = t.node("paint_gradient_dynamic", {"input": t.gray(0.25), "gradient_source": t.pattern(True)},
                        gradient_input_position=0.9)
        self.assertFalse(np.allclose(px(t.read(ramp)), px(t.read(ramp_h))))  # orientation matters


class Quantize(unittest.TestCase):
    def test_at_most_color_amount_colors(self):
        t = Graph(6)
        for amount in (2, 5):
            n = t.node("paint_quantize", {"input": t.pattern(True)}, color_amount=amount, contour_smoothing=0.0)
            colors = np.unique(np.round(t.read(n)[..., :3].reshape(-1, 3), 5), axis=0)
            used = t.read(n, "palette_color_amount")[0, 0, 0]
            self.assertLessEqual(len(colors), amount)
            self.assertEqual(len(colors), int(used))

    def test_uniform_image_has_one_color(self):  # "the actual count may be lower"
        t = Graph()
        n = t.node("paint_quantize", {"input": t.color((0.3, 0.5, 0.7, 1))}, color_amount=8)
        self.assertEqual(t.read(n, "palette_color_amount")[0, 0, 0], 1.0)
        np.testing.assert_allclose(px(t.read(n))[:3], [0.3, 0.5, 0.7], atol=2e-3)

    def test_alpha(self):
        t = Graph()
        n = t.node("paint_quantize", {"input": t.color((0.3, 0.5, 0.7, 0.6))}, apply_to_alpha=True, alpha_threshold=0.5)
        self.assertEqual(px(t.read(n))[3], 1.0)
        n = t.node("paint_quantize", {"input": t.color((0.3, 0.5, 0.7, 0.6))})
        self.assertAlmostEqual(px(t.read(n))[3], 0.6, places=5)


class ColorMatch(unittest.TestCase):
    def test_source_goes_to_target(self):
        t = Graph()
        src = (0.6, 0.3, 0.2)
        n = t.node("paint_color_match", {"input": t.color((*src, 1))}, source_color_mode="Parameter",
                   source_color=list(src), target_color=[0.2, 0.4, 0.8])
        np.testing.assert_allclose(px(t.read(n))[:3], [0.2, 0.4, 0.8], atol=2e-3)
        avg = t.node("paint_color_match", {"input": t.color((*src, 1))}, target_color=[0.2, 0.4, 0.8])
        np.testing.assert_allclose(px(t.read(avg))[:3], [0.2, 0.4, 0.8], atol=2e-3)

    def test_details_kept_or_flattened(self):
        t = Graph(6)
        pat = t.pattern(True)
        kept = t.read(t.node("paint_color_match", {"input": pat}))
        flat = t.read(t.node("paint_color_match", {"input": pat}, custom_color_variation=True, hue=0.0, chroma=0.0,
                             luma=0.0))
        self.assertGreater(kept[..., :3].reshape(-1, 3).std(axis=0).max(), 0.05)  # "maintain details"
        self.assertLess(flat[..., :3].reshape(-1, 3).std(axis=0).max(), 1e-3)


class PBRValidate(unittest.TestCase):
    def check(self, base, metal, **params):
        t = Graph()
        n = t.node("paint_pbr_validate", {"basecolor": t.color((base, base, base, 1)), "metallic": t.gray(metal)},
                   overlay_map=False, **params)
        return tuple(np.round(px(t.read(n, "basecolor"))[:3], 3))

    def test_albedo_and_metal(self):
        red, green, black = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)
        self.assertEqual(self.check(0.05, 0.0), red)  # too dark for a non-metal
        self.assertEqual(self.check(0.5, 0.0), green)
        self.assertEqual(self.check(0.5, 1.0), red)  # too dark for a metal reflectance
        self.assertEqual(self.check(0.9, 1.0), green)
        self.assertEqual(self.check(0.15, 0.0), green)  # 38 sRGB: above the tolerant 30...
        self.assertEqual(self.check(0.15, 0.0, albedo_dark_range_threshold="Strict (50 sRGB)"), red)  # ...not 50
        self.assertEqual(self.check(0.5, 1.0, validation_mode="Albedo"), black)  # metals not checked

    def test_overlay_keeps_the_map(self):
        t = Graph()
        n = t.node("paint_pbr_validate", {"basecolor": t.color((0.5, 0.5, 0.5, 1)), "metallic": t.gray(0.0)})
        np.testing.assert_allclose(px(t.read(n, "basecolor"))[:3], [0.25, 0.75, 0.25], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
