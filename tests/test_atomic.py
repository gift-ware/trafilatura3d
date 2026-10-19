"""Atomic nodes behave as documented."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph, imageio

EV = Evaluator()
ABS = lambda log2: {"mode": "absolute", "value": [log2, log2]}


class Builder:
    """A tiny graph: nodes with params, connected in a line, read at 32 px."""

    def __init__(self, log2=5):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", log2)

    def node(self, type, inputs=None, **params):
        n = self.g.add_node(type)
        for k, v in params.items():
            self.g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            self.g.connect(src, "output", n, port)
        return n

    def color(self, rgba):
        return self.node("uniform_color", output_color=list(rgba))

    def gray(self, v):
        return self.node("uniform_color", color_mode="Grayscale", output_color=[v, v, v, 1])

    def read(self, n):
        a = EV.read(compile_graph(self.pkg, "t").jobs[n], "output")
        return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)

    def base(self, n, name, value):
        self.g.set_base(n, name, value)


def px(a, x=0, y=0):
    return a[y, x]


class BlendModes(unittest.TestCase):
    def blend(self, mode, f, b, opacity=1.0, **kw):
        t = Builder()
        for n in (fg := t.color(f)), (bg := t.color(b)):
            t.base(n, "output_format", {"mode": "absolute", "value": "32f"})
        n = t.node("blend", {"foreground": fg, "background": bg}, blending_mode=mode, opacity=opacity, **kw)
        t.base(n, "output_format", {"mode": "absolute", "value": "32f"})
        return px(t.read(n))[:3]

    def test_modes(self):
        f, b = (0.25, 0.5, 0.75, 1), (0.5, 0.5, 0.5, 1)
        cases = {"Copy": [0.25, 0.5, 0.75], "Add": [0.75, 1.0, 1.25], "Subtract": [0.25, 0.0, -0.25],
                 "Multiply": [0.125, 0.25, 0.375], "Max": [0.5, 0.5, 0.75], "Min": [0.25, 0.5, 0.5],
                 "Divide": [2.0, 1.0, 0.5 / 0.75], "Screen": [0.625, 0.75, 0.875],
                 "Add sub": [0.0, 0.5, 1.0], "Overlay": [0.25, 0.5, 0.75],
                 "Soft light": [0.375, 0.5, 0.25 + 0.5 ** 0.5 * 0.5]}  # Photoshop's, measured on the original
        for mode, expected in cases.items():
            with self.subTest(mode=mode):
                np.testing.assert_allclose(self.blend(mode, f, b), expected, atol=1e-3)

    def test_alpha_is_the_foreground_over_the_background(self):
        # Every alpha blending gives s + ba (1 - s), s = foreground alpha x opacity: 0.5 + 0.4 x 0.5 = 0.7.
        t = Builder()
        fg, bg = t.color((1, 0, 0, 0.5)), t.color((0, 0, 1, 0.4))
        for mode in ("Use source alpha", "Ignore alpha", "Straight alpha blending", "Premultiplied alpha blending"):
            with self.subTest(alpha_blending=mode):
                n = t.node("blend", {"foreground": fg, "background": bg}, alpha_blending=mode)
                self.assertAlmostEqual(float(px(t.read(n))[3]), 0.7, delta=0.005)

    def test_opacity_zero_is_background(self):
        np.testing.assert_allclose(self.blend("Multiply", (1, 0, 0, 1), (0.2, 0.4, 0.6, 1), opacity=0.0),
                                   [0.2, 0.4, 0.6], atol=1e-3)

    def test_cropping_area_shows_background_outside(self):
        t = Builder()
        n = t.node("blend", {"foreground": t.color((1, 1, 1, 1)), "background": t.color((0, 0, 0, 1))},
                   cropping_area=[0.0, 0.5, 0.0, 1.0])
        a = t.read(n)
        self.assertAlmostEqual(a[16, 4, 0], 1.0, places=2)
        self.assertAlmostEqual(a[16, 28, 0], 0.0, places=2)


class Adjustments(unittest.TestCase):
    def test_levels(self):
        t = Builder()
        n = t.node("levels", {"input": t.gray(0.5)}, level_in_low=[0.25] * 4, level_in_high=[0.75] * 4,
                   level_in_mid=[0.5] * 4)
        self.assertAlmostEqual(px(t.read(n))[0], 0.5, delta=0.01)
        # In mid is a gamma: exponent e -> 0.5 + (e - 1) / 16 (e >= 1), else
        # (9 - 1 / e) / 16. 0.6 sits half way into [0.5, 0.7].
        for e, mid in ((2.0, 0.5625), (0.5, (9 - 2) / 16)):
            n2 = t.node("levels", {"input": t.gray(0.6)}, level_in_low=[0.5] * 4, level_in_high=[0.7] * 4,
                        level_in_mid=[mid] * 4, level_out_low=[0.2] * 4, level_out_high=[0.4] * 4)
            self.assertAlmostEqual(px(t.read(n2))[0], 0.2 + 0.2 * 0.5 ** e, delta=0.01)

    def test_grayscale_conversion_extracts_a_channel(self):
        t = Builder()
        n = t.node("grayscale_conversion", {"input": t.color((0.2, 0.6, 0.9, 1))}, channel_weights=[0, 1, 0, 0])
        self.assertAlmostEqual(px(t.read(n))[0], 0.6, delta=0.01)

    def test_hsl_neutral_is_identity(self):
        t = Builder()
        n = t.node("hsl", {"input": t.color((0.2, 0.6, 0.9, 1))})
        np.testing.assert_allclose(px(t.read(n))[:3], [0.2, 0.6, 0.9], atol=0.01)
        # Each slider adds 2 (value - 0.5): Lightness 0.75 lifts a mid gray to 1, Saturation 0 makes any colour gray.
        self.assertAlmostEqual(px(t.read(t.node("hsl", {"input": t.color((0.5, 0.5, 0.5, 1))}, lightness=0.75)))[0],
                               1.0, delta=0.01)
        g = px(t.read(t.node("hsl", {"input": t.color((0.2, 0.6, 0.9, 1))}, saturation=0.0)))[:3]
        self.assertAlmostEqual(np.ptp(g), 0.0, delta=0.01)

    def test_hsl_takes_a_float_colour_as_it_is(self):
        # A Subtract at 32 bits float pushes blue to -0.02: HSL reads it so (its saturation past 1), then clamps the
        # result; measured on the original (a clamped input would give 0.324, 0.362, 0.102).
        t = Builder()
        f32 = {"mode": "absolute", "value": "32f"}
        dark, blue = t.color((0.157, 0.184, 0.0, 1)), t.color((0.0, 0.0, 0.02, 1))
        sub = t.node("blend", {"background": dark, "foreground": blue}, blending_mode="Subtract")
        hsl = t.node("hsl", {"input": sub}, saturation=0.28, lightness=0.57)
        for n in (dark, blue, sub, hsl):
            t.base(n, "output_format", f32)
        np.testing.assert_allclose(px(t.read(hsl))[:3], [0.354, 0.4, 0.044], atol=0.003)

    def test_channel_shuffle(self):
        t = Builder()
        n = t.node("channel_shuffle", {"input_1": t.color((0.1, 0.2, 0.3, 1)), "input_2": t.gray(0.7)},
                   red_channel="Input 1 Blue", green_channel="Input 2 Red", blue_channel="Input 1 Red",
                   alpha_channel="Input 2 Alpha")
        np.testing.assert_allclose(px(t.read(n)), [0.3, 0.7, 0.1, 0.7], atol=0.01)

    def test_curve_identity_and_fold(self):
        t = Builder()
        n = t.node("curve", {"input": t.gray(0.3)})
        self.assertAlmostEqual(px(t.read(n))[0], 0.3, delta=0.01)
        n2 = t.node("curve", {"input": t.gray(0.3)}, curve=[[0, 1], [1, 0]])
        self.assertAlmostEqual(px(t.read(n2))[0], 0.7, delta=0.01)

    def test_curve_bezier_handles(self):
        # Handles on their points: a straight segment. Flat handles at a third: an S through (0.5, 0.5), below the
        # line before it. Points without handles: the curve through them as before.
        t = Builder()
        def at(curve, v):
            return px(t.read(t.node("curve", {"input": t.gray(v)}, curve=curve)))[0]
        line = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
        s = [[0, 0, 0, 0, 1 / 3, 0], [1, 1, 2 / 3, 1, 1, 1]]
        self.assertAlmostEqual(at(line, 0.3), 0.3, delta=0.01)
        self.assertAlmostEqual(at(s, 0.5), 0.5, delta=0.01)
        self.assertLess(at(s, 0.25), 0.2)
        self.assertAlmostEqual(at([[0, 0], [1, 1]], 0.3), 0.3, delta=0.01)

    def test_gradient_map_ends(self):
        t = Builder()
        grad = {"keys": [{"pos": 0, "color": [1, 0, 0]}, {"pos": 1, "color": [0, 0, 1]}]}
        lo = t.node("gradient_map", {"input": t.gray(0.0)}, gradient=grad)
        hi = t.node("gradient_map", {"input": t.gray(1.0)}, gradient=grad)
        np.testing.assert_allclose(px(t.read(lo))[:3], [1, 0, 0], atol=0.01)
        np.testing.assert_allclose(px(t.read(hi))[:3], [0, 0, 1], atol=0.01)

    def test_gradient_no_interpolation_steps(self):
        t = Builder()
        grad = {"interpolation": "none", "keys": [{"pos": 0, "color": [0, 0, 0]}, {"pos": 0.5, "color": [1, 1, 1]}]}
        n = t.node("gradient_map", {"input": t.gray(0.4)}, gradient=grad, color_mode="Grayscale")
        self.assertAlmostEqual(px(t.read(n))[0], 0.0, delta=0.01)


class Spatial(unittest.TestCase):
    def test_distance_fades_from_white_and_stops(self):
        t = Builder(6)
        sq = t.node("transformation_2d", {"input": t.gray(1.0)}, transformation_matrix=[10, 0, 0, 10])  # sampling: 1/10 size
        t.base(sq, "tiling", {"mode": "absolute", "value": "none"})
        n = t.node("distance", {"mask_input": sq}, maximum_distance=64.0)
        a = t.read(n)[:, :, 0]
        self.assertAlmostEqual(a[32, 32], 1.0, delta=0.01)       # inside the white square
        self.assertGreater(a[32, 40], a[32, 50])                   # fades outwards
        far = Builder(6)
        n2 = far.node("distance", {"mask_input": far.gray(0.0)})
        self.assertEqual(far.read(n2).max(), 0.0)                  # no white: nothing

    def test_transformation_offset_and_matte(self):
        t = Builder()
        src = t.node("test_pattern")
        moved = t.node("transformation_2d", {"input": src}, offset=[0.25, 0.0])
        a, b = t.read(src), t.read(moved)
        np.testing.assert_allclose(b[:, :24], a[:, 8:], atol=0.02)  # a positive offset moves it left (tiled)
        t.base(moved, "tiling", {"mode": "absolute", "value": "none"})
        t.g.set_param(moved, "matte_color", [1, 0, 0, 1])
        self.assertAlmostEqual(t.read(moved)[16, 29, 0], 1.0, delta=0.01)  # uncovered: matte

    def test_transformation_matrix_samples_column_major(self):
        t = Builder()
        src = t.node("test_pattern")
        a = t.read(src)
        twice = t.read(t.node("transformation_2d", {"input": src}, transformation_matrix=[2, 0, 0, 2]))
        np.testing.assert_allclose(twice[:, :16], twice[:, 16:], atol=1e-4)  # Noise Upscale's 2 0 0 2: two repeats
        np.testing.assert_allclose(twice[:16, :], twice[16:, :], atol=1e-4)
        # a b c d = first column (a, b): 0 -1 1 0 samples (x, y) -> (-y, x) about the centre.
        rot = t.read(t.node("transformation_2d", {"input": src}, transformation_matrix=[0, -1, 1, 0]))
        np.testing.assert_allclose(rot[5, 9], a[22, 5], atol=0.02)

    def test_box_blur_keeps_the_mean(self):
        t = Builder()
        src = t.node("test_pattern")
        n = t.node("blur", {"input": src}, intensity=20.0)
        self.assertAlmostEqual(t.read(n).mean(), t.read(src).mean(), delta=0.01)


    def test_warp_moves_intensity_over_256_of_the_image_per_unit_slope(self):
        # Gradient Linear 1 falls from 1 to 0 down the image (slope -1 in uv): Warp 64 reads 64/256 of the image
        # uphill, the content moves down by 16 px of 64; Pixel Size 0.5 halves the move (warp.py).
        def shift(pixel_size, node="warp"):
            t = Builder(6)
            src = t.node("test_pattern")
            if node == "warp":
                n = t.node("warp", {"input": src, "gradient_input": t.node("gradient_linear_ramp")}, intensity=64.0)
            else:  # Intensity input 0.5 at Intensity 64: 1/8 of the image along the angle (a quarter turn: down)
                n = t.node("directional_warp", {"input": src, "intensity_input": t.gray(0.5)}, intensity=64.0,
                           warp_angle=0.25)
            t.base(n, "pixel_size", {"mode": "absolute", "value": [pixel_size, pixel_size]})
            a, b = t.read(src), t.read(n)
            rows = slice(16, 48)  # away from the ramp's seam
            err = [np.abs(b[rows] - np.roll(a, -k, axis=0)[rows]).mean() for k in range(-32, 33)]
            return int(np.argmin(err)) - 32, min(err)
        for ps, want in ((1.0, -16), (0.5, -8)):
            k, err = shift(ps)
            self.assertEqual(k, want)
            self.assertLess(err, 0.02)
        self.assertEqual(shift(1.0, "directional")[0], 8)

class Raster(unittest.TestCase):
    def test_bitmap_reads_a_file_at_its_own_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((16, 64, 3), np.uint8)
            img[:, 32:] = 255
            imageio.write(Path(tmp) / "half.png", img)
            pkg = Package(Path(tmp) / "p.trafilatura3d")
            g = pkg.new_graph("t", 5)
            n = g.add_node("bitmap")
            g.set_param(n, "pkg_resource_path", "half.png")
            job = compile_graph(pkg, "t").jobs[n]
            self.assertEqual(job.size, (64, 16))  # absolute: the image's size
            a = EV.read(job, "output")
            self.assertEqual(a[8, 4, 0], 0)
            self.assertEqual(a[8, 60, 0], 255)

    def test_text_layout(self):
        """Font size is the line height (ascent + descent) as a fraction of the image; the first baseline starts at
        the centre moved by Position; lines go down; the text stretches with the image; Left starts at the anchor."""
        from trafilatura3d import fonts
        found = fonts.find("")
        if found is None:
            self.skipTest("no system font")
        file, face = found
        if face:
            self.skipTest("the default font is a face of a collection")
        _, _, (ascent, descent, gap) = fonts.face_info(file)

        def rows_cols(log2=(8, 8), **params):
            t = Builder(8)
            n = t.node("text", **{"font": str(file), "text": "H", "font_size": 0.2, "color_mode": "Grayscale",
                                  **params})
            t.base(n, "output_size", {"mode": "absolute", "value": list(log2)})
            a = t.read(n)[..., 0]
            ys, xs = np.nonzero(a > 0.5)
            return ys, xs
        ys, xs = rows_cols()
        self.assertLessEqual(abs(ys.max() + 1 - 128), 1)  # the baseline at the centre
        cap = (ys.max() - ys.min() + 1) / 256
        self.assertTrue(0.4 * 0.2 / (ascent + descent) < cap < 0.85 * 0.2 / (ascent + descent))  # a cap height in em
        moved, _ = rows_cols(position=[0.0, 0.25])
        self.assertLessEqual(abs(moved.max() - ys.max() - 64), 1)
        two, _ = rows_cols(text="H\nH")
        self.assertAlmostEqual((two.max() - ys.max()) / 256, 0.2 * (ascent + descent + gap) / (ascent + descent),
                               delta=2 / 256)
        _, left = rows_cols(alignment="Left")
        self.assertGreaterEqual(left.min(), 128)
        wide_y, wide_x = rows_cols(log2=(9, 8))
        np.testing.assert_array_equal(np.unique(wide_y), np.unique(ys))  # same rows, twice the columns
        self.assertAlmostEqual(np.ptp(wide_x), 2 * np.ptp(xs), delta=2)
        _, narrow = rows_cols(transformation=[2.0, 0.0, 0.0, 1.0])  # maps the output to the text: half as wide
        self.assertAlmostEqual(np.ptp(narrow), np.ptp(xs) / 2, delta=2)

    def test_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "r.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
                                             '<rect x="0" y="0" width="5" height="10" fill="#ff0000"/></svg>', encoding="utf-8")
            pkg = Package(Path(tmp) / "p.trafilatura3d")
            g = pkg.new_graph("t", 5)
            n = g.add_node("svg")
            g.set_param(n, "pkg_resource_path", "r.svg")
            a = EV.read(compile_graph(pkg, "t").jobs[n], "output").astype(float) / 255
            self.assertAlmostEqual(a[16, 4, 0], 1.0, delta=0.02)
            self.assertAlmostEqual(a[16, 28, 0], 0.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
