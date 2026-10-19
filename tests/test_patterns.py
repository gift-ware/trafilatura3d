"""Texture generators/Patterns behave as documented."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph

EV = Evaluator()


def render(node_type, params=None, inputs=None, log2=7, output=None, base=None):
    """Cook `node_type` with `params` at 2^log2 px; `inputs` maps port -> (source type, its params).
    Returns float64 HxWxC (8 bit outputs scaled to 0-1)."""
    pkg = Package()
    g = pkg.new_graph("t", log2)
    n = g.add_node(node_type)
    for k, v in (params or {}).items():
        g.set_param(n, k, v)
    for k, v in (base or {}).items():
        g.set_base(n, k, v)
    for port, (src_type, src_params) in (inputs or {}).items():
        s = g.add_node(src_type)
        for k, v in (src_params or {}).items():
            g.set_param(s, k, v)
        g.connect(s, "output", n, port)
    job = compile_graph(pkg, "t").jobs[n]
    a = EV.read(job, output or next(iter(job.node.outputs)))
    return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)


def gray(a):
    return a[..., 0]


def runs(line, threshold=0.5):
    """Number of bright runs along a 1D profile (wrapping)."""
    b = line > threshold
    return int(np.sum(b & ~np.roll(b, 1)))


class ShapeNode(unittest.TestCase):
    def test_tiling_repeats(self):
        for n in (1, 3, 5):
            a = gray(render("shape", {"tiling": n, "pattern": "Disc", "scale": 0.8}))
            self.assertEqual(runs(a[a.shape[0] // (2 * n)]), n)

    def test_square_fills_with_scale_1(self):
        a = gray(render("shape", {"pattern": "Square", "scale": 1.0}))
        self.assertGreater(a[2:-2, 2:-2].min(), 0.99)

    def test_cone_peaks_at_centre(self):
        a = gray(render("shape", {"pattern": "Cone", "scale": 1.0}, log2=6))
        self.assertGreater(a[31:33, 31:33].min(), 0.95)
        self.assertLess(a[0, 0], 0.01)

    def test_angle_rotates(self):
        a = gray(render("shape", {"pattern": "Brick", "scale": 1.0}))
        b = gray(render("shape", {"pattern": "Brick", "scale": 1.0, "angle": 0.25}))
        np.testing.assert_allclose(b, np.rot90(a), atol=0.1)


class TileGeneratorNode(unittest.TestCase):
    def test_amounts_set_the_grid(self):
        for x, y in ((4, 4), (8, 3)):
            a = gray(render("tile_generator", {"x_amount": x, "y_amount": y, "pattern": "Disc"}))
            self.assertEqual(runs(a[a.shape[0] // (2 * y)]), x)
            self.assertEqual(runs(a[:, a.shape[1] // (2 * x)]), y)

    def test_seed_changes_randomness_only_when_random(self):
        p = {"pattern": "Disc", "luminance_random": 0.8, "position_random": [0.3, 0.3]}
        a = render("tile_generator", p)
        b = render("tile_generator", p, base={"seed": {"mode": "absolute", "value": 7}})
        self.assertGreater(np.abs(a - b).mean(), 0.01)
        c = render("tile_generator", {"pattern": "Disc"})
        d = render("tile_generator", {"pattern": "Disc"}, base={"seed": {"mode": "absolute", "value": 7}})
        np.testing.assert_array_equal(c, d)

    def test_checker_mask_hides_half(self):
        full = gray(render("tile_generator", {"pattern": "Square", "x_amount": 8, "y_amount": 8}))
        half = gray(render("tile_generator", {"pattern": "Square", "x_amount": 8, "y_amount": 8, "checker_mask": True}))
        self.assertAlmostEqual(half.mean() / full.mean(), 0.5, delta=0.02)
        inv = gray(render("tile_generator", {"pattern": "Square", "x_amount": 8, "y_amount": 8, "checker_mask": True,
                                             "invert_mask": True}))
        np.testing.assert_allclose(half + inv, full, atol=0.02)

    def test_background_and_opacity(self):
        a = gray(render("tile_generator", {"global_opacity": 0.5, "background_color": [0.2, 0.2, 0.2, 1.0],
                                           "blending_mode": "Add", "pattern": "Square",
                                           "interstice_x_y": [0.5, 0.0, 0.5, 0.0]}))
        self.assertAlmostEqual(a.min(), 0.2, delta=0.01)
        # A tile adds (luminance - background) x opacity, 0.2 + 0.5 x 0.8.
        self.assertAlmostEqual(a.max(), 0.6, delta=0.01)

    def test_horizontal_mask_hides_odd_rows(self):
        # Rows of tiles remain (the doc text says columns).
        a = gray(render("tile_generator", {"pattern": "Square", "x_amount": 4, "y_amount": 4, "horizontal_mask": True}))
        self.assertGreater(a[16].mean(), 0.5)
        self.assertLess(a[48].max(), 0.01)

    def test_middle_size_keeps_lines_packed(self):
        # Even columns 2 x 0.25 cells wide, odd ones the rest (1.5 cells), tiles centred in them.
        a = gray(render("tile_generator", {"pattern": "Square", "x_amount": 4, "y_amount": 4,
                                           "middle_size": [0.25, 0.5]}))
        row = a[16] > 0.5
        np.testing.assert_array_equal(np.flatnonzero(np.diff(row.astype(int))) + 1, [1, 15, 18, 62, 65, 79, 82, 126])

    def test_offset_shifts_rows(self):
        a = gray(render("tile_generator", {"pattern": "Square", "x_amount": 4, "y_amount": 4}))
        b = gray(render("tile_generator", {"pattern": "Square", "x_amount": 4, "y_amount": 4, "offset": 0.5}))
        np.testing.assert_allclose(a[:32], b[:32], atol=0.02)               # row 0 stays
        np.testing.assert_allclose(np.roll(a[32:64], 16, axis=1), b[32:64], atol=0.02)  # row 1 by half a tile

    def test_size_modes_and_interstice_random(self):
        # Normal - Size in cells, Absolute in image units, Pixel in whole pixels; the interstice
        # random (2nd / 4th component) shrinks the gap per tile.
        p = {"pattern": "Square", "x_amount": 2, "y_amount": 2}
        a = gray(render("tile_generator", {**p, "size_mode": "Normal - Size", "size": [0.5, 0.5]}))
        self.assertAlmostEqual(a.mean(), 0.25, delta=0.02)
        a = gray(render("tile_generator", {**p, "size_mode": "Absolute", "size_absolute": [0.25, 0.25]}))
        self.assertAlmostEqual(a.mean(), 4 * 0.25 ** 2, delta=0.02)
        a = gray(render("tile_generator", {**p, "size_mode": "Pixel", "size_pixel": [1.5, 1.5]}))
        self.assertEqual(int(round(a.sum())), 4 * 2 * 2)
        gap = {"pattern": "Square", "interstice_x_y": [0.5, 0.0, 0.5, 0.0]}
        fixed = gray(render("tile_generator", gap)).mean()
        rand = gray(render("tile_generator", {**gap, "interstice_x_y": [0.5, 1.0, 0.5, 1.0]})).mean()
        self.assertAlmostEqual(fixed, 0.25, delta=0.02)
        self.assertGreater(rand, fixed + 0.1)

    def test_pattern_specific_random(self):
        p = {"pattern": "Gradation", "pattern_specific": [1.0, 0.0]}
        a = render("tile_generator", p)
        b = render("tile_generator", {**p, "pattern_specific": [1.0, 1.0]})
        self.assertGreater(np.abs(a - b).mean(), 0.005)

    def test_color_variant(self):
        # Color Variant: Color and Background Color are colours, tiles add their colour; By Number multiplies
        # the channel of its parametrisation (here alpha), Color Random is hue/saturation/lightness/alpha.
        bg = ("uniform_color", {"output_color": [0.0, 0.0, 0.5, 1.0]})
        p = {"pattern": "Square", "color": [1.0, 0.0, 0.0, 1.0], "x_amount": 2, "y_amount": 2,
             "interstice_x_y": [0.0, 0.0, 0.0, 0.0]}
        a = render("tile_generator", p, {"background": bg})[..., :3]
        np.testing.assert_allclose(a[10, 10], [1.0, 0.0, 0.5], atol=0.01)
        a = render("tile_generator", {**p, "luminance_by_number": True, "by_number_parametrisation": "Alpha"},
                   {"background": bg})[..., :3]
        self.assertAlmostEqual(a[10, 10, 0], 0.25, delta=0.01)  # tile 0 of 4: alpha 1/4
        a = render("tile_generator", {**p, "color_random": [1.0, 0.0, 0.0, 0.0], "blending_mode": "Alpha Blend"},
                   {"background": bg})[..., :3]
        self.assertGreater(a.reshape(-1, 3).std(axis=0).max(), 0.05)


class NonSquareExpansion(unittest.TestCase):
    def test_square_frame_on_2_to_1(self):
        size = {"output_size": {"mode": "absolute", "value": [8, 7]}}
        a = gray(render("tile_generator", {"pattern": "Disc", "x_amount": 4, "y_amount": 4}, base=size))
        self.assertEqual(runs(a[a.shape[0] // 8]), 8)
        disc = {"pattern": "Disc", "scale": 0.75}  # discs apart (at Scale 1 they touch)
        self.assertEqual(runs(gray(render("shape", {**disc, "non_square_tiling": True}, base=size))[64]), 2)
        self.assertEqual(runs(gray(render("shape", disc, base=size))[64]), 1)  # default: tiling off
        self.assertEqual(runs(gray(render("shape", {**disc, "non_square_expansion": False}, base=size))[64]), 1)


class PointGradients(unittest.TestCase):
    def test_axial_black_to_white(self):
        a = gray(render("gradient_axial", {"point_1": [0.0, 0.5], "point_2": [1.0, 0.5]}, log2=6))
        row = a[32]
        self.assertLess(row[0], 0.03)
        self.assertGreater(row[-1], 0.97)
        self.assertTrue(np.all(np.diff(row) >= -1e-6))
        np.testing.assert_allclose(a[0], a[-1])              # constant across the axis

    def test_axial_reflected_peaks_in_the_middle(self):
        row = gray(render("gradient_axial_reflected", {"point_1": [0.0, 0.5], "point_2": [1.0, 0.5]}, log2=6))[32]
        self.assertLess(max(row[0], row[-1]), 0.05)
        self.assertGreater(row[31:33].max(), 0.95)

    def test_radial_white_centre_black_at_radius(self):
        a = gray(render("gradient_radial", {"point_1": [0.5, 0.5], "point_2": [0.75, 0.5]}, log2=6))
        self.assertGreater(a[31:33, 31:33].min(), 0.9)
        self.assertEqual(a[0, 0], 0.0)
        cone = gray(render("gradient_radial", {}, log2=6))
        hemi = gray(render("gradient_radial", {"shape": "Hemisphere"}, log2=6))
        self.assertGreater(hemi.mean(), cone.mean())          # rounder: brighter on average

    def test_circular_starts_and_ends_at_point_2(self):
        a = gray(render("gradient_circular", {}, log2=6))     # around the centre, from the right
        self.assertLess(a[33, 60], 0.1)                        # just after the start (clockwise)
        self.assertGreater(a[30, 60], 0.9)                     # just before coming back
        b = gray(render("gradient_circular", {"angle": 0.25}, log2=6))
        np.testing.assert_allclose(b, (a + 0.25) % 1.0, atol=0.01)   # Angle added to the value


class LinearGradients(unittest.TestCase):
    def test_linear_1_ramps_and_rotates(self):
        a = gray(render("gradient_linear_ramp", {}, log2=6))
        self.assertTrue(np.all(np.diff(a[:, 0]) < 0))           # white at the top, black at the bottom
        b = gray(render("gradient_linear_ramp", {"rotation": "180°"}, log2=6))
        np.testing.assert_allclose(b, a[::-1], atol=0.01)
        r = gray(render("gradient_linear_ramp", {"rotation": "90°"}, log2=6))
        self.assertTrue(np.all(np.diff(r[0]) > 0))              # turned clockwise: white on the right
        c = gray(render("gradient_linear_ramp", {"tiling": 4}, log2=6))
        self.assertEqual(int(np.sum(np.diff(c[:, 0]) > 0.5)), 3)  # three resets inside the image

    def test_linear_2_and_3_black_white_black(self):
        for t in ("gradient_linear_rounded", "gradient_linear_peak"):
            col = gray(render(t, {}, log2=6))[:, 0]               # Rotation 0 varies top-bottom
            self.assertLess(max(col[0], col[-1]), 0.05)
            self.assertGreater(col[31:33].max(), 0.95)
        col = gray(render("gradient_linear_peak", {"position": 0.25}, log2=6))[:, 0]
        self.assertIn(int(np.argmax(col)), (15, 16))
        row = gray(render("gradient_linear_rounded", {"rotation": "90°", "tiling": 3}, log2=6))[0]
        self.assertEqual(runs(row), 3)


class ThreeD(unittest.TestCase):
    def test_volume_mask_sphere_feather(self):
        a = gray(render("3d_volume_mask", {"shape": "Sphere", "feather_width": 0.0}, log2=6))
        self.assertEqual(a[32, 32], 1.0)
        self.assertEqual(a[0, 0], 0.0)
        self.assertAlmostEqual((a > 0.5).mean(), np.pi * 0.25 ** 2, delta=0.01)
        f = gray(render("3d_volume_mask", {"shape": "Sphere", "feather_width": 0.3}, log2=6))
        self.assertLess(f[32, 18], 0.9)                       # feathered inwards (fraction of the radius)
        cube = gray(render("3d_volume_mask", {"feather_width": 0.0}, log2=6))   # default Cube, Scale 0.25
        self.assertAlmostEqual((cube > 0.5).mean(), 0.25, delta=0.01)

    def test_volume_mask_follows_position_input(self):
        p = {"shape": "Sphere", "feather_width": 0.0, "position_uv": [0.25, 0.25]}
        a = gray(render("3d_volume_mask", p, log2=6))
        self.assertGreater(a[16, 16], 0.99)
        self.assertLess(a[48, 48], 0.01)

    def test_linear_gradient_3d_contrast(self):
        pts = {"point_1_uv": [0.25, 0.5], "point_2_uv": [0.75, 0.5]}
        a = gray(render("3d_linear_gradient", pts, log2=6))[0]
        self.assertAlmostEqual(a[32], 0.5, delta=0.05)
        self.assertLess(a[15], 0.02)
        b = gray(render("3d_linear_gradient", {**pts, "contrast": 1.0}, log2=6))[0]
        self.assertTrue(set(np.unique(b)) <= {0.0, 1.0})


class Polygons(unittest.TestCase):
    def test_sides_set_the_area(self):
        for n in (3, 5, 8):
            a = gray(render("polygon_smooth", {"sides": n, "gradient": 0.01}, log2=8))
            r = 128                                            # circumradius (Auto Scale, Scale 1)
            self.assertAlmostEqual((a > 0.5).sum() / (0.5 * n * r * r * np.sin(2 * np.pi / n)), 1.0, delta=0.02)

    def test_polygon_1_solid_and_explode(self):
        a = gray(render("polygon_slices", {}, log2=7))
        self.assertEqual(a[64, 64], 1.0)
        b = gray(render("polygon_slices", {"explode": 0.5}, log2=7))
        self.assertLess(b[64, 64], 0.01)                       # slices moved off the centre
        c = gray(render("polygon_slices", {"gradient": True}, log2=7))
        self.assertGreater(c[64, 64], c[64, 100])              # white at the centre

    def test_polygon_2_invert(self):
        a = gray(render("polygon_smooth", {}, log2=7))
        b = gray(render("polygon_smooth", {"invert_gradient": True}, log2=7))
        self.assertGreater(a[64, 64], 0.95)
        self.assertLess(b[64, 64], 0.05)

    def test_polygon_2_and_weave_1_tiling(self):
        one = gray(render("polygon_smooth", {}, log2=6))
        two = gray(render("polygon_smooth", {"tiling": 2}, log2=7))    # Tiling copies, centred in their cells
        np.testing.assert_allclose(two[:64, :64], one, atol=0.05)
        w = gray(render("weave_basket", {}, log2=6))
        w4 = gray(render("weave_basket", {"tiling": 4}, log2=8))       # 2^int(sqrt(4)) = 4 copies
        np.testing.assert_allclose(w4[:64, :64], w, atol=0.02)

    def test_star_has_five_points(self):
        a = gray(render("star", {}, log2=8))
        ys, xs = np.nonzero(a > 0.5)
        cy, cx = ys.mean(), xs.mean()
        ang = np.arctan2(ys - cy, xs - cx)
        r = np.hypot(ys - cy, xs - cx)
        prof = np.array([r[(ang >= b) & (ang < b + np.pi / 90)].max() for b in np.linspace(-np.pi, np.pi, 180, endpoint=False)])
        peaks = np.sum((prof > np.roll(prof, 1)) & (prof >= np.roll(prof, -1)) & (prof > np.median(prof)))
        self.assertEqual(peaks, 5)


class StarburstAndStripes(unittest.TestCase):
    def test_branch_count(self):
        for n in (8, 20):
            a = gray(render("starburst", {"branches": n, "blur_amount": 0.0}, log2=8))
            t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
            ring = a[(128 + 40 * np.sin(t)).astype(int), (128 + 40 * np.cos(t)).astype(int)]
            self.assertEqual(runs(ring, 0.3), n)                # branches fade towards their tip

    def test_size_variation_seeded(self):
        p = {"size_variation": 1.0, "randomness": 0.5}
        a = render("starburst", p)
        b = render("starburst", p, base={"seed": {"mode": "absolute", "value": 3}})
        self.assertGreater(np.abs(a - b).mean(), 0.01)

    def test_stripes_count_width_and_shift(self):
        a = gray(render("stripes", {"stripes": 7, "width": 0.3}, log2=8))
        self.assertEqual(runs(a[0]), 7)
        self.assertAlmostEqual(a.mean(), 0.3, delta=0.01)
        b = gray(render("stripes", {"stripes": 4, "shift": 2}, log2=8))
        self.assertEqual(runs(b[:, 0]), 2)                     # tilted: crossing columns too
        self.assertLess(np.abs(b[0] - b[-1]).mean(), 2 * np.abs(b[1] - b[0]).mean() + 1e-3)  # tiles vertically


SEED7 = {"seed": {"mode": "absolute", "value": 7}}
class SimplePatterns(unittest.TestCase):
    def test_brick_rows_follow_scale(self):
        for n in (4, 8):                                    # Scale: Brick 1 rows, Brick 2 row pairs
            a = gray(render("brick_running", {"scale": n}))
            self.assertEqual(runs(a[:, 192 // n]), n)
            b = gray(render("brick_mixed", {"scale": n // 2}))
            self.assertEqual(runs(b[:, 4]), n)

    def test_brick_interstice_widens_joints(self):
        for t in ("brick_running", "brick_mixed"):
            thin = gray(render(t, {"interstice_width": 0.05})).mean()
            wide = gray(render(t, {"interstice_width": 0.5})).mean()
            self.assertLess(wide, thin - 0.05)

    def test_brick_generator_height_varies_per_brick(self):
        bevel = [0.1, 0.1, 0.0, 0.0]
        flat = gray(render("brick_generator", {"bevel": bevel}))
        self.assertGreater(flat.max(), 0.99)
        for height in ([1.0, 1.0, 0.0, 1.0], [0.2, 1.0, 0.0, 0.0]):   # luminance variation, two levels
            a = gray(render("brick_generator", {"bevel": bevel, "height": height}))
            peaks = [a[y, x] for y in range(5, 128, 11) for x in range(5, 128, 21)]
            self.assertGreater(np.std(peaks), 0.1)
            b = gray(render("brick_generator", {"bevel": bevel, "height": height}, base=SEED7))
            self.assertGreater(np.abs(a - b).mean(), 0.02)

    def test_brick_generator_bricks_count(self):
        a = gray(render("brick_generator", {"bricks": [4, 8], "offset": [0.0, 0.0]}))
        self.assertEqual(runs(a[8], 0.3), 4)
        self.assertEqual(runs(a[:, 8], 0.3), 8)                # Bricks y = rows

    def test_brick_generator_slope_tilts_bricks(self):
        a = gray(render("brick_generator", {"bevel": [0.05, 0.05, 0.0, 0.0], "slope": [1.0, 0.0, 1.0, 1.0]}))
        row = a[16, 4:28]                                       # inside the first brick: a ramp along x
        self.assertGreater(row[-1] - row[0], 0.5)

    def test_checker(self):
        for n in (1, 3):
            a = gray(render("checker", {"tiling": n}))
            self.assertAlmostEqual(a.mean(), 0.5, delta=0.01)
            self.assertEqual(runs(a[5]) + runs(1 - a[5]), 2 * n)
        r = gray(render("checker", {"rotate_45_degrees": True}))
        self.assertAlmostEqual(r.mean(), 0.5, delta=0.01)

    def test_gaussian_blob_counts(self):
        for n in (1, 4):
            a = gray(render("gaussian_diagonal", {"tiling": n}))
            self.assertEqual(runs(a[64 // n]), n)
        b = gray(render("gaussian_staggered", {"tiling": 2}))
        self.assertEqual(runs(b[16]), 4)            # two blobs per row per tile
        self.assertEqual(runs(b[48]), 4)            # the next row, staggered
        self.assertLess(np.abs(b[48] - np.roll(b[16], 16)).max(), 0.01)

    def test_alveolus_interstice_and_mode(self):
        a = gray(render("alveolus", {"interstice_width": 0.0}))
        b = gray(render("alveolus", {"interstice_width": 0.5}))
        self.assertLess((b > 0.01).mean(), (a > 0.01).mean() - 0.2)
        h = gray(render("alveolus", {"gradient_filled_cells": True}))
        self.assertGreater((h > 0.01).mean(), 0.9)  # hexagons cover the plane

    def test_weave_2_disorder_is_random(self):
        a = render("weave_twill")
        np.testing.assert_array_equal(a, render("weave_twill", base=SEED7))
        b = render("weave_twill", {"disorder": 0.8})
        self.assertGreater(np.abs(b - render("weave_twill", {"disorder": 0.8}, base=SEED7)).mean(), 0.02)
        self.assertGreater(np.abs(a - b).mean(), 0.05)

    def test_rotate_45_changes_but_keeps_density(self):
        for t in ("mesh_thin", "mesh_thick", "weave_basket", "weave_twill", "checker"):
            a = gray(render(t))
            r = gray(render(t, {"rotate_45_degrees": True}))
            self.assertGreater(np.abs(a - r).mean(), 0.05, t)
            self.assertAlmostEqual(a.mean(), r.mean(), delta=0.05, msg=t)

    def test_weave_generator_blocks(self):
        a = gray(render("weave_generator", {"tile_x": 4, "tile_y": 4, "weave": 1, "gap": [0.3, 0.3]}))
        np.testing.assert_allclose(np.roll(a, (32, 32), (0, 1)), a, atol=1 / 255)  # twill: one cell per row
        b = gray(render("weave_generator", {"tile_x": 4, "tile_y": 4, "weave": 3, "gap": [0.3, 0.3]}))
        self.assertEqual(runs(b[:, 16], 0.6), 2 * 3 + 1)   # column 0: two horizontal blocks of 3 stitches, one vertical

    def test_weave_generator_outputs_and_gap_axes(self):
        p = {"tile_x": 4, "tile_y": 4, "weave": 3, "gap": [0.0, 0.5]}
        m1 = gray(render("weave_generator", p, output="mask_1"))
        m2 = gray(render("weave_generator", p, output="mask_2"))
        op = gray(render("weave_generator", p, output="opacity"))
        self.assertLess((m1 * m2).max(), 1e-6)                          # one stitch per pixel
        np.testing.assert_allclose(np.maximum(m1, m2), op, atol=1 / 255)
        self.assertLess(m1.mean(), m2.mean() - 0.1)                     # gap y thins the horizontal stitches

    def test_fibers_differ(self):
        self.assertGreater(np.abs(render("fibers_rope") - render("fibers_cloth")).mean(), 0.05)
        np.testing.assert_array_equal(render("fibers_rope"), render("fibers_rope", base=SEED7))  # not random


SEED7 = {"seed": {"mode": "absolute", "value": 7}}
def ring_profile(a, radius):
    """Values along a circle of `radius` (uv) around the image centre."""
    n = a.shape[0]
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    x = ((0.5 + radius * np.cos(t)) * n).astype(int) % n
    y = ((0.5 + radius * np.sin(t)) * n).astype(int) % n
    return a[y, x]


class TileSamplerNode(unittest.TestCase):
    def test_amounts_set_the_grid(self):
        for x, y in ((4, 4), (8, 3)):
            a = gray(render("tile_sampler", {"x_amount": x, "y_amount": y, "pattern": "Disc"}))
            self.assertEqual(runs(a[a.shape[0] // (2 * y)]), x)
            self.assertEqual(runs(a[:, a.shape[1] // (2 * x)]), y)

    def test_global_offset_is_in_cells(self):
        # Global Offset 1 moves the grid by one cell, onto itself.
        a = gray(render("tile_sampler", {"x_amount": 4, "y_amount": 4}))
        np.testing.assert_allclose(gray(render("tile_sampler", {"x_amount": 4, "y_amount": 4,
                                                                "global_offset": [1.0, 1.0]})), a, atol=0.01)

    def test_seed_changes_random_tiles(self):
        p = {"pattern": "Disc", "color_luminance_random": 0.8, "position_random": 0.5}
        self.assertGreater(np.abs(render("tile_sampler", p) - render("tile_sampler", p, base=SEED7)).mean(), 0.01)

    def test_mask_map_hides_tiles_on_dark_areas(self):
        black = ("uniform_color", {"color_mode": "Grayscale", "output_color": [0.2, 0.2, 0.2, 1]})
        p = {"pattern": "Square", "mask_map_threshold": 0.5}
        self.assertEqual(gray(render("tile_sampler", p, {"mask_map_input": black})).max(), 0.0)
        p["mask_map_invert"] = True
        self.assertGreater(gray(render("tile_sampler", p, {"mask_map_input": black})).mean(), 0.3)

    def test_scale_map_scales_tiles(self):
        half = ("uniform_color", {"color_mode": "Grayscale", "output_color": [0.5, 0.5, 0.5, 1]})
        p = {"pattern": "Square", "scale": 1.0}
        full = gray(render("tile_sampler", p)).mean()
        p["scale_map_multiplier"] = 1.0
        quarter = gray(render("tile_sampler", p, {"scale_map_input": half})).mean()
        self.assertAlmostEqual(quarter / full, 0.25, delta=0.03)

    def test_shapes_have_hard_edges(self):
        """Built-in shapes are drawn without antialiasing: every pixel is in or out (a Distance fed with tiles then
        gives each cell the tile's own value)."""
        for pattern in ("Disc", "Square"):
            a = gray(render("tile_sampler", {"pattern": pattern, "x_amount": 3, "y_amount": 3, "rotation": 0.07,
                                             "scale_random": 0.6}))
            self.assertEqual(int(((a > 0.01) & (a < 0.99)).sum()), 0, msg=pattern)
            self.assertGreater(a.mean(), 0.1, msg=pattern)

    def test_large_tiles_are_whole(self):
        """Tiles of any size are whole: with only the tiles of the cells at the centre kept (a small disc as mask),
        discs 16 cells wide reach 8 cells from their centre (gathering the cells around each pixel stopped at 6)."""
        mask = ("shape", {"pattern": "Disc", "scale": 0.08})
        p = {"pattern": "Disc", "x_amount": 32, "y_amount": 32, "scale": 16.0, "mask_map_threshold": 0.5}
        a = gray(render("tile_sampler", p, {"mask_map_input": mask}, log2=8))  # 8 px per cell
        self.assertEqual(a[128, 128 + 7 * 8], 1.0)
        self.assertEqual(a[128 - 7 * 8, 128], 1.0)
        self.assertEqual(a[128, 128 + 10 * 8], 0.0)
        self.assertEqual(a[0, 0], 0.0)

    def test_unconnected_vector_map_reads_black(self):
        """An unconnected Vector Map Input is black: the vector (1, 1), so its multipliers still act (tiles sqrt(2)
        larger, turned by an eighth) instead of shrinking every tile to nothing."""
        black = ("uniform_color", {"output_color": [0.0, 0.0, 0.0, 1.0]})
        p = {"pattern": "Square", "x_amount": 1, "y_amount": 1, "scale": 0.5, "size": [1.0, 0.25]}
        for extra in ({"scale_vector_map_multiplier": 1.0}, {"vector_map_multiplier": 1.0},
                      {"vector_map_displacement": 0.2}):
            a = gray(render("tile_sampler", {**p, **extra}))
            np.testing.assert_allclose(a, gray(render("tile_sampler", {**p, **extra}, {"vector_map_input": black})))
        plain = gray(render("tile_sampler", p)).mean()
        self.assertAlmostEqual(gray(render("tile_sampler", {**p, "scale_vector_map_multiplier": 1.0})).mean() / plain,
                               2.0, delta=0.1)

    def test_reverse_rendering_order_flips_overlaps(self):
        p = {"pattern": "Square", "scale": 1.6, "blending_mode": "Alpha Blend", "x_amount": 4, "y_amount": 4,
             "color_luminance_random": 1.0}
        a = render("tile_sampler", p)
        b = render("tile_sampler", {**p, "reverse_rendering_order": True})
        self.assertGreater(np.abs(a - b).mean(), 0.01)

    def test_quincunx_offset_shifts_every_other_row(self):
        p = {"pattern": "Square", "x_amount": 4, "y_amount": 4, "scale": 1.0, "size": [0.5, 0.5]}
        a = gray(render("tile_sampler", p))
        b = gray(render("tile_sampler", {**p, "offset": 0.5}))
        np.testing.assert_allclose(a[:32], b[:32], atol=0.02)
        np.testing.assert_allclose(np.roll(a[32:64], 16, axis=1), b[32:64], atol=0.02)
        np.testing.assert_allclose(a[64:96], b[64:96], atol=0.02)

    def test_size_modes(self):
        # Absolute = image units, Pixel = whole pixels (ceil of size x scale), Normal = cells.
        p = {"pattern": "Square", "scale": 1.0, "x_amount": 2, "y_amount": 2}
        self.assertAlmostEqual(gray(render("tile_sampler", {**p, "size": [0.5, 0.5]})).mean(), 0.25, delta=0.02)
        a = gray(render("tile_sampler", {**p, "size_mode": "Absolute", "size_absolute": [0.25, 0.25]}))
        self.assertAlmostEqual(a.mean(), 4 * 0.25 ** 2, delta=0.02)
        a = gray(render("tile_sampler", {**p, "size_mode": "Pixel", "size_pixel": [3.4, 3.4]}))
        self.assertEqual(int(round(a.sum())), 4 * 4 * 4)

    def test_size_random_per_axis(self):
        p = {"pattern": "Square", "scale": 1.0, "size_random": [1.0, 0.0]}
        for axis, sr in ((1, [1.0, 0.0]), (0, [0.0, 1.0])):
            a = gray(render("tile_sampler", {**p, "size_random": sr}))
            self.assertLess(a.mean(), 0.75)  # random widths (or heights)...
            self.assertGreater((a > 0.5).any(axis=axis).mean(), 0.97)  # ...the other axis is untouched

    def test_color_random_is_hsl(self):
        # Colour variant: Color Random = hue / saturation / lightness / alpha randomness per tile.
        red = ("uniform_color", {"output_color": [1.0, 0.0, 0.0, 1.0]})
        p = {"pattern": "Square", "color_random": [0.0, 0.0, 1.0, 0.0], "color": [1.0, 0.0, 0.0, 1.0],
             "blending_mode": "Alpha Blend"}
        a = render("tile_sampler", p, {"background_input": red})[..., :3]
        self.assertLess(np.abs(a[..., 1] - a[..., 2]).max(), 0.02)  # the red hue is kept: g == b
        self.assertGreater(a.sum(axis=2).std(), 0.1)  # lightness varies per tile


class HardEdges(unittest.TestCase):
    def test_pattern_nodes_draw_shapes_with_hard_edges(self):
        # A disc's curved border shows any antialiasing: every pixel must be in or out.
        for node, params in (("shape", {"pattern": "Disc", "scale": 0.7}),
                             ("tile_generator", {"pattern": "Disc"}),
                             ("splatter_circular", {"pattern": "Disc"}),
                             ("tile_random", {"pattern": "Disc"}),
                             ("arc_pavement", {"pattern": "Disc"})):
            with self.subTest(node=node):
                a = gray(render(node, params))
                self.assertEqual(int(((a > 0.01) & (a < 0.99)).sum()), 0)
                self.assertGreater(a.mean(), 0.02)


class SplatterNode(unittest.TestCase):
    def test_grid_number_is_a_power_of_two(self):
        for g in (1, 2, 3):
            a = gray(render("splatter", {"grid_number": g, "pattern_size_width": 80.0, "pattern_size_height": 80.0}))
            self.assertEqual(runs(a[a.shape[0] // 2 ** (g + 1)]), 2 ** g)

    def test_default_is_not_random_until_disorder(self):
        a = render("splatter")
        np.testing.assert_array_equal(a, render("splatter", base=SEED7))
        p = {"disorder": 50.0, "disorder_random": True}
        self.assertGreater(np.abs(render("splatter", p) - render("splatter", p, base=SEED7)).mean(), 0.01)

    def test_output_levels_remap(self):
        # The input levels of a final Levels, background 0.4 -> (0.4 - 0.2) / 0.4.
        a = gray(render("splatter", {"output_level_min": 0.2, "output_level_max": 0.6, "background_color": [0.4, 0.4, 0.4, 1.0],
                                     "pattern_size_width": 80.0, "pattern_size_height": 80.0}))
        self.assertAlmostEqual(a.min(), 0.5, delta=0.01)
        self.assertAlmostEqual(a.max(), 1.0, delta=0.01)

    def test_pattern_size_width_scales_copies(self):
        a = gray(render("splatter", {"pattern_size_width": 100.0, "pattern_size_height": 100.0})).mean()
        b = gray(render("splatter", {"pattern_size_width": 50.0, "pattern_size_height": 50.0})).mean()
        self.assertAlmostEqual(b / a, 0.25, delta=0.03)


class SplatterCircularNode(unittest.TestCase):
    def test_pattern_amount_on_the_ring(self):
        for n in (5, 12):
            a = gray(render("splatter_circular", {"pattern_amount": n, "pattern": "Square", "size": [0.08, 0.08]}))
            self.assertEqual(runs(ring_profile(a, 0.25)), n)

    def test_patterns_start_on_angle_0(self):
        # 4 patterns on a full turn sit on the axes, not on the diagonals.
        a = gray(render("splatter_circular", {"pattern_amount": 4, "size": [0.1, 0.1]}))
        d = int(0.25 / np.sqrt(2) * 128)
        self.assertLess(max(a[64 + d, 64 + d], a[64 - d, 64 - d]), 0.1)
        self.assertGreater(min(a[64, 64 + 32], a[64 + 32, 64], a[64, 64 - 32], a[64 - 32, 64]), 0.9)

    def test_half_spread_covers_the_left_side(self):
        a = gray(render("splatter_circular", {"pattern_amount": 16, "pattern": "Square", "size": [0.06, 0.06],
                                              "spread": 0.5}))
        self.assertGreater(a[:, :60].sum(), 0)
        self.assertEqual(a[:, 76:].sum(), 0)  # the arc starts half a step right of the vertical

    def test_connected_patterns_close_the_ring(self):
        a = gray(render("splatter_circular", {"pattern_amount": 24, "connect_patterns": True, "pattern": "Square",
                                              "blending_mode": "Add"}))  # Max would halve the joints' edges
        self.assertGreater(ring_profile(a, 0.24).min(), 0.9)  # just inside: the joints are notched outside

    def test_ring_amount_adds_rings_inside(self):
        a = gray(render("splatter_circular", {"ring_amount": 3, "pattern_amount": 64, "connect_patterns": True,
                                              "size_connected": [1.0, 0.04], "pattern": "Square"}))
        row = a[64, 64:]  # from the centre to the right edge
        self.assertEqual(runs(np.concatenate([[0], row, [0]])), 3)

    def test_scale_by_ring_number_hides_the_outer_ring(self):
        p = {"ring_amount": 2, "pattern": "Square", "pattern_amount": 32, "scale_by_ring_number": 1.0}
        a = gray(render("splatter_circular", p))
        self.assertEqual(ring_profile(a, 0.25).max(), 0.0)
        self.assertGreater(ring_profile(a, 0.125).max(), 0.9)

    def test_add_and_add_sub_are_the_same(self):
        p = {"pattern_amount": 24, "size": [0.15, 0.15], "background_color": [0.2, 0.2, 0.2, 1.0],
             "color": [0.5, 0.5, 0.5, 1.0]}
        add = gray(render("splatter_circular", {**p, "blending_mode": "Add"}))
        np.testing.assert_array_equal(add, gray(render("splatter_circular", {**p, "blending_mode": "Add Sub"})))
        self.assertGreater(add.max(), gray(render("splatter_circular", {**p, "blending_mode": "Max"})).max())

    def test_random_mask_hides_patterns(self):
        p = {"pattern_amount": 32, "size": [0.05, 0.05]}
        full = gray(render("splatter_circular", p)).mean()
        half = gray(render("splatter_circular", {**p, "random_mask": 0.5})).mean()
        self.assertLess(half, full * 0.8)
        self.assertGreater(half, full * 0.2)

    def test_layout_wraps_around(self):
        a = gray(render("splatter_circular", {"radius": 0.5, "pattern_amount": 8, "size": [0.2, 0.2]}))
        np.testing.assert_allclose(a[:, 0], a[:, -1], atol=0.25)

    def test_color_variant(self):
        # Color Variant: Color is a colour, Alpha Blend lays later patterns over earlier ones, Color Random
        # randomizes the hue.
        bg = ("uniform_color", {"output_color": [0.0, 0.0, 0.0, 1.0]})
        p = {"pattern_amount": 24, "size": [0.2, 0.2], "color": [0.0, 0.0, 1.0, 0.5], "blending_mode": "Alpha Blend"}
        a = render("splatter_circular", {**p, "pattern_amount": 4}, {"background": bg})[..., :3]
        self.assertAlmostEqual(a[..., 2].max(), 0.5, delta=0.01)
        a = render("splatter_circular", p, {"background": bg})[..., :3]
        k = np.log2(1.0 - a[..., 2].max())  # k overlapping halves: 1 - 0.5^k
        self.assertAlmostEqual(k, round(k), delta=0.1)
        self.assertLess(a[..., :2].max(), 0.01)
        a = render("splatter_circular", {**p, "color_random": [1.0, 0.0, 0.0, 0.0]}, {"background": bg})[..., :3]
        self.assertGreater(a[..., :2].max(), 0.2)


SEED7 = {"seed": {"mode": "absolute", "value": 7}}
class TileRandomNode(unittest.TestCase):
    def test_amounts_set_the_grid_without_randomness(self):
        p = {"split_mode": "none", "random_x": 0.0, "random_y": 0.0, "offset_random": 0.0,
             "interstice_mode": "Relative to largest brick", "interstice_amount": 0.3}
        for x, y in ((4, 4), (8, 3)):
            a = gray(render("tile_random", {**p, "x_amount": x, "y_amount": y}))
            self.assertEqual(runs(a[a.shape[0] // (2 * y)]), x)
            self.assertEqual(runs(a[:, a.shape[1] // (2 * x)]), y)

    def test_split_cuts_into_ceil_aspect_plus_multiplier_pieces(self):
        # 2 x 4 bricks of 2:1: auto cuts each across its width into ceil(2 + Multiplier) pieces.
        p = {"random_x": 0.0, "random_y": 0.0, "x_amount": 2, "y_amount": 4, "interstice_amount": 0.3}
        for mode, mult, n in (("none", 0, 2), ("auto", 0, 4), ("auto", 1, 6), ("auto vertical", 0, 2)):
            a = gray(render("tile_random", {**p, "split_mode": mode, "multiplier": mult}, log2=8))
            self.assertEqual(runs(a[32]), n, (mode, mult))
            self.assertEqual(runs(a[:, 16]), 4, (mode, mult))

    def test_offset_accumulates_per_row(self):
        p = {"split_mode": "none", "random_x": 0.0, "random_y": 0.0, "offset": 0.25, "interstice_amount": 0.4}
        a = gray(render("tile_random", p, log2=8)) > 0.5
        starts = [int(np.argmax(a[32 + 64 * j] & ~np.roll(a[32 + 64 * j], 1))) % 64 for j in range(4)]
        self.assertEqual([(s - starts[0]) % 64 for s in starts], [0, 16, 32, 48])  # j x Offset bricks

    def test_interstice_is_quadratic_in_the_largest_leaf(self):
        p = {"split_mode": "none", "random_x": 0.0, "random_y": 0.0}
        dark = [1.0 - (gray(render("tile_random", {**p, "interstice_amount": k}, log2=8))[32] > 0.5).mean()
                for k in (0.25, 0.5)]
        self.assertAlmostEqual(dark[0], 4 * 4 / 256, delta=0.01)   # 0.25^2 x 0.25 uv = 4 px per gap
        self.assertAlmostEqual(dark[1], 4 * 16 / 256, delta=0.01)

    def test_add_sub_lerps_background_to_color(self):
        p = {"split_mode": "none", "random_x": 0.0, "random_y": 0.0, "background_color": [0.3, 0.3, 0.3, 1.0],
             "color": [0.8, 0.8, 0.8, 1.0], "interstice_amount": 0.5}
        a = gray(render("tile_random", p, log2=8))
        self.assertAlmostEqual(a[32, 32], 0.8, delta=0.01)
        self.assertAlmostEqual(a[0, 0], 0.3, delta=0.01)

    def test_non_square_squares_have_their_own_layout(self):
        size = {"output_size": {"mode": "absolute", "value": [8, 7]}}
        a = gray(render("tile_random", {}, base=size))
        self.assertEqual(a.shape, (128, 256))
        self.assertGreater(np.abs(a[:, :128] - a[:, 128:]).mean(), 0.05)

    def test_seed_changes_layout(self):
        a = render("tile_random", {})
        b = render("tile_random", {}, base=SEED7)
        self.assertGreater(np.abs(a - b).mean(), 0.02)

    def test_random_mask_and_invert(self):
        p = {"mask_random": 0.5}
        full = gray(render("tile_random", {}))
        a = gray(render("tile_random", p))
        b = gray(render("tile_random", {**p, "mask_invert": True}))
        self.assertLess(a.mean(), full.mean() * 0.8)
        np.testing.assert_allclose(np.maximum(a, b), full, atol=0.02)

    def test_background_color(self):
        a = gray(render("tile_random", {"background_color": [0.3, 0.3, 0.3, 1.0], "mask_random": 1.0}))
        self.assertAlmostEqual(a.mean(), 0.3, delta=0.01)

    def test_color_random(self):
        # Grayscale: the lightness component only; colour: hue / saturation / lightness / alpha.
        p = {"split_mode": "none", "random_x": 0.0, "random_y": 0.0, "interstice_amount": 0.5}
        a = gray(render("tile_random", {**p, "color_random": [1.0, 1.0, 0.0, 0.0]}))
        np.testing.assert_array_equal(a, gray(render("tile_random", p)))
        self.assertGreater(gray(render("tile_random", {**p, "color_random": [0.0, 0.0, 1.0, 0.0]})).std(), 0.2)
        red = ("uniform_color", {"output_color": [0.0, 0.0, 0.0, 1.0]})
        c = {**p, "color": [1.0, 0.0, 0.0, 1.0], "blending_mode": "Alpha Blend", "color_random": [1.0, 0.0, 0.0, 0.0]}
        a = render("tile_random", c, {"background_input": red})[..., :3]
        self.assertGreater(a[..., 1:].max(), 0.3)  # hues away from red


class TileRandom2Node(unittest.TestCase):
    def test_regular_grid_without_randomness(self):
        a = gray(render("tile_random_sizes", {"amount_x": 4, "amount_y": 6, "random_size_multiplier": 0.0}))
        self.assertEqual(runs(a[a.shape[0] // 12]), 4)
        self.assertEqual(runs(a[:, a.shape[1] // 8]), 6)

    def test_random_sizes_vary_the_cells(self):
        a = render("tile_random_sizes", {"amount_x": 8, "amount_y": 8, "bevel_distance_multiplier": 0.0}, output="uv")
        rnd = a[..., 2][a[..., 3] > 0.5]
        cells = len(np.unique(np.round(rnd, 3)))
        self.assertLess(cells, 64)                    # some cells merged
        self.assertGreater(cells, 20)

    def test_uv_output_spans_each_shape(self):
        a = render("tile_random_sizes", {"random_size_multiplier": 0.0, "amount_x": 2, "amount_y": 2,
                                     "interstice_thickness": 0.0}, output="uv")
        self.assertLess(a[1:63, 1:63, 0].min(), 0.03)
        self.assertGreater(a[1:63, 1:63, 0].max(), 0.97)

    def test_bevel_ramps_from_the_border(self):
        flat = gray(render("tile_random_sizes", {"random_size_multiplier": 0.0, "bevel_distance_multiplier": 0.0}))
        bev = gray(render("tile_random_sizes", {"random_size_multiplier": 0.0, "bevel_distance_multiplier": 1.0}))
        self.assertLess(bev.mean(), flat.mean() * 0.8)
        self.assertGreater(bev.max(), 0.7)

    def test_mask_random_range(self):
        # The range hides the shapes in it (default 0 / 0: none), Invert the others.
        full = gray(render("tile_random_sizes", {}))
        none = gray(render("tile_random_sizes", {"mask_random_start": 0.0, "mask_random_end": 1.0}))
        inv = gray(render("tile_random_sizes", {"mask_random_start": 0.0, "mask_random_end": 1.0,
                                            "mask_random_invert": True}))
        half = gray(render("tile_random_sizes", {"mask_random_start": 0.5, "mask_random_end": 1.0}))
        self.assertGreater(full.mean(), 0.3)
        self.assertLess(none.max(), 0.01)
        np.testing.assert_array_equal(inv, full)
        self.assertLess(half.mean(), full.mean() * 0.8)


class ArcPavementNode(unittest.TestCase):
    def test_scale_repeats_fans(self):
        a = gray(render("arc_pavement", {"scale": 1, "color_random": 0.0}, log2=7))
        b = gray(render("arc_pavement", {"scale": 2, "color_random": 0.0}, log2=7))
        half = gray(render("arc_pavement", {"scale": 1, "color_random": 0.0}, log2=6))
        # The upper rows only: Arcs Amount + 1 = 7 bands per cell stagger differently across the wrap.
        self.assertLess(np.abs(b[:40, :64] - half[:40]).mean(), 0.03)
        self.assertGreater(a.mean(), 0.2)

    def test_color_random_varies_bricks(self):
        a = gray(render("arc_pavement", {"color_random": 0.0}, log2=9))
        b = gray(render("arc_pavement", {"color_random": 0.8}, log2=9))
        self.assertGreater(np.mean(a[a > 0.5] > 0.95), 0.9)
        self.assertLess(np.mean(b[b > 0.1] > 0.95), 0.5)

    def test_more_bricks_per_arc(self):
        p = {"scale": 1, "arcs_amount": 1, "color_random": 0.0}
        few = gray(render("arc_pavement", {**p, "pattern_amount": 4}, log2=8))
        many = gray(render("arc_pavement", {**p, "pattern_amount": 16}, log2=8))
        # A horizontal line above a fan centre crosses its bricks.
        self.assertGreater(runs(many[51, 64:192]), runs(few[51, 64:192]))


def splatter_chain(ss_params=None, tail=None, tail_params=None, output="output", bg=None, log2=7):
    """Shape Splatter (no patterns: built-in paraboloid), optionally a uniform gray background and a
    data consumer `tail` reading both Splatter Data outputs. Returns (ss height, data1, data2, tail out)."""
    pkg = Package()
    g = pkg.new_graph("t", log2)
    ss = g.add_node("shape_splatter")
    for k, v in (ss_params or {}).items():
        g.set_param(ss, k, v)
    if bg is not None:
        b = g.add_node("uniform_color")
        g.set_param(b, "color_mode", "Grayscale")
        g.set_param(b, "output_color", [bg, bg, bg, 1.0])
        g.connect(b, "output", ss, "background_height")
    t = None
    if tail:
        t = g.add_node(tail)
        for k, v in (tail_params or {}).items():
            g.set_param(t, k, v)
        g.connect(ss, "splatter_data_1", t, "splatter_data_1")
        g.connect(ss, "splatter_data_2", t, "splatter_data_2")
    jobs = compile_graph(pkg, "t").jobs
    f = lambda a: a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)
    res = [f(EV.read(jobs[ss], o)) for o in ("height", "splatter_data_1", "splatter_data_2")]
    res.append(f(EV.read(jobs[t], output)) if t is not None else None)
    return res


class ShapeSplatterNode(unittest.TestCase):
    def test_amounts_set_the_grid(self):
        for x, y in ((4, 4), (6, 3)):
            h, *_ = splatter_chain({"x_amount": x, "y_amount": y, "scale": 0.8})
            a = gray(h)
            self.assertEqual(runs(a[a.shape[0] // (2 * y)], 0.3), x)
            self.assertEqual(runs(a[:, a.shape[1] // (2 * x)], 0.3), y)

    def test_seed_changes_randomness(self):
        p = {"x_amount": 6, "y_amount": 6, "position_random": 0.8, "rotation_random": 0.5}
        a = render("shape_splatter", p, output="height")
        b = render("shape_splatter", p, output="height", base={"seed": {"mode": "absolute", "value": 9}})
        self.assertGreater(np.abs(a - b).mean(), 0.01)

    def test_data_ids_follow_column_index(self):
        _, d1, d2, _ = splatter_chain({"x_amount": 4, "y_amount": 4, "pattern_input_number": 3,
                                       "pattern_distribution_mode": "Column Index", "scale": 0.8})
        s = d1.shape[1] // 4
        for col in range(4):
            self.assertAlmostEqual(d1[s // 2, col * s + s // 2, 2], col % 3 + 1)
        self.assertTrue(set(np.unique(d1[..., 2])) <= {0.0, 1.0, 2.0, 3.0})
        covered = d1[..., 2] > 0
        self.assertTrue(np.all(d1[covered, 3] > 0))                         # random in (0, 1]
        self.assertTrue(np.all((d1[covered, :2] >= 0) & (d1[covered, :2] <= 1)))  # pattern uv
        # tile index / count: 16 distinct values
        self.assertEqual(len(np.unique(np.round(d2[covered, 3] * 16))), 16)

    def test_cutout_leaves_no_data_outside_shapes(self):
        h, d1, _, _ = splatter_chain({"x_amount": 4, "y_amount": 4, "scale": 0.6})
        self.assertTrue(np.all(d1[gray(h) <= 0.0, 2] == 0))

    def test_mask_random_hides_everything(self):
        h, d1, _, _ = splatter_chain({"mask_random": 1.0})
        self.assertEqual(gray(h).max(), 0.0)
        self.assertEqual(d1[..., 2].max(), 0.0)

    def test_tiles_sit_on_background(self):
        h, *_ = splatter_chain({"x_amount": 8, "y_amount": 8, "scale": 0.8}, bg=0.4)
        self.assertAlmostEqual(gray(h).min(), 0.4, delta=0.005)
        self.assertAlmostEqual(gray(h).max(), 1.4, delta=0.02)

    def test_preserve_aspect_ratio(self):
        # 0 = tiles stretched to their cells, 1 = squares of the cell's shorter side.
        p = {"x_amount": 8, "y_amount": 2, "scale": 0.8}
        stretched, *_ = splatter_chain(p)
        square, *_ = splatter_chain({**p, "preserve_aspect_ratio": 1.0})
        self.assertGreater((gray(stretched) > 0).mean(), 3 * (gray(square) > 0).mean())

    def test_flatten_pattern_base(self):
        p = {"height_scale_auto_adjust": False, "height_offset": 0.1, "height_scale": 0.2, "scale": 0.8}
        h, *_ = splatter_chain(p, bg=0.5)
        self.assertAlmostEqual(gray(h).min(), 0.5, delta=0.005)            # tiles below the background vanish
        h2, *_ = splatter_chain({**p, "flatten_pattern_base": True}, bg=0.5)
        self.assertLess(gray(h2).min(), 0.15)                              # they override it


class ShapeSplatterCompanions(unittest.TestCase):
    P = {"x_amount": 5, "y_amount": 5, "pattern_input_number": 4, "scale": 0.8}

    def test_to_mask_modes(self):
        _, d1, _, m = splatter_chain(self.P, "shape_splatter_to_mask", {"pattern_id_end_range": 8}, output="output")
        np.testing.assert_array_equal(gray(m) > 0.5, d1[..., 2] > 0)
        _, d1, _, m = splatter_chain(self.P, "shape_splatter_to_mask", output="output")   # default: pattern 1 only
        np.testing.assert_array_equal(gray(m) > 0.5, d1[..., 2] == 1)
        _, d1, _, m = splatter_chain(self.P, "shape_splatter_to_mask", {
            "pattern_id_start_range": 3, "pattern_id_end_range": 2, "output_mode": "Integer Mask"})
        ids = d1[..., 2]
        np.testing.assert_allclose(gray(m), np.where((ids == 2) | (ids == 3), ids, 0.0))
        _, _, _, m = splatter_chain(self.P, "shape_splatter_to_mask", {"random_mask": 1.0})
        self.assertEqual(gray(m).max(), 0.0)

    def test_data_extract(self):
        _, d1, d2, e = splatter_chain(self.P, "shape_splatter_data_extract", output="pattern_id")
        np.testing.assert_allclose(gray(e), d1[..., 2] / 8.0, atol=1e-6)
        _, _, d2, e = splatter_chain(self.P, "shape_splatter_data_extract", output="transform")
        np.testing.assert_allclose(e[..., :3], d2[..., :3], atol=1e-6)

    def test_blend_shows_pattern_over_background(self):
        h, d1, _, b = splatter_chain(self.P, "shape_splatter_blend", output="output")
        out = gray(b)
        self.assertTrue(np.all(out[d1[..., 2] == 0] == 0.0))               # black background
        # unconnected patterns: the paraboloid, as in the height (background 0, height scale 1)
        cov = d1[..., 2] > 0
        np.testing.assert_allclose(out[cov], gray(h)[cov], atol=0.03)
        _, _, _, b2 = splatter_chain(self.P, "shape_splatter_blend", {"luminance_adjustment": -0.5})
        self.assertLess(gray(b2)[cov].mean(), out[cov].mean() - 0.2)


SEED7 = {"seed": {"mode": "absolute", "value": 7}}
class ShapeMapperNode(unittest.TestCase):
    DISC = {"input": ("shape", {"pattern": "Disc", "scale": 0.5})}

    def ring(self, a, radius):
        n = a.shape[0]
        ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        x = (n / 2 + np.cos(ang) * radius * n / 2).astype(int)
        y = (n / 2 + np.sin(ang) * radius * n / 2).astype(int)
        return a[y, x]

    def test_pattern_amount_copies(self):
        for amount in (3, 7, 12):
            a = gray(render("shape_mapper", {"pattern_amount": amount, "radius": 0.6, "width": 0.3}, self.DISC, log2=8))
            self.assertEqual(runs(self.ring(a, 0.6)), amount)

    def test_band_between_radius_and_width(self):
        white = {"input": ("uniform_color", {"color_mode": "Grayscale", "output_color": [1, 1, 1, 1]})}
        a = gray(render("shape_mapper", {"radius": 0.6, "width": 0.2}, white, log2=8))
        self.assertLess(self.ring(a, 0.3).max(), 0.01)
        self.assertGreater(self.ring(a, 0.6).min(), 0.99)
        self.assertLess(self.ring(a, 0.8).max(), 0.01)

    def test_polygon_linked_segments(self):
        a = gray(render("shape_mapper", {"shape": "Polygon", "pattern_amount": 5, "radius": 0.6}, self.DISC, log2=8))
        # Walk the polygon's edges (circumradius 0.6): the middle of the band.
        corners = [np.array([np.cos(k * 2 * np.pi / 5), np.sin(k * 2 * np.pi / 5)]) * 0.6 for k in range(6)]
        pts = np.concatenate([np.linspace(corners[k], corners[k + 1], 200, endpoint=False) for k in range(5)])
        idx = (128 + pts * 128).astype(int)
        self.assertEqual(runs(a[idx[:, 1], idx[:, 0]]), 5)


class PanoramaShapeNode(unittest.TestCase):
    # A shape at the panorama's centre: the inverse Matrix (1/3) times Offset moves it from (0.5, 0.25) down 0.25.
    CENTRE = {"matrix": [3.0, 0.0, 0.0, 3.0], "offset": [0.0, 0.75], "shape_color": [1.0, 1.0, 1.0, 1.0]}

    def test_background_and_hdr_shape(self):
        p = {**self.CENTRE, "enable_background": True, "background_color": [0.1, 0.2, 0.3, 1.0],
             "shape_intensity": 4.0, "hotspot_intensity": 0.0}
        a = render("panorama_shape", p)
        np.testing.assert_allclose(a[5, 5, :3], [0.1, 0.2, 0.3], atol=1e-3)
        self.assertAlmostEqual(a[64, 64, 0], 4.0, delta=0.05)   # HDR: 16f by default

    def test_default_centre_45_degrees_up(self):
        a = render("panorama_shape", {})
        y, x = np.unravel_index(np.argmax(a[..., 0]), a.shape[:2])
        self.assertAlmostEqual(x / a.shape[1], 0.5, delta=0.05)
        self.assertAlmostEqual(y / a.shape[0], 0.25, delta=0.05)

    def test_hotspot_adds(self):
        p = {**self.CENTRE, "shape_intensity": 1.0, "hotspot_intensity": 3.0, "hotspot_size": 0.3,
             "hotspot_falloff": 0.5}
        a = render("panorama_shape", p)
        self.assertAlmostEqual(a[64, 64, 0], 4.0, delta=0.05)

    def test_chains_over_background(self):
        src = {"background": ("uniform_color", {"output_color": [0.0, 0.5, 0.0, 1.0]})}
        a = render("panorama_shape", self.CENTRE, src)
        np.testing.assert_allclose(a[5, 5, :3], [0.0, 0.5, 0.0], atol=3e-3)
        self.assertGreater(a[64, 64, 0], 1.0)


class TriangleGridNode(unittest.TestCase):
    P = {"color_output": "Per Triangle", "color_source": "Random Grayscale"}

    def test_triangle_count_follows_amounts(self):
        for n in (3, 6):
            a = render("triangle_grid", {**self.P, "x_amount": n, "y_amount": n, "random_position": [0, 0]},
                       base={"output_format": {"mode": "absolute", "value": "32f"}})
            values = np.unique(np.round(a[..., 0], 5))
            self.assertEqual(len(values), 2 * n * n)

    def test_seed_and_manual_seed(self):
        a = render("triangle_grid", self.P)
        b = render("triangle_grid", self.P, base=SEED7)
        self.assertGreater(np.abs(a - b).mean(), 0.01)
        m = {**self.P, "random_color_seed_mode": "Manual Seed", "random_position": [0, 0], "height_random": 0}
        np.testing.assert_array_equal(render("triangle_grid", m), render("triangle_grid", m, base=SEED7))

    def test_thin_line_is_mostly_background(self):
        a = gray(render("triangle_grid", {"color_output": "Thin Line", "x_amount": 8, "y_amount": 8}, log2=8))
        self.assertLess(a.mean(), 0.15)
        self.assertGreater(a.max(), 0.9)

    def test_height_drives_per_vertex(self):
        flat = {"height": ("uniform_color", {"color_mode": "Grayscale", "output_color": [0.6, 0.6, 0.6, 1]})}
        a = gray(render("triangle_grid", {"color_output": "Per Vertex", "color_source": "Height", "height_random": 0.0,
                                          "height_input_multiplier": 1.0}, flat,
                        base={"output_format": {"mode": "absolute", "value": "32f"}}))
        np.testing.assert_allclose(a, 0.6, atol=1e-4)

    def test_distance_and_center_by_the_rule(self):
        # Distance to Edge = uv distance / Thickness (Min: nearest edge), Center = 3 x min barycentric.
        f = {"output_format": {"mode": "absolute", "value": "32f"}}
        flat = {"random_position": [0, 0], "x_amount": 4, "y_amount": 4}
        d = gray(render("triangle_grid", {**flat, "color_output": "Distance to Edge",
                                          "distance_to_edge_thickness": 0.1}, base=f))
        self.assertAlmostEqual(d.max(), 0.0732 / 0.1, delta=0.02)  # inradius of half a 1/4 uv cell, in uv
        self.assertLess(d.min(), 0.02)
        ce = gray(render("triangle_grid", {**flat, "color_output": "Center"}, base=f))
        self.assertGreater(ce.max(), 0.95)
        self.assertLess(ce.min(), 0.05)


class ScratchesGeneratorNode(unittest.TestCase):
    def test_spline_number_adds_scratches(self):
        few = gray(render("scratches_generator", {"spline_number": 10}))
        many = gray(render("scratches_generator", {"spline_number": 100}))
        self.assertGreater(many.mean(), few.mean() * 4)

    def test_low_position_random_clusters_at_centre(self):
        a = gray(render("scratches_generator", {"spline_position_random": 0.2, "spline_scale": 0.1}))
        self.assertGreater(a[48:80, 48:80].mean(), 5 * (a.mean() - a[48:80, 48:80].mean() / 16 + 1e-6))
        self.assertLess(a[:24].max() + a[-24:].max(), 0.01)

    def test_seed(self):
        self.assertGreater(np.abs(render("scratches_generator") - render("scratches_generator", base=SEED7)).mean(), 0.005)

    def test_normal_formats_flip_green(self):
        gl = render("scratches_generator", {"normal_format": "OpenGL"}, output="normal")
        dx = render("scratches_generator", {"normal_format": "DirectX"}, output="normal")
        np.testing.assert_allclose(gl[..., 0], dx[..., 0], atol=1 / 255)
        np.testing.assert_allclose(gl[..., 1], 1 - dx[..., 1], atol=2 / 255)
        flat = gray(render("scratches_generator")) == 0
        np.testing.assert_allclose(gl[flat][:, :3], np.tile([0.5, 0.5, 1.0], (flat.sum(), 1)), atol=1 / 255)


F32 = {"output_format": {"mode": "absolute", "value": "32f"}}
class Cube3DNode(unittest.TestCase):
    def test_centred_cube_on_black(self):
        a = gray(render("cube_3d", log2=7))
        self.assertEqual(a[0, 0], 0.0)
        self.assertGreater(a[64, 64], 0.5)                  # nearer than the cube centre: brighter
        big = gray(render("cube_3d", {"scale": 0.9}, log2=7))
        self.assertGreater((big > 0).mean(), (a > 0).mean() * 2)

    def test_size_is_per_axis(self):
        a = gray(render("cube_3d", {"size": [1.0, 0.3, 1.0], "orientation_offset": [0.0, 0.0]}, log2=7))
        rows, cols = np.where(a > 0)
        self.assertAlmostEqual((cols.max() - cols.min()) / (rows.max() - rows.min()), 1 / 0.3, delta=0.4)

    def test_gbuffers(self):
        h = gray(render("cube_3d_gbuffers", output="depth", log2=7))
        np.testing.assert_allclose(h, gray(render("cube_3d", log2=7)))
        n = render("cube_3d_gbuffers", output="normal", log2=7)
        np.testing.assert_allclose(n[0, 0, :3], [0.5, 0.5, 1.0], atol=0.01)
        self.assertGreater(n[..., 1].max(), 0.7)             # the top face points up (OpenGL green)
        pos = render("cube_3d_gbuffers", output="position", log2=7)
        self.assertGreater(pos[..., :3].max(), 0.95)


class HeightExtrudeNode(unittest.TestCase):
    SRC = {"input": ("shape", {"pattern": "Paraboloid"})}
    TOP = {"camera_angle": [0.5, 1.0]}       # the position widget: y = 1 is the top view

    def test_top_view_is_the_scaled_height(self):
        src = gray(render("shape", {"pattern": "Paraboloid"}, log2=6))
        a = gray(render("height_extrude", {**self.TOP, "height_scale": 0.5, "camera_scale": 1.0}, self.SRC,
                        log2=6, output="depth", base=F32))
        np.testing.assert_allclose(a, src * 0.5, atol=0.02)

    def test_tilt_and_ground(self):
        p = {"camera_scale": 1.0, "height_scale": 0.5}
        on = gray(render("height_extrude", {**p, "ground": True}, self.SRC, log2=6, output="depth"))
        off = gray(render("height_extrude", p, self.SRC, log2=6, output="depth"))
        self.assertGreater(on[60, 32], 0.2)                  # ground plane in front of the shape
        self.assertEqual(off[60, 32], 0.0)
        self.assertGreater(off[28, 32], 0.3)                  # the shape itself stays

    def test_normal_flat_on_flat_ground(self):
        n = render("height_extrude", {**self.TOP, "camera_scale": 1.0}, self.SRC, log2=6, output="normal")
        np.testing.assert_allclose(n[1, 1, :3], [0.5, 0.5, 1.0], atol=0.01)


class ShapeExtrudeNode(unittest.TestCase):
    BOX = {"extrude_shape": "Cube", "extrude_depth": 0.0, "scale": 0.5, "position": [0.5, 0.5]}   # top view
    TOP = {**BOX, "bevel_intensity": 0.0}

    def test_top_view_height(self):
        a = gray(render("shape_extrude", {**self.TOP, "extrude_height": 0.5}, log2=6, base=F32))
        # The profile: the top at half the frame x Scale x Extrude Height = 0.125; [0, 1]: 0.5 + 0.5 x that
        self.assertAlmostEqual(a[32, 32], 0.5625, delta=0.01)
        self.assertEqual(a[0, 0], 0.0)
        b = gray(render("shape_extrude", {**self.TOP, "extrude_height": 0.5, "output_range": "[-1, 1]"},
                        log2=6, base=F32))
        self.assertAlmostEqual(b[32, 32], 0.125, delta=0.01)
        self.assertAlmostEqual((a > 0).mean(), 0.25, delta=0.03)  # Scale 0.5 square section

    def test_bevel_lowers_the_border(self):
        a = gray(render("shape_extrude", {**self.BOX, "bevel_intensity": 0.5,
                                          "bevel_height": 1.0}, log2=6, base=F32))
        self.assertGreater(a[32, 32], a[32, 18] + 0.02)

    def test_custom_and_mask(self):
        disc = {"extrude_shape_input": ("shape", {"pattern": "Disc", "scale": 0.5})}
        a = gray(render("shape_extrude", {**self.TOP, "extrude_shape": "Custom Input", "scale": 1.0}, disc, log2=6))
        self.assertAlmostEqual((a > 0).mean(), np.pi / 16, delta=0.02)
        hidden = gray(render("shape_extrude", {**self.TOP, "profile_type": "Mask"},
                             {"profile_mask": ("uniform_color", {"color_mode": "Grayscale",
                                                                 "output_color": [0, 0, 0, 1]})}, log2=6))
        self.assertEqual(hidden.max(), 0.0)

    def test_downscale_blocks_and_default_tilt(self):
        a = gray(render("shape_extrude", {"downscale_multiplier": 2}, log2=6, base=F32))
        np.testing.assert_array_equal(a[::4, ::4], a[3::4, 3::4])         # one ray per 4 x 4 block
        top = gray(render("shape_extrude", {"position": [0.5, 0.5]}, log2=6, base=F32))
        tilt = gray(render("shape_extrude", {}, log2=6, base=F32))         # Position 0.75, 0.25: seen from a side
        self.assertGreater((tilt > 0).mean(), (top > 0).mean() + 0.05)


if __name__ == "__main__":
    unittest.main()
