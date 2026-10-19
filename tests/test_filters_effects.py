"""Filters/Effects and Filters/Normal Map behave as documented."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph

EV = Evaluator()
F32 = {"mode": "absolute", "value": "32f"}


class Builder:
    """A small graph read back as float arrays."""

    def __init__(self, log2=6):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", log2)

    def node(self, type, inputs=None, f32=False, **params):
        n = self.g.add_node(type)
        for k, v in params.items():
            self.g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            src, out = src if isinstance(src, tuple) else (src, "output")
            self.g.connect(src, out, n, port)
        if f32:
            self.g.set_base(n, "output_format", F32)
        return n

    def gray(self, v):
        return self.node("uniform_color", color_mode="Grayscale", output_color=[v, v, v, 1])

    def color(self, rgba):
        return self.node("uniform_color", f32=True, output_color=list(rgba))

    def disc(self, scale=0.5):
        return self.node("shape", pattern="Disc", scale=scale)

    def read(self, n, out="output"):
        a = EV.read(compile_graph(self.pkg, "t").jobs[n], out)
        return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)


FLAT = (0.5, 0.5, 1.0, 1.0)


class NormalMaps(unittest.TestCase):
    def test_flat_height_gives_flat_normal(self):
        for node in ("normal_sobel", "height_to_normal_world_units"):
            t = Builder()
            n = t.node(node, {"input": t.gray(0.4)})
            np.testing.assert_allclose(t.read(n)[10, 10, :3], FLAT[:3], atol=0.01, err_msg=node)

    def test_normal_color_follows_direction_and_slope(self):
        # Normal Color: R = 0.5 - 0.5 cos(a) s, G = 0.5 +- 0.5 sin(a) s, B = 1.
        t = Builder(4)
        np.testing.assert_allclose(t.read(t.node("normal_color"))[5, 5], FLAT, atol=0.005)
        cases = (({"slope_angle": 0.25}, (0.0, 0.5)), ({"slope_angle": 0.25, "direction": 0.5}, (1.0, 0.5)),
                 ({"slope_angle": 0.25, "direction": 0.25}, (0.5, 0.0)),
                 ({"slope_angle": 0.25, "direction": 0.25, "normal_format": "OpenGL"}, (0.5, 1.0)),
                 ({"slope_angle": 1 / 12}, (0.25, 0.5)))
        for params, rg in cases:
            a = t.read(t.node("normal_color", **params))
            np.testing.assert_allclose(a[5, 5], (*rg, 1.0, 1.0), atol=0.005, err_msg=str(params))
            self.assertLess(np.ptp(a[..., :3], axis=(0, 1)).max(), 1e-6)  # uniform

    def test_normal_format_flips_green(self):
        t = Builder()
        h = t.node("test_pattern")
        gl = t.read(t.node("normal_sobel", {"input": h}))
        dx = t.read(t.node("normal_sobel", {"input": h}, normal_format="DirectX"))
        np.testing.assert_allclose(dx[..., 1], 1.0 - gl[..., 1], atol=0.01)
        np.testing.assert_allclose(dx[..., 0], gl[..., 0], atol=0.01)

    def test_slope_direction(self):
        t = Builder()
        h = t.disc()
        a = t.read(t.node("normal_sobel", {"input": h}, intensity=3.0))
        # Disc of radius 0.25 at the centre: its right edge faces +X, its top edge +Y (OpenGL).
        self.assertGreater(a[32, 48, 0], 0.6)
        self.assertGreater(a[16, 32, 1], 0.6)

    def test_invert_green(self):
        t = Builder()
        n = t.node("normal_invert", {"input": t.color((0.2, 0.3, 0.9, 1.0))})
        np.testing.assert_allclose(t.read(n)[0, 0], [0.2, 0.7, 0.9, 1.0], atol=0.01)

    def test_intensity_scales_the_tilt(self):
        # Tilt x (1 + i) below 0, towards the unit tilt above 0, z rebuilt; flat stays flat.
        tilted = (0.65, 0.5, 0.5 + 0.5 * np.sqrt(1 - 0.3 ** 2), 1.0)  # tilt 0.3 along x
        for i, x, z in ((0.0, 0.3, np.sqrt(1 - 0.09)), (-0.5, 0.15, np.sqrt(1 - 0.0225)), (-1.0, 0.0, 1.0),
                        (1.0, 1.0, 0.0), (0.5, 0.65, np.sqrt(1 - 0.4225))):
            t = Builder()
            a = t.read(t.node("normal_intensity", {"input": t.color(tilted)}, f32=True, intensity=i))
            np.testing.assert_allclose(a[0, 0], [0.5 + x / 2, 0.5, 0.5 + z / 2, 1.0], atol=0.01, err_msg=str(i))
        t = Builder()
        a = t.read(t.node("normal_intensity", {"input": t.color((0.502, 0.5, 1.0, 0.4))}, intensity=1.0))
        np.testing.assert_allclose(a[0, 0], [0.5, 0.5, 1.0, 0.4], atol=0.01)  # nearly flat: flat

    def test_normalize(self):
        t = Builder()
        n = t.node("normal_normalize", {"input": t.color((0.9, 0.5, 0.6, 1.0))}, f32=True)
        v = t.read(n)[0, 0, :3] * 2 - 1
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=3)

    def test_facing_flat_is_white(self):
        t = Builder()
        self.assertAlmostEqual(t.read(t.node("facing_normal", {"input": t.color(FLAT)}))[0, 0, 0], 1.0, places=2)

    def test_combine_with_flat_detail_keeps_base(self):
        t = Builder()
        base = t.node("normal_sobel", {"input": t.node("test_pattern")})
        for tech in ("Whiteout", "Channel mixer", "Detail oriented"):
            n = t.node("normal_combine", {"normal_1": base, "normal_2": t.color(FLAT)}, technique=tech)
            np.testing.assert_allclose(t.read(n)[..., :3], t.read(base)[..., :3], atol=0.02, err_msg=tech)

    def test_combine_techniques_on_encoded_values(self):
        # Works on encoded values: XY added around 0.5; Whiteout multiplies Z then
        # normalises, Channel mixer keeps the lower Z and does not normalise.
        t = Builder()
        a, b = t.color((0.6, 0.5, 0.9, 0.3)), t.color((0.7, 0.5, 0.8, 1.0))
        expected = {"Whiteout": [0.903, 0.5, 0.796, 1.0], "Channel mixer": [0.8, 0.5, 0.8, 1.0]}
        for tech, want in expected.items():
            n = t.node("normal_combine", {"normal_1": a, "normal_2": b}, technique=tech)
            np.testing.assert_allclose(t.read(n)[0, 0], want, atol=0.01, err_msg=tech)

    def test_vector_rotation_half_turn_flips_xy(self):
        t = Builder()
        n = t.node("normal_vector_rotation", {"normal": t.color((0.8, 0.3, 0.8, 1.0))}, rotation_angle=0.5)
        np.testing.assert_allclose(t.read(n)[0, 0, :3], [0.2, 0.7, 0.8], atol=0.01)

    def test_transform_rotation_matches_rotated_height(self):
        t = Builder()
        h = t.node("test_pattern")
        rot = [0.0, -1.0, 1.0, 0.0]
        a = t.read(t.node("normal_transform", {"input": t.node("normal_sobel", {"input": h})}, matrix2x2=rot))
        b = t.read(t.node("normal_sobel", {"input": t.node("transformation_2d", {"input": h}, transformation_matrix=rot)}))
        self.assertLess(np.abs(a[..., :3] - b[..., :3]).mean(), 0.02)

    def test_normal_to_height_recovers_the_height(self):
        for node in ("normal_to_height", "normal_to_height_hq"):
            t = Builder(7)
            h = t.disc(0.4)
            out = t.read(t.node(node, {"input": t.node("normal_sobel", {"input": h})}, f32=True))[..., 0]
            ref = t.read(h)[..., 0]
            corr = np.corrcoef(out.ravel(), ref.ravel())[0, 1]
            self.assertGreater(corr, 0.95, node)

    def test_bent_normal_flat(self):
        t = Builder(5)
        n = t.node("bent_normal", {"input": t.gray(0.5)})
        np.testing.assert_allclose(t.read(n)[8, 8, :3], FLAT[:3], atol=0.02)


class Curvature(unittest.TestCase):
    def test_gain_per_pixel_of_divergence(self):
        # A ramp as the normal's y component changes it by 2/N per pixel down the image; the output moves from 0.5 by
        # 0.4 x that x Intensity, at any resolution.
        for log2 in (7, 8):
            t = Builder(log2)
            normal = t.node("rgba_merge", {"r": t.gray(0.5), "g": t.node("gradient_linear_ramp", f32=True),
                                           "b": t.gray(1.0), "a": t.gray(1.0)}, f32=True)
            n = t.node("curvature", {"input": normal}, f32=True, intensity=10.0)
            v = t.read(n)[2 ** log2 // 2, 2 ** log2 // 2, 0]
            self.assertAlmostEqual(abs(v - 0.5), 0.4 * 2 / 2 ** log2 * 10.0, delta=0.002)

    def test_flat_is_mid_gray_and_bump_is_convex(self):
        t = Builder()
        flat = t.color(FLAT)
        for node in ("curvature", "curvature_sobel"):
            self.assertAlmostEqual(t.read(t.node(node, {"input": flat}))[5, 5, 0], 0.5, delta=0.01)
        self.assertAlmostEqual(t.read(t.node("curvature_smooth", {"normal": flat}), "curvature")[5, 5, 0], 0.5,
                               delta=0.01)
        nrm = t.node("normal_sobel", {"input": t.node("blur", {"input": t.disc()}, intensity=8.0)}, intensity=2.0)
        c = t.read(t.node("curvature_smooth", {"normal": nrm}), "curvature")[..., 0]
        self.assertGreater(c[32, 32 - 14], 0.5)   # inside the rim: convex
        self.assertLess(c[32, 32 - 20], 0.5)      # at the foot: concave


class FloodFill(unittest.TestCase):
    def tiles(self, t):
        return t.node("flood_fill", {"input": t.node("tile_generator", x_amount=4, y_amount=4)})

    def test_one_value_per_shape_and_distinct_shapes(self):
        t = Builder(7)
        ff = self.tiles(t)
        g = t.read(t.node("flood_fill_to_random_grayscale", {"flood_fill": ff}, f32=True))[..., 0]
        cells = [g[16 + 32 * j - 6:16 + 32 * j + 6, 16 + 32 * i - 6:16 + 32 * i + 6] for j in range(4) for i in range(4)]
        for c in cells:
            self.assertLess(np.ptp(c), 1e-6)  # uniform inside a tile
        self.assertEqual(len({round(float(c[0, 0]), 5) for c in cells}), 16)

    def test_index_counts_and_orders_shapes(self):
        t = Builder(7)
        ff = self.tiles(t)
        n = t.node("flood_fill_to_index", {"flood_fill_bbox": ff, "special_shape_info": (ff, "special_shape_info")},
                   output="Integer")
        idx = t.read(n, "shape_index")[..., 0]
        self.assertAlmostEqual(t.read(n, "shape_count")[0, 0, 0], 16.0)
        self.assertAlmostEqual(idx[16, 16], 0.0)
        self.assertAlmostEqual(idx[16, 48], 1.0)
        self.assertAlmostEqual(idx[112, 112], 15.0)

    def test_output_smaller_than_input(self):
        """The input at twice the node's size gives the same shapes as an input at its size."""
        t = Builder(6)
        tiles = t.node("tile_generator", x_amount=4, y_amount=4)
        t.g.set_base(tiles, "output_size", {"mode": "relative_parent", "value": [1, 1]})
        ff = t.node("flood_fill", {"input": tiles})
        t.g.set_base(ff, "output_size", {"mode": "relative_input", "value": [-1, -1]})
        g = t.read(t.node("flood_fill_to_random_grayscale", {"flood_fill": ff}, f32=True))[..., 0]
        self.assertEqual(g.shape, (64, 64))
        cells = [g[8 + 16 * j - 3:8 + 16 * j + 3, 8 + 16 * i - 3:8 + 16 * i + 3] for j in range(4) for i in range(4)]
        for c in cells:
            self.assertLess(np.ptp(c), 1e-6)
        self.assertEqual(len({round(float(c[0, 0]), 5) for c in cells}), 16)
        bbox = t.read(ff)[8, 8]
        self.assertAlmostEqual(bbox[2], bbox[3], places=2)  # a square tile's box
        self.assertGreater(bbox[2], 0.15)
        self.assertLess(bbox[2], 0.25)

    def test_winding_shapes_match_a_cpu_labelling(self):
        """Shapes of a thresholded noise (winding ones, that take several rounds) against a breadth-first labelling
        on the CPU: each pixel's box size, the shapes' count and order. The rounds stop once nothing changes: every
        trade-off that reaches that point gives the same result."""
        from collections import deque
        t = Builder(7)
        t.g.set_base(None, "tiling", {"mode": "absolute", "value": "none"})
        mask = t.node("threshold", {"input": t.node("perlin_noise", scale=6)}, threshold=0.55)
        ff = {trade: t.node("flood_fill", {"input": mask}, safety_speed_trade_off=trade)
              for trade in ("Complex or big shapes", "No failure mode")}
        m = t.read(mask)[..., 0] >= 0.5
        h, w = m.shape
        label = -np.ones((h, w), int)
        boxes = []
        for y in range(h):
            for x in range(w):
                if m[y, x] and label[y, x] < 0:
                    label[y, x], todo, pts = len(boxes), deque([(y, x)]), []
                    while todo:
                        cy, cx = todo.popleft()
                        pts.append((cy, cx))
                        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                            if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and label[ny, nx] < 0:
                                label[ny, nx] = len(boxes)
                                todo.append((ny, nx))
                    ys, xs = zip(*pts)
                    boxes.append(((max(xs) - min(xs) + 1) / w, (max(ys) - min(ys) + 1) / h))
        self.assertGreater(len(boxes), 3)
        out = {k: (t.read(n), t.read(n, "special_shape_info")) for k, n in ff.items()}
        np.testing.assert_array_equal(*[o[0] for o in out.values()])
        bbox, info = out["Complex or big shapes"]
        expected = np.zeros((h, w, 2))
        expected[m] = np.array(boxes)[label[m]]
        np.testing.assert_allclose(bbox[..., 2:4], expected, atol=1e-6)
        np.testing.assert_allclose(info[..., 2][m], label[m], atol=1e-6)  # the index: row-major by first pixel
        self.assertAlmostEqual(info[m][0, 3], len(boxes))

    def test_shapes_across_the_tiling_edges_match_a_cpu_labelling(self):
        """A tiled thresholded noise (shapes across the edges, some going round the image) against a breadth-first
        labelling on the torus that follows each pixel's unwrapped position: each pixel's box size (the whole image
        along an axis a shape goes round), its position in the box, the shapes' count and order."""
        from collections import deque
        t = Builder(7)
        mask = t.node("threshold", {"input": t.node("perlin_noise", scale=6)}, threshold=0.55)
        ff = t.node("flood_fill", {"input": mask})
        m = t.read(mask)[..., 0] >= 0.5
        h, w = m.shape
        label = -np.ones((h, w), int)
        size = np.zeros((h, w, 2))
        local = np.zeros((h, w, 2))
        shapes = 0
        for y in range(h):
            for x in range(w):
                if not m[y, x] or label[y, x] >= 0:
                    continue
                label[y, x], at, todo, loops = shapes, {(y, x): (y, x)}, deque([(y, x)]), [False, False]
                while todo:
                    cy, cx = todo.popleft()
                    uy, ux = at[(cy, cx)]
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = (cy + dy) % h, (cx + dx) % w
                        if not m[ny, nx]:
                            continue
                        u = (uy + dy, ux + dx)
                        if (ny, nx) in at:
                            loops[0] |= at[(ny, nx)][1] != u[1]
                            loops[1] |= at[(ny, nx)][0] != u[0]
                            continue
                        label[ny, nx], at[(ny, nx)] = shapes, u
                        todo.append((ny, nx))
                us = np.array(list(at.values()))
                lo, span = us.min(0), us.max(0) - us.min(0) + 1
                for (py, px), (uy, ux) in at.items():
                    for k, (n, wraps, pos, low, s) in enumerate(((w, loops[0], ux, lo[1], span[1]),
                                                                 (h, loops[1], uy, lo[0], span[0]))):
                        whole = wraps or s >= n
                        size[py, px, k] = 1.0 if whole else s / n
                        local[py, px, k] = ((px, py)[k] + 0.5) / n if whole else (pos - low + 0.5) / s
                shapes += 1
        crossing = (label[:, 0] >= 0) & (label[:, 0] == label[:, -1])
        self.assertTrue(crossing.any())  # shapes across the edge, the case this is about
        self.assertGreater(shapes, 3)
        bbox, info = t.read(ff), t.read(ff, "special_shape_info")
        np.testing.assert_allclose(bbox[..., 2:4][m], size[m], atol=1e-6)
        np.testing.assert_allclose(bbox[..., 0:2][m], local[m], atol=1e-6)
        np.testing.assert_allclose(info[..., 2][m], label[m], atol=1e-6)
        self.assertAlmostEqual(info[m][0, 3], shapes)

    def test_bbox_size(self):
        t = Builder(7)
        v = t.read(t.node("flood_fill_to_bbox_size", {"flood_fill": self.tiles(t)}))[16, 16, 0]
        self.assertGreater(v, 0.15)
        self.assertLess(v, 0.25)


class Occlusion(unittest.TestCase):
    def test_ao_darker_in_pits(self):
        for node in ("ambient_occlusion_hbao", "ambient_occlusion_rtao"):
            t = Builder()
            pit = t.node("invert", {"input": t.disc(0.3)})
            a = t.read(t.node(node, {"input": pit}))[..., 0]
            self.assertLess(a[32, 32 - 8], a[2, 2] - 0.02, node)  # floor near the pit's wall vs open plateau

    def test_hbao_flat_is_white_and_radius_limits_the_reach(self):
        t = Builder()
        self.assertAlmostEqual(t.read(t.node("ambient_occlusion_hbao", {"input": t.gray(0.4)}))[5, 5, 0], 1.0,
                               places=3)
        # A wide pit: its centre sees the walls 19 px away only with a radius reaching that far (64 px image).
        pit = t.node("invert", {"input": t.disc(0.6)})
        far = t.read(t.node("ambient_occlusion_hbao", {"input": pit}, radius=1.0))[32, 32, 0]
        near = t.read(t.node("ambient_occlusion_hbao", {"input": pit}, radius=0.1))[32, 32, 0]
        self.assertLess(far, near - 0.05)
        self.assertAlmostEqual(near, 1.0, places=3)

    def test_rt_shadow_behind_a_block(self):
        t = Builder()
        n = t.node("rt_shadow", {"input": t.disc(0.3)}, mode="Shadow Only", light_position=[0.95, 0.5],
                   max_shadows_length=0.5)
        a = t.read(n)[..., 0]
        # Light position on the right, near the edge (low light): the shadow falls on the left of the disc.
        self.assertLess(a[32, 14], 0.5)
        self.assertGreater(a[32, 54], 0.9)
        inverted = t.read(t.node("rt_shadow", {"input": t.disc(0.3)}, mode="Shadow Only", invert=True,
                                 light_position=[0.95, 0.5], max_shadows_length=0.5))[..., 0]
        np.testing.assert_allclose(inverted, 1.0 - a, atol=0.01)


class Misc(unittest.TestCase):
    def test_summed_area_table(self):
        t = Builder(5)
        src = t.node("test_pattern")
        sat = t.read(t.node("summed_area_table", {"input": src}))[..., 0]
        img = t.read(src)[..., 0]
        self.assertAlmostEqual(sat[-1, -1], img.mean(), places=4)
        self.assertAlmostEqual(sat[3, 5], img[:4, :6].sum() / img.size, places=5)

    def test_identities(self):
        t = Builder()
        src = t.node("test_pattern_color")
        ref = t.read(src)
        for n in (t.node("vector_warp", {"input": src, "vector_map": t.color((0.5, 0.5, 0.5, 1))}),
                  t.node("swirl", {"input": src}, amount=0.0),
                  t.node("mosaic", {"color": src, "mosaic_map": t.node("test_pattern")}, samples=0)):
            np.testing.assert_allclose(t.read(n), ref, atol=0.01)

    def test_vector_warp_scales_by_length(self):
        # The input read at uv + v |v| Intensity, v = red/green centred on 0.5.
        t = Builder()
        src = t.node("test_pattern_color")
        ref = t.read(src)
        for rg, shift in (((0.75, 0.5), (0, 16)), ((0.5, 0.25), (-16, 0))):
            with self.subTest(rg=rg):
                n = t.node("vector_warp", {"input": src, "vector_map": t.color((*rg, 0.5, 1))})
                np.testing.assert_allclose(t.read(n), np.roll(ref, (-shift[0], -shift[1]), axis=(0, 1)), atol=0.01)

    def test_diffusion_keeps_known_and_fills_between(self):
        t = Builder(5)
        src = t.node("test_pattern")
        mask = t.node("shape", pattern="Square", scale=0.5)
        n = t.node("diffusion_grayscale", {"source": src, "mask": mask}, f32=True)
        out, s, m = t.read(n)[..., 0], t.read(src)[..., 0], t.read(mask)[..., 0]
        known = m >= 0.5
        np.testing.assert_allclose(out[known], s[known], atol=0.01)
        self.assertGreater(out[~known].min(), s[known].min() - 0.05)  # values come from the known area
        self.assertLess(out[~known].max(), s[known].max() + 0.05)

    def test_shape_stroke_outside_the_shape(self):
        t = Builder()
        n = t.node("shape_stroke", {"input": t.disc()}, width=0.5, outline_color=[1, 0, 0, 1], mask_color=[1, 1, 1, 1])
        a, m = t.read(n), t.read(n, "mask")[..., 0]
        np.testing.assert_allclose(a[32, 32], [1, 1, 1, 1], atol=0.01)      # the shape (Mask Color)
        # The reach: 128 x Width^2 pixels at 256 = 8 pixels here, beyond the disc's radius of 16.
        np.testing.assert_allclose(a[32, 32 + 22], [1, 0, 0, 1], atol=0.01)
        self.assertAlmostEqual(a[32, 32 + 26, 3], 0.0, places=2)            # transparent beyond it
        self.assertAlmostEqual(m[32, 32], 0.0, places=2)                     # Mask: the stroke alone
        self.assertAlmostEqual(m[32, 32 + 22], 1.0, places=2)

    def test_shape_glow_and_shadow_masks_outside_the_shape(self):
        t = Builder()
        for node, params in (("shape_glow", {"width": 0.5}), ("shape_drop_shadow", {"distance": 0.2})):
            m = t.read(t.node(node, {"input": t.disc()}, opacity=1.0, **params), "mask")[..., 0]
            self.assertAlmostEqual(m[32, 32], 0.0, places=2, msg=node)
            self.assertGreater(m[32 + 17, 32], 0.2, msg=node)  # below the disc (the shadow's default Angle)

    def test_edge_detect_on_edges_only(self):
        # Black edges on white; Invert gives white edges on black.
        t = Builder(8)
        a = t.read(t.node("edge_detect", {"input": t.disc()}))[..., 0]
        self.assertLess(a[128, 192], 0.1)
        self.assertGreater(a[128, 128], 0.9)
        self.assertGreater(a[2, 2], 0.9)
        b = t.read(t.node("edge_detect", {"input": t.disc()}, invert=True))[..., 0]
        np.testing.assert_allclose(b, 1.0 - a, atol=1e-6)

    def test_3d_sdf_sign(self):
        t = Builder(8)
        pos = t.node("3d_texture_position")
        mask = t.node("3d_volume_mask", {"position": pos}, position_input="World Space Position",
                      position_world_space=[0.5, 0.5, 0.5], scale=0.6)
        sdf = t.read(t.node("3d_texture_sdf", {"mask_input": mask}))[..., 0]
        m = t.read(mask)[..., 0]
        self.assertTrue(np.all(sdf[m > 0.6] < 0))
        self.assertTrue(np.all(sdf[m < 0.4] > 0))

    def test_3d_texture_position_grid(self):
        # X, Y = fract(pos x Grid Size), Z = slice index / slices (row by row).
        t = Builder(5)
        a = t.read(t.node("3d_texture_position", f32=True, grid_size=[4, 2]))
        np.testing.assert_allclose(a[0, 0, :3], [0.0625, 0.0625 / 2, 0.0], atol=1e-4)
        np.testing.assert_allclose(a[0, 8, 2], 1 / 8, atol=1e-4)    # second column
        np.testing.assert_allclose(a[16, 0, 2], 4 / 8, atol=1e-4)   # second row

    def test_reaction_diffusion_is_contrasted(self):
        t = Builder(7)
        a = t.read(t.node("reaction_diffusion_fast", {"input": t.node("perlin_noise")}))[..., 0]
        self.assertGreater(((a < 0.1) | (a > 0.9)).mean(), 0.6)


if __name__ == "__main__":
    unittest.main()
