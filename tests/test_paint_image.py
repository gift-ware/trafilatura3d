"""The Paint workspace's image filters (nodes/paint/image/): behaviour checked on images
made with numpy and fed through Bitmap nodes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from trafilatura3d.compiler import compile_graph
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.package import Package

EV = None
TOL = 1.5 / 255.0


def ev():
    global EV
    if EV is None:
        EV = Evaluator()
    return EV


def run(node_type: str, inputs: dict, params: dict | None = None) -> dict:
    """Cook `node_type` with numpy images on its inputs (HxW grayscale or HxWx4 RGBA, float 0-1): its outputs as
    float arrays HxWxC."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Package()
        size = next(iter(inputs.values())).shape[0]
        g = pkg.new_graph("t", int(np.log2(size)))
        node = g.add_node(node_type)
        for name, value in (params or {}).items():
            g.set_param(node, name, value)
        for port, image in inputs.items():
            a = np.asarray(image, np.float32)
            gray = a.ndim == 2
            a = np.ascontiguousarray(a[..., None] if gray else a)  # OIIO wants an explicit channel axis
            path = str(Path(tmp) / f"{port}.exr")
            out = oiio.ImageOutput.create(path)
            out.open(path, oiio.ImageSpec(a.shape[1], a.shape[0], a.shape[2], oiio.FLOAT))
            out.write_image(a)
            out.close()
            b = g.add_node("bitmap")
            g.set_param(b, "pkg_resource_path", path)
            g.set_param(b, "color_mode", "Grayscale" if gray else "Color")
            g.connect(b, "output", node, port)
        job = compile_graph(pkg, "t").jobs[node]
        res = {}
        for o in job.node.outputs:
            r = ev().read(job, o)
            res[o] = r.astype(np.float64) / (255.0 if r.dtype == np.uint8 else 1.0)
        return res


def disc(n=64, r=0.25, cx=0.5, cy=0.5):
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    return (np.hypot(x - cx, y - cy) < r).astype(np.float32)


def square(n=64, lo=0.3, hi=0.7):
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    return ((x > lo) & (x < hi) & (y > lo) & (y < hi)).astype(np.float32)


def rgba(rgb, alpha):
    a = np.zeros(alpha.shape + (4,), np.float32)
    a[..., :3] = rgb
    a[..., 3] = alpha
    return a


def noise(n=64, seed=1):
    return np.random.default_rng(seed).random((n, n)).astype(np.float32)


class Transforms(unittest.TestCase):
    def test_transform(self):
        img = noise()
        self.assertLess(np.abs(run("paint_transform", {"input": img})["output"][..., 0] - img).max(), TOL)
        rolled = run("paint_transform", {"input": img}, {"offset_x": 0.5})["output"][..., 0]
        self.assertLess(np.abs(rolled - np.roll(img, 32, axis=1)).max(), TOL)  # Repeat: the offset wraps around
        flipped = run("paint_transform", {"input": img}, {"flip_mode": "Horizontal"})["output"][..., 0]
        self.assertLess(np.abs(flipped - img[:, ::-1]).max(), TOL)
        small = run("paint_transform", {"input": np.ones((64, 64), np.float32)},
                    {"scaling": 0.5, "repeat_mode": "None"})["output"][..., 0]
        self.assertLess(small[2, 2], TOL)  # None: nothing outside the scaled down input
        self.assertGreater(small[32, 32], 1 - TOL)

    def test_blur_directional_follows_angle(self):
        rows = np.repeat(noise()[:, :1], 64, axis=1)  # constant along each row
        along = run("paint_blur_directional", {"input": rows}, {"angle": 0.0, "intensity": 1.0})["output"][..., 0]
        self.assertLess(np.abs(along - rows).max(), 2 * TOL)  # blurring along the rows changes nothing
        across = run("paint_blur_directional", {"input": rows}, {"angle": 0.25, "intensity": 1.0})["output"][..., 0]
        self.assertLess(across.std(), rows.std() * 0.7)

    def test_warp(self):
        flat = np.full((64, 64), 0.4, np.float32)
        img = noise()
        self.assertLess(np.abs(run("paint_warp", {"input": flat})["output"][..., 0] - 0.4).max(), TOL)  # uniform stays
        self.assertGreater(np.abs(run("paint_warp", {"input": img})["output"][..., 0] - img).mean(), 0.05)
        still = run("paint_warp", {"input": img}, {"intensity": 0.0})["output"][..., 0]
        self.assertLess(np.abs(still - img).max(), TOL)
        a = run("paint_warp", {"input": img}, {"warp_mode": "Directional"})["output"][..., 0]
        b = run("paint_warp", {"input": img}, {"warp_mode": "Directional", "intensity_divider": "100"})["output"][..., 0]
        self.assertGreater(np.abs(a - img).mean(), np.abs(b - img).mean() * 3)  # the divider weakens it

    def test_kuwahara_flattens_but_keeps_edges(self):
        rng = np.random.default_rng(3)
        img = np.clip(np.where(np.arange(64)[None, :] < 32, 0.8, 0.2) + rng.normal(0, 0.05, (64, 64)), 0, 1)
        out = run("paint_anisotropic_kuwahara", {"input": img.astype(np.float32)})["output"][..., 0]
        self.assertLess(out[:, 8:24].std(), img[:, 8:24].std() * 0.6)  # flatter areas
        self.assertGreater(out[:, 28].mean() - out[:, 35].mean(), 0.5)  # the edge stays sharp


class Effects(unittest.TestCase):
    def test_fxaa(self):
        n = 64
        y, x = np.mgrid[0:n, 0:n]
        stairs = (x > y // 4 * 4 * 0.7 + 10).astype(np.float32)  # a jagged edge
        out = run("paint_fxaa_anti_aliasing", {"input": stairs})["output"][..., 0]
        self.assertGreater(((out > 0.1) & (out < 0.9)).sum(), 10)  # in-between values smooth the steps
        flat = np.full((n, n), 0.3, np.float32)
        self.assertLess(np.abs(run("paint_fxaa_anti_aliasing", {"input": flat})["output"][..., 0] - 0.3).max(), TOL)
        img = rgba(0.5, stairs)
        kept = run("paint_fxaa_anti_aliasing", {"input": img})["output"][..., 3]
        self.assertLess(np.abs(kept - stairs).max(), TOL)  # alpha untouched by default
        aa = run("paint_fxaa_anti_aliasing", {"input": img}, {"affect_alpha_channel": True})["output"][..., 3]
        self.assertGreater(((aa > 0.1) & (aa < 0.9)).sum(), 10)

    def test_drop_shadow(self):
        """A shadow of the painted shape, towards Angle, in Color."""
        shape = square(lo=0.3, hi=0.5)
        out = run("paint_drop_shadow", {"input": rgba((1.0, 1.0, 1.0), shape)},
                  {"angle": 0.125, "distance": 32.0, "size": 2.0, "opacity": 1.0})["output"]
        self.assertGreater(out[36, 36, 3], 0.8)  # below right of the square: shadow
        self.assertLess(out[36, 36, 0], 0.1)     # black
        self.assertLess(out[14, 14, 3], 0.05)    # above left: nothing
        self.assertGreater(out[25, 25, 0], 1 - TOL)  # the shape itself is kept

    def test_glow(self):
        shape = disc(r=0.2)
        for mode in ("Soft", "Precise"):
            with self.subTest(mode=mode):
                out = run("paint_glow", {"input": rgba((0.2, 0.4, 0.9), shape)}, {"mode": mode})["output"]
                ring = out[32, 32 + 14]  # just outside the disc (radius 12.8 px)
                self.assertGreater(ring[3], 0.15)
                self.assertGreater(ring[0], ring[2])  # in the glow color (warm), not the shape's
                self.assertLess(out[2, 2, 3], 0.05)  # fades away
                np.testing.assert_allclose(out[32, 32], [0.2, 0.4, 0.9, 1.0], atol=2 * TOL)
        mask = run("paint_glow", {"input": shape}, {"used_on_a_mask": True})["output"][..., 0]
        self.assertGreater(mask[32, 46], 0.15)
        self.assertGreater(mask[32, 32], 1 - TOL)


class Distances(unittest.TestCase):
    def test_bevel_smooth_directions(self):
        m = square(lo=0.3, hi=0.7)  # left edge between pixels 18 and 19
        kw = {"distance": 16.0, "smoothing": 0.0}  # 4 px here
        outward = run("paint_bevel_smooth", {"input": m}, {**kw, "direction": "Outward"})["output"][..., 0]
        self.assertGreater(outward[32, 32], 1 - TOL)
        self.assertTrue(0.05 < outward[32, 16] < 0.95)  # a gradient outside
        self.assertLess(outward[32, 10], TOL)
        inward = run("paint_bevel_smooth", {"input": m}, {**kw, "direction": "Inward"})["output"][..., 0]
        self.assertLess(inward[32, 16], TOL)
        self.assertTrue(0.05 < inward[32, 20] < 0.95)  # a gradient inside
        self.assertGreater(inward[32, 32], 1 - TOL)

    def test_directional_distance(self):
        """A gradient travelling along Angle (0 = to the right) over Distance (1 = the shorter side)."""
        dot = disc(r=0.06, cx=0.25)
        out = run("paint_directional_distance", {"input": dot}, {"angle": 0.0, "distance": 0.5})["output"][..., 0]
        trail = out[32, 20:52]  # from the dot's right edge
        self.assertGreater(trail[0], 0.9)
        self.assertTrue(np.all(np.diff(trail) <= TOL))  # falling along the direction
        self.assertGreater(out[32, 30], 0.6)
        self.assertLess(out[32, 8], TOL)  # nothing behind
        self.assertLess(out[32, 50], 0.2)

    def test_mask_outline(self):
        m = square(lo=0.3, hi=0.7)
        kw = {"width": 16.0, "blur": 0.0}
        out = run("paint_mask_outline", {"input": m}, {**kw, "mask_position": "Outside"})["output"][..., 0]
        self.assertGreater(out[32, 18], 0.5)  # just outside the edge
        self.assertLess(out[32, 32], TOL)     # inside
        self.assertLess(out[32, 2], TOL)      # beyond Width
        inside = run("paint_mask_outline", {"input": m}, {**kw, "mask_position": "Inside"})["output"][..., 0]
        self.assertGreater(inside[32, 21], 0.5)
        self.assertLess(inside[32, 16], TOL)
        black = run("paint_mask_outline", {"input": m}, {**kw, "start_from_black": True})["output"][..., 0]
        np.testing.assert_allclose(black, 1.0 - out, atol=TOL)


class Normals(unittest.TestCase):
    def test_height_to_normal(self):
        flat = np.full((64, 64), 0.5, np.float32)
        v = np.array([0.2, 0.0, 0.8]) / np.linalg.norm([0.2, 0.0, 0.8])
        base = rgba(v * 0.5 + 0.5, np.ones((64, 64), np.float32))
        n = run("paint_height_to_normal", {"height": flat, "normal": base},
                {"overwrite_existing_normal": True})["normal"]
        np.testing.assert_allclose(n[10, 10, :3], [0.5, 0.5, 1.0], atol=TOL)  # flat height, flat normal
        kept = run("paint_height_to_normal", {"height": flat, "normal": base})["normal"]
        np.testing.assert_allclose(kept[10, 10, :3], base[10, 10, :3], atol=0.01)  # laid on the existing normal
        ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :], 64, axis=0)  # rising to the right
        r = run("paint_height_to_normal", {"height": ramp, "normal": base}, {"overwrite_existing_normal": True})
        self.assertLess(r["normal"][32, 32, 0], 0.49)  # facing left: -X (gently: 5 cm over 100 cm)
        world = run("paint_height_to_normal", {"height": ramp, "normal": base},
                    {"overwrite_existing_normal": True, "use_world_units": True, "height_depth_cm": 50.0})
        self.assertLess(world["normal"][32, 32, 0], r["normal"][32, 32, 0])  # deeper relief, steeper

    def test_tri_planar_identity(self):
        """Seen along Z with a position map that is the UV square, the Z projection gives the input back."""
        n = 64
        y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
        pos = rgba(np.stack([x, 1.0 - y, np.full_like(x, 0.5)], -1), np.ones((n, n), np.float32))
        nrm = rgba((0.5, 0.5, 1.0), np.ones((n, n), np.float32))
        smooth = 0.5 + 0.4 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)  # 8 bit positions are not texel exact
        img = rgba(np.stack([smooth] * 3, -1), np.ones((n, n), np.float32))
        out = run("paint_tri_planar_advanced_filter", {"input": img, "world_space_normals": nrm, "position": pos})
        self.assertLess(np.abs(out["output"][..., 0] - img[..., 0]).max(), 2 * TOL)
        only_x = run("paint_tri_planar_advanced_filter", {"input": img, "world_space_normals": nrm, "position": pos},
                     {"projection": "X"})["output"][..., 0]
        self.assertLess(np.abs(only_x - only_x[:, :1]).max(), 2 * TOL)  # the X projection reads (z, y): constant rows


class FillArea(unittest.TestCase):
    def ring(self, gap=False):
        y, x = (np.mgrid[0:64, 0:64] + 0.5) / 64
        d = np.hypot(x - 0.5, y - 0.5)
        r = ((d > 0.25) & (d < 0.3)).astype(np.float32)
        if gap:
            r[30:34, 40:] = 0.0
        return r

    def test_mask(self):
        out = run("paint_fill_area_mask", {"input": self.ring()})["output"][..., 0]
        self.assertGreater(out[32, 32], 1 - TOL)  # closed: filled
        self.assertLess(out[3, 3], TOL)           # outside: as it was
        open_ = run("paint_fill_area_mask", {"input": self.ring(gap=True)})["output"][..., 0]
        self.assertLess(open_[32, 32], TOL)       # not closed: not filled
        dbg = run("paint_fill_area_mask", {"input": self.ring()}, {"debug_area_detection": True})["output"][..., 0]
        self.assertAlmostEqual(dbg[32, 32], 0.35, delta=0.01)

    def test_color(self):
        img = rgba((1.0, 0.0, 0.0), self.ring())
        out = run("paint_fill_area_color", {"input": img}, {"fill_color": [0.0, 1.0, 0.0]})["output"]
        np.testing.assert_allclose(out[32, 32], [0.0, 1.0, 0.0, 1.0], atol=TOL)
        self.assertLess(out[3, 3, 3], TOL)
        for mode in ("Blur", "Diffusion"):
            with self.subTest(mode=mode):
                o = run("paint_fill_area_color", {"input": img}, {"color_mode": mode, "blur_intensity": 24.0})["output"]
                np.testing.assert_allclose(o[32, 32], [1.0, 0.0, 0.0, 1.0], atol=2 * TOL)  # the strokes' color


if __name__ == "__main__":
    unittest.main()
