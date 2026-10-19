"""The Paint generators of nodes/paint/generators/ on synthetic mesh maps of a unit cube (six square UV islands,
one per face, a dent in the top face): facts from the documentation's descriptions."""

import hashlib
import unittest

import numpy as np

from trafilatura3d.evaluator import Evaluator, Job
from trafilatura3d.nodes import NODES
from trafilatura3d.nodes.base import Input, Node, Output, PortType

EV = Evaluator()
N = 192  # map size

COPY = """
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    store(c, load_src(c));
}"""


class _Array(Node):
    """An uploaded numpy image (job.extra['array'], HxWx4 float32)."""
    type = "gen_test_array"
    label = "Array"
    registered = False
    inputs = {"src": Input(PortType.COLOR)}
    outputs = {"output": Output(PortType.COLOR)}

    def encode(self, ctx, job):
        tex = ctx.ev.upload(job.key, lambda: job.extra["array"])
        ctx.dispatch(self, job, {}, srcs={"src": (tex, False)})

    def wgsl(self):
        return COPY


ARRAY = _Array()


def source(name: str, a: np.ndarray, gray: bool) -> Job:
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a, np.ones_like(a)], axis=2)
    elif a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], axis=2)
    key = "gen_test:" + name + ":" + hashlib.sha1(a.tobytes()).hexdigest()
    return Job(key, name, ARRAY, {}, {}, {"output": PortType.GRAY if gray else PortType.COLOR}, (a.shape[1], a.shape[0]),
               {"output": "32f"}, 0, 0.0, {"array": a})


def generator(node_type: str, maps: dict, params: dict | None = None, seed: float = 0.0, size: int = N) -> np.ndarray:
    """The node's output (HxWxC float) with the maps on the inputs of the same name (others unconnected)."""
    node = NODES[node_type]
    values = {k: p.default for k, p in node.params.items()}
    for k, v in (params or {}).items():
        values[k] = node.params[k].validate(v)
    inputs = {port: (source(port, maps[port], spec.type is PortType.GRAY), "output")
              for port, spec in node.inputs.items() if port in maps}
    key = f"gen:{node_type}:{values}:{sorted((p, j.key) for p, (j, _) in inputs.items())}:{seed}:{size}"
    types = {o: (PortType.GRAY if out.type is PortType.GRAY else PortType.COLOR) for o, out in node.outputs.items()}
    job = Job(key, node_type, node, values, inputs, types, (size, size), {o: "32f" for o in node.outputs}, 0, seed)
    return EV.read(job, "output").astype(np.float64)


# --- a unit cube unwrapped: faces +X -X +Y / -Y +Z -Z on a 3x2 grid of cells --------------------------------------

FACES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
MARGIN = 0.03
DENSITY = (-1.0, -0.5, 0.0, 0.5, 1.0, 0.0)


def face_rect(k: int, n: int = N):
    """Pixel rectangle (x0, y0, x1, y1) of face k's island."""
    i, j = k % 3, k // 3
    s = 1.0 / 3.0 - 2 * MARGIN
    x0, y0 = i / 3.0 + MARGIN, j / 2.0 + MARGIN
    return int(round(x0 * n)), int(round(y0 * n)), int(round((x0 + s) * n)), int(round((y0 + s) * n))


def cube_maps(n: int = N) -> dict:
    """Mesh maps of a unit cube: position (0-1 box), world space normals, curvature (convex edges, a concave ring
    around a dent in the top face), ambient occlusion (dark in the dent), thickness, uv islands, texel density."""
    pos = np.zeros((n, n, 3))
    nrm = np.full((n, n, 3), 0.5)
    curv = np.full((n, n), 0.5)
    ao = np.ones((n, n))
    thick = np.full((n, n), 0.5)
    isl = np.zeros((n, n, 4))
    dens = np.zeros((n, n))
    rng = np.random.default_rng(3)
    for k, face in enumerate(FACES):
        x0, y0, x1, y1 = face_rect(k, n)
        a = ((np.arange(x0, x1) + 0.5 - x0) / (x1 - x0))[None, :].repeat(y1 - y0, 0)
        b = ((np.arange(y0, y1) + 0.5 - y0) / (y1 - y0))[:, None].repeat(x1 - x0, 1)
        one, zero = np.ones_like(a), np.zeros_like(a)
        p = {"+X": (one, 1 - b, a), "-X": (zero, 1 - b, 1 - a), "+Y": (a, one, b), "-Y": (a, zero, b),
             "+Z": (a, 1 - b, one), "-Z": (1 - a, 1 - b, zero)}[face]
        pos[y0:y1, x0:x1] = np.stack(p, axis=2)
        v = np.zeros(3)
        v["XYZ".index(face[1])] = 1.0 if face[0] == "+" else -1.0
        nrm[y0:y1, x0:x1] = v * 0.5 + 0.5
        edge = np.minimum(np.minimum(a, 1 - a), np.minimum(b, 1 - b))
        c = 0.5 + 0.45 * np.exp(-edge / 0.04)
        if face == "+Y":
            r = np.hypot(a - 0.5, b - 0.5)
            c -= 0.4 * np.exp(-((r - 0.2) / 0.03) ** 2)
            ao[y0:y1, x0:x1] = 1 - 0.7 * (1 - np.clip(r / 0.2, 0, 1)) ** 2
        curv[y0:y1, x0:x1] = c
        thick[y0:y1, x0:x1] = 0.3 + 0.4 * b
        isl[y0:y1, x0:x1] = [*rng.random(2), 0.0, 1.0]
        dens[y0:y1, x0:x1] = DENSITY[k]
    return {"position": pos, "world_space_normals": nrm, "curvature": curv, "baked_ambient_occlusion": ao,
            "thickness": thick, "uv_islands": isl, "texel_density": dens}


MAPS = cube_maps()


def face_values(img: np.ndarray, k: int, inset: int = 4) -> np.ndarray:
    x0, y0, x1, y1 = face_rect(k, img.shape[0])
    return img[y0 + inset:y1 - inset, x0 + inset:x1 - inset]


def gray(img):
    return img[..., 0]


class TestUVGenerators(unittest.TestCase):
    def test_uv_random_color_one_flat_color_per_island(self):
        img = generator("paint_uv_random_color", MAPS)
        colors = []
        for k in range(6):
            f = face_values(img, k, 0)[..., :3]
            self.assertLess(np.ptp(f.reshape(-1, 3), axis=0).max(), 1e-5, "one flat color per island")
            colors.append(f[0, 0])
        for i in range(6):
            for j in range(i + 1, 6):
                self.assertGreater(np.abs(colors[i] - colors[j]).max(), 0.02, "islands get different colors")
        self.assertTrue(np.all(img[0, 0, :3] == 0), "black outside the UVs")
        other = generator("paint_uv_random_color", MAPS, seed=5.0)
        self.assertGreater(np.abs(face_values(other, 0)[0, 0, :3] - colors[0]).max(), 1e-3, "the seed changes colors")

    def test_uv_random_color_custom_gradient(self):
        ramp = np.zeros((4, 64, 3))
        ramp[..., 0] = np.linspace(0, 1, 64)  # black to red
        img = generator("paint_uv_random_color", {**MAPS, "custom_gradient": ramp},
                        {"color_source_mode": "Custom Gradient"})
        f = face_values(img, 2)[0, 0]
        self.assertLess(f[1] + f[2], 1e-4, "colors picked along the gradient")

    def test_uv_border_distance(self):
        img = gray(generator("paint_uv_border_distance", MAPS, {"distance": 16.0}))
        x0, y0, x1, y1 = face_rect(2)
        row = img[(y0 + y1) // 2, x0:x1]
        self.assertGreater(row[0], 0.9, "white on the border")
        self.assertTrue(np.all(np.diff(row[: len(row) // 2]) <= 1e-6), "fades inwards")
        self.assertLess(row[len(row) // 2], 1e-3, "black beyond Distance")
        inv = gray(generator("paint_uv_border_distance", MAPS, {"distance": 16.0, "invert": True}))
        np.testing.assert_allclose(inv[(y0 + y1) // 2, x0:x1], 1 - row, atol=1e-5)

    def test_uv_checker(self):
        img = generator("paint_uv_checker", MAPS, {"checker_grid_opacity": 0.0, "uv_borders_width": 0.0})
        f = face_values(img, 4)[..., :3].reshape(-1, 3)
        shades = np.unique(np.round(f, 4), axis=0)
        self.assertEqual(len(shades), 2, "a checker of two shades")
        borders = generator("paint_uv_checker", MAPS, {"uv_borders_color": [0.0, 1.0, 0.0], "uv_borders_width": 2.0})
        x0, y0, x1, y1 = face_rect(4)
        np.testing.assert_allclose(borders[(y0 + y1) // 2, x0, :3], [0, 1, 0], atol=1e-5)
        rnd = generator("paint_uv_checker", MAPS, {"checker_color_mode": "Random", "checker_grid_opacity": 0.0,
                                                     "uv_borders_width": 0.0})
        self.assertGreater(np.abs(face_values(rnd, 0).max((0, 1)) - face_values(rnd, 1).max((0, 1))).max(), 0.02,
                           "Random: a color per island")

    def test_uv_texel_density(self):
        img = generator("paint_uv_texel_density", MAPS)
        lo, mid, hi = NODES["paint_uv_texel_density"].params["color_low"].default, \
            NODES["paint_uv_texel_density"].params["color_medium"].default, \
            NODES["paint_uv_texel_density"].params["color_high"].default
        np.testing.assert_allclose(face_values(img, 0)[0, 0, :3], lo, atol=1e-5)  # half the mean density
        np.testing.assert_allclose(face_values(img, 2)[0, 0, :3], mid, atol=1e-5)
        np.testing.assert_allclose(face_values(img, 4)[0, 0, :3], hi, atol=1e-5)  # twice the mean


class TestMaskGenerators(unittest.TestCase):
    def test_position_top_to_bottom(self):
        img = gray(generator("paint_position", MAPS))
        self.assertGreater(face_values(img, 2).mean(), 0.99, "Top To Bottom: 1 at the top")
        self.assertLess(face_values(img, 3).mean(), 0.01, "0 at the bottom")
        side = face_values(img, 4)
        self.assertGreater(side[0].mean(), side[-1].mean(), "a gradient down the sides")
        inv = gray(generator("paint_position", MAPS, {"global_invert": True}))
        np.testing.assert_allclose(inv, 1 - img, atol=1e-5)

    def test_position_right_to_left(self):
        img = gray(generator("paint_position", MAPS, {"top_to_bottom": 0.0, "right_to_left": 1.0}))
        self.assertGreater(face_values(img, 4).mean(), 0.99)  # +Z = right
        self.assertLess(face_values(img, 5).mean(), 0.01)

    def test_world_space_normals_facing(self):
        img = gray(generator("paint_world_space_normals", MAPS))
        self.assertGreater(face_values(img, 2).mean(), 0.99, "faces looking up are white")
        self.assertLess(face_values(img, 3).mean(), 0.01, "faces looking down are black")
        self.assertAlmostEqual(face_values(img, 0).mean(), 0.5, delta=0.01)

    def test_ambient_occlusion(self):
        img = gray(generator("paint_ambient_occlusion", MAPS))
        np.testing.assert_allclose(img, MAPS["baked_ambient_occlusion"], atol=1e-4,
                                   err_msg="occluded dark, unoccluded light: the baked map with the defaults")
        blurred = gray(generator("paint_ambient_occlusion", MAPS, {"blur": 4.0}))
        self.assertLess(np.abs(np.diff(blurred, axis=1)).max(), np.abs(np.diff(img, axis=1)).max())

    def test_ambient_occlusion_texture(self):
        tex = np.zeros((32, 32))
        tex[:, :16] = 1.0
        img = gray(generator("paint_ambient_occlusion", {**MAPS, "texture": tex}, {"use_texture": True}))
        self.assertLess(img.min(), 0.01, "the texture multiplied in")
        off = gray(generator("paint_ambient_occlusion", {**MAPS, "texture": tex}))
        np.testing.assert_allclose(off, MAPS["baked_ambient_occlusion"], atol=1e-4)

    def test_curvature_modes(self):
        un = gray(generator("paint_curvature", MAPS, {"mode": "Unprocessed"}))
        np.testing.assert_allclose(un, MAPS["curvature"], atol=2e-3, err_msg="all bands at 1: the map")
        edges = gray(generator("paint_curvature", MAPS))
        top = face_values(edges, 0, 0)
        self.assertGreater(top[:, 0].mean(), 0.6, "Edges: the convex edges white")
        self.assertLess(top[top.shape[0] // 2, top.shape[1] // 2], 0.05, "flat areas black")
        cav = gray(generator("paint_curvature", MAPS, {"mode": "Cavities"}))
        self.assertGreater(face_values(cav, 2, 0).max(), 0.5, "Cavities: the dent's ring")
        self.assertLess(face_values(cav, 0, 0).max(), 0.01)
        soft = gray(generator("paint_curvature", MAPS, {"mode": "Unprocessed", "sharp": 0.0, "fine": 0.0}))
        self.assertLess(np.abs(np.diff(soft, axis=1)).max(), np.abs(np.diff(un, axis=1)).max(),
                        "without the sharp and fine bands the map is softer")

    def test_mask_editor(self):
        img = gray(generator("paint_mask_editor", MAPS))
        edges = gray(generator("paint_curvature", MAPS))
        np.testing.assert_allclose(img, edges, atol=1e-4, err_msg="default: the curvature edges")
        ao = gray(generator("paint_mask_editor", MAPS, {"curvature_opacity": 0.0, "ambient_occlusion_opacity": 1.0}))
        np.testing.assert_allclose(ao, MAPS["baked_ambient_occlusion"], atol=1e-4)
        both = gray(generator("paint_mask_editor", MAPS, {"ambient_occlusion_opacity": 1.0,
                                                            "ao_blending_mode": "Copy",
                                                            "curvature_blending_mode": "Multiply"}))
        np.testing.assert_allclose(both, MAPS["baked_ambient_occlusion"] * edges, atol=1e-4,
                                   err_msg="layers blended bottom up")
        thick = gray(generator("paint_mask_editor", MAPS, {"curvature_opacity": 0.0, "thickness_opacity": 1.0}))
        np.testing.assert_allclose(face_values(thick, 1), face_values(MAPS["thickness"], 1), atol=1e-4)

    def test_mask_builder(self):
        img = gray(generator("paint_mask_builder", MAPS, {"grunge": 0.0}))
        f = face_values(img, 0, 0)
        self.assertGreater(f[:, 0].mean(), 0.8, "curvature: the edges")
        self.assertLess(f[f.shape[0] // 2, f.shape[1] // 2], 0.05)
        grad = gray(generator("paint_mask_builder", MAPS, {"grunge": 0.0, "curvature": 0.0, "top_down_gradient": 1.0}))
        self.assertGreater(face_values(grad, 2).mean(), 0.99, "top/down gradient: white at the top")
        self.assertLess(face_values(grad, 3).mean(), 0.01)
        sc = gray(generator("paint_mask_builder", MAPS, {"grunge": 0.0, "curvature": 0.0, "scatter": 1.0}))
        self.assertGreater(sc.max(), 0.5, "scattered stamps")
        self.assertGreater((sc < 0.01).mean(), 0.2)

    def test_fiber_glass_edge_wear(self):
        img = gray(generator("paint_fiber_glass_edge_wear", MAPS))
        f = face_values(img, 0, 0)
        self.assertGreater(f[:, :3].mean(), f[20:-20, 20:-20].mean() + 0.3, "wear on the edges")

    def test_dripping_rust_runs_down(self):
        dent = np.full((N, N), 0.5)
        x0, y0, x1, y1 = face_rect(4)  # a side face: a cavity near its top
        cy, cx = y0 + (y1 - y0) // 5, (x0 + x1) // 2
        yy, xx = np.mgrid[0:N, 0:N]
        dent -= 0.45 * (np.hypot(yy - cy, xx - cx) < 5)
        img = gray(generator("paint_dripping_rust", {**MAPS, "curvature": dent}, {"drips_intensity": 0.8}))
        below = img[cy + 10:y1 - 2, cx - 6:cx + 7].max()
        above = img[y0 + 1:cy - 8, cx - 6:cx + 7].max()
        self.assertGreater(img[cy, cx], 0.5, "rust in the cavity")
        self.assertGreater(below, 0.2, "drips under it")
        self.assertLess(above, 0.05, "nothing above it")

    def test_tri_planar_advanced(self):
        stripes = np.zeros((64, 64))
        stripes[:, :32] = 1.0
        img = gray(generator("paint_tri_planar_advanced", {**MAPS, "mask": stripes}))
        top = face_values(img, 2)  # +Y face: projected on XZ, stripes along X
        self.assertGreater(np.ptp(top), 0.9)
        none = gray(generator("paint_tri_planar_advanced", MAPS))
        self.assertEqual(none.max(), 0.0, "no mask: black")
        x_only = gray(generator("paint_tri_planar_advanced", {**MAPS, "mask": stripes}, {"projection": "X Axis"}))
        side = face_values(x_only, 0)  # +X face sees the X projection (Z, Y): stripes along its width
        self.assertGreater(np.ptp(side), 0.9)

    def test_auto_stitcher_uv_mask(self):
        img = gray(generator("paint_auto_stitcher", MAPS, {"path_smoothness": 0.0}))
        x0, y0, x1, y1 = face_rect(4)
        pp = int(round(4.0 * N / 256))  # Path Position, px at 256
        band = img[y0 + pp - 1:y0 + pp + 2, x0 + 8:x1 - 8].max(axis=0)  # along the top border's path
        on = band > 0.3
        self.assertGreaterEqual(int(np.sum(on[1:] & ~on[:-1])), 4, "a row of separate stitches along the path")
        self.assertLess(face_values(img, 4, 14).max(), 0.01, "nothing away from the path")
        self.assertEqual(img[0, 0], 0.0, "nothing outside the UVs")

    def test_auto_stitcher_custom_input(self):
        half = np.zeros((N, N))
        half[:, N // 2:] = 1.0
        img = gray(generator("paint_auto_stitcher", {**MAPS, "custom_input": half},
                             {"mask_mode": "Custom Input", "path_smoothness": 0.0}))
        cols = np.where(img.max(axis=0) > 0.3)[0]
        self.assertTrue(len(cols) > 0)
        self.assertFalse(np.any((cols >= N // 2 - 1) & (cols <= N // 2)), "the paths never cross the border")
        self.assertTrue(np.any(cols < N // 2) and np.any(cols > N // 2), "paths on both sides")

    def test_inflate_shrinkwrap(self):
        img = gray(generator("paint_inflate_shrinkwrap", MAPS, {"wrinkle_scale": 0.0, "wrinkle_range": 0.0}))
        x0, y0, x1, y1 = face_rect(0)
        row = img[(y0 + y1) // 2, x0:x1]
        self.assertGreater(row[len(row) // 2], row[1] + 0.3, "inflated: the middle higher than the seam")
        wr = gray(generator("paint_inflate_shrinkwrap", MAPS))
        along = (slice(y0 + 3, y0 + 4), slice(x0 + 10, x1 - 10))  # a line parallel to the seam
        self.assertGreater(np.ptp(wr[along]), np.ptp(img[along]) + 0.05, "wrinkles across the seams")


if __name__ == "__main__":
    unittest.main()
