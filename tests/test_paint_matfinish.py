"""The Metal Finish and Baked Lighting filters (nodes/paint/matfinish) behave as their documentation says:
finishes that look like their name (brushed streaks along a direction, perforated holes sized by Holes Size,
spangles...), tile, follow their intensity and Seed; lighting baked into the color."""

import unittest

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES

EV = Evaluator()
F32 = {"mode": "absolute", "value": "32f"}
FINISHES = [t for t, d in NODES.items() if d.category == "Paint/Filters/Finishes"]
LIGHTING = [t for t, d in NODES.items() if d.category == "Paint/Filters/Lighting"]


class Graph:
    def __init__(self, log2=8):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", log2)
        self.log2 = log2

    def node(self, type, inputs=None, seed=None, **params):
        n = self.g.add_node(type)
        self.g.set_base(n, "output_format", F32)
        self.g.set_base(n, "output_size", {"mode": "absolute", "value": [self.log2, self.log2]})
        if seed is not None:
            self.g.set_base(n, "seed", {"mode": "absolute", "value": seed})
        for k, v in params.items():
            self.g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            self.g.connect(src, "output", n, port)
        return n

    def color(self, rgb):
        return self.node("uniform_color", output_color=list(rgb) + [1.0])

    def gray(self, v):
        return self.node("uniform_color", color_mode="Grayscale", output_color=[v, v, v, 1.0])

    def read(self, n, out):
        a = EV.read(compile_graph(self.pkg, "t").jobs[n], out).astype(np.float64)
        return a[:, :, :3] if a.shape[2] >= 3 else np.repeat(a, 3, axis=2)


def finish(type, log2=8, seed=None, **params):
    """A finish on a uniform grey metal: {output: image}."""
    g = Graph(log2)
    ins = {p: (g.color([0.6, 0.6, 0.6]) if NODES[type].inputs[p].type.value == "color" and p != "normal"
               else g.gray(0.4)) for p in ("basecolor", "roughness", "metallic") if p in NODES[type].inputs}
    n = g.node(type, ins, seed, **params)
    return {o: g.read(n, o) for o in NODES[type].outputs}


def slopes(normal):
    """Mean |x| and |y| of a tangent space normal map."""
    v = normal * 2.0 - 1.0
    return np.abs(v[:, :, 0]).mean(), np.abs(v[:, :, 1]).mean()


class Finishes(unittest.TestCase):
    def test_all_eleven_nodes(self):
        self.assertEqual(len(FINISHES), 9)
        self.assertEqual(len(LIGHTING), 2)

    def test_finishes_add_relief_and_tile(self):
        """Every finish puts its pattern in the normal channel, and the pattern tiles (UV mapping)."""
        for t in FINISHES:
            with self.subTest(node=t):
                n = finish(t)["normal"]
                inner_x = np.abs(np.diff(n, axis=1)).mean() + 1e-4
                inner_y = np.abs(np.diff(n, axis=0)).mean() + 1e-4
                self.assertGreater(np.abs(n[:, :, :2] - 0.5).mean(), 0.003, "no relief")
                self.assertLess(np.abs(n[:, 0] - n[:, -1]).mean(), inner_x * 3, "seam on the vertical edge")
                self.assertLess(np.abs(n[0, :] - n[-1, :]).mean(), inner_y * 3, "seam on the horizontal edge")

    def test_seed_changes_the_pattern(self):
        for t in FINISHES:
            if "Seed" not in NODES[t].unsupported:
                continue  # Perforated Circles: a regular grid, no Seed in the documentation
            with self.subTest(node=t):
                a, b = finish(t, seed=1)["normal"], finish(t, seed=2)["normal"]
                self.assertGreater(np.abs(a - b).mean(), 1e-3)

    def test_zero_intensity_leaves_the_material(self):
        for t, p in (("paint_metal_finish_brushed_linear", "brushing_intensity"),
                     ("paint_metal_finish_grainy", "grain_intensity"),
                     ("paint_metal_finish_grinded", "grinding_intensity"),
                     ("paint_metal_finish_hammered", "details_intensity"),
                     ("paint_metal_finish_raw", "details_intensity")):
            with self.subTest(node=t):
                r = finish(t, 6, **{p: 0.0})
                np.testing.assert_allclose(r["normal"], np.broadcast_to([0.5, 0.5, 1.0], r["normal"].shape), atol=2e-3)
                np.testing.assert_allclose(r["basecolor"], 0.6, atol=1e-4)
                np.testing.assert_allclose(r["roughness"], 0.4, atol=1e-4)

    def test_brushed_streaks_run_along_rotation(self):
        """Brushed Linear: the relief changes across the brushing direction, hardly along it; Rotation turns it."""
        x0, y0 = slopes(finish("paint_metal_finish_brushed_linear", rotation=0.0)["normal"])
        self.assertGreater(y0, x0 * 4)  # streaks along U: slopes across, in V
        x9, y9 = slopes(finish("paint_metal_finish_brushed_linear", rotation=90.0)["normal"])
        self.assertGreater(x9, y9 * 4)

    def test_brushed_length(self):
        """Longer streaks: the relief varies less along them."""
        def along(length):
            n = finish("paint_metal_finish_brushed_linear", length=length)["normal"][:, :, 1]
            return np.abs(np.diff(n, axis=1)).mean()
        self.assertLess(along(1.0), along(0.2))

    def test_perforated_holes_follow_holes_size(self):
        """Holes cut in the opacity (and dark in base color); bigger Holes Size, more open area; 0 = no holes."""
        def open_area(size):
            r = finish("paint_metal_finish_perforated_circles", holes_size=size)
            return (r["opacity"][:, :, 0] < 0.5).mean(), r
        a0, _ = open_area(0.0)
        a3, _ = open_area(0.3)
        a7, r = open_area(0.7)
        self.assertEqual(a0, 0.0)
        self.assertLess(a3, a7)
        # Circles of diameter Holes Size x spacing on a near hexagonal grid: area ~ 0.785 * size^2 / 0.866.
        self.assertAlmostEqual(a7, 0.785 * 0.49 / 0.866, delta=0.06)
        holes = r["opacity"][:, :, 0] < 0.5
        self.assertLess(r["basecolor"][holes].mean(), 0.1)

    def test_galvanized_spangles(self):
        """Galvanized: crystals of uniform color, differing from one another (Flakes Intensity)."""
        bc = finish("paint_metal_finish_galvanized", flakes_intensity=1.0)["basecolor"][:, :, 0]
        self.assertGreater(bc.std(), 0.03)
        self.assertLess(np.median(np.abs(np.diff(bc, axis=1))), 1e-3)  # flat inside a crystal
        flat = finish("paint_metal_finish_galvanized", flakes_intensity=0.0)["basecolor"][:, :, 0]
        self.assertLess(flat.std(), 1e-4)

    def test_hammered_dents_follow_intensity(self):
        weak = sum(slopes(finish("paint_metal_finish_hammered", details_intensity=0.2)["normal"]))
        strong = sum(slopes(finish("paint_metal_finish_hammered", details_intensity=1.0)["normal"]))
        self.assertGreater(strong, weak * 2)

    def test_powder_flakes_are_metallic(self):
        r = finish("paint_metal_finish_powder_coated", flakes_intensity=1.0)
        m = r["metallic"][:, :, 0]
        self.assertGreater(m.max(), 0.9)
        self.assertLess(np.median(m), 0.45)  # most of the coat keeps the input (0.4)

    def test_custom_flake_and_grunge_inputs(self):
        """Use Custom Flake / Use Custom Grunge take the connected texture instead of the procedural one."""
        for t, port, toggle in (("paint_metal_finish_galvanized", "custom_flake", "use_custom_flake"),
                                ("paint_metal_finish_powder_coated", "custom_flake", "use_custom_flake"),
                                ("paint_metal_finish_rough", "custom_flake", "use_custom_flake"),
                                ("paint_metal_finish_rough", "custom_grunge", "use_custom_grunge")):
            with self.subTest(node=t, port=port):
                outs = []
                for on in (False, True):
                    g = Graph(7)
                    n = g.node(t, {port: g.node("test_pattern")}, **{toggle: on})
                    outs.append(g.read(n, "roughness"))
                self.assertGreater(np.abs(outs[0] - outs[1]).mean(), 1e-3)

    def test_triplanar_mapping_uses_the_mesh_maps(self):
        for t in FINISHES:
            if "triplanar_mapping" not in NODES[t].params:
                continue
            with self.subTest(node=t):
                outs = []
                for on in (False, True):
                    g = Graph(7)
                    n = g.node(t, {"position": g.node("test_pattern_color"),
                                   "world_space_normals": g.color([0.5, 1.0, 0.5])}, triplanar_mapping=on)
                    outs.append(g.read(n, "normal"))
                self.assertGreater(np.abs(outs[0] - outs[1]).mean(), 1e-3)


def srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Lighting(unittest.TestCase):
    def stylized(self, normal=(0.5, 1.0, 0.5), ao=1.0, **params):
        g = Graph(5)
        n = g.node("paint_baked_lighting_stylized", {"basecolor": g.color([0.8, 0.3, 0.2]),
                                                       "world_space_normals": g.color(list(normal)),
                                                       "baked_ambient_occlusion": g.gray(ao)}, **params)
        return g.read(n, "basecolor")[16, 16]

    def test_stylized_lit_from_the_sun(self):
        """The baked color is brighter where the surface faces the sun than where it turns away."""
        sun = dict(sun_horizontal_angle=0.0, sun_vertical_angle=0.0, sky_intensity=0.0)
        facing = self.stylized((0.5, 0.5, 1.0), **sun)  # normal +Z, the sun at the horizon along +Z
        away = self.stylized((0.5, 0.5, 0.0), **sun)
        self.assertGreater(facing.sum(), away.sum() + 0.5)

    def test_stylized_ambient_occlusion(self):
        self.assertLess(self.stylized(ao=0.3).sum(), self.stylized(ao=1.0).sum() - 0.1)
        np.testing.assert_allclose(self.stylized(ao=0.3, diffuse_ao=0.0, specular_ao=0.0), self.stylized(ao=1.0),
                                   atol=1e-5)

    def test_stylized_lighting_only_ignores_the_color(self):
        c = self.stylized(output="Lighting Only")
        self.assertLess(np.ptp(c), 0.1)  # grey-ish lighting, not the red base color
        self.assertGreater(np.ptp(self.stylized()), 0.3)

    def environment(self, env=None, **params):
        g = Graph(5)
        ins = {"basecolor": g.color([0.5, 0.5, 0.5]), "metallic": g.gray(0.0), "roughness": g.gray(0.5),
               "world_space_normals": g.color([0.5, 1.0, 0.5])}
        if env is not None:
            ins["environment"] = env(g)
        n = g.node("paint_baked_lighting_environment", ins, **params)
        return g.read(n, "basecolor")[16, 16]

    def test_environment_uniform_white(self):
        """A uniform white environment: diffuse = the albedo, reflection = F0 (0.04)."""
        want = srgb(lin(0.5) + 0.04)
        np.testing.assert_allclose(self.environment(), want, atol=0.01)

    def test_environment_exposure_and_rotation(self):
        self.assertGreater(self.environment(exposure=1.0).sum(), self.environment().sum() + 0.1)
        env = lambda g: g.node("test_pattern_color")  # noqa: E731
        a = self.environment(env)
        b = self.environment(env, horizontal_rotation=180.0, vertical_rotation=60.0)
        self.assertGreater(np.abs(a - b).sum(), 0.01)


if __name__ == "__main__":
    unittest.main()
