"""Material filters, mesh based generators and HDRI tools behave as documented."""

import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES

EV = Evaluator()
F32 = {"mode": "absolute", "value": "32f"}
CATALOG = json.loads((Path(__file__).parent.parent / "reference" / "graph" / "material_mesh_hdri.json").read_text(encoding="utf-8"))["nodes"]


class Graph:
    """A small graph read at 32 px (or `size` = (log2 w, log2 h)), every node in 32-bit float."""

    def __init__(self, size=(5, 5)):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", size[0])
        self.size = size

    def node(self, type, inputs=None, **params):
        n = self.g.add_node(type)
        self.g.set_base(n, "output_format", F32)
        self.g.set_base(n, "output_size", {"mode": "absolute", "value": list(self.size)})
        for k, v in params.items():
            self.g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            src, out = src if isinstance(src, tuple) else (src, "output")
            self.g.connect(src, out, n, port)
        return n

    def color(self, rgb):
        return self.node("uniform_color", output_color=list(rgb) + [1.0])

    def gray(self, v):
        return self.node("uniform_color", color_mode="Grayscale", output_color=[v, v, v, 1.0])

    def read(self, n, out="output"):
        a = EV.read(compile_graph(self.pkg, "t").jobs[n], out)
        return a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)


def mid(a):
    return a[a.shape[0] // 2, a.shape[1] // 2]


class Conformance(unittest.TestCase):
    def test_repeated_labels_all_present(self):
        """Labels repeated in a node's catalog entry (one per group) exist as many times."""
        nodes = {d.catalog: d for d in NODES.values() if d.catalog}
        for entry in CATALOG:
            d = nodes[entry["id"]]
            want = Counter(p["name"] for p in entry["parameters"] if p["name"] not in d.unsupported)
            have = Counter(p.label for p in d.params.values())
            with self.subTest(node=entry["id"]):
                self.assertEqual({k: v for k, v in want.items() if have[k] < v}, {})

    def test_every_catalog_node_exists(self):
        have = {d.catalog for d in NODES.values()}
        self.assertEqual([n["id"] for n in CATALOG if n["id"] not in have], [])


class Blending(unittest.TestCase):
    def blend(self, mask):
        t = Graph()
        fg, bg = t.color((1.0, 0.0, 0.0)), t.color((0.0, 0.0, 1.0))
        rf, rb = t.gray(0.9), t.gray(0.1)
        n = t.node("material_blend", {"basecolor_fg": fg, "basecolor_bg": bg, "roughness_fg": rf, "roughness_bg": rb,
                                      "greyscale_mask": t.gray(mask)})
        return mid(t.read(n, "basecolor"))[:3], mid(t.read(n, "roughness"))[0]

    def test_mask_selects_material(self):
        col, r = self.blend(0.0)
        np.testing.assert_allclose(col, [0, 0, 1], atol=1e-3)
        self.assertAlmostEqual(r, 0.1, places=3)
        col, r = self.blend(1.0)
        np.testing.assert_allclose(col, [1, 0, 0], atol=1e-3)
        self.assertAlmostEqual(r, 0.9, places=3)

    def test_material_switch(self):
        t = Graph()
        a, b = t.gray(0.2), t.gray(0.7)
        for switch, want in ((False, 0.2), (True, 0.7)):
            n = t.node("material_switch", {"height_1": a, "height_2": b}, switch_=switch)
            self.assertAlmostEqual(mid(t.read(n, "height"))[0], want, places=3)

    def test_channel_switches(self):
        """The Channels group: a blend's switched-off channel gives its default value (the ones off by
        default too: Ambient Occlusion), an effect's passes its input through."""
        t = Graph()
        ao, h = t.gray(0.3), t.gray(0.8)
        blend = t.node("material_blend", {"ambient_occlusion_fg": ao, "ambient_occlusion_bg": ao, "height_fg": h,
                                          "height_bg": h})
        self.assertAlmostEqual(mid(t.read(blend, "ambient_occlusion"))[0], 1.0, places=3)
        self.assertAlmostEqual(mid(t.read(blend, "height"))[0], 0.8, places=3)
        off = t.node("material_blend", {"height_fg": h, "height_bg": h}, channel_height=False)
        self.assertAlmostEqual(mid(t.read(off, "height"))[0], 0.5, places=3)
        rough = t.gray(0.2)
        fx = t.node("moss_weathering", {"roughness": rough}, channel_roughness=False)
        self.assertAlmostEqual(mid(t.read(fx, "roughness"))[0], 0.2, places=3)

    def test_adjustment_blend_defaults_leave_material(self):
        t = Graph()
        src = t.color((0.6, 0.3, 0.2))
        n = t.node("material_adjustment_blend", {"basecolor": src})
        np.testing.assert_allclose(mid(t.read(n, "basecolor"))[:3], [0.6, 0.3, 0.2], atol=3e-3)
        n = t.node("material_adjustment_blend", {"basecolor": src}, basecolor_luminosity=0.2)
        np.testing.assert_allclose(mid(t.read(n, "basecolor"))[:3], [0.8, 0.5, 0.4], atol=3e-3)

    def test_multi_material_blend_picks_by_id(self):
        t = Graph()
        n = t.node("multi_material_blend", {"roughness_1": t.gray(0.1), "roughness_2": t.gray(0.8),
                                            "color_id": t.color((1.0, 0.0, 0.0))}, materials="2", color_2=[1, 0, 0, 1])
        self.assertAlmostEqual(mid(t.read(n, "roughness"))[0], 0.8, places=3)

    def test_height_blend_follows_heights(self):
        t = Graph()
        n = t.node("height_blend", {"height_top": t.gray(0.9), "height_bottom": t.gray(0.2)})
        self.assertAlmostEqual(mid(t.read(n, "mask"))[0], 1.0, places=3)
        self.assertAlmostEqual(mid(t.read(n, "height"))[0], 0.9, places=3)
        n = t.node("height_blend", {"height_top": t.gray(0.1), "height_bottom": t.gray(0.8)})
        self.assertAlmostEqual(mid(t.read(n, "mask"))[0], 0.0, places=3)
        # The mask falls linearly over 1 - Contrast below the bottom; Offset 0.6 scales the bottom by 0.8 and leaves
        # the top (Balanced) or raises the top towards 1 by 0.2 (Bottom height priority).
        top, bottom = t.gray(0.5), t.gray(0.55)
        for params, mask, height in (({}, 0.5, 0.525), ({"height_offset": 0.6}, 1.0, 0.5),
                                     ({"height_offset": 0.6, "mode": "Bottom height priority"}, 1.0, 0.6),
                                     ({"contrast": 1.0}, 0.0, 0.55), ({"opacity": 0.5}, 0.25, 0.5375)):
            with self.subTest(params=params):
                n = t.node("height_blend", {"height_top": top, "height_bottom": bottom}, **params)
                self.assertAlmostEqual(mid(t.read(n, "mask"))[0], mask, places=3)
                self.assertAlmostEqual(mid(t.read(n, "height"))[0], height, places=3)
        # Offset below 0.5 scales the top down; the user mask scales the top and the bottom's lowering.
        n = t.node("height_blend", {"height_top": t.gray(0.8), "height_bottom": t.gray(0.5)}, height_offset=0.25)
        self.assertAlmostEqual(mid(t.read(n, "mask"))[0], 0.0, places=3)  # top 0.4, 1 - Contrast below the bottom
        self.assertAlmostEqual(mid(t.read(n, "height"))[0], 0.5, places=3)
        n = t.node("height_blend", {"height_top": t.gray(0.9), "height_bottom": t.gray(0.2), "mask": t.gray(0.5)},
                   height_offset=0.75)
        self.assertAlmostEqual(mid(t.read(n, "mask"))[0], 1.0, places=3)
        self.assertAlmostEqual(mid(t.read(n, "height"))[0], 0.45, places=3)  # top 0.9 x 0.5 over bottom 0.15


class MaskGenerators(unittest.TestCase):
    def test_unconnected_normal_faces_nowhere(self):
        # An unconnected World Space Normal is black (no direction): nothing faces up, so no dust settles and no light
        # falls; an upward normal brings them.
        t = Graph()
        for node in ("dust", "light"):
            with self.subTest(node=node):
                self.assertAlmostEqual(mid(t.read(t.node(node)))[0], 0.0, places=3)
                up = t.node(node, {"world_space_normal": t.color((0.5, 1.0, 0.5))})
                self.assertGreater(mid(t.read(up))[0], 0.1)
        bleach, up = t.node("sun_bleach"), t.node("sun_bleach", {"normal_world_space": t.color((0.5, 1.0, 0.5))})
        self.assertLess(mid(t.read(bleach))[0], mid(t.read(up))[0] - 0.3)

    def test_metal_edge_wear_response(self):
        # Without grunge: the edge mask is twice the convexity (full at curvature 0.75), Wear Level 0.5 keeps it,
        # 0.75 doubles it; the occlusion factor scales it at AO Masking 0.5 (measured response: 0.75 at AO 0.5).
        t = Graph()
        def wear(curv, ao, **params):
            n = t.node("metal_edge_wear", {"curvature": t.gray(curv), "ambient_occlusion": t.gray(ao)}, grunge_amount=0.0,
                       wear_contrast=0.0, **params)
            return mid(t.read(n))[0]
        self.assertAlmostEqual(wear(0.5, 1.0), 0.0, places=2)
        self.assertAlmostEqual(wear(0.8, 1.0), 1.0, places=2)
        self.assertAlmostEqual(wear(0.6, 1.0, wear_level=0.5), 0.4, places=2)
        self.assertAlmostEqual(wear(0.6, 1.0, wear_level=0.75), 0.8, places=2)
        self.assertAlmostEqual(wear(0.8, 0.5, wear_level=0.5), 0.75, delta=0.01)
        self.assertAlmostEqual(wear(0.8, 0.5, ambient_occlusion_masking=1.0), 0.0, places=2)

    def test_fiber_glass_edge_wear_occlusion_takes_away(self):
        # Without fibres: occlusion takes 1 - its factor away from the edges (Metal Edge Wear scales them instead).
        t = Graph()
        def wear(node, curv, ao):
            n = t.node(node, {"curvature": t.gray(curv), "ambient_occlusion": t.gray(ao)}, grunge_amount=0.0,
                       wear_contrast=0.0, wear_level=0.5, curvature_weight=1.0)
            return mid(t.read(n))[0]
        self.assertAlmostEqual(wear("fiber_glass_edge_wear", 0.8, 1.0), 1.0, places=2)
        # measured occlusion factor 0.75 at AO 0.5
        self.assertAlmostEqual(wear("fiber_glass_edge_wear", 0.65, 0.5), 0.35, delta=0.01)  # 0.6 + 0.75 - 1
        self.assertAlmostEqual(wear("metal_edge_wear", 0.65, 0.5), 0.45, delta=0.01)  # 0.6 x 0.75

    def test_dirt_without_curvature_covers_everything(self):
        # Unconnected curvature reads 0 (no raised edge to clean): dirt everywhere at the default level.
        t = Graph()
        self.assertGreater(mid(t.read(t.node("dirt")))[0], 0.9)


class Pbr(unittest.TestCase):
    def test_albedo_safe_color_lifts_too_dark(self):
        t = Graph()
        # Tolerance 0: the strict 50 sRGB limit (the default 0.5 allows darker).
        n = t.node("pbr_albedo_safe_color", {"basecolor": t.color((0.02, 0.02, 0.02))}, tolerance=0.0)
        v = mid(t.read(n, "basecolor"))[:3]
        self.assertGreaterEqual(v.mean(), 49 / 255)
        ok = t.node("pbr_albedo_safe_color", {"basecolor": t.color((0.5, 0.4, 0.3))})
        np.testing.assert_allclose(mid(t.read(ok, "basecolor"))[:3], [0.5, 0.4, 0.3], atol=2e-3)

    def test_validate_flags_out_of_range_albedo(self):
        t = Graph()
        bad = mid(t.read(t.node("pbr_basecolor_metallic_validate", {"basecolor": t.color((0.03, 0.03, 0.03))},
                                validation_mode="Albedo")))
        good = mid(t.read(t.node("pbr_basecolor_metallic_validate", {"basecolor": t.color((0.5, 0.5, 0.5))},
                                 validation_mode="Albedo")))
        self.assertGreater(bad[0], 0.9)
        self.assertLess(bad[1], 0.1)  # red
        self.assertGreater(good[1], 0.5)
        self.assertLess(good[0], 0.1)  # green

    def test_dielectric_f0_from_ior(self):
        t = Graph()
        v = mid(t.read(t.node("pbr_dielectric_f0", specular_f0="Custom IOR", ior=1.5)))[0]
        self.assertAlmostEqual(v, 0.2209, places=2)  # F0 0.04 in sRGB

    def test_metal_reflectance_gold_is_yellow(self):
        t = Graph()
        v = mid(t.read(t.node("pbr_metal_reflectance", metal_color="Gold")))
        self.assertGreater(v[0], v[2] + 0.2)

    def test_converter_metal_has_no_diffuse(self):
        t = Graph()
        n = t.node("basecolor_metallic_roughness_converter",
                   {"basecolor": t.color((0.9, 0.6, 0.3)), "metallic": t.gray(1.0), "roughness": t.gray(0.3)})
        np.testing.assert_allclose(mid(t.read(n, "diffuse"))[:3], [0, 0, 0], atol=1e-3)
        np.testing.assert_allclose(mid(t.read(n, "specular"))[:3], [0.9, 0.6, 0.3], atol=2e-3)
        self.assertAlmostEqual(mid(t.read(n, "glossiness"))[0], 0.7, places=3)


class Hdri(unittest.TestCase):
    def test_blackbody_red_low_blue_high(self):
        t = Graph()
        low = mid(t.read(t.node("blackbody", temperature=1500.0)))
        high = mid(t.read(t.node("blackbody", temperature=15000.0)))
        self.assertGreater(low[0], low[2] * 4)
        self.assertGreater(high[2], high[0])

    def test_exposure_stops(self):
        t = Graph()
        v = mid(t.read(t.node("exposure", {"input": t.color((0.25, 0.5, 1.0))}, exposure=1.0)))
        np.testing.assert_allclose(v[:3], [0.5, 1.0, 2.0], atol=1e-3)

    def test_sky_panorama_is_2_to_1_with_horizon_in_the_middle(self):
        t = Graph(size=(7, 6))
        a = t.read(t.node("physical_sun_sky"))
        self.assertEqual(a.shape[:2], (64, 128))
        sky, ground = a[10, 10, :3], a[54, 10, :3]
        self.assertGreater(sky[2], sky[0])  # blue sky above the horizon
        self.assertGreater(luma(sky), luma(ground) * 0.5)
        # The sun disc is the brightest point, where Sun Position puts it.
        y, x = np.unravel_index(np.argmax(a[:, :, :3].sum(axis=2)), a.shape[:2])
        self.assertLess(abs(x / 128 - 0.5), 0.05)
        self.assertLess(abs(y / 64 - 0.3), 0.05)

    def test_lights_are_hdr(self):
        t = Graph(size=(7, 6))
        a = t.read(t.node("shape_light", shape_exposure=4.0, offset=[0.0, 0.0]))
        self.assertGreater(a[:, :, :3].max(), 8.0)

    def test_panorama_rotation_identity(self):
        t = Graph(size=(6, 5))
        src = t.node("panorama_3d_position", up_vector="Y Up")
        a = t.read(src)
        b = t.read(t.node("panorama_rotation", {"input": src}))
        np.testing.assert_allclose(a[4:-4, 4:-4], b[4:-4, 4:-4], atol=0.02)

    def test_panorama_position_up_at_top(self):
        t = Graph(size=(6, 5))
        a = t.read(t.node("panorama_3d_position", up_vector="Y Up"))
        self.assertGreater(a[0, 10, 1], 0.95)
        self.assertLess(a[-1, 10, 1], 0.05)


class Masks(unittest.TestCase):
    def test_edge_select_convex_only(self):
        t = Graph()
        # The defaults select both kinds of edges at zero width: ask for convex ones only.
        convex = {"concave_intensity": 0.0, "convex_edges_width": 0.5, "contrast": 0.5}
        flat = mid(t.read(t.node("edge_select", {"curvature": t.gray(0.5)}, **convex)))[0]
        edge = mid(t.read(t.node("edge_select", {"curvature": t.gray(1.0)}, **convex)))[0]
        self.assertLess(flat, 0.05)
        self.assertGreater(edge, 0.95)

    def test_bottom_to_top_white_at_bottom(self):
        t = Graph()
        bottom = mid(t.read(t.node("bottom_to_top", {"position": t.color((0.5, 0.0, 0.5))})))[0]
        top = mid(t.read(t.node("bottom_to_top", {"position": t.color((0.5, 1.0, 0.5))})))[0]
        self.assertGreater(bottom, top + 0.5)

    def test_dust_on_upward_faces(self):
        t = Graph()
        up = t.read(t.node("dust", {"world_space_normal": t.color((0.5, 1.0, 0.5))}, noise_opacity=0.0)).mean()
        down = t.read(t.node("dust", {"world_space_normal": t.color((0.5, 0.0, 0.5))}, noise_opacity=0.0)).mean()
        self.assertGreater(up, down + 0.3)

    def test_optional_mask_masks(self):
        t = Graph()
        v = mid(t.read(t.node("edge_select", {"curvature": t.gray(1.0), "mask": t.gray(0.0)})))[0]
        self.assertLess(v, 1e-3)


class Scan(unittest.TestCase):
    def test_crop_identity(self):
        t = Graph()
        src = t.node("test_pattern")
        # The crop: Transform 1 is the input's whole short side (the default 2 is its centre half).
        n = t.node("crop", {"input": src}, transform=[1.0, 0.0, 0.0, 1.0], input_size=[32, 32])
        np.testing.assert_allclose(t.read(n), t.read(src), atol=0.02)
        half = t.read(t.node("crop", {"input": src}, input_size=[32, 32]))
        np.testing.assert_allclose(half[16, 16], t.read(src)[16, 16], atol=0.05)  # the centre stays in place

    def test_color_equalizer_mask(self):
        """The mask keeps the pixels near the reference color in hue, chroma and luma (all or nothing), leaves
        the others unchanged and is the Mask output; Blur softens it; Input cuts the Mask Input at 0.5."""
        red, blue = [0.8, 0.3, 0.2], [0.2, 0.4, 0.8]
        t = Graph()
        ramp = t.node("gradient_axial")
        src = t.node("blend", {"foreground": t.color(red), "background": t.color(blue), "opacity": ramp})
        img = t.read(src)
        n = t.node("color_equalizer", {"input": src}, mask_source="Color Parameter", color=red + [1.0], radius=2.0)
        m = t.read(n, "mask")[..., 0]
        self.assertEqual(set(np.unique(m)), {0.0, 1.0})
        near = np.abs(img[..., :3] - red).max(-1) < 0.05
        self.assertTrue(m[near].all() and not m[np.abs(img[..., :3] - blue).max(-1) < 0.05].any())
        out = t.read(n)
        np.testing.assert_allclose(out[m == 0], img[m == 0], atol=1e-6)
        soft = t.read(t.node("color_equalizer", {"input": src}, mask_source="Color Parameter", color=red + [1.0],
                             blur=2.0), "mask")[..., 0]
        self.assertTrue(((soft > 0.01) & (soft < 0.99)).any())
        self.assertAlmostEqual(soft.mean(), m.mean(), delta=0.05)
        cut = t.read(t.node("color_equalizer", {"input": src, "mask_input": ramp}, mask_source="Input"), "mask")
        np.testing.assert_array_equal(cut[..., 0], (t.read(ramp)[..., 0] >= 0.5).astype(float))

    def test_multi_angle_albedo_rejects_shadow(self):
        t = Graph()
        lit = [t.color((0.5, 0.5, 0.5)) for _ in range(3)] + [t.color((0.0, 0.0, 0.0))]
        n = t.node("multi_angle_to_albedo", {f"input_{i + 1}": s for i, s in enumerate(lit)}, samples_amount=4)
        np.testing.assert_allclose(mid(t.read(n))[:3], [0.5, 0.5, 0.5], atol=0.01)

    def test_multi_angle_normal_flat_when_evenly_lit(self):
        t = Graph()
        s = t.color((0.5, 0.5, 0.5))
        n = t.node("multi_angle_to_normal", {f"input_{i}": s for i in range(1, 5)})
        np.testing.assert_allclose(mid(t.read(n))[:3], [0.5, 0.5, 1.0], atol=0.02)

    def test_atlas_splitter_finds_shape(self):
        t = Graph(size=(6, 6))
        n = t.node("atlas_splitter", {"opacity": t.node("test_pattern")})
        self.assertGreater(t.read(n, "opacity").mean(), 0.2)

    def test_atlas_scatter_covers(self):
        t = Graph(size=(6, 6))
        n = t.node("atlas_scatter", {"atlas_opacity": t.node("test_pattern")})
        cov = t.read(n, "opacity")
        self.assertGreater(cov.mean(), 0.05)


class Materials(unittest.TestCase):
    def test_material_transform_identity(self):
        t = Graph()
        src = t.node("test_pattern_color")
        a = t.read(t.node("material_transform", {"basecolor": src}), "basecolor")
        np.testing.assert_allclose(a, t.read(src), atol=0.02)

    def test_base_material_metal_preset(self):
        t = Graph()
        n = t.node("base_material", material_preset="Gold")
        self.assertAlmostEqual(mid(t.read(n, "metallic"))[0], 1.0, places=3)
        bc = mid(t.read(n, "basecolor"))
        self.assertGreater(bc[0], bc[2])

    def test_water_level_floods_low_areas(self):
        t = Graph()
        n = t.node("water_level", {"height": t.gray(0.1), "roughness": t.gray(0.8)}, water_level=0.5)
        self.assertLess(mid(t.read(n, "roughness"))[0], 0.1)
        dry = t.node("water_level", {"height": t.gray(0.9), "roughness": t.gray(0.8)}, water_level=0.5,
                     edges_wetness=0.0)
        self.assertAlmostEqual(mid(t.read(dry, "roughness"))[0], 0.8, places=3)

    def test_water_level_mask(self):
        """The Mask output is the wetness: Edges Wetness under water and up to half Edges Wetness Distance above
        it, dry from 1.45 times the distance; the Mask input scales the water level."""
        t = Graph()

        def wetness(height, mask=1.0):
            n = t.node("water_level", {"height": t.gray(height), "mask": t.gray(mask)}, water_level=0.5,
                       edges_wetness=0.75, edges_wetness_distance=0.2)
            return mid(t.read(n, "mask"))[0]
        self.assertAlmostEqual(wetness(0.2), 0.75, places=3)
        self.assertAlmostEqual(wetness(0.58), 0.75, places=3)
        self.assertAlmostEqual(wetness(0.79), 0.0, places=3)
        self.assertTrue(0.0 < wetness(0.7) < 0.75)
        self.assertAlmostEqual(wetness(0.3, mask=0.5), 0.75, places=3)  # the level is 0.25: 0.3 is wet shore
        self.assertAlmostEqual(wetness(0.6, mask=0.5), 0.0, places=3)  # dry, though under Water Level


def luma(c):
    return float(np.dot(c, [0.2126, 0.7152, 0.0722]))


if __name__ == "__main__":
    unittest.main()
