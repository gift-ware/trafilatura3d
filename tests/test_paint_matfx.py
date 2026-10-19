"""The Paint workspace's MatFX filters and Stylization (nodes/paint/matfx/) behave as described."""

import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from trafilatura3d import Evaluator, Package, compile_graph
from trafilatura3d.nodes import NODES
from trafilatura3d.paint import graph_nodes as dn

EV = Evaluator()
F32 = {"mode": "absolute", "value": "32f"}
CATALOG = {n["id"]: n for n in json.loads((Path(__file__).parent.parent / "reference" / "paint" / "filters.json")
                                           .read_text(encoding="utf-8"))["items"]}
MATFX = sorted(t for t, d in NODES.items() if t.startswith("paint_material_fx_")) + ["paint_stylization"]


class Graph:
    """A small graph at 128 px, every node in 32-bit float."""

    def __init__(self, log2=7):
        self.pkg = Package()
        self.g = self.pkg.new_graph("t", log2)

    def node(self, type, inputs=None, **params):
        n = self.g.add_node(type)
        self.g.set_base(n, "output_format", F32)
        for k, v in params.items():
            self.g.set_param(n, k, v)
        for port, src in (inputs or {}).items():
            self.g.connect(src, "output", n, port)
        return n

    def color(self, rgb, a=1.0):
        return self.node("uniform_color", output_color=list(rgb) + [a])

    def gray(self, v):
        return self.node("uniform_color", color_mode="Grayscale", output_color=[v, v, v, 1.0])

    def bricks(self):
        h = self.node("brick_running")
        return h, self.node("normal", {"input": h}, intensity=4.0)

    def read(self, n, out):
        a = EV.read(compile_graph(self.pkg, "t").jobs[n], out)
        a = a.astype(np.float64) / (255.0 if a.dtype == np.uint8 else 1.0)
        return a if a.shape[2] == 4 else np.concatenate([np.repeat(a, 3, axis=2), np.ones_like(a)], axis=2)


def flat_material(t, **extra):
    m = {"basecolor": t.color((0.6, 0.2, 0.1)), "roughness": t.gray(0.3), "metallic": t.gray(0.0),
         "normal": t.color((0.5, 0.5, 1.0)), "height": t.gray(0.5), "ambient_occlusion": t.gray(1.0),
         "diffuse": t.color((0.6, 0.2, 0.1)), "specular": t.color((0.22, 0.22, 0.22)), "glossiness": t.gray(0.7)}
    m.update(extra)
    return m


def run(node_type, inputs, outs, **params):
    """Outputs of `node_type` on `inputs` (port -> node of the graph made by `inputs(t)`)."""
    t = Graph()
    src = inputs(t)
    n = t.node(node_type, {p: s for p, s in src.items() if p in NODES[node_type].inputs}, **params)
    return [t.read(n, o) for o in outs]


class Conformance(unittest.TestCase):
    def test_all_eleven_nodes(self):
        want = {i for i in CATALOG if i.startswith("painter.filter.matfx_")} | {"painter.filter.stylization"}
        self.assertEqual({NODES[t].catalog for t in MATFX}, want)

    def test_repeated_labels_all_present(self):
        """Labels repeated in an entry (Stylization's Color, Oil Paint's per-layer parameters) exist as often."""
        for t in MATFX:
            d = NODES[t]
            want = Counter(p["name"] for p in CATALOG[d.catalog]["parameters"] if p["name"] not in d.unsupported)
            have = Counter(p.label for p in d.params.values())
            with self.subTest(node=t):
                self.assertEqual({k: v for k, v in want.items() if have[k] < v}, {})

    def test_material_ports_in_paint(self):
        """Each channel port takes the stack below of that channel, mesh maps their slot, the rest are resources."""
        hbao = NODES["paint_material_fx_hbao"]
        self.assertEqual(dn.role(hbao, "input", "filter"), ("stack", None))
        self.assertEqual(dn.role(hbao, "height", "filter"), ("channel", "height"))
        rust = NODES["paint_material_fx_rust_weathering"]
        self.assertEqual(dn.role(rust, "basecolor", "filter"), ("channel", "basecolor"))
        self.assertEqual(dn.role(rust, "position", "filter"), ("mesh", "position"))
        self.assertEqual(dn.resource_ports(NODES["paint_material_fx_oil_paint"], "filter"), ["image_input"])
        self.assertEqual(dn.resource_ports(NODES["paint_material_fx_watercolor"], "filter"), ["paper"])
        for t in MATFX:
            self.assertIn(NODES[t], dn.filters())

    def test_edge_damages_leaves_color(self):
        """Edge Damages doesn't change the color of the damaged area: no base color output."""
        d = NODES["paint_material_fx_edge_damages"]
        self.assertIsNone(dn.output_for(d, "basecolor"))
        self.assertEqual(dn.output_for(d, "roughness"), "roughness")


class Behaviour(unittest.TestCase):
    def test_detail_edge_wear_needs_varied_normals(self):
        """No variety in the normal channel: no edges, no visible effect; bricks' normals: worn edges."""
        bc, = run("paint_material_fx_detail_edge_wear", flat_material, ["basecolor"])
        np.testing.assert_allclose(bc[..., :3], np.broadcast_to([0.6, 0.2, 0.1], bc[..., :3].shape), atol=1e-4)

        def bricks(t):
            h, n = t.bricks()
            return flat_material(t, normal=n, height=h)
        bc, m = run("paint_material_fx_detail_edge_wear", bricks, ["basecolor", "metallic"])
        worn = m[..., 0] > 0.5
        self.assertGreater(worn.mean(), 0.01)
        self.assertLess(worn.mean(), 0.5)
        self.assertGreater(bc[worn][:, 1].mean(), 0.5)  # the bare metal color

    def test_edge_damages_on_edges_only(self):
        r, h = run("paint_material_fx_edge_damages", flat_material, ["roughness", "height"])
        np.testing.assert_allclose(r[..., 0], 0.3, atol=1e-4)
        np.testing.assert_allclose(h[..., 0], 0.5, atol=1e-4)

        def bricks(t):
            hh, n = t.bricks()
            return flat_material(t, normal=n, height=hh)
        (r,) = run("paint_material_fx_edge_damages", bricks, ["roughness"], damages_roughness=0.9)
        self.assertGreater((r[..., 0] > 0.6).mean(), 0.01)

    def test_hbao_from_height(self):
        def flat(t):
            return {"input": t.gray(1.0), "height": t.gray(0.5)}
        (o,) = run("paint_material_fx_hbao", flat, ["output"])
        np.testing.assert_allclose(o[..., 0], 1.0, atol=1e-4)

        def bricks(t):
            return {"input": t.gray(1.0), "height": t.bricks()[0]}
        (o,) = run("paint_material_fx_hbao", bricks, ["output"])
        self.assertLess(o[..., 0].min(), 0.7)
        self.assertGreater(np.median(o[..., 0]), 0.8)
        (o2,) = run("paint_material_fx_hbao", bricks, ["output"], intensity=0.0)
        np.testing.assert_allclose(o2[..., 0], 1.0, atol=1e-4)

    def test_peeling_paint_exposes_below(self):
        """Peeled paint pulls away: its coverage goes to 0 there (the layers below show), elsewhere it stays."""
        bc, = run("paint_material_fx_peeling_paint", flat_material, ["basecolor"], peeling_level=0.5)
        a = bc[..., 3]
        self.assertGreater((a < 0.01).mean(), 0.05)
        self.assertGreater((a > 0.99).mean(), 0.3)
        bc, = run("paint_material_fx_peeling_paint", flat_material, ["basecolor"], peeling_level=0.0)
        np.testing.assert_allclose(bc[..., 3], 1.0, atol=1e-4)

    def test_shut_line_follows_height_steps(self):
        bc, = run("paint_material_fx_shut_line", flat_material, ["basecolor"])
        np.testing.assert_allclose(bc[..., :3], np.broadcast_to([0.6, 0.2, 0.1], bc[..., :3].shape), atol=1e-4)

        def steps(t):
            return flat_material(t, height=t.bricks()[0])
        bc, = run("paint_material_fx_shut_line", steps, ["basecolor"], line_color=[0.0, 1.0, 0.0, 1.0],
                  source_material_opacity=0.0)
        line = bc[..., 1] > 0.8
        self.assertGreater(line.mean(), 0.02)
        self.assertLess(line.mean(), 0.4)
        self.assertGreater((bc[..., 3] < 0.01).mean(), 0.3)  # away from the lines the source is gone

    def test_water_drops(self):
        (r,) = run("paint_material_fx_water_drops", flat_material, ["roughness"], drops_amount=0.0, intensity=0.0)
        np.testing.assert_allclose(r[..., 0], 0.3, atol=1e-4)
        r, h = run("paint_material_fx_water_drops", flat_material, ["roughness", "height"], intensity=0.0,
                   drops_roughness=0.0)
        drops = r[..., 0] < 0.05
        self.assertGreater(drops.mean(), 0.05)
        self.assertGreater(h[..., 0][drops].mean(), 0.5)  # drops raise the height

    def test_rust_weathering(self):
        def mat(t):
            return {**flat_material(t, basecolor=t.color((0.5, 0.5, 0.55))),
                    "baked_ambient_occlusion": t.node("ambient_occlusion_hbao", {"input": t.bricks()[0]})}
        bc, me = run("paint_material_fx_rust_weathering", mat, ["basecolor", "metallic"], metallic_intensity=1.0)
        rusty = bc[..., 0] > bc[..., 2] + 0.05
        self.assertGreater(rusty.mean(), 0.05)
        bc, = run("paint_material_fx_rust_weathering", mat, ["basecolor"], base_color_intensity=0.0)
        np.testing.assert_allclose(bc[..., :3], np.broadcast_to([0.5, 0.5, 0.55], bc[..., :3].shape), atol=1e-4)

    def test_comic_book(self):
        def mat(t):
            return {"basecolor": t.node("test_pattern_color"), "curvature": t.gray(0.5),
                    "baked_ambient_occlusion": t.gray(1.0)}
        bc, = run("paint_material_fx_comic_book", mat, ["basecolor"], input_type="Grayscale", border_opacity=0.0)
        self.assertLess(np.abs(bc[..., 0] - bc[..., 2]).max(), 0.02)  # monochrome
        flat, = run("paint_material_fx_comic_book", mat, ["basecolor"], border_opacity=0.0, sharpness=0.0,
                    warp=0.0, paint_intensity=1.0)
        inked, = run("paint_material_fx_comic_book", mat, ["basecolor"], border_opacity=1.0, sharpness=0.0, warp=0.0)
        self.assertGreater((inked[..., :3].sum(-1) < flat[..., :3].sum(-1) - 0.3).mean(), 0.01)  # ink borders

    def test_oil_paint(self):
        def mat(t):
            return flat_material(t, basecolor=t.node("test_pattern_color"))
        bc0, = run("paint_material_fx_oil_paint", mat, ["basecolor"], effect_intensity=0.0)
        src, = run("paint_material_fx_oil_paint", mat, ["basecolor"], effect_intensity=0.0, input_type="Image Input")
        np.testing.assert_allclose(bc0, src, atol=1e-5)  # nothing at 0
        bc, n = run("paint_material_fx_oil_paint", mat, ["basecolor", "normal"])
        self.assertGreater(np.abs(bc - bc0)[..., :3].mean(), 0.01)
        self.assertGreater(np.abs(n[..., :2] - 0.5).mean(), 0.01)  # strokes show in the normal

    def test_watercolor(self):
        def mat(t):
            return {"basecolor": t.color((0.2, 0.4, 0.8))}
        bc, = run("paint_material_fx_watercolor", mat, ["basecolor"], paint_opacity=0.0, hide_paper=True)
        np.testing.assert_allclose(bc[..., :3], 1.0, atol=1e-4)  # no paint, no paper: white
        bc, = run("paint_material_fx_watercolor", mat, ["basecolor"])
        self.assertGreater(bc[..., :3].std(axis=(0, 1)).max(), 0.005)  # the paper's grain
        self.assertGreater(bc[..., 2].mean(), bc[..., 0].mean())  # still the paint's hue

    def test_stylization(self):
        def mat(t):
            h, n = t.bricks()
            return {**flat_material(t, normal=n, height=h), "curvature": t.node("curvature", {"input": n})}
        outs = ["basecolor", "roughness", "normal"]
        off = run("paint_stylization", mat, outs, stylization=0.0)
        np.testing.assert_allclose(off[0][..., :3], np.broadcast_to([0.6, 0.2, 0.1], off[0][..., :3].shape), atol=1e-4)
        np.testing.assert_allclose(off[1][..., 0], 0.3, atol=1e-4)
        on = run("paint_stylization", mat, outs)
        self.assertGreater(np.abs(on[0] - off[0])[..., :3].mean(), 0.01)
        helper, = run("paint_stylization", mat, ["basecolor"], display_helpers="Brush Strokes")
        self.assertLess(np.abs(helper[..., 0] - helper[..., 1]).max(), 1e-4)  # a gray mask
        self.assertGreater(helper[..., 0].mean(), 0.2)


if __name__ == "__main__":
    unittest.main()
