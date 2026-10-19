"""Filters and generators of the Paint workspace (library nodes, paint/graph_nodes.py): the Paint workspace's
own (nodes/paint/, catalog conformance is in test_nodes) and the Graph workspace's, on the GPU."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_paint import ev, read, stack, write_obj
from trafilatura3d.nodes.base import Input, Node, Output, PortType
from trafilatura3d.paint import graph_nodes as dn
from trafilatura3d.paint.model import ChannelBlend, ChannelSource, Effect, Mask, new_layer
from trafilatura3d.paint.session import Session


def P(node_type: str) -> str:
    return dn.PREFIX + "paint_" + node_type


class TestPaintNodes(unittest.TestCase):
    def test_listed_first(self):
        """The Paint nodes come first and hide the Graph nodes of the same label."""
        fl = dn.filters()
        self.assertEqual(fl[0].category.split("/")[0], "Paint")
        labels = [n.label for n in fl]
        self.assertEqual(labels.count("Blur"), 1)
        self.assertEqual(next(n for n in fl if n.label == "Blur").type, "paint_blur")
        self.assertEqual(next(n for n in dn.generators() if n.label == "Light").type, "paint_light")

    def filtered(self, procedural, values, base=0.2, channel="roughness"):
        p, ts, store, sc = stack()
        layer = new_layer(ts, "fill", "F")
        layer.material[channel].color = [base] * 3
        ts.layers.append(layer)
        layer.effects.append(Effect("f", "filter", "Filter", procedural=P(procedural), values=values))
        return read(sc, sc.channel(ts, channel))

    def test_filters(self):
        self.assertAlmostEqual(self.filtered("invert", {})[0, 0, 0], 0.8, delta=1.5 / 255)
        self.assertAlmostEqual(self.filtered("invert", {"invert": False})[0, 0, 0], 0.2, delta=1.5 / 255)
        self.assertAlmostEqual(self.filtered("clamp", {"min": 0.3})[0, 0, 0], 0.3, delta=1.5 / 255)
        self.assertAlmostEqual(self.filtered("threshold", {"threshold": 0.1})[0, 0, 0], 1.0, delta=1.5 / 255)
        self.assertAlmostEqual(self.filtered("histogram_shift", {"position": 0.5})[0, 0, 0], 0.7, delta=1.5 / 255)
        self.assertAlmostEqual(self.filtered("blur", {})[5, 5, 0], 0.2, delta=1.5 / 255)  # uniform stays uniform
        self.assertAlmostEqual(self.filtered("highpass", {})[5, 5, 0], 0.5, delta=1.5 / 255)  # uniform: flat 0.5
        self.assertAlmostEqual(self.filtered("height_adjust", {"offset": 0.3}, 0.1, "height")[0, 0, 0], 0.4, delta=1e-2)

    def test_filter_opacity(self):
        p, ts, store, sc = stack()
        layer = new_layer(ts, "fill", "F")
        layer.material["roughness"].color = [0.2] * 3
        ts.layers.append(layer)
        layer.effects.append(Effect("f", "filter", "Filter", procedural=P("invert"),
                                    blend={"roughness": ChannelBlend("normal", 0.5)}))
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.5, delta=1.5 / 255)

    def test_generators_on_baked_maps(self):
        """3D Linear Gradient along X on the two quads: varies; Light from above on +Y faces: lit; a generator
        without its mesh map is logged."""
        from trafilatura3d.paint import baking
        with tempfile.TemporaryDirectory() as tmp:
            s = Session.new(write_obj(Path(tmp)), resolution=64, evaluator=ev())
            ts = s.project.texture_set("Left")
            layer = new_layer(ts, "fill", "F")
            ts.layers.append(layer)
            layer.material["roughness"].color = [1.0] * 3
            layer.mask = Mask(0.0, effects=[Effect("g", "generator", "Light", procedural=P("light"),
                                                   values={"vertical_angle": 90.0, "highlight_level": 0.0})])
            read(s.compiler(1001), s.compiler(1001).mask(ts, layer))
            self.assertTrue(s.log_lines)  # no world space normal map yet
            cfg = baking.config(ts)
            cfg["common"]["size"] = [64, 64]
            baking.set_config(ts, cfg)
            baking.bake_texture_set(s, ts, slots=["world_space_normals", "position"])
            m = read(s.compiler(1001), s.compiler(1001).mask(ts, layer))
            self.assertGreater(m[32, 32, 0], 0.9)  # light straight above, normals up
            layer.mask.effects[0].values["vertical_angle"] = -90.0
            m = read(s.compiler(1001), s.compiler(1001).mask(ts, layer))
            self.assertLess(m[32, 32, 0], 0.1)
            layer.mask.effects[0] = Effect("g2", "generator", "Gradient", procedural=P("linear_gradient_3d"),
                                           values={"position_start": [0, 0, 0], "position_end": [1, 0, 0]})
            m = read(s.compiler(1001), s.compiler(1001).mask(ts, layer))
            row = m[32, 4:60, 0]
            self.assertGreater(abs(float(row[-1] - row[0])), 0.2)  # varies across the quad (along X)


class _MaterialProbe(Node):
    """A test material filter: base color := its height input (0.5 = flat), height := 2 x height;
    roughness has no output; `texture` is a resource slot added to the base color."""
    type = "test_material_probe"
    registered = False
    inputs = {"basecolor": Input(PortType.COLOR), "height": Input(PortType.GRAY), "texture": Input(PortType.GRAY)}
    outputs = {"basecolor": Output(PortType.COLOR), "height": Output(PortType.GRAY)}

    def wgsl(self):
        return """@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let h = load_height(c).r;
    let t = select(0.0, load_texture(c).r, TEXTURE_CONNECTED);
    store_basecolor(c, vec4f(vec3f(h + t) + load_basecolor(c).rgb * 0.0, 1.0));
    store_height(c, vec4f(vec3f((h - 0.5) * 2.0 + 0.5), 1.0));
}"""


class TestMaterialFilters(unittest.TestCase):
    def setUp(self):
        from trafilatura3d.nodes import NODES
        NODES[_MaterialProbe.type] = _MaterialProbe()
        self.addCleanup(NODES.pop, _MaterialProbe.type)

    def test_channels_in_and_out(self):
        """Each channel takes its namesake output, reads the other channels' stack below at the same point,
        heights converted (Paint signed, Graph 0.5 = flat); a channel without an output is left as is; a
        resource slot feeds the node's other input."""
        p, ts, store, sc = stack()
        self.assertTrue({"height", "basecolor", "roughness"} <= set(ts.channel_ids()))
        fill = new_layer(ts, "fill", "F")
        fill.material["basecolor"].color = [0.8] * 3
        fill.material["height"].color = [0.2] * 3
        fill.material["roughness"].color = [0.4] * 3
        ts.layers.append(fill)
        top = new_layer(ts, "fill", "PT")
        top.material.clear()
        for c in ts.channel_ids():
            top.blend[c] = ChannelBlend("passthrough", 1.0)
        fx = Effect("m", "filter", "Probe", procedural=dn.PREFIX + _MaterialProbe.type)
        top.effects.append(fx)
        ts.layers.append(top)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "basecolor"))[0, 0, 0], 0.6, delta=2 / 255)  # 0.2 * .5 + .5
        self.assertAlmostEqual(read(sc, sc.channel(ts, "height"))[0, 0, 0], 0.4, delta=1e-2)  # doubled, signed
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.4, delta=2 / 255)  # untouched
        self.assertEqual(dn.resource_ports(_MaterialProbe(), "filter"), ["texture"])
        fx.inputs["texture"] = ChannelSource([0.25] * 3)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "basecolor"))[0, 0, 0], 0.85, delta=2 / 255)


class TestComputedMaps(unittest.TestCase):
    def test_islands_and_density(self):
        from trafilatura3d.paint.uvmaps import density_values, island_values
        quad = np.array([[[0, 0], [0.4, 0], [0.4, 0.4]], [[0, 0], [0.4, 0.4], [0, 0.4]]], float)
        uv = np.concatenate([quad, quad + 0.5])  # two islands of two triangles
        isl = island_values(uv)
        np.testing.assert_array_equal(isl[0], isl[1])
        np.testing.assert_array_equal(isl[2], isl[3])
        self.assertFalse(np.array_equal(isl[0], isl[2]))
        pos = np.concatenate([np.pad(quad, ((0, 0), (0, 0), (0, 1))), np.pad(quad, ((0, 0), (0, 0), (0, 1))) * 2])
        dens = density_values(uv, pos, (64, 64))
        self.assertAlmostEqual(dens[0, 0] - dens[2, 0], 1.0, places=5)  # half the surface: twice the density

    def test_in_the_stack(self):
        """UV islands of a Texture Set (one quad: one island, alpha 1 inside the UVs)."""
        with tempfile.TemporaryDirectory() as tmp:
            s = Session.new(write_obj(Path(tmp)), resolution=64, evaluator=ev())
            ts = s.project.texture_set("Left")
            m = read(s.compiler(1001), s.compiler(1001).mesh_map(ts, "uv_islands"))
            self.assertTrue((m[8:56, 8:56, 3] == 1).all())
            self.assertLess(np.ptp(m[8:56, 8:56, 0]), 1e-6)
            d = read(s.compiler(1001), s.compiler(1001).mesh_map(ts, "texel_density"))
            self.assertLess(np.abs(d[8:56, 8:56, 0]).max(), 1e-4)  # uniform density: the mean


class TestGraphNodes(unittest.TestCase):
    def test_lists(self):
        self.assertGreater(len(dn.procedurals()), 50)
        self.assertGreater(len(dn.filters()), 50)
        self.assertGreater(len(dn.generators()), 10)

    def test_procedural_source_filter_and_passthrough(self):
        """A Graph noise as a fill channel's source; Invert on it keeps the coverage; a Passthrough layer's
        effect filters the whole stack below."""
        noise = next(n for n in dn.procedurals() if "noise" in n.label.lower())
        p, ts, store, sc = stack()
        layer = new_layer(ts, "fill", "F")
        layer.material["roughness"] = ChannelSource([0.5] * 3, procedural=dn.PREFIX + noise.type)
        ts.layers.append(layer)
        a = read(sc, sc.channel(ts, "roughness"))
        self.assertGreater(a[..., 0].std(), 0.01)
        inv = next(n for n in dn.filters() if n.label == "Invert")
        layer.effects.append(Effect("i", "filter", "Invert", procedural=dn.PREFIX + inv.type))
        b = read(sc, sc.channel(ts, "roughness"))
        np.testing.assert_allclose(b[..., 0], 1 - a[..., 0], atol=2 / 255)
        layer.effects.clear()
        top = new_layer(ts, "fill", "PT")
        top.material.clear()
        top.blend["roughness"] = ChannelBlend("passthrough", 1.0)
        top.effects.append(Effect("i2", "filter", "Invert", procedural=dn.PREFIX + inv.type))
        ts.layers.append(top)
        c = read(sc, sc.channel(ts, "roughness"))
        np.testing.assert_allclose(c[..., 0], 1 - a[..., 0], atol=2 / 255)

    def test_every_paint_filter_in_the_stack(self):
        """Every Paint filter on a Passthrough layer over a fill: every channel cooks finite values."""
        with tempfile.TemporaryDirectory() as tmp:
            s = Session.new(write_obj(Path(tmp)), resolution=64, evaluator=ev())
            ts = s.project.texture_set("Left")
            fill = new_layer(ts, "fill", "F")
            ts.layers.append(fill)
            top = new_layer(ts, "fill", "PT")
            top.material.clear()
            for c in ts.channel_ids():
                top.blend[c] = ChannelBlend("passthrough", 1.0)
            ts.layers.append(top)
            for f in (n for n in dn.filters() if n.category.startswith("Paint/")):
                top.effects[:] = [Effect("f", "filter", f.label, procedural=dn.PREFIX + f.type)]
                for c in ts.channel_ids():
                    with self.subTest(node=f.label, channel=c):
                        self.assertTrue(np.isfinite(read(s.compiler(1001), s.compiler(1001).channel(ts, c))).all())

    def test_every_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session.new(write_obj(Path(tmp)), resolution=64, evaluator=ev())
            ts = s.project.texture_set("Left")
            layer = new_layer(ts, "fill", "F")
            ts.layers.append(layer)
            for g in dn.generators():  # every generator cooks (missing mesh maps read their defaults)
                layer.mask = Mask(0.0, effects=[Effect("g", "generator", g.label, procedural=dn.PREFIX + g.type)])
                m = read(s.compiler(1001), s.compiler(1001).mask(ts, layer))
                self.assertTrue(np.isfinite(m).all(), g.label)


if __name__ == "__main__":
    unittest.main()
