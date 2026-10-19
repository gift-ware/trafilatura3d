"""Paint core: the project document, the layer stack on the GPU (blending modes from the documentation's
formulas, masks, folders), the brush (stamps, Stroke Opacity, Eraser, occlusion), project files and the
texture export."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import imageio
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.paint import blending
from trafilatura3d.paint.brush import BrushSettings, Stroke, Target, TileTargets, View, _read
from trafilatura3d.paint.export import ExportSettings, Exporter, OutputMap, PREDEFINED, Template, _encode
from trafilatura3d.paint.model import (ChannelBlend, Mask, Project, ProjectError, TextureSet, new_layer,
                                   new_texture_set)
from trafilatura3d.paint.paint import MASK, PaintStore, storage_for
from trafilatura3d.paint.session import Session
from trafilatura3d.paint.stack import Resources, StackCompiler

EV = None


def ev():
    global EV
    if EV is None:
        EV = Evaluator()
    return EV


def write_obj(folder: Path) -> Path:
    """Two quads in the XZ plane facing +Y, materials "Left" and "Right", each with the full 0-1 UV square."""
    (folder / "two.mtl").write_text("newmtl Left\nKd 0.8 0.2 0.2\nnewmtl Right\nKd 0.2 0.2 0.8\n", encoding="utf-8")
    lines = ["mtllib two.mtl"]
    for i, (x0, mat) in enumerate(((-1.0, "Left"), (0.0, "Right"))):
        lines += [f"o {mat}", f"v {x0} 0 -0.5", f"v {x0 + 1} 0 -0.5", f"v {x0 + 1} 0 0.5", f"v {x0} 0 0.5",
                  "vt 0 1", "vt 1 1", "vt 1 0", "vt 0 0", "vn 0 1 0", f"usemtl {mat}"]
        b, t = 4 * i + 1, 4 * i + 1
        lines.append(f"f {b}/{t}/{i + 1} {b + 3}/{t + 3}/{i + 1} {b + 2}/{t + 2}/{i + 1}")
        lines.append(f"f {b}/{t}/{i + 1} {b + 2}/{t + 2}/{i + 1} {b + 1}/{t + 1}/{i + 1}")
    path = folder / "two.obj"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def stack(size=64):
    p = Project(resolution=size)
    ts = new_texture_set("mat", size)
    p.texture_sets.append(ts)
    store = PaintStore(ev().device)
    return p, ts, store, StackCompiler(p, store, Resources(), ev(), 1001)


def read(sc, job):
    a = sc.ev.read(job, "output")
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


class TestModel(unittest.TestCase):
    def test_unknown_nodes_are_emptied_on_load(self):
        """A saved project naming library nodes that no longer exist opens: the filter is left with no node, the
        channel source falls back to its uniform value, one warning each; the strict check stays otherwise."""
        from trafilatura3d.paint.model import ChannelSource, Effect, drop_unknown_nodes, new_uid
        p, ts, _, _ = stack()
        fill = new_layer(ts, "fill", "Fill layer 1")
        fill.material["basecolor"] = ChannelSource(procedural="node:clouds_soft")  # exists
        ts.layers.append(fill)
        data = p.to_dict()
        layer = data["texture_sets"][0]["layers"][0]
        layer["effects"] = [Effect(new_uid(), "filter", "Gone", procedural="node:no_such_filter",
                                   values={"x": 1}).to_dict()]
        layer["material"]["roughness"]["procedural"] = "node:no_such_noise"
        with self.assertRaises(ProjectError):
            Project.from_dict(data)
        warnings = drop_unknown_nodes(data)
        self.assertEqual(len(warnings), 2)
        self.assertIn("no_such_filter", warnings[0] + warnings[1])
        q = Project.from_dict(data)
        fx = q.texture_sets[0].layers[0].effects[0]
        self.assertEqual((fx.procedural, fx.values), ("", {}))
        self.assertEqual(q.texture_sets[0].layers[0].material["roughness"].procedural, "")
        self.assertEqual(q.texture_sets[0].layers[0].material["basecolor"].procedural, "node:clouds_soft")

    def test_round_trip(self):
        p, ts, _, _ = stack()
        fill = new_layer(ts, "fill", "Fill layer 1")
        folder = new_layer(ts, "folder", "Folder 1")
        folder.children.append(new_layer(ts, "paint", "Layer 1"))
        folder.mask = Mask(0.0)
        ts.layers += [fill, folder]
        fill.blend["roughness"] = ChannelBlend("multiply", 0.4)
        q = Project.from_dict(p.to_dict())
        self.assertEqual(q.to_dict(), p.to_dict())

    def test_validation(self):
        p, ts, _, _ = stack()
        layer = new_layer(ts, "paint", "Layer 1")
        ts.layers.append(layer)
        layer.blend["basecolor"] = ChannelBlend("normaldetail", 1.0)  # Normal map modes are for Normal
        with self.assertRaises(ProjectError):
            Project.from_dict(p.to_dict())
        layer.blend["basecolor"] = ChannelBlend("nope", 1.0)
        with self.assertRaises(ProjectError):
            Project.from_dict(p.to_dict())

    def test_default_blending(self):
        """Layer stack documentation: Height Linear Dodge (Add), Normal Normal Map Detail, AO Multiply."""
        _, ts, _, _ = stack()
        layer = new_layer(ts, "paint", "Layer 1")
        self.assertEqual(layer.blend_of("height").mode, "lineardodge")
        self.assertEqual(layer.blend_of("normal").mode, "normaldetail")
        self.assertEqual(layer.blend_of("ambientocclusion").mode, "multiply")
        self.assertEqual(layer.blend_of("basecolor").mode, "normal")
        self.assertEqual(len(blending.MODES), 32)


class TestStack(unittest.TestCase):
    def two_fills(self, below, top, channel="roughness"):
        p, ts, store, sc = stack()
        a, b = new_layer(ts, "fill", "A"), new_layer(ts, "fill", "B")
        a.material[channel].color = list(below)
        b.material[channel].color = list(top)
        ts.layers += [a, b]
        return ts, b, sc

    def test_unpainted_is_default_with_zero_alpha(self):
        _, ts, _, sc = stack()
        v = read(sc, sc.channel(ts, "roughness"))[0, 0]  # a channel without layers is a 1x1 uniform job
        self.assertAlmostEqual(v[0], 0.3, delta=1 / 255)  # lib-sampler DEFAULT_ROUGHNESS
        self.assertEqual(v[3], 0.0)

    def test_gray_modes(self):
        """Documentation formulas on a non color managed channel (Roughness, 8 bit)."""
        b, t = 0.6, 0.3
        expect = {"normal": t, "multiply": t * b, "divide": b / t, "darken": min(t, b), "lighten": max(t, b),
                  "lineardodge": min(b + t, 1), "subtract": b - t, "inversesubtract": 0.0, "difference": abs(b - t),
                  "screen": 1 - (1 - t) * (1 - b), "linearburn": max(t + b - 1, 0), "replace": t,
                  "inversedivide": t / b, "exclusion": t + b - 2 * t * b}
        for mode, want in expect.items():
            ts, top, sc = self.two_fills((b, b, b), (t, t, t))
            top.blend["roughness"] = ChannelBlend(mode, 1.0)
            got = read(sc, sc.channel(ts, "roughness"))[5, 5, 0]
            self.assertAlmostEqual(got, min(max(want, 0), 1), delta=1.5 / 255, msg=mode)

    def test_opacity_and_disable(self):
        ts, top, sc = self.two_fills((0.2,) * 3, (1.0,) * 3)
        top.blend["roughness"] = ChannelBlend("normal", 0.5)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.6, delta=1.5 / 255)
        top.blend["roughness"] = ChannelBlend("disable", 1.0)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.2, delta=1.5 / 255)
        top.blend["roughness"] = ChannelBlend("normal", 1.0)
        top.visible = False
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.2, delta=1.5 / 255)

    def test_base_color_blends_in_linear(self):
        """Blending modes are performed in linear space (documentation): Multiply 50% of sRGB values."""
        ts, top, sc = self.two_fills((0.8, 0.8, 0.8), (0.5, 0.5, 0.5), "basecolor")
        top.blend["basecolor"] = ChannelBlend("multiply", 1.0)
        got = read(sc, sc.channel(ts, "basecolor"))[0, 0, 0]
        self.assertAlmostEqual(got, srgb(lin(0.8) * lin(0.5)), delta=1.5 / 255)

    def test_hdr_channel_not_clamped(self):
        """Linear Dodge results above 1 are clamped unless the channel is HDR (Height is L16F)."""
        ts, top, sc = self.two_fills((0.8,) * 3, (0.7,) * 3, "height")
        self.assertAlmostEqual(read(sc, sc.channel(ts, "height"))[0, 0, 0], 1.5, delta=1e-2)

    def test_height_signed(self):
        """Height: a uniform -0.4 stays negative; an image source maps mid grey to 0 (Height map painting)."""
        p, ts, store, sc = stack()
        layer = new_layer(ts, "fill", "F")
        layer.material["height"].color = [-0.4] * 3
        ts.layers.append(layer)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "height"))[0, 0, 0], -0.4, delta=1e-2)
        sc.resources.add("grey", np.full((8, 8, 3), 0.5, np.float32))
        layer.material["height"].image = "grey"
        self.assertAlmostEqual(read(sc, sc.channel(ts, "height"))[3, 3, 0], 0.0, delta=1e-2)

    def test_hsv_modes(self):
        ts, top, sc = self.two_fills((0.2, 0.4, 0.8), (0.9, 0.1, 0.1), "basecolor")
        top.blend["basecolor"] = ChannelBlend("value", 1.0)  # value of the top, hue/saturation of the bottom
        r, g, b = read(sc, sc.channel(ts, "basecolor"))[0, 0, :3]
        self.assertGreater(b, g)
        self.assertGreater(g, r)  # still blue
        top.blend["basecolor"] = ChannelBlend("tint", 1.0)  # hue of the top
        r, g, b = read(sc, sc.channel(ts, "basecolor"))[0, 0, :3]
        self.assertGreater(r, b)

    def test_normal_modes(self):
        """A flat normal on top changes nothing with Normal Map Detail or Combine."""
        tilted = np.array([0.3, 0.1, 0.95])
        tilted /= np.linalg.norm(tilted)
        enc = tuple(tilted * 0.5 + 0.5)
        for mode in ("normaldetail", "normalcombine"):
            ts, top, sc = self.two_fills(enc, (0.5, 0.5, 1.0), "normal")
            top.blend["normal"] = ChannelBlend(mode, 1.0)
            got = read(sc, sc.channel(ts, "normal"))[0, 0, :3]
            np.testing.assert_allclose(got, enc, atol=3e-3, err_msg=mode)

    def test_mask_and_paint(self):
        p, ts, store, sc = stack()
        layer = new_layer(ts, "paint", "Layer 1")
        ts.layers.append(layer)
        img = np.zeros((64, 64, 4), np.float32)
        img[:, :32] = (1, 1, 1, 1)  # left half painted white
        store.write(layer.uid, "roughness", 1001, img, storage_for("L8"))
        a = read(sc, sc.channel(ts, "roughness"))
        self.assertAlmostEqual(a[10, 5, 0], 1.0, delta=1 / 255)
        self.assertAlmostEqual(a[10, 50, 0], 0.3, delta=1 / 255)
        self.assertEqual(a[10, 50, 3], 0.0)
        layer.mask = Mask(0.0)  # black mask hides it
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[10, 5, 0], 0.3, delta=1 / 255)
        m = np.zeros((64, 64, 4), np.float32)
        m[:32] = (1, 1, 1, 1)  # mask painted white on the top half
        store.write(layer.uid, MASK, 1001, m, storage_for(None))
        a = read(sc, sc.channel(ts, "roughness"))
        self.assertAlmostEqual(a[5, 5, 0], 1.0, delta=1 / 255)
        self.assertAlmostEqual(a[50, 5, 0], 0.3, delta=1 / 255)
        layer.mask.enabled = False  # Shift+click toggles the mask
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[50, 5, 0], 1.0, delta=1 / 255)

    def test_folder(self):
        p, ts, store, sc = stack()
        base = new_layer(ts, "fill", "Base")
        base.material["roughness"].color = [0.2] * 3
        folder = new_layer(ts, "folder", "Folder")
        inner = new_layer(ts, "fill", "Inner")
        inner.material["roughness"].color = [1.0] * 3
        folder.children.append(inner)
        folder.blend["roughness"] = ChannelBlend("normal", 0.5)
        ts.layers += [base, folder]
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.6, delta=1.5 / 255)

    def test_effects(self):
        """Effects on the content and on the mask, bottom first: Levels on its Affected Channel, a Fill effect
        blended with its own mode, a Paint effect's pixels."""
        from trafilatura3d.paint.model import ChannelSource, Effect, Levels
        p, ts, store, sc = stack()
        layer = new_layer(ts, "fill", "F")
        layer.material["roughness"].color = [0.4] * 3
        ts.layers.append(layer)
        layer.effects.append(Effect("lv", "levels", "Levels", levels=Levels("roughness", out_low=0.5, out_high=1.0)))
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.7, delta=1.5 / 255)
        layer.effects[0].levels.channel = "metallic"  # another channel: no change
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.4, delta=1.5 / 255)
        fx = Effect("fx", "fill", "Fill", material={"roughness": ChannelSource([1.0] * 3)},
                    blend={"roughness": ChannelBlend("multiply", 1.0)})
        layer.effects.append(fx)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.4, delta=1.5 / 255)
        fx.material["roughness"].color = [0.5] * 3
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[0, 0, 0], 0.2, delta=1.5 / 255)
        pe = Effect("pe", "paint", "Paint")
        layer.effects.append(pe)
        img = np.zeros((64, 64, 4), np.float32)
        img[:8] = (0.9, 0.9, 0.9, 1)
        store.write("pe", "roughness", 1001, img, storage_for("L8"))
        a = read(sc, sc.channel(ts, "roughness"))
        self.assertAlmostEqual(a[2, 2, 0], 0.9, delta=1.5 / 255)
        self.assertAlmostEqual(a[40, 2, 0], 0.2, delta=1.5 / 255)
        layer.mask = Mask(1.0, effects=[Effect("mf", "fill", "Fill", material={"mask": ChannelSource([0.0] * 3)})])
        self.assertAlmostEqual(read(sc, sc.channel(ts, "roughness"))[2, 2, 0], 0.3, delta=1.5 / 255)  # masked out
        q = Project.from_dict(p.to_dict())
        self.assertEqual(q.to_dict(), p.to_dict())
        self.assertIn("mf", q.texture_sets[0].all_uids())

    def test_anchor_points(self):
        """An anchor point is a live reference to the stack where it is: on a Passthrough layer the layers under
        it, on a Normal layer that layer's content, in a mask the mask's effects under it; it follows what is
        painted under it; moved above the layer using it, the reference breaks."""
        from trafilatura3d.paint.model import ChannelSource, Effect
        p, ts, store, sc = stack()
        base = new_layer(ts, "fill", "Base")
        base.material["roughness"].color = [0.2] * 3
        base.material["metallic"].color = [0.7] * 3
        paint = new_layer(ts, "paint", "Paint")
        img = np.zeros((64, 64, 4), np.float32)
        img[:, :32] = (0.9, 0.9, 0.9, 1)
        store.write(paint.uid, "roughness", 1001, img, storage_for("L8"))
        holder = new_layer(ts, "fill", "Holder")
        holder.material = {}  # a Passthrough layer with no content of its own
        holder.blend = {c: ChannelBlend("passthrough", 1.0) for c in ts.channel_ids()}
        holder.effects.append(Effect("anchor", "anchor", "Anchor Point"))
        user = new_layer(ts, "fill", "User")
        user.material["roughness"] = ChannelSource(anchor="anchor")
        user.material["metallic"] = ChannelSource(anchor="anchor", anchor_channel="roughness")
        ts.layers += [base, paint, holder, user]
        rough = lambda: read(sc, sc.channel(ts, "roughness"))
        a = rough()
        self.assertAlmostEqual(a[5, 5, 0], 0.9, delta=1.5 / 255)  # the three layers under the anchor
        self.assertAlmostEqual(a[5, 50, 0], 0.2, delta=1.5 / 255)
        self.assertAlmostEqual(read(sc, sc.channel(ts, "metallic"))[5, 50, 0], 0.2, delta=1.5 / 255)  # roughness
        img[:, 32:] = (0.5, 0.5, 0.5, 1)  # painted under the anchor: the reference follows
        store.write(paint.uid, "roughness", 1001, img, storage_for("L8"))
        self.assertAlmostEqual(read(sc, sc.channel(ts, "metallic"))[5, 50, 0], 0.5, delta=1.5 / 255)
        # On a Normal layer at 100%: that layer only (its own content, not the stack under it)
        holder.blend = {}
        holder.material = {"roughness": ChannelSource([0.6] * 3)}
        user.material["metallic"].anchor_channel = "roughness"
        self.assertAlmostEqual(read(sc, sc.channel(ts, "metallic"))[5, 5, 0], 0.6, delta=1.5 / 255)
        # In a mask: the mask's effects under the anchor, not those above it
        holder.mask = Mask(0.0, effects=[Effect("w", "fill", "White", material={"mask": ChannelSource([1.0] * 3)}),
                                         Effect("manchor", "anchor", "Mask Anchor"),
                                         Effect("b", "fill", "Black", material={"mask": ChannelSource([0.0] * 3)})])
        user.material["roughness"] = ChannelSource(anchor="manchor")
        user.mask = Mask(0.0, effects=[Effect("uf", "fill", "Fill",
                                              material={"mask": ChannelSource(anchor="manchor")})])
        a = rough()
        self.assertAlmostEqual(a[5, 50, 0], 1.0, delta=1.5 / 255)  # the white under the anchor, masked by it
        self.assertEqual([(l.name, f.name) for l, f, _m in ts.anchors(below=user.uid)],
                         [("Holder", "Anchor Point"), ("Holder", "Mask Anchor")])
        self.assertEqual({(l.name, slot) for l, _f, slot in ts.references("manchor")}, {("User", "roughness"),
                                                                                         ("User", "mask")})
        # Moved above the layer using it: broken, the layer gives nothing
        holder.mask, holder.material, user.mask = None, {}, None
        user.material["roughness"] = ChannelSource(anchor="anchor")
        ts.layers[:] = [base, paint, user, holder]
        self.assertEqual(ts.anchors(below=user.uid), [])
        self.assertAlmostEqual(rough()[5, 50, 0], 0.5, delta=1.5 / 255)  # the paint layer under User
        from trafilatura3d.paint.smart import fresh  # a copy (smart material, paste): its references follow it
        folder = new_layer(ts, "folder", "Folder")
        folder.children = [holder, user]
        user.material["roughness"].anchor = "anchor"
        copied = fresh(folder)
        self.assertEqual(copied.children[1].material["roughness"].anchor, copied.children[0].effects[0].uid)
        self.assertNotEqual(copied.children[0].effects[0].uid, "anchor")
        q = Project.from_dict(p.to_dict())
        self.assertEqual(q.to_dict(), p.to_dict())

    def test_material_mode(self):
        """A fill layer's Material mode: a Trafilatura 3D graph's outputs fill the channels of their usages; channels
        the material lacks keep their own source; a missing package falls back and is logged."""
        from tests.scenes import MATERIAL_GRAPH, material_package
        from trafilatura3d.paint.materials import Materials
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        package = material_package(Path(tmp.name) / "material.trafilatura3d")
        p, ts, store, sc = stack(128)
        logged = []
        sc.materials, sc.log = Materials(), logged.append
        layer = new_layer(ts, "fill", "F")
        ts.layers.append(layer)
        layer.resource = {"package": str(package), "graph": MATERIAL_GRAPH, "values": {}}
        a = read(sc, sc.channel(ts, "basecolor"))
        self.assertGreater(a[..., :3].std(), 0.01)  # a texture, not the uniform default
        self.assertTrue((a[..., 3] > 0.99).all())
        n = read(sc, sc.channel(ts, "normal"))
        self.assertGreater(np.abs(n[..., 0] - 0.5).max(), 0.02)
        layer.material.pop("metallic")  # a channel turned off is not filled even by the material
        self.assertEqual(read(sc, sc.channel(ts, "metallic"))[0, 0, 3], 0.0)
        layer.resource = {"package": "/nowhere/missing.trafilatura3d", "graph": "g", "values": {}}
        r = read(sc, sc.channel(ts, "roughness"))
        self.assertAlmostEqual(r[0, 0, 0], 0.3, delta=1.5 / 255)  # the channel's uniform value
        self.assertTrue(logged)
        q = Project.from_dict(p.to_dict())
        self.assertEqual(q.texture_sets[0].layers[0].resource, layer.resource)

    def test_cache_recooks_only_above(self):
        p, ts, store, sc = stack()
        layers = [new_layer(ts, "fill", f"F{i}") for i in range(4)]
        ts.layers += layers
        sc.ev.cook([sc.channel(ts, "roughness")])
        layers[3].material["roughness"].color = [0.9] * 3
        stats = sc.ev.cook([sc.channel(ts, "roughness")])
        # the top fill and the layers' blends in one fused pass (the uniform fills below, 1x1, read again by it)
        self.assertEqual([c for c in stats.cooked if c != "paint_fill"], ["F3/roughness"])

    def test_fused_blends_give_the_blends_texels(self):
        """Layers of a channel fused into one pass (nodes.BlendChain) give exactly what the blends one after the other
        give: 8-bit base color, grayscale roughness, half float height, normal; modes, opacities, masks."""
        from trafilatura3d import evaluator as E
        p, ts, store, sc = stack(64)
        rng = np.random.default_rng(3)
        modes = ["normal", "multiply", "overlay", "screen", "lineardodge", "softlight", "difference", "normal"]
        for i, mode in enumerate(modes):
            layer = new_layer(ts, "paint", f"P{i}")
            ts.layers.append(layer)
            for ch in ("basecolor", "roughness", "height", "normal"):
                img = rng.random((64, 64, 4)).astype(np.float32)
                store.write(layer.uid, ch, 1001, img, storage_for(ts.format_of(ch)))
                layer.blend[ch] = ChannelBlend("normalcombine" if ch == "normal" and i % 2 else mode,
                                               0.3 + 0.7 * rng.random())
            if i % 3 == 1:
                layer.mask = Mask()
                store.write(layer.uid, MASK, 1001, rng.random((64, 64, 4)).astype(np.float32), storage_for(None))
        got = {}
        for fused in (True, False):
            sc.ev.cache.clear()
            chains, E._chains = E._chains, (E._chains if fused else lambda *a: {})
            try:
                got[fused] = {ch: read(sc, sc.channel(ts, ch)) for ch in ("basecolor", "roughness", "height")}
                got[fused]["normal"] = read(sc, sc.normal(ts))
            finally:
                E._chains = chains
        for ch in got[True]:
            np.testing.assert_array_equal(got[True][ch], got[False][ch], err_msg=ch)

    def test_region_recook_matches_a_whole_cook(self):
        """A painting changed in a rectangle (not on workgroup boundaries): its chain to the channel computed again
        over the rectangle with region intermediates (Evaluator.recook_region) gives the channel a whole cook
        gives."""
        p, ts, store, sc = stack(64)
        paint, fill = new_layer(ts, "paint", "P"), new_layer(ts, "fill", "F")
        fill.material["basecolor"].color = [0.2, 0.6, 0.9]
        fill.blend["basecolor"] = ChannelBlend("multiply", 0.5)
        ts.layers += [paint, fill]
        img = np.zeros((64, 64, 4), np.float32)
        img[10:20, 10:20] = [1, 0, 0, 1]
        store.write(paint.uid, "basecolor", 1001, img, storage_for("sRGB8"))
        entry = store.entries[(paint.uid, "basecolor", 1001)]
        top = sc.channel(ts, "basecolor")
        memo, order = {}, []

        def reads(job):
            if job.key not in memo:
                read = [reads(src[0]) for src in job.inputs.values() if src]
                memo[job.key] = job.extra.get("pages") is entry.pages or any(read)
                if memo[job.key]:
                    order.append(job)
            return memo[job.key]
        reads(top)
        chain = {j.key for j in order}
        for job in order:  # what the chain reads from outside it stays cached
            for src in job.inputs.values():
                if src and src[0].key not in chain:
                    src[0].keep = True
        sc.ev.cook([top])
        img[30:40, 31:43] = [0, 1, 0, 1]
        entry.pages.write(img)
        sc.ev.recook_region(order, (29, 29, 45, 42), {top.key}, set())
        frozen = read(sc, top)
        store.touched(paint.uid, "basecolor", 1001)
        np.testing.assert_array_equal(frozen, read(sc, sc.channel(ts, "basecolor")))

    def test_region_recook_of_a_painted_mask_and_the_normal(self):
        """A mask painted (a paint effect) under a fill's height, the normal made from the height: frames over large
        rectangles one after the other (region textures larger than the regions, neighbors read by the normal at a
        region's edge) give the normal a whole cook gives."""
        from trafilatura3d.paint.model import Effect, new_uid
        from trafilatura3d.paint.paint import MASK
        n = 512
        p, ts, store, sc = stack(n)
        fill = new_layer(ts, "fill", "F")
        fill.material["height"].color = [0.8, 0.8, 0.8]
        fill.mask = Mask(0.0)
        fx = Effect(new_uid(), "paint", "Paint")
        fill.mask.effects.append(fx)
        ts.layers.append(fill)
        img = np.zeros((n, n, 4), np.float32)
        img[100:140, 100:140] = [1, 1, 1, 1]
        store.write(fx.uid, MASK, 1001, img, storage_for(None))
        entry = store.entries[(fx.uid, MASK, 1001)]
        top = sc.normal(ts)
        memo, order = {}, []

        def reads(job):
            if job.key not in memo:
                read = [reads(src[0]) for src in job.inputs.values() if src]
                memo[job.key] = job.extra.get("pages") is entry.pages or any(read)
                if memo[job.key]:
                    order.append(job)
            return memo[job.key]
        reads(top)
        chain = {j.key for j in order}
        outside = [src[0] for j in order for src in j.inputs.values() if src and src[0].key not in chain]
        for job in outside + [top]:
            job.keep = True
        sc.ev.hold("test", outside + [top])
        sc.ev.cook([top] + outside)
        for x0, y0, x1, y1 in ((150, 160, 240, 400), (230, 200, 330, 300), (90, 90, 160, 160)):
            img[y0 + 10:y1 - 10, x0 + 10:x1 - 10] = [0.7, 0.7, 0.7, 1]
            entry.pages.write(img)
            sc.ev.recook_region(order, (x0, y0, x1, y1), {top.key}, set())
        frozen = read(sc, top)
        sc.ev.hold("test", [])
        store.touched(fx.uid, MASK, 1001)
        np.testing.assert_array_equal(frozen, read(sc, sc.normal(ts)))

    def test_padding_fills_outside_islands(self):
        p, ts, store, sc = stack()
        import wgpu
        mask = np.zeros((64, 64, 4), np.uint8)
        mask[20:40, 20:40] = 255
        t = sc.ev.device.create_texture(size=(64, 64, 1), format="rgba8unorm",
                                        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        sc.ev.device.queue.write_texture({"texture": t}, mask.tobytes(), {"bytes_per_row": 256}, (64, 64, 1))
        sc.uv_masks[(ts.name, 1001)] = t
        layer = new_layer(ts, "fill", "F")
        layer.material["roughness"].color = [0.9] * 3
        ts.layers.append(layer)
        sc.ev.clear()
        a = read(sc, sc.padded(ts, sc.channel(ts, "roughness")))
        self.assertAlmostEqual(a[0, 0, 0], 0.9, delta=1 / 255)
        b = read(sc, sc.padded(ts, sc.channel(ts, "roughness"), distance=4, outside=0))
        self.assertEqual(b[0, 0, 3], 0.0)
        self.assertAlmostEqual(b[18, 30, 0], 0.9, delta=1 / 255)


class TestViewPages(unittest.TestCase):
    def test_viewport_computes_near_the_islands_only(self):
        """A quad on a corner of the UV square: the viewport's stack (StackCompiler.uv_only) computes the pages near
        it only, and its padded slots, the normal included, are the whole stack's texel for texel."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "q.obj").write_text("v -1 0 -1\nv 1 0 -1\nv 1 0 1\nv -1 0 1\nvt 0.1 0.1\nvt 0.35 0.1\n"
                                      "vt 0.35 0.35\nvt 0.1 0.35\nvn 0 1 0\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n", encoding="utf-8")
            s = Session.new(tmp / "q.obj", resolution=256, evaluator=ev())
            ts = s.project.texture_sets[0]
            rng = np.random.default_rng(5)
            for i in range(3):
                layer = new_layer(ts, "paint", f"P{i}")
                ts.layers.append(layer)
                for ch in ("basecolor", "height"):
                    s.store.write(layer.uid, ch, 1001, rng.random((256, 256, 4)).astype(np.float32),
                                  storage_for(ts.format_of(ch)))
                    layer.blend[ch] = ChannelBlend("multiply" if i else "normal", 0.8)
            s.uv_mask(ts, 1001)
            sc = s.compiler(1001)
            on = sc._pages_on(ts)
            self.assertIsNotNone(on)
            self.assertLess(sum(bin(w & 0xFFFFFFFF).count("1") for w in on), 64)  # of 256
            for job in (sc.channel(ts, "basecolor"), sc.normal(ts)):
                view = sc.uv_only(ts, job)
                self.assertNotEqual(view.key, job.key)
                np.testing.assert_array_equal(read(sc, sc.padded(ts, view)), read(sc, sc.padded(ts, job)))


class TestMaskStorage(unittest.TestCase):
    def test_mask_keeps_value_and_coverage(self):
        """A painted mask is stored in two 8-bit channels (value, how much is painted) and read whole as RGBA, the
        value repeated; nothing painted reads 0 coverage."""
        store = PaintStore(ev().device)
        e = store.ensure("layer", MASK, 1001, (64, 64), storage_for(None))
        self.assertIn(e.pages.format, ("rg8unorm", "rgba8unorm"))
        px = np.zeros((64, 64, 4), np.float32)
        px[:32, :32] = (0.5, 0.5, 0.5, 0.25)
        e.pages.write(px)
        back = e.pages.read()
        np.testing.assert_allclose(back[:32, :32], px[:32, :32], atol=0.5 / 255)
        self.assertEqual(float(back[40:, 40:, 3].max()), 0.0)


    def test_pages_from_another_storage(self):
        """A file's pages saved in another storage (an older mask format, a device without two channel formats) are
        read through the whole pixels into the entry's storage."""
        src = PaintStore(ev().device)
        e = src.ensure("l", "basecolor", 1001, (64, 64), "rgba16float")
        px = np.zeros((64, 64, 4), np.float32)
        px[:10, :10] = (0.25, 0.25, 0.25, 0.5)
        e.pages.write(px)
        table, pages = e.pages.raw()
        dst = PaintStore(ev().device)
        dst.write_raw("l", MASK, 1001, (64, 64), storage_for(None), table, pages, "rgba16float")
        np.testing.assert_allclose(dst.read("l", MASK, 1001), px, atol=0.5 / 255)


class TestBrush(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Session.new(write_obj(Path(self.tmp.name)), resolution=128, evaluator=ev())
        self.ts = self.s.project.texture_sets[0]
        self.layer = new_layer(self.ts, "paint", "Layer 1")
        self.ts.layers.append(self.layer)

    def tearDown(self):
        self.tmp.cleanup()

    def view(self, eye=(0.0, 3.0, 0.001)):
        from trafilatura3d.editor.render3d import look_at, perspective
        eye = np.array(eye, np.float32)
        vp = perspective(math.radians(40), 1.0, 0.05, 100) @ look_at(eye, [0, 0, 0])
        return View(vp, eye, math.radians(40), 256, 256)

    def meshes(self):
        import wgpu
        d = self.s.ev.device
        mk = lambda a, u: d.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(), usage=u)
        V, I = wgpu.BufferUsage.VERTEX, wgpu.BufferUsage.INDEX
        return [([mk(o.pos, V), mk(o.nrm, V), mk(o.tan, V), mk(o.uv, V)], mk(o.idx, I),
                 [(x.start, x.count, x.material) for x in o.submeshes]) for o in self.s.objects]

    def stroke(self, points, settings=None, erase=False, value=(1.0, 0.0, 0.0), view=None, symmetry=None):
        e = self.s.store.ensure(self.layer.uid, "basecolor", 1001, self.ts.size, storage_for("sRGB8"))
        tile = TileTargets(1001, self.s.uv_maps(self.ts, 1001), [Target((self.layer.uid, "basecolor", 1001), e.pages,
                                                                        value, erase)])
        st = Stroke(self.s.brush_engine(), [tile], view or self.view(), self.meshes(), self.s.material_index(self.ts),
                    settings or BrushSettings(), self.s.bbox_diagonal(), symmetry=symmetry)
        n = st.add(points)
        st.end()
        self.s.store.touched(self.layer.uid, "basecolor", 1001)
        return n, self.s.store.read(self.layer.uid, "basecolor", 1001)

    def test_paints_only_its_texture_set(self):
        # The Left texture set covers the left half of the screen (x < 128).
        n, a = self.stroke([(40, 128, 1.0), (100, 128, 1.0)])
        self.assertGreater(n, 3)
        self.assertGreater((a[..., 3] > 0.5).mean(), 0.01)
        n, _ = self.stroke([(160, 128, 1.0), (220, 128, 1.0)])
        self.assertEqual(n, 0)

    def test_stroke_saves_only_the_blocks_it_changes(self):
        """At 1024 (blocks of 256: 16 per texture) a small stroke saves only the blocks it changes, and the region
        its undo puts back is the entry as it was before the stroke, texel for texel."""
        self.s = Session.new(write_obj(Path(self.tmp.name)), resolution=1024, evaluator=ev())
        self.ts = self.s.project.texture_sets[0]
        self.layer = new_layer(self.ts, "paint", "Layer 1")
        self.ts.layers.append(self.layer)
        _, before = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(size=10))  # what was there
        key = (self.layer.uid, "basecolor", 1001)
        store = self.s.store
        tile = TileTargets(1001, self.s.uv_maps(self.ts, 1001), [Target(key, store.entries[key].pages,
                                                                        (0.0, 1.0, 0.0), False)])
        st = Stroke(self.s.brush_engine(), [tile], self.view(), self.meshes(), self.s.material_index(self.ts),
                    BrushSettings(size=5), self.s.bbox_diagonal())
        self.assertGreater(st.add([(60, 110, 1.0), (66, 110, 1.0)]), 0)
        saved = st.end()
        store.touched(*key)
        self.assertFalse(np.array_equal(store.read(*key), before))  # the stroke painted
        self.assertTrue(0 < len(saved.layers[key]) < 16)  # a few blocks saved, not the whole texture
        snap = store.stroke_undo(key, saved)
        saved.destroy()
        store.apply_undo(key, snap, after=False)
        np.testing.assert_array_equal(store.read(*key), before)

    def test_undo_of_a_stroke_entering_painted_pages(self):
        """A stroke starting where nothing is painted (its first pass saves nothing) and then reaching painted pages
        one at a time: its undo gives back what they held."""
        self.s = Session.new(write_obj(Path(self.tmp.name)), resolution=1024, evaluator=ev())
        self.ts = self.s.project.texture_sets[0]
        self.layer = new_layer(self.ts, "paint", "Layer 1")
        self.ts.layers.append(self.layer)
        _, before = self.stroke([(80, 128, 1.0), (100, 128, 1.0)], BrushSettings(size=10))
        key = (self.layer.uid, "basecolor", 1001)
        store = self.s.store
        p = store.entries[key].pages
        st = Stroke(self.s.brush_engine(), [TileTargets(1001, self.s.uv_maps(self.ts, 1001),
                                                        [Target(key, p, (0.0, 1.0, 0.0), False)])],
                    self.view(), self.meshes(), self.s.material_index(self.ts), BrushSettings(size=5),
                    self.s.bbox_diagonal())
        for k in range(21):
            st.add([(110 - k, 170 - 2.1 * k, 1.0)])
        saved = st.end()
        store.touched(*key)
        self.assertIn(None, saved.layers[key].values())  # pages added
        self.assertGreater(sum(v is not None for v in saved.layers[key].values()), 1)  # and painted ones saved
        snap = store.stroke_undo(key, saved)
        saved.destroy()
        store.apply_undo(key, snap, after=False)
        np.testing.assert_array_equal(store.read(*key), before)

    def test_painting_is_stored_by_pages(self):
        """At 1024 (pages of 256: 16 per tile) a small stroke on an empty entry adds only the pages it reaches; its
        undo empties them again (nothing painted, layers left for reuse) and the redo brings them back."""
        self.s = Session.new(write_obj(Path(self.tmp.name)), resolution=1024, evaluator=ev())
        self.ts = self.s.project.texture_sets[0]
        self.layer = new_layer(self.ts, "paint", "Layer 1")
        self.ts.layers.append(self.layer)
        key = (self.layer.uid, "basecolor", 1001)
        store = self.s.store
        p = store.ensure(*key, self.ts.size, storage_for("sRGB8")).pages
        self.assertEqual(p.painted(), [])
        st = Stroke(self.s.brush_engine(), [TileTargets(1001, self.s.uv_maps(self.ts, 1001),
                                                        [Target(key, p, (0.0, 1.0, 0.0), False)])],
                    self.view(), self.meshes(), self.s.material_index(self.ts), BrushSettings(size=5),
                    self.s.bbox_diagonal())
        self.assertGreater(st.add([(60, 110, 1.0), (66, 110, 1.0)]), 0)
        saved = st.end()
        store.touched(*key)
        after = store.read(*key)
        pages = p.painted()
        self.assertTrue(0 < len(pages) < 16)
        self.assertLess(p.nbytes, 1024 * 1024 * 4)  # less than the whole texture
        for b in range(16):  # outside its pages nothing is painted
            x, y, w, h = p.box(b)
            if b not in pages:
                self.assertFalse(after[y:y + h, x:x + w, 3].any())
        snap = store.stroke_undo(key, saved)
        saved.destroy()
        store.apply_undo(key, snap, after=False)
        self.assertEqual(p.painted(), [])
        self.assertFalse(store.read(*key).any())
        store.apply_undo(key, snap, after=True)
        self.assertEqual(sorted(p.painted()), sorted(pages))
        np.testing.assert_array_equal(store.read(*key), after)

    def test_stroke_opacity_caps(self):
        _, a = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(stroke_opacity=50))
        self.assertLessEqual(a[..., 3].max(), 0.5 + 1 / 255)
        self.assertGreater(a[..., 3].max(), 0.45)

    def test_stamps_blending_lighten(self):
        """Low-flow stamps build up to near full coverage; with Lighten overlaps keep the strongest stamp."""
        pts = [(40, 128, 1.0), (100, 128, 1.0)]
        _, normal = self.stroke(pts, BrushSettings(flow=30, spacing=5))
        self.layer.uid = "lighten"
        _, lighten = self.stroke(pts, BrushSettings(flow=30, spacing=5, stamps_blending="Lighten"))
        self.assertGreater(normal[..., 3].max(), 0.9)
        self.assertLessEqual(lighten[..., 3].max(), 0.3 + 1 / 255)
        self.assertGreater(lighten[..., 3].max(), 0.25)

    def test_eraser(self):
        _, before = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(size=40))
        core = before[..., 3] > 0.99
        self.assertGreater(core.sum(), 10)
        _, a = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(size=40), erase=True)
        self.assertLess(a[..., 3][core].max(), 0.05)  # erased where the stroke was solid
        self.assertLessEqual(a[..., 3].max(), before[..., 3].max())

    def test_spacing(self):
        few, _ = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(spacing=100))
        many, _ = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(spacing=10))
        self.assertGreater(many, few * 5)

    def test_mirror_symmetry(self):
        """Mirror symmetry across Z = 0: a stroke on one half also paints the other half."""
        pts = [(40, 95, 1.0), (100, 95, 1.0)]
        _, a = self.stroke(pts, BrushSettings(size=10))
        rows = (a[..., 3] > 0.5).any(1)
        h = len(rows)
        self.assertTrue(rows[: h // 2].any() != rows[h // 2:].any())  # one half only
        self.layer.uid = "sym"
        from trafilatura3d.paint.brush import symmetry_copies
        _, b = self.stroke(pts, BrushSettings(size=10), symmetry=symmetry_copies("Mirror", 2, 0.0))
        rows = (b[..., 3] > 0.5).any(1)
        self.assertTrue(rows[: h // 2].any() and rows[h // 2:].any())  # both halves
        np.testing.assert_allclose(b[..., 3], b[::-1, :, 3], atol=0.08)  # mirrored in V
        # Radial around Y through the painted quad's center (Texture Set 0 is "Right", x 0..1) with 2 copies: the
        # half turn lands on the other half
        self.layer.uid = "radial"
        _, c = self.stroke(pts, BrushSettings(size=10), symmetry=symmetry_copies("Radial", 1, [0.5, 0.0, 0.0], 2))
        rows = (c[..., 3] > 0.5).any(1)
        self.assertTrue(rows[: h // 2].any() and rows[h // 2:].any())
        self.assertGreater((c[..., 3] > 0.5).sum(), (a[..., 3] > 0.5).sum() * 1.5)
        self.assertEqual(len(symmetry_copies("Radial", 1, [0, 0, 0], 6, "U")), 5)
        self.assertEqual([f for *_r, f in symmetry_copies("Radial", 1, [0, 0, 0], 4, "U")], [1, 0, 1])

    def test_texture_space(self):
        """Painting in the 2D view: stamps in texels, only inside the Texture Set's UV islands."""
        e = self.s.store.ensure(self.layer.uid, "basecolor", 1001, self.ts.size, storage_for("sRGB8"))
        tile = TileTargets(1001, self.s.uv_maps(self.ts, 1001), [Target((self.layer.uid, "basecolor", 1001), e.pages,
                                                                        (1.0, 0.0, 0.0))])
        st = Stroke(self.s.brush_engine(), [tile], None, [], self.s.material_index(self.ts), BrushSettings(), 1.0)
        n = st.add_texture([(20.0, 64.0, 1.0), (100.0, 64.0, 1.0)], 6.0, self.ts.size)
        st.end()
        self.assertGreater(n, 10)
        a = self.s.store.read(self.layer.uid, "basecolor", 1001)
        self.assertGreater(a[64, 60, 3], 0.9)
        self.assertEqual(a[20, 60, 3], 0.0)
        rows = np.where((a[..., 3] > 0.5).any(1))[0]
        self.assertLessEqual(rows.max() - rows.min(), 14)  # a band about the stamp's diameter

    def test_backface_culled(self):
        """Seen from below the plane faces away: with 90° backface culling nothing is painted."""
        _, a = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], view=self.view((0.0, -3.0, 0.001)))
        self.assertEqual((a[..., 3] > 0).sum(), 0)
        for align in ("Camera", "Tangent | Planar"):
            self.layer.uid = f"x{align}"
            _, a = self.stroke([(40, 128, 1.0), (100, 128, 1.0)], BrushSettings(alignment=align))
            self.assertGreater((a[..., 3] > 0.5).mean(), 0.01, align)


class TestFiles(unittest.TestCase):
    def test_new_project_texture_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = Session.new(write_obj(Path(tmp)), resolution=256, evaluator=ev())
            self.assertEqual(sorted(ts.name for ts in s.project.texture_sets), ["Left", "Right"])
            self.assertEqual(s.project.texture_sets[0].size, [256, 256])

    def test_import_baked_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            imageio.write(tmp / "Left_ambient_occlusion.png", np.full((32, 32, 3), 128, np.uint8))
            s = Session.new(write_obj(tmp), resolution=128, baked_maps=[tmp / "Left_ambient_occlusion.png"],
                            evaluator=ev())
            self.assertIn("ambient_occlusion", s.project.texture_set("Left").mesh_maps)
            self.assertNotIn("ambient_occlusion", s.project.texture_set("Right").mesh_maps)

    def test_save_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s = Session.new(write_obj(tmp), resolution=64, evaluator=ev())
            ts = s.project.texture_sets[0]
            layer = new_layer(ts, "paint", "Layer 1")
            ts.layers.append(layer)
            img = np.random.default_rng(1).random((64, 64, 4)).astype(np.float32)
            s.store.write(layer.uid, "roughness", 1001, img, storage_for("L8"))
            mask = np.zeros((64, 64, 4), np.float32)
            mask[:20, 40:] = (0.6, 0.6, 0.6, 0.3)  # a corner painted: a page of it
            s.store.write(layer.uid, MASK, 1001, mask, storage_for(None))
            s.save(tmp / "p.trafilatura3dpaint")
            t = Session.open(tmp / "p.trafilatura3dpaint", evaluator=ev())
            self.assertEqual(t.project.to_dict(), s.project.to_dict())
            for key in ("roughness", MASK):  # the pages as they were, exactly
                np.testing.assert_array_equal(t.store.read(layer.uid, key, 1001), s.store.read(layer.uid, key, 1001))
                self.assertEqual(t.store.get(layer.uid, key, 1001).pages.painted(),
                                 s.store.get(layer.uid, key, 1001).pages.painted())
            self.assertEqual(len(t.objects), len(s.objects))
            np.testing.assert_allclose(t.objects[0].pos, s.objects[0].pos)

    def test_new_project_template(self):
        """New Project > Template (shaders.TEMPLATES): the Texture Sets get the channels its shader reads (Opacity
        only when it sees through), the project its shader; an unknown one is refused; a saved project naming a
        shader that no longer exists opens with the default one, reported."""
        import json
        import zipfile
        from trafilatura3d import shaders
        from trafilatura3d.paint.channels import render_slots
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s = Session.new(write_obj(tmp), resolution=64, evaluator=ev(), template="Specular glossiness - Alpha blend")
            t = shaders.TEMPLATES["Specular glossiness - Alpha blend"]
            self.assertEqual(s.project.shader.shader, t.name)
            for ts in s.project.texture_sets:
                self.assertEqual(ts.channel_ids(), [u.lower() for u in t.channels])
            self.assertIn("opacity", s.project.texture_sets[0].channel_ids())
            self.assertEqual(set(render_slots(t.shader)), set(s.project.texture_sets[0].channel_ids()))
            opaque = Session.new(write_obj(tmp), resolution=64, evaluator=ev(), template="OpenPBR")
            self.assertNotIn("opacity", opaque.project.texture_sets[0].channel_ids())
            self.assertIn("specularroughness", opaque.project.texture_sets[0].channel_ids())
            with self.assertRaises(ProjectError):
                Session.new(write_obj(tmp), resolution=64, evaluator=ev(), template="nothing")
            s.save(tmp / "p.trafilatura3dpaint")
            with zipfile.ZipFile(tmp / "p.trafilatura3dpaint") as z:
                files = {n: z.read(n) for n in z.namelist()}
            data = json.loads(files["project.json"])
            data["shader"]["shader"] = "gone"
            files["project.json"] = json.dumps(data).encode()
            with zipfile.ZipFile(tmp / "old.trafilatura3dpaint", "w") as z:
                for n, b in files.items():
                    z.writestr(n, b)
            old = Session.open(tmp / "old.trafilatura3dpaint", evaluator=ev())
            self.assertEqual(old.project.shader.shader, shaders.DEFAULT_TEMPLATE)
            self.assertTrue(any("'gone'" in w for w in old.load_warnings))

    def test_open_rejects_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.trafilatura3dpaint"
            bad.write_text("not a zip", encoding="utf-8")
            with self.assertRaises(ProjectError):
                Session.open(bad, evaluator=ev())


class TestBaking(unittest.TestCase):
    def test_bake_mesh_maps(self):
        """Low poly baked onto itself: flat quads give a flat normal and open ambient occlusion; the maps land
        in the Texture Set's mesh map slots and the stack reads them."""
        from trafilatura3d.paint import baking
        with tempfile.TemporaryDirectory() as tmp:
            s = Session.new(write_obj(Path(tmp)), resolution=64, evaluator=ev())
            ts = s.project.texture_set("Left")
            cfg = baking.config(ts)
            cfg["common"]["size"] = [64, 64]
            for slot in cfg["bakers"]:
                cfg["bakers"][slot]["enabled"] = slot in ("normal", "ambient_occlusion", "opacity")
            baking.set_config(ts, cfg)
            out = baking.bake_texture_set(s, ts)
            self.assertEqual(set(out), {"normal", "ambient_occlusion", "opacity"})
            self.assertEqual(ts.mesh_maps["normal"], out["normal"])
            n = s.resources.images[out["normal"]["1001"]]
            np.testing.assert_allclose(n[32, 32, :3], [0.5, 0.5, 1.0], atol=0.02)
            self.assertGreater(s.resources.images[out["ambient_occlusion"]["1001"]][32, 32, 0], 0.9)
            self.assertNotIn("normal", s.project.texture_set("Right").mesh_maps)
            a = read(s.compiler(1001), s.compiler(1001).normal(ts))
            np.testing.assert_allclose(a[32, 32, :3], [0.5, 0.5, 1.0], atol=0.02)
            s.save(Path(tmp) / "b.trafilatura3dpaint")  # the baked maps and the configuration are saved
            t = Session.open(Path(tmp) / "b.trafilatura3dpaint", evaluator=ev())
            self.assertEqual(t.project.texture_set("Left").baking, ts.baking)
            self.assertIn(out["normal"]["1001"], t.resources.images)
            np.testing.assert_allclose(t.fit[1], s.fit[1])


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.s = Session.new(write_obj(self.dir), resolution=64, evaluator=ev())
        ts = self.s.project.texture_set("Left")
        fill = new_layer(ts, "fill", "Fill")
        fill.material["basecolor"].color = [0.8, 0.4, 0.2]
        fill.material["metallic"].color = [1.0] * 3
        fill.material["roughness"].color = [0.25] * 3
        ts.layers.append(fill)

    def tearDown(self):
        self.tmp.cleanup()

    def test_document_channels(self):
        out = self.dir / "out"
        files = Exporter(self.s).run(PREDEFINED[0], ExportSettings(output_dir=str(out)))
        names = sorted(p.name for p in files)
        self.assertIn("two_Left_BaseColor.png", names)
        self.assertIn("two_Left_Normal.png", names)
        self.assertIn("two_Right_AO.png", names)
        img = imageio.read(out / "two_Left_Roughness.png")
        self.assertEqual(img.shape[:2], (64, 64))
        self.assertAlmostEqual(img[0, 0, 0] / 255, 0.25, delta=2 / 255)

    def test_packing_and_converted(self):
        t = Template("Packed", [OutputMap("$textureSet_ORM", "R+G+B", {
            "R": {"input": "converted:Glossiness", "component": "R"},
            "G": {"input": "channel:roughness", "component": "Grayscale"},
            "B": {"input": "channel:metallic", "component": "R"}})])
        out = self.dir / "packed"
        Exporter(self.s).run(t, ExportSettings(output_dir=str(out), texture_sets=["Left"]))
        img = imageio.read(out / "Left_ORM.png")
        np.testing.assert_allclose(img[3, 3, :3] / 255, [0.75, 0.25, 1.0], atol=2 / 255)
        self.assertFalse((out / "Right_ORM.png").exists())

    def test_dithering_and_depths(self):
        x = np.full((8, 8, 1), 0.5, np.float32)
        self.assertEqual(_encode(x, "png", "16").dtype, np.uint16)
        self.assertEqual(_encode(x, "exr", "16 (floating)").dtype, np.float16)
        d = _encode(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8, 1), "png", "8 + dithering")
        self.assertEqual(d.dtype, np.uint8)


class TestGeometryMask(unittest.TestCase):
    def test_mesh_names(self):
        """Two meshes on one material (A on the left half of the UVs, B on the right): a fill layer excluding A
        lays nothing on A's texels; the mask's list is saved with the layer."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        (folder / "m.mtl").write_text("newmtl M\nKd 1 1 1\n", encoding="utf-8")
        lines = ["mtllib m.mtl"]
        for i, (name, x0, u0) in enumerate((("A", -1.0, 0.0), ("B", 0.0, 0.5))):
            lines += [f"o {name}", f"v {x0} 0 -0.5", f"v {x0 + 1} 0 -0.5", f"v {x0 + 1} 0 0.5", f"v {x0} 0 0.5",
                      f"vt {u0} 1", f"vt {u0 + 0.5} 1", f"vt {u0 + 0.5} 0", f"vt {u0} 0", "vn 0 1 0", "usemtl M"]
            b = 4 * i + 1
            lines += [f"f {b}/{b}/{i + 1} {b + 3}/{b + 3}/{i + 1} {b + 2}/{b + 2}/{i + 1}",
                      f"f {b}/{b}/{i + 1} {b + 2}/{b + 2}/{i + 1} {b + 1}/{b + 1}/{i + 1}"]
        (folder / "ab.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
        s = Session.new(folder / "ab.obj", resolution=64, evaluator=ev())
        ts = s.project.texture_sets[0]
        self.assertEqual(s.mesh_names(ts), ["A", "B"])
        layer = new_layer(ts, "fill", "F")
        ts.layers.append(layer)
        cov = read(s.compiler(1001), s.compiler(1001).channel(ts, "basecolor"))[..., 3]
        self.assertGreater(cov[32, 8], 0.9)
        self.assertGreater(cov[32, 56], 0.9)
        layer.geometry_excluded = ["A"]
        cov = read(s.compiler(1001), s.compiler(1001).channel(ts, "basecolor"))[..., 3]
        self.assertLess(cov[32, 8], 0.05)  # A (left half): nothing
        self.assertGreater(cov[32, 56], 0.9)  # B
        from trafilatura3d.paint.model import Layer
        self.assertEqual(Layer.from_dict(layer.to_dict()).geometry_excluded, ["A"])


class TestPaths(unittest.TestCase):
    def session(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = Session.new(write_obj(Path(tmp.name)), resolution=64, evaluator=ev())
        ts = s.project.texture_set("Left")
        layer = new_layer(ts, "paint", "P")
        ts.layers.append(layer)
        return s, ts, layer

    def coverage(self, s, ts, channel="basecolor"):
        return read(s.compiler(1001), s.compiler(1001).channel(ts, channel))[..., 3]

    def test_paint_filled_erase_ribbon(self):
        """A line across the Left quad (x -1..0 at z 0) paints a band along v = 0.5; a closed square fills its
        inside; Erase along the same line removes the band from the fill; a Ribbon lays a band of its width; the
        layer's data keep the paths."""
        from trafilatura3d.paint.model import Layer, PaintPath
        s, ts, layer = self.session()
        up = [0.0, 1.0, 0.0, 1.0, 0.0]
        line = [[-0.9, 0.0, 0.0, *up], [-0.1, 0.0, 0.0, *up]]
        p = PaintPath("p1", "Path 1", points=line, brush={"size": 10.0, "spacing": 10.0}, values={"basecolor": [1, 0, 0]})
        layer.paths.append(p)
        cov = self.coverage(s, ts)
        self.assertGreater(cov[32, 10:54].min(), 0.5)  # along the line
        self.assertLess(cov[4, 32], 0.05)  # far from it
        square = [[-0.8, 0.0, -0.3, *up], [-0.2, 0.0, -0.3, *up], [-0.2, 0.0, 0.3, *up], [-0.8, 0.0, 0.3, *up]]
        for pt in square:
            pt[7] = 1.0  # corners: straight sides
        layer.paths[0] = PaintPath("p2", "Fill", mode="Filled path", points=square, closed=True,
                                   values={"basecolor": [0, 1, 0]})
        cov = self.coverage(s, ts)
        self.assertGreater(cov[32, 32], 0.9)
        self.assertLess(cov[4, 4], 0.05)
        layer.paths.append(PaintPath("p3", "Erase", mode="Erase along path", points=line,
                                     brush={"size": 5.0, "spacing": 10.0}))
        cov = self.coverage(s, ts)
        self.assertLess(cov[32, 32], 0.2)  # erased on the line
        self.assertGreater(cov[24, 32], 0.9)  # the fill elsewhere
        layer.paths[:] = [PaintPath("p4", "Ribbon", mode="Ribbon path", points=line, values={"basecolor": [0, 0, 1]},
                                    ribbon={"stroke_width": 20.0})]
        cov = self.coverage(s, ts)
        self.assertGreater(cov[32, 32], 0.9)
        self.assertLess(cov[4, 32], 0.05)
        d = layer.to_dict()
        self.assertEqual(d["paths"][0]["mode"], "Ribbon path")
        self.assertEqual(Layer.from_dict(d).paths[0].points, line)

    def test_tangents_and_symmetry(self):
        """Custom handles bend the curve (the middle moves off the chord); Symmetry across Z paints the mirror
        image too; the automatic handles keep the Catmull-Rom curve."""
        from trafilatura3d.paint import paths
        from trafilatura3d.paint.model import PaintPath
        up = [0.0, 1.0, 0.0, 1.0, 0.0]
        line = [[-0.9, 0.0, 0.2, *up], [-0.1, 0.0, 0.2, *up]]
        straight = paths.curve(line, False, 0.01)
        self.assertLess(np.abs(straight.pos[:, 2] - 0.2).max(), 1e-6)
        bent = [line[0] + [0, 0, 0, 0.0, 0.0, 0.3, 1.0], line[1] + [0.0, 0.0, 0.3, 0, 0, 0, 1.0]]
        c = paths.curve(bent, False, 0.01)
        self.assertGreater(c.pos[len(c.pos) // 2, 2], 0.35)  # pulled toward +Z by both handles
        self.assertEqual(len(paths.transformed(bent, np.eye(3), np.zeros(3))[0]), 15)
        s, ts, layer = self.session()
        layer.paths.append(PaintPath("p", "Sym", points=line, brush={"size": 6.0, "spacing": 10.0},
                                     symmetry={"type": "Mirror", "axis": 2, "position": 0.0},
                                     values={"basecolor": [1, 0, 0]}))
        cov = self.coverage(s, ts)
        # z 0.2 -> v 0.3 (row 19), its mirror z -0.2 -> v 0.7 (row 44): both painted
        self.assertGreater(cov[19, 32], 0.5)
        self.assertGreater(cov[44, 32], 0.5)

    def test_smudge_along_path(self):
        """Smudge along path drags the layer's painting: a red half smudged across the line's middle."""
        from trafilatura3d.paint.model import PaintPath
        s, ts, layer = self.session()
        e = s.store.ensure(layer.uid, "basecolor", 1001, ts.size, storage_for(ts.format_of("basecolor")))
        a = np.zeros((64, 64, 4), np.float32)
        a[:, :32] = (1, 0, 0, 1)  # the left half painted
        e.pages.write(a)
        s.store.touched(layer.uid, "basecolor", 1001)
        before = self.coverage(s, ts)
        up = [0.0, 1.0, 0.0, 1.0, 0.0]
        layer.paths.append(PaintPath("p", "Smudge", mode="Smudge along path",
                                     points=[[-0.6, 0.0, 0.0, *up], [-0.2, 0.0, 0.0, *up]],
                                     brush={"size": 10.0, "spacing": 5.0}))
        after = self.coverage(s, ts)
        self.assertGreater(after[32, 34:44].mean(), before[32, 34:44].mean() + 0.2)  # color dragged right


class TestProjections(unittest.TestCase):
    def fill(self, **projection):
        """The Left quad (XZ plane, x -1..0, u along x) filled with a ramp image through a projection."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        s = Session.new(write_obj(Path(tmp.name)), resolution=64, evaluator=ev())
        ts = s.project.texture_set("Left")
        ramp = np.zeros((32, 32, 4), np.float32)
        ramp[..., :3] = np.linspace(0, 1, 32)[None, :, None]
        ramp[..., 3] = 1
        s.resources.add("ramp", ramp)
        layer = new_layer(ts, "fill", "F")
        layer.material["roughness"].image = "ramp"
        for k, v in projection.items():
            setattr(layer.projection, k, v)
        layer.validate()
        ts.layers.append(layer)
        return read(s.compiler(1001), s.compiler(1001).channel(ts, "roughness"))

    def test_planar_crop(self):
        """Planar from above (Z rotated -90 degrees about X onto +Y), a quarter of the scene's largest side,
        cropped: only the square around the scene center (x, z in -0.25..0.25) is covered; from below the faces
        up are culled."""
        top = dict(mode="Planar projection", rotation3d=[-90.0, 0.0, 0.0], scale3d=[0.25, 0.25, 0.25])
        a = self.fill(**top, shape_crop="Project cropped to shape")
        self.assertGreater(a[32, 60, 3], 0.9)  # u 0.94: x -0.06, z 0
        self.assertLess(a[32, 20, 3], 0.1)  # u 0.3: x -0.7
        self.assertLess(a[4, 60, 3], 0.1)  # v 0.06: z outside
        self.assertGreater(self.fill(**top)[32, 20, 3], 0.9)  # extends outside the shape
        below = self.fill(mode="Planar projection", rotation3d=[90.0, 0.0, 0.0])
        self.assertLess(below[32, 32, 3], 0.1)
        self.assertGreater(self.fill(mode="Planar projection", rotation3d=[90.0, 0.0, 0.0],
                                     backface_culling=False)[32, 32, 3], 0.9)

    def test_warp(self):
        """The flat warp grid projects like Planar (cropped square from above); moving a grid point sideways
        shifts the image under it; beyond Projection depth nothing is laid."""
        from trafilatura3d.paint.model import warp_grid
        top = dict(mode="Warp projection", rotation3d=[-90.0, 0.0, 0.0], scale3d=[0.25, 0.25, 0.25],
                   shape_crop="Project cropped to shape")
        flat = self.fill(**top)
        planar = self.fill(**{**top, "mode": "Planar projection"})
        np.testing.assert_allclose(flat[8:56, 8:56, 3], planar[8:56, 8:56, 3], atol=0.05)
        pts = warp_grid(3, 3)
        for p in pts:
            if abs(p[0]) < 0.4 and abs(p[1]) < 0.4:  # the inner points pushed along +X
                p[0] += 0.15
        bent = self.fill(**top, warp_points=pts)
        self.assertFalse(np.allclose(bent[..., 0], flat[..., 0], atol=0.02))
        far = self.fill(**top, warp_depth=0.0001, offset3d=[0.0, 0.3, 0.0])  # the grid above the surface
        self.assertLess(far[..., 3].max(), 0.1)

    def test_fill_symmetry(self):
        """A cropped planar square from above off the center (x -0.7): Mirror across X at -0.5 adds its copy at
        x -0.3; Radial around Y with 2 copies through the quad's center does the same."""
        top = dict(mode="Planar projection", rotation3d=[-90.0, 0.0, 0.0], scale3d=[0.15, 0.15, 0.15],
                   shape_crop="Project cropped to shape", offset3d=[-0.35, 0.0, 0.0])
        alone = self.fill(**top)
        self.assertGreater(alone[32, 19, 3], 0.9)  # x -0.7: u 0.3
        self.assertLess(alone[32, 45, 3], 0.1)  # x -0.3: u 0.7
        mirror = self.fill(**top, symmetry={"type": "Mirror", "axis": 0, "position": -0.5})
        self.assertGreater(mirror[32, 19, 3], 0.9)
        self.assertGreater(mirror[32, 45, 3], 0.9)
        radial = self.fill(**top, symmetry={"type": "Radial", "axis": 1, "position": [-0.5, 0.0, 0.0], "copies": 2})
        self.assertGreater(radial[32, 45, 3], 0.9)

    def test_spherical_and_cylindrical(self):
        """Spherical: the ramp follows the longitude around the center (the quad, from above, spans a range of
        angles); Cylindrical with crop and a small Angle covers only a wedge."""
        a = self.fill(mode="Spherical projection")
        col = a[2:62, 32, 0]  # x -0.5, z from -0.5 to 0.5: the longitude turns
        self.assertGreater(abs(float(col[-1] - col[0])), 0.05)
        b = self.fill(mode="Cylindrical projection", angle=30.0, shape_crop="Project cropped to shape",
                      backface_culling=False)
        self.assertGreater((b[..., 3] < 0.1).mean(), 0.5)  # most of the quad outside the wedge
        self.assertTrue((b[..., 3] > 0.9).any())

    def test_triplanar_follows_world(self):
        """Tri-planar on faces up: the ramp runs along world X as the plane spans it (half the scene: x -1..0 is
        the first half of the bounding box)."""
        a = self.fill(mode="Tri-planar projection", hardness=1.0)
        row = a[32, 2:62, 0]
        self.assertGreater(abs(float(row[-1] - row[0])), 0.3)
        self.assertTrue(np.all(np.diff(row) * np.sign(row[-1] - row[0]) >= -1e-3))  # monotonic
        self.assertLess(abs(float(row[-1] - row[0])), 0.7)  # the quad is half the projection, not all of it


def write_udim_obj(folder: Path) -> Path:
    """Material "Body": two quads in the UDIM tiles 1001 and 1002; material "Head": one quad in 1011."""
    (folder / "udim.mtl").write_text("newmtl Body\nKd 0.8 0.2 0.2\nnewmtl Head\nKd 0.2 0.2 0.8\n", encoding="utf-8")
    lines = ["mtllib udim.mtl"]
    for i, (mat, (du, dv)) in enumerate((("Body", (0, 0)), ("Body", (1, 0)), ("Head", (0, 1)))):
        x0 = float(i)
        lines += [f"v {x0} 0 -0.5", f"v {x0 + 1} 0 -0.5", f"v {x0 + 1} 0 0.5", f"v {x0} 0 0.5",
                  f"vt {du + 0.1} {dv + 0.9}", f"vt {du + 0.9} {dv + 0.9}", f"vt {du + 0.9} {dv + 0.1}",
                  f"vt {du + 0.1} {dv + 0.1}", "vn 0 1 0", f"usemtl {mat}"]
        b = 4 * i + 1
        lines.append(f"f {b}/{b}/{i + 1} {b + 3}/{b + 3}/{i + 1} {b + 2}/{b + 2}/{i + 1}")
        lines.append(f"f {b}/{b}/{i + 1} {b + 2}/{b + 2}/{i + 1} {b + 1}/{b + 1}/{i + 1}")
    path = folder / "udim.obj"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestUVTiles(unittest.TestCase):
    def test_new_project_keeps_each_materials_tiles(self):
        from trafilatura3d.paint.model import UV_TILE_MODES
        with tempfile.TemporaryDirectory() as tmp:
            mesh = write_udim_obj(Path(tmp))
            plain = Session.new(mesh, resolution=64, evaluator=ev())
            self.assertEqual([ts.tiles for ts in plain.project.texture_sets], [[1001], [1001]])  # the 0-1 range
            s = Session.new(mesh, resolution=64, evaluator=ev(), uv_tiles=UV_TILE_MODES[0])
            self.assertEqual({ts.name: ts.tiles for ts in s.project.texture_sets}, {"Body": [1001, 1002], "Head": [1011]})
            s.save(Path(tmp) / "p.trafilatura3dpaint")
            back = Session.open(Path(tmp) / "p.trafilatura3dpaint", evaluator=ev())
            self.assertEqual(back.project.uv_tiles, UV_TILE_MODES[0])
            self.assertEqual(back.project.texture_set("Body").tiles, [1001, 1002])
            with self.assertRaises(ProjectError):  # the legacy mode is not offered
                Session.new(mesh, resolution=64, evaluator=ev(), uv_tiles=UV_TILE_MODES[1])
        p = Project()
        p.texture_sets.append(new_texture_set("m", 64))
        p.texture_sets[0].tiles = [1001, 1002]
        with self.assertRaises(ProjectError):  # without the UV Tile workflow only 1001
            p.validate()
        p.uv_tiles = UV_TILE_MODES[0]
        p.validate()
        p.texture_sets[0].tiles = [1002, 1001]
        with self.assertRaises(ProjectError):  # sorted
            p.validate()
        p.texture_sets[0].tiles = [1001, 1010]
        with self.assertRaises(ProjectError):  # 1010 is no UDIM (U goes 1 to 9 then the next row)
            p.validate()
        p.texture_sets[0].tiles = [1001, 1011]
        p.validate()

    def test_each_tile_is_a_texture(self):
        """A UV Tile project: each tile has its own mesh in texture space, painting, baked maps and exported files."""
        from trafilatura3d.paint import baking
        from trafilatura3d.paint.model import UV_TILE_MODES
        from trafilatura3d.paint.paint import storage_for
        from trafilatura3d.paint.uvmaps import read_mask
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s = Session.new(write_udim_obj(tmp), resolution=64, evaluator=ev(), uv_tiles=UV_TILE_MODES[0])
            body = s.project.texture_set("Body")
            xs = []
            for udim in body.tiles:  # the quad of the tile fills 0.1-0.9 of its texture
                maps = s.uv_maps(body, udim)
                self.assertAlmostEqual(read_mask(s.ev.device, maps).mean(), 0.64, delta=0.05)
                pos = _read(s.ev.device, maps.position)
                xs.append(pos[..., 0][pos[..., 3] > 0].mean())
            self.assertLess(xs[0], xs[1])  # 1001: the first quad, 1002: the second, to its right
            layer = new_layer(body, "paint", "P")
            body.layers.append(layer)
            for udim, v in ((1001, 0.25), (1002, 0.75)):
                img = np.full((64, 64, 4), v, np.float32)
                img[..., 3] = 1.0
                s.store.write(layer.uid, "roughness", udim, img, storage_for("L8"))
            out = baking.bake_texture_set(s, body, slots=["normal"])
            self.assertEqual(out, {"normal": {"1001": "Body_normal.1001", "1002": "Body_normal.1002"}})
            s.save(tmp / "p.trafilatura3dpaint")
            t = Session.open(tmp / "p.trafilatura3dpaint", evaluator=ev())
            tb = t.project.texture_set("Body")
            for udim, v in ((1001, 0.25), (1002, 0.75)):
                r = read(t.compiler(udim), t.compiler(udim).channel(tb, "roughness"))
                self.assertAlmostEqual(float(r[32, 32, 0]), v, delta=1 / 255)
            files = Exporter(t).run(PREDEFINED[0], ExportSettings(output_dir=str(tmp / "out")))
            names = {f.name for f in files}
            for name in ("two_Body_Roughness.1001.png", "two_Body_Roughness.1002.png", "two_Head_Roughness.1011.png"):
                self.assertIn(name.replace("two", "udim"), names)

    def tile_session(self, tmp):
        from trafilatura3d.paint.model import UV_TILE_MODES
        s = Session.new(write_udim_obj(tmp), resolution=64, evaluator=ev(), uv_tiles=UV_TILE_MODES[0])
        return s, s.project.texture_set("Body")

    def test_geometry_mask_skips_tiles(self):
        """Geometry mask, Masking type UV Tiles: the layer does nothing in the excluded tiles."""
        with tempfile.TemporaryDirectory() as tmp:
            s, body = self.tile_session(Path(tmp))
            layer = new_layer(body, "fill", "F")
            layer.material["roughness"].color = [0.9] * 3
            layer.geometry_type, layer.geometry_tiles = "UV Tiles", [1002]
            body.layers.append(layer)
            s.project.validate()
            r1 = read(s.compiler(1001), s.compiler(1001).channel(body, "roughness"))
            r2 = read(s.compiler(1002), s.compiler(1002).channel(body, "roughness"))
            self.assertAlmostEqual(float(r1[0, 0, 0]), 0.9, delta=1 / 255)
            self.assertNotAlmostEqual(float(r2[0, 0, 0]), 0.9, delta=0.05)

    def test_image_sequence_match_per_tile(self):
        """Importing one file of a UDIM sequence brings the sequence as one resource; Fill (match per UV Tile) gives
        each tile its image, the UV projection the first one everywhere."""
        from trafilatura3d.paint.model import MATCH_PER_TILE
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s, body = self.tile_session(tmp)
            for udim, v in ((1001, 0.2), (1002, 0.8)):
                imageio.write(tmp / f"grunge_{udim}.png", np.full((8, 8, 3), int(v * 255), np.uint8))
            imageio.write(tmp / "grunge_0032.png", np.zeros((8, 8, 3), np.uint8))  # not a UDIM: its own image
            name = s.import_image(tmp / "grunge_1002.png")
            self.assertEqual(name, "grunge")
            self.assertEqual(s.resources.sequence("grunge"), {1001: "grunge.1001", 1002: "grunge.1002"})
            self.assertEqual([n for n, _i in s.resources.listing()], ["grunge"])
            layer = new_layer(body, "fill", "F")
            layer.material["roughness"].image = "grunge"
            layer.projection.mode = MATCH_PER_TILE
            body.layers.append(layer)
            s.project.validate()
            for udim, v in ((1001, 0.2), (1002, 0.8)):
                r = read(s.compiler(udim), s.compiler(udim).channel(body, "roughness"))
                self.assertAlmostEqual(float(r[32, 32, 0]), v, delta=0.02)
            layer.projection.mode = "UV projection"
            r = read(s.compiler(1002), s.compiler(1002).channel(body, "roughness"))
            self.assertAlmostEqual(float(r[32, 32, 0]), 0.2, delta=0.02)  # the first image
            self.assertEqual(s.used_resources(), {"grunge", "grunge.1001", "grunge.1002"})

    def test_tile_resolution(self):
        """A tile with its own resolution: its mesh in texture space, its stack and its exported files."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            s, body = self.tile_session(tmp)
            body.tile_sizes = {"1002": [128, 128]}
            s.project.validate()
            self.assertEqual(s.uv_maps(body, 1001).size, (64, 64))
            self.assertEqual(s.uv_maps(body, 1002).size, (128, 128))
            job = s.compiler(1002).channel(body, "roughness")
            self.assertEqual(read(s.compiler(1002), s.compiler(1002).padded(body, job)).shape[:2], (128, 128))
            files = Exporter(s).run(PREDEFINED[0], ExportSettings(output_dir=str(tmp / "out"),
                                                                    texture_sets=["Body"]))
            sizes = {f.name.split(".")[-2]: imageio.read(f).shape[:2] for f in files if "Roughness" in f.name}
            self.assertEqual(sizes, {"1001": (64, 64), "1002": (128, 128)})
            body.tile_sizes = {"1011": [128, 128]}
            with self.assertRaises(ProjectError):  # not a tile of the Texture Set
                s.project.validate()


if __name__ == "__main__":
    unittest.main()
