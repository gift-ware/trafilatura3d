"""The 3D View's renderer, offscreen: environments, material upload, shadows, parallax."""

import math
import unittest

import numpy as np
import wgpu

from trafilatura3d.editor import render3d as r
from trafilatura3d.editor.meshes import PRIMITIVES
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.sample import sample_job

W, H = 160, 120


class Render3DTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = Evaluator()
        cls.renderer = r.Renderer(cls.ev.device, "rgba8unorm")
        cls.target = cls.ev.device.create_texture(
            size=(W, H, 1), format="rgba8unorm", usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)

    def draw(self, look=None, mesh="Sphere"):
        self.renderer.set_mesh(*PRIMITIVES[mesh]())
        eye = np.array([0.0, 1.0, 4.0], np.float32)
        vp = r.perspective(math.radians(40), W / H, 0.05, 100.0) @ r.look_at(eye, (0, 0, 0))
        self.renderer.draw(self.target, eye, vp, look or r.Look())
        data = self.ev.device.queue.read_texture({"texture": self.target}, {"bytes_per_row": W * 4}, (W, H, 1))
        return np.frombuffer(data, np.uint8).reshape(H, W, 4)[..., :3].astype(int)

    def test_studio_lights_the_mesh(self):
        self.renderer.set_studio()
        img = self.draw()
        centre, corner = img[H // 2, W // 2], img[2, 2]
        self.assertGreater(img.std(), 5)
        self.assertFalse(np.array_equal(centre, corner))

    def test_material_and_parallax(self):
        job = sample_job("test_pattern", size_log2=7)
        tex = self.ev.result(job, "output")
        self.renderer.set_studio()
        self.renderer.set_textures({"baseColor": (tex, True), "height": (tex, True)})
        flat = self.draw(r.Look(parallax=0.0, displacement=0.0), "Plane")
        deep = self.draw(r.Look(parallax=0.1, displacement=0.0), "Plane")
        self.assertGreater(np.abs(flat - deep).sum(), 0)
        self.renderer.set_textures({})

    def test_normal_maps_face_the_right_way_on_every_primitive(self):
        """A uniform normal map tilted along +U (red) or -V (green, OpenGL) lights the side it
        points to; a point seen head on, key light only."""
        s, c = 0.6, 0.8
        cases = {  # mesh: (eye, world +U, world -V) at the point under the eye
            "Plane": ((0, 5, 0), (1, 0, 0), (0, 0, -1)),
            "Cube": ((0, 0, 5), (1, 0, 0), (0, 1, 0)),
            "Sphere": ((5, 0, 0), (0, 0, 1), (0, 1, 0)),
            "Cylinder": ((5, 0, 0), (0, 0, 1), (0, 1, 0)),
            "Torus": ((5, 0, 0), (0, 0, 1), (0, -1, 0)),
        }
        for mesh, (eye, plus_u, minus_v) in cases.items():
            n = np.array(eye, float) / np.linalg.norm(eye)
            for x, y, d in ((1, 0, plus_u), (0, 1, minus_v)):
                nmap = self.constant([x * s * 0.5 + 0.5, y * s * 0.5 + 0.5, c * 0.5 + 0.5, 1])
                self.renderer.set_textures({"normal": (nmap, False)})
                lit = []
                for sign in (1, -1):
                    self.renderer._key(n * 0.3 + sign * np.array(d), (1, 1, 1), 3.0)
                    eye_v = np.array(eye, np.float32)
                    vp = r.perspective(math.radians(10), W / H, 0.05, 100.0) @ r.look_at(eye_v, eye_v * 0.82)
                    self.renderer.set_mesh(*PRIMITIVES[mesh]())
                    self.renderer.draw(self.target, eye_v, vp, r.Look(env_intensity=0.0, shadows=False, ground=False,
                                                                      show_env=False, displacement=0.0,
                                                                      opengl_normals=True))
                    lit.append(self.pixel())
                self.assertGreater(lit[0], lit[1] + 50, f"{mesh} {'red' if x else 'green'}")
        self.renderer.set_textures({})
        self.renderer.set_studio()

    def constant(self, value):
        t = self.ev.device.create_texture(size=(1, 1, 1), format="rgba32float",
                                          usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.ev.device.queue.write_texture({"texture": t}, np.array(value, np.float32).tobytes(),
                                           {"bytes_per_row": 16}, (1, 1, 1))
        return t

    def pixel(self):
        data = self.ev.device.queue.read_texture({"texture": self.target}, {"bytes_per_row": W * 4}, (W, H, 1))
        return int(np.frombuffer(data, np.uint8).reshape(H, W, 4)[H // 2, W // 2, 0])

    def test_image_based_light_conserves_energy(self):
        """Albedo 0.5 under a uniform environment of radiance 1 (no key light, Linear) reads 0.5."""
        srgb = 1.055 * 0.5 ** (1 / 2.4) - 0.055
        self.renderer.set_environment_image(np.ones((32, 64, 3), np.float32))
        self.renderer.set_materials([{"constants": {"baseColor": (srgb, srgb, srgb, 1), "roughness": (1, 1, 1, 1)}}])
        img = self.draw(r.Look(light_intensity=0.0, shadows=False, ground=False, displacement=0.0, tonemap=2))
        self.assertAlmostEqual(img[H // 2, W // 2, 0], srgb * 255, delta=4)
        self.renderer.set_materials([{}])
        self.renderer.set_studio()

    def test_unbound_slots_read_back(self):
        """An unbound slot (a constant) reads back like a bound one: the path tracer reads the height to displace."""
        self.renderer.set_materials([{"constants": {"height": (0.25, 0.25, 0.25, 1)}}])
        self.assertAlmostEqual(float(self.renderer.read_material(0, "height")[0, 0, 0]), 0.25, delta=1e-3)
        self.renderer.set_materials([{}])

    def test_hdr_image_sets_the_key_light(self):
        pixels = np.full((64, 128, 3), 0.2, np.float32)
        pixels[10:14, 30:34] = 50.0  # a bright spot above the horizon, left of centre
        self.renderer.set_environment_image(pixels)
        d = self.renderer.light_dir
        self.assertGreater(d[1], 0.5)
        self.assertLess(d[0], 0.0)
        unlit = self.draw(r.Look(light_intensity=0.0, shadows=False))
        lit = self.draw(r.Look(light_intensity=1.0))
        self.assertGreater(lit.sum(), unlit.sum())
        self.renderer.set_studio()


class ShadersTest(unittest.TestCase):
    """The shaders of shaders.py on one layered model (render3d SURFACE): each renders, the documented
    behaviours hold (legacy usages, Lambert without image based light, Mesh Info as is, Unlit), the lobes conserve
    energy under a uniform environment."""

    @classmethod
    def setUpClass(cls):
        cls.ev = Evaluator()
        cls.renderer = r.Renderer(cls.ev.device, "rgba8unorm")
        cls.target = cls.ev.device.create_texture(
            size=(W, H, 1), format="rgba8unorm", usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        cls.hdr = cls.ev.device.create_texture(
            size=(W, H, 1), format="rgba16float", usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)

    def draw(self, shader, values=None, look=None, linear=False):
        from trafilatura3d.editor.scene3d import Material
        self.renderer.set_materials([Material(shader=shader, values=dict(values or {})).spec()])
        self.renderer.set_mesh(*PRIMITIVES["Sphere"]())
        eye = np.array([0.0, 1.0, 4.0], np.float32)
        vp = r.perspective(math.radians(40), W / H, 0.05, 100.0) @ r.look_at(eye, (0, 0, 0))
        target = self.hdr if linear else self.target
        self.renderer.draw(target, eye, vp, look or r.Look(ground=False, displacement=0.0), linear=linear)
        if linear:
            data = self.ev.device.queue.read_texture({"texture": target}, {"bytes_per_row": W * 8}, (W, H, 1))
            return np.frombuffer(data, np.float16).reshape(H, W, 4)[..., :3].astype(np.float32)
        data = self.ev.device.queue.read_texture({"texture": target}, {"bytes_per_row": W * 4}, (W, H, 1))
        return np.frombuffer(data, np.uint8).reshape(H, W, 4)[..., :3].astype(int)

    def furnace(self, shader, values):
        """The centre of the sphere under a uniform environment of radiance 1, no key light: scene linear value."""
        self.renderer.set_environment_image(np.ones((32, 64, 3), np.float32))
        img = self.draw(shader, values, r.Look(light_intensity=0.0, shadows=False, ground=False, displacement=0.0,
                                               show_env=False), linear=True)
        self.renderer.set_studio()
        return img[H // 2, W // 2]

    def test_every_shader_renders(self):
        from trafilatura3d import shaders
        self.renderer.set_studio()
        for name in shaders.SHADERS:
            img = self.draw(name)
            self.assertTrue(np.isfinite(img).all(), name)
            if name != "Lambert":
                self.assertGreater(img[H // 2, W // 2].sum(), 30, name)

    def test_lambert_has_no_image_based_light(self):
        """Lambert uses only the point lights, no HDR image lighting."""
        self.renderer.set_studio()
        self.renderer.ambient = (0.0, 0.0, 0.0)
        self.renderer.point_lights = []
        self.assertLess(self.draw("Lambert")[H // 2, W // 2].sum(), 5)
        self.renderer.point_lights = [((0.0, 2.0, 4.0), (1.0, 1.0, 1.0), 20.0)]
        self.assertGreater(self.draw("Lambert")[H // 2, W // 2].sum(), 100)
        self.renderer.point_lights = []

    def test_mesh_info_shows_the_normal_as_is(self):
        """Mesh Info (Normal): the world normal clamped to [0, 1]; the sphere's centre faces +Z, a bit up."""
        c = self.draw("Mesh Info", {"mode": 0})[H // 2, W // 2]
        self.assertLess(c[0], 30)
        self.assertGreater(c[2], 200)

    def test_unlit_shows_the_color(self):
        c = self.draw("Unlit", {"color": (0.2, 0.6, 0.9)})[H // 2, W // 2]
        np.testing.assert_allclose(c, np.array([0.2, 0.6, 0.9]) * 255, atol=2)

    def test_white_furnace(self):
        """White dielectric and white rough metal reflect about all of a uniform environment, never more."""
        for shader, values in (("OpenPBR", {"base_color": (1.0, 1.0, 1.0), "specular_roughness": 0.5}),
                               ("OpenPBR", {"base_color": (1.0, 1.0, 1.0), "metalness": 1.0,
                                            "specular_roughness": 0.8}),
                               ("Adobe Standard Material", {"base_color": (1.0, 1.0, 1.0)})):
            v = self.furnace(shader, values)
            self.assertGreater(v.mean(), 0.88, (shader, values))
            self.assertLess(v.mean(), 1.06, (shader, values))

    def test_metallic_roughness_is_openpbr_at_its_defaults(self):
        """Metallic roughness = OpenPBR with base weight 1, specular weight 1, IOR 1.5 (same base color and roughness)."""
        a = self.furnace("Metallic roughness", {"base_color": (0.5, 0.5, 0.5), "roughness": 0.5})
        b = self.furnace("OpenPBR", {"base_color": (0.5, 0.5, 0.5), "specular_roughness": 0.5})
        np.testing.assert_allclose(a, b, atol=0.01)

    def test_coat_darkening(self):
        """OpenPBR: Darkening 0 boosts the coated base back to its uncoated color, 1 darkens it physically."""
        base = {"base_color": (0.8, 0.2, 0.2), "coat_weight": 1.0}
        dark = self.furnace("OpenPBR", {**base, "coat_darkening": 1.0})
        light = self.furnace("OpenPBR", {**base, "coat_darkening": 0.0})
        self.assertLess(dark[0], light[0] - 0.02)

    def test_fuzz_and_specular_weight(self):
        black = {"base_color": (0.0, 0.0, 0.0)}
        plain = self.furnace("OpenPBR", black).mean()
        self.assertGreater(self.furnace("OpenPBR", {**black, "fuzz_weight": 1.0}).mean(), plain + 0.01)
        self.assertLess(self.furnace("OpenPBR", {**black, "specular_weight": 0.0}).mean(), 0.01)

    def test_legacy_usages_feed_openpbr(self):
        """Legacy usages map to OpenPBR channels; the legacy shaders keep their own usages."""
        from trafilatura3d import shaders
        self.assertEqual(shaders.channel_of("OpenPBR", "metallic"), "metalness")
        self.assertEqual(shaders.channel_of("OpenPBR", "roughness"), "specularRoughness")
        self.assertEqual(shaders.channel_of("OpenPBR", "emissive"), "emissionColor")
        self.assertEqual(shaders.channel_of("OpenPBR", "sheenColor"), "fuzzColor")
        self.assertEqual(shaders.channel_of("OpenPBR", "baseColor"), "baseColor")
        self.assertIsNone(shaders.channel_of("Metallic roughness", "metalness"))
        self.assertEqual(shaders.channel_of("Specular glossiness", "glossiness"), "glossiness")
        self.assertEqual(shaders.DEFAULT, "OpenPBR")

    def test_blending_modes(self):
        """Opacity 0.2 everywhere: Opaque ignores it, Alpha test below the threshold shows the background, above it the
        opaque surface; Alpha blend lets the background through (sorted and order independent alike); a material that
        cannot see through (opacity 1) is drawn opaque whatever its mode."""
        from trafilatura3d import shaders
        self.renderer.set_studio()
        look = r.Look(ground=False, displacement=0.0, shadows=False, show_env=False)  # a flat background
        o = {"opacity": 0.2}
        opaque = self.draw("OpenPBR", {**o, "blending": shaders.OPAQUE}, look)
        empty = self.draw("OpenPBR", {**o, "blending": shaders.ALPHA_TEST}, look)
        kept = self.draw("OpenPBR", {**o, "blending": shaders.ALPHA_TEST, "alpha_threshold": 0.1}, look)
        blend = self.draw("OpenPBR", {**o, "blending": shaders.ALPHA_BLEND}, look)
        np.testing.assert_array_equal(opaque, kept)
        c = (H // 2, W // 2)
        background = empty[2, 2]
        np.testing.assert_allclose(empty[c], background, atol=1)
        self.assertGreater(np.abs(opaque[c] - background).sum(), 60)
        self.assertGreater(np.abs(blend[c] - opaque[c]).sum(), 30)
        self.assertGreater(np.abs(blend[c] - background).sum(), 5)
        oit = self.draw("OpenPBR", {**o, "blending": shaders.ALPHA_BLEND},
                        r.Look(ground=False, displacement=0.0, shadows=False, show_env=False,
                               order_independent_transparency=True))
        self.assertGreater(np.abs(oit[c] - opaque[c]).sum(), 30)  # alpha to coverage: quantized, but see-through
        self.assertEqual(self.renderer.materials[0]["blending"], shaders.ALPHA_BLEND)
        self.draw("OpenPBR", {"blending": shaders.ALPHA_BLEND}, look)
        self.assertEqual(self.renderer.materials[0]["blending"], shaders.OPAQUE)

    def test_blending_defaults(self):
        """Shaders with an Opacity channel blend; UsdPreviewSurface by its opacityThreshold (0
        translucent, above a cutout); those without opacity are opaque; the Paint shader files carry their mode."""
        from trafilatura3d import shaders
        self.assertEqual(shaders.uniforms("OpenPBR", {})["blending"], shaders.ALPHA_BLEND)
        self.assertEqual(shaders.uniforms("Metallic roughness", {})["blending"], shaders.ALPHA_BLEND)
        self.assertEqual(shaders.uniforms("UsdPreviewSurface", {})["blending"], shaders.ALPHA_BLEND)
        usd = shaders.uniforms("UsdPreviewSurface", {"opacity_threshold": 0.5})
        self.assertEqual((usd["blending"], usd["alpha_threshold"]), (shaders.ALPHA_TEST, 0.5))
        self.assertEqual(shaders.uniforms("Lambert", {})["blending"], shaders.OPAQUE)
        for name, mode in (("Metallic roughness", shaders.OPAQUE), ("Metallic roughness - Alpha blend",
                                                                    shaders.ALPHA_BLEND),
                           ("Metallic roughness - Alpha test", shaders.ALPHA_TEST), ("Lambert", shaders.OPAQUE)):
            t = shaders.TEMPLATES[name]  # the templates carry their mode
            self.assertEqual(shaders.uniforms(t.shader, t.values(0.33))["blending"], mode, name)

    def test_sorted_blending_back_to_front(self):
        """Two blended quads, red near and blue far, both half opaque: the near one's color dominates whatever
        order the meshes are given in (sorted by distance)."""
        from trafilatura3d import shaders
        from trafilatura3d.meshio import MeshObject, Submesh
        quad = lambda z: (np.array([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], np.float32),
                          np.tile(np.float32([0, 0, 1]), (4, 1)), np.tile(np.float32([1, 0, 0, 1]), (4, 1)),
                          np.float32([[0, 1], [1, 1], [1, 0], [0, 0]]), np.uint32([0, 1, 2, 0, 2, 3]))
        mats = [{"shader": shaders.index("Unlit"), "uniforms": {"blending": shaders.ALPHA_BLEND},
                 "constants": {"baseColor": color, "opacity": (0.5, 0.5, 0.5, 1)}}
                for color in ((1, 0, 0, 1), (0, 0, 1, 1))]
        self.renderer.set_materials(mats)
        eye = np.array([0.0, 0.0, 4.0], np.float32)
        vp = r.perspective(math.radians(40), W / H, 0.05, 100.0) @ r.look_at(eye, (0, 0, 0))
        near, far = MeshObject("near", *quad(0.5), [Submesh(0, 6, 0)]), MeshObject("far", *quad(-0.5), [Submesh(0, 6, 1)])
        for objects in ([near, far], [far, near]):
            self.renderer.set_meshes(objects)
            self.renderer.draw(self.target, eye, vp, r.Look(ground=False, shadows=False, displacement=0.0,
                                                            show_env=False))
            data = self.ev.device.queue.read_texture({"texture": self.target}, {"bytes_per_row": W * 4}, (W, H, 1))
            c = np.frombuffer(data, np.uint8).reshape(H, W, 4)[H // 2, W // 2].astype(int)
            self.assertGreater(c[0], c[2] + 40, [o.name for o in objects])
        self.renderer.set_materials([{}])

    def test_path_tracer_follows_the_blending_mode(self):
        """The path tracer: Opaque ignores an opacity of 0, Alpha test and Alpha blend see through it."""
        from trafilatura3d import shaders
        from trafilatura3d.editor.pathtrace import PathTracer
        from trafilatura3d.editor.scene3d import Material
        from trafilatura3d.meshio import MeshObject, Submesh
        self.renderer.set_studio()
        pt = PathTracer(self.renderer)
        pos, nrm, tan, uv, idx = PRIMITIVES["Sphere"]()
        obj = MeshObject("s", pos, nrm, tan, uv, idx, [Submesh(0, idx.size, 0)])
        eye = np.array([0.0, 1.0, 4.0], np.float32)
        vp = r.perspective(math.radians(40), W / H, 0.05, 100.0) @ r.look_at(eye, (0, 0, 0))
        target = self.ev.device.create_texture(size=(W, H, 1), format="rgba16float",
                                               usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        out = {}
        for mode in (shaders.OPAQUE, shaders.ALPHA_TEST, shaders.ALPHA_BLEND):
            self.renderer.set_materials([Material(values={"opacity": 0.0, "blending": mode,
                                                          "base_color": (1.0, 0.0, 0.0)}).spec()])
            look = r.Look(ground=False, displacement=0.0)
            pt.set_scene([obj], look)
            pt.key = None
            pt.draw(target, eye, vp, look, [], 1, 4, linear=True, spp=4)
            data = self.ev.device.queue.read_texture({"texture": target}, {"bytes_per_row": W * 8}, (W, H, 1))
            out[mode] = np.frombuffer(data, np.float16).reshape(H, W, 4)[H // 2, W // 2, :3].astype(np.float32)
        self.assertGreater(out[shaders.OPAQUE][0], out[shaders.OPAQUE][2] * 2)  # red
        for mode in (shaders.ALPHA_TEST, shaders.ALPHA_BLEND):  # the gray environment behind
            self.assertLess(abs(out[mode][0] - out[mode][2]), 0.1, mode)
        self.renderer.set_materials([{}])

    def test_every_shader_path_traces(self):
        from trafilatura3d import shaders
        from trafilatura3d.editor.pathtrace import PathTracer
        from trafilatura3d.editor.scene3d import Material
        from trafilatura3d.meshio import MeshObject, Submesh
        self.renderer.set_studio()
        pt = PathTracer(self.renderer)
        pos, nrm, tan, uv, idx = PRIMITIVES["Sphere"]()
        obj = MeshObject("s", pos, nrm, tan, uv, idx, [Submesh(0, idx.size, 0)])
        eye = np.array([0.0, 1.0, 4.0], np.float32)
        vp = r.perspective(math.radians(40), W / H, 0.05, 100.0) @ r.look_at(eye, (0, 0, 0))
        target = self.ev.device.create_texture(size=(W, H, 1), format="rgba16float",
                                               usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        for name in shaders.SHADERS:
            values = {"transmission_weight": 1.0, "coat_weight": 1.0} if name == "OpenPBR" else {}
            self.renderer.set_materials([Material(shader=name, values=values).spec()])
            look = r.Look(ground=False, displacement=0.0)
            pt.set_scene([obj], look)
            pt.key = None
            pt.draw(target, eye, vp, look, [], 1, 4, linear=True, spp=4)
            data = self.ev.device.queue.read_texture({"texture": target}, {"bytes_per_row": W * 8}, (W, H, 1))
            img = np.frombuffer(data, np.float16).reshape(H, W, 4)[..., :3].astype(np.float32)
            self.assertTrue(np.isfinite(img).all(), name)
            if name != "Lambert":
                self.assertGreater(img[H // 2, W // 2].sum(), 0.02, name)


if __name__ == "__main__":
    unittest.main()
