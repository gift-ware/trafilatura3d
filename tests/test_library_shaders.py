"""Library shaders (shaders.py, project.LIBRARY_KINDS): a project's library `shaders` folder adds .wgsl shaders to the
table at start. The header is checked line by line; a library shader that fills the lobes like a table shader renders
like it (rasterizer and path tracer), its parameters reach `M.custom`, an unlit one shows its color, and its channels
become Texture Set channels and a template. The render runs in a child process: the table is read when the module is
imported, from the project configuration of the start (TRAFILATURA3D_CONFIG)."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from trafilatura3d import shaders

# Metallic roughness again (case 6u of lobes()), with a tint parameter on the base color.
METAL = '''
//: shader "Test Metal"
//: channel baseColor baseColor "Base color" = 0.5,0.5,0.5
//: channel metallic metallic "Metallic"
//: channel roughness roughness "Roughness" = 0.5
//: channel normal normal "Normal"
//: param color tint "Tint" = 1,1,1
//: param float unused "Unused" = 0.25 [0,4]
//: blending opaque
fn helper(x: vec3f) -> vec3f { return x; }
fn shade(m: M, c: Ch, L: Lobes) -> Lobes {
    var out = L;
    out.metal = c.metal;
    out.albedo = helper(L.albedo * tint(m));
    out.f0m = out.albedo;
    return out;
}
'''
UNLIT = '''
//: shader "Test Unlit"
//: param color color "Color" = 0.2,0.6,0.9
//: blending alpha_test 0.5
fn shade(m: M, c: Ch, L: Lobes) -> Lobes {
    var out = L;
    out.model = MODEL_UNLIT;
    out.albedo = srgb_to_linear(color(m));
    return out;
}
'''
CHILD = '''
import json, math
import numpy as np
import wgpu
from trafilatura3d import shaders
from trafilatura3d.editor import render3d as r
from trafilatura3d.editor.meshes import PRIMITIVES
from trafilatura3d.editor.scene3d import Material
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.paint.channels import CHANNELS
W, H = 96, 72
ev = Evaluator()
ren = r.Renderer(ev.device, "rgba8unorm")
target = ev.device.create_texture(size=(W, H, 1), format="rgba8unorm",
                                  usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
ren.set_studio()
ren.set_mesh(*PRIMITIVES["Sphere"]())
eye = np.array([0.0, 1.0, 4.0], np.float32)
vp = r.perspective(math.radians(40), W / H, 0.05, 100.0) @ r.look_at(eye, (0, 0, 0))
def draw(shader, values):
    ren.set_materials([Material(shader=shader, values=values).spec()])
    ren.draw(target, eye, vp, r.Look(ground=False, displacement=0.0))
    data = ev.device.queue.read_texture({"texture": target}, {"bytes_per_row": W * 4}, (W, H, 1))
    return np.frombuffer(data, np.uint8).reshape(H, W, 4)[..., :3].astype(int)
core = draw("Metallic roughness", {"base_color": (0.7, 0.3, 0.2), "metallic": 0.4, "roughness": 0.35})
lib = draw("Test Metal", {"baseColor": (0.7, 0.3, 0.2), "metallic": 0.4, "roughness": 0.35})
tinted = draw("Test Metal", {"baseColor": (0.7, 0.3, 0.2), "metallic": 0.4, "roughness": 0.35, "tint": (0.0, 1.0, 1.0)})
unlit = draw("Test Unlit", {})
from trafilatura3d.editor.pathtrace import PathTracer
from trafilatura3d.meshio import MeshObject, Submesh
pt = PathTracer(ren)
pos, nrm, tan, uv, idx = PRIMITIVES["Sphere"]()
hdr = ev.device.create_texture(size=(W, H, 1), format="rgba16float",
                               usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
traced = []
for name in ("Test Metal", "Test Unlit"):
    ren.set_materials([Material(shader=name, values={}).spec()])
    look = r.Look(ground=False, displacement=0.0)
    pt.set_scene([MeshObject("s", pos, nrm, tan, uv, idx, [Submesh(0, idx.size, 0)])], look)
    pt.key = None
    pt.draw(hdr, eye, vp, look, [], 1, 4, linear=True, spp=4)
    data = ev.device.queue.read_texture({"texture": hdr}, {"bytes_per_row": W * 8}, (W, H, 1))
    img = np.frombuffer(data, np.float16).reshape(H, W, 4)[..., :3].astype(np.float32)
    traced.append(float(img[H // 2, W // 2].sum()) if np.isfinite(img).all() else -1.0)
from trafilatura3d.paint.export import library_templates
print(json.dumps({
    "presets": [(t.name, t.predefined, t.maps[0].name) for t in library_templates()],
    "traced": traced,
    "library": [s.name for s in shaders.LIBRARY], "templates": [t for t in shaders.TEMPLATES if t.startswith("Test")],
    "same": int(np.abs(core - lib).max()), "tint_red": int(tinted[H // 2, W // 2, 0]),
    "lib_red": int(lib[H // 2, W // 2, 0]), "unlit": unlit[H // 2, W // 2].tolist(),
    "uniforms": shaders.uniforms("Test Unlit", {}), "channels": sorted(k for k in ("basecolor", "metallic") if k in CHANNELS),
}))
'''


class LibraryShaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, name="s.wgsl"):
        path = self.dir / name
        path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
        return path

    def test_header(self):
        s = shaders.load_shader(self.write(METAL), 3)
        self.assertEqual(s.name, "Test Metal")
        self.assertEqual(s.channels, {"baseColor": "baseColor", "metallic": "metallic", "roughness": "roughness",
                                      "normal": "normal"})
        params = {p.key: p for p in s.params}
        self.assertEqual((params["tint"].target, params["unused"].target), (("custom", 0), ("custom", 1)))
        self.assertEqual((params["unused"].lo, params["unused"].hi, params["unused"].default), (0.0, 4.0, 0.25))
        self.assertEqual(params["roughness"].target, ("slot", "roughness"))
        self.assertEqual(s.fixed, {"blending": shaders.OPAQUE})
        self.assertIn("fn shade_l3(", s.wgsl)
        self.assertIn("helper_l3(L.albedo * tint_l3(m))", s.wgsl)  # functions renamed, not the lobes' fields
        self.assertIn("out.albedo", s.wgsl)
        for bad, message in (('//: shader "X"\n//: channel base nowhere "B"\nfn shade() {}', "slot one of"),
                             ('//: shader "X"\n//: blending sometimes\nfn shade() {}', "blending"),
                             ('//: shader "X"\n//: colour c "C"\nfn shade() {}', "unknown header"),
                             ('//: shader "X"\nfn other() {}', "no fn shade"),
                             ("fn shade() {}", "no //: shader")):
            with self.assertRaisesRegex(shaders.ShaderFileError, message):
                shaders.load_shader(self.write(bad, "bad.wgsl"), 0)

    def test_render_through_the_project_library(self):
        library = self.dir / "lib"
        (library / "shaders").mkdir(parents=True)
        (library / "shaders" / "metal.wgsl").write_text(textwrap.dedent(METAL).lstrip(), encoding="utf-8")
        (library / "shaders" / "unlit.wgsl").write_text(textwrap.dedent(UNLIT).lstrip(), encoding="utf-8")
        (library / "shaders" / "broken.wgsl").write_text("fn shade() {}", encoding="utf-8")  # reported, left out
        proj = self.dir / "studio.trafilatura3dproject"
        proj.write_text(json.dumps({"trafilatura3d_project": 1, "libraries": {"paths": ["lib"]}}), encoding="utf-8")
        cfg = self.dir / "studio.trafilatura3dconfig"
        cfg.write_text(json.dumps({"projects": [proj.name]}), encoding="utf-8")
        (library / "export_presets").mkdir()
        (library / "export_presets" / "studio.json").write_text(json.dumps([{"name": "Studio", "maps": [
            {"name": "$mesh_BaseMap", "type": "RGB+A", "slots": {}}]}]), encoding="utf-8")
        env = {**os.environ, "TRAFILATURA3D_CONFIG": str(cfg), "TRAFILATURA3D_USER_DIR": str(self.dir / "user")}
        run = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("broken.wgsl", run.stderr)
        out = json.loads(run.stdout.strip().splitlines()[-1])
        self.assertEqual(out["library"], ["Test Metal", "Test Unlit"])
        self.assertEqual(out["templates"], ["Test Metal"])  # no channels, nothing to output: no template
        self.assertLessEqual(out["same"], 1)  # the same lobes as the table's Metallic roughness
        self.assertLess(out["tint_red"], out["lib_red"] // 2)  # the tint parameter reaches the shader
        for got, want in zip(out["unlit"], (0.2, 0.6, 0.9)):
            self.assertAlmostEqual(got / 255, want, delta=2 / 255)
        self.assertEqual((out["uniforms"]["blending"], out["uniforms"]["alpha_threshold"]), (shaders.ALPHA_TEST, 0.5))
        self.assertEqual(out["channels"], ["basecolor", "metallic"])
        self.assertEqual(out["presets"], [["Studio", True, "$mesh_BaseMap"]])  # the library's, read only
        metal, unlit = out["traced"]
        self.assertGreater(metal, 0.02)  # path traced, finite
        self.assertAlmostEqual(unlit, 0.2 ** 2.2 + 0.6 ** 2.2 + 0.9 ** 2.2, delta=0.15)  # its color, emitted


if __name__ == "__main__":
    unittest.main()
