"""The 3D View's scene: primitives, 3D scene import/export, state files, picking, the path tracer."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import wgpu

from trafilatura3d.editor import render3d as r
from trafilatura3d.editor.meshes import EXTENDED, PRIMITIVES, STANDARD
from trafilatura3d.editor.pathtrace import PathTracer, bvh_bounds, morton_bvh
from trafilatura3d.editor.scene3d import Material, Scene, export_scene, load_scene
from trafilatura3d.evaluator import Evaluator


class FakeView:
    def __init__(self):
        self.yaw, self.pitch, self.distance = 0.1, 0.2, 3.0
        self.target = np.zeros(3, np.float32)
        self.look = r.Look()


class SceneTest(unittest.TestCase):
    def test_standard_primitives(self):
        self.assertEqual(list(STANDARD), ["Cube", "Cylinder", "Hollow Box", "Inner Box", "Plane", "Plane (hi-res)",
                                          "Sphere"])
        self.assertEqual(list(EXTENDED), ["Cloth", "Mat Ball", "Rounded Cube", "Rounded Cylinder", "Sphere 2 tiles",
                                          "Torus"])
        for name, make in PRIMITIVES.items():
            pos, nrm, tan, uv, idx = make()
            self.assertEqual(tan.shape[1], 4, name)
            self.assertTrue(np.isfinite(pos).all() and np.isfinite(tan).all(), name)
            self.assertEqual(idx.max() + 1, len(pos), name)

    def test_export_and_import(self):
        sc = Scene()
        sc.set_primitive("Torus")
        sc.materials = [Material(name="Rock")]
        with tempfile.TemporaryDirectory() as d:
            for fmt in ("obj", "glb", "ply", "usda", "usdz"):
                path = Path(d) / f"torus.{fmt}"
                export_scene(path, sc.objects, sc.materials)
                objects, materials, _, _ = load_scene(path)
                faces = sum(o.idx.size // 3 for o in objects)
                self.assertEqual(faces, sc.objects[0].idx.size // 3, fmt)
                self.assertEqual(objects[0].tan.shape[1], 4, fmt)
                if fmt in ("glb", "usda", "usdz"):
                    self.assertEqual(materials[0].name, "Rock", fmt)

    def test_state_round_trip(self):
        sc, view = Scene(), FakeView()
        sc.set_primitive("Cube")
        sc.materials[0].shader = "Specular glossiness"
        sc.lights.points[1].enabled = True
        view.look.grid = True
        data = sc.state(view)
        sc2, view2 = Scene(), FakeView()
        sc2.apply_state(data, view2, with_mesh=True)
        self.assertEqual(sc2.mesh_name, "Cube")
        self.assertEqual(sc2.materials[0].shader, "Specular glossiness")
        self.assertTrue(sc2.lights.points[1].enabled)
        self.assertTrue(view2.look.grid)
        self.assertAlmostEqual(view2.distance, 3.0)

    def test_pick(self):
        sc = Scene()
        sc.set_primitive("Cube")
        self.assertEqual(sc.pick(np.array([0, 0, 5.0]), np.array([0, 0, -1.0])), 0)
        self.assertIsNone(sc.pick(np.array([0, 3, 5.0]), np.array([0, 0, -1.0])))

    def test_bvh_bounds_contain_their_triangles(self):
        rng = np.random.default_rng(1)
        tri = rng.normal(size=(37, 3, 3)).astype(np.float32)
        order, leaves = morton_bvh(tri.mean(1))
        nodes = bvh_bounds(tri[order], leaves).reshape(-1, 2, 4)
        lo, hi = nodes[0, 0, :3], nodes[0, 1, :3]
        self.assertTrue((tri.min((0, 1)) >= lo - 1e-6).all() and (tri.max((0, 1)) <= hi + 1e-6).all())


class PathTracerTest(unittest.TestCase):
    def test_converges_without_nans_and_casts_a_shadow(self):
        ev = Evaluator()
        R = r.Renderer(ev.device, "rgba8unorm")
        sc = Scene()
        sc.set_primitive("Sphere")
        R.set_meshes(sc.objects)
        pt = PathTracer(R)
        look = r.Look(displacement=0.0, ground_reflections=False)
        pt.set_scene(sc.objects, look)
        W = H = 64
        t = ev.device.create_texture(size=(W, H, 1), format="rgba16float",
                                     usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        eye = np.array([0.0, 4.0, 0.01], np.float32)  # from above: the sphere and the floor around it
        vp = r.perspective(math.radians(60), 1.0, 0.05, 100) @ r.look_at(eye, (0, 0, 0))
        n = pt.draw(t, eye, vp, look, [], 0, 8, linear=True, spp=8)
        self.assertEqual(n, 8)
        img = np.frombuffer(ev.device.queue.read_texture({"texture": t}, {"bytes_per_row": W * 8}, (W, H, 1)),
                            np.float16).reshape(H, W, 4).astype(np.float32)
        self.assertTrue(np.isfinite(img).all())
        self.assertGreater(img[..., :3].mean(), 0.0)


if __name__ == "__main__":
    unittest.main()
