"""FBX import through ufbx (trafilatura3d/fbx.py), the 3D View's scenes and the bakers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d import fbx, meshio

DATA = Path(__file__).parent / "data" / "fbx"


@unittest.skipUnless(fbx.available(), "FBX library not built for this platform (tools/build_ufbx.py)")
class FbxTest(unittest.TestCase):
    def test_blender_materials_and_camera(self):
        objects, materials, cams = meshio.load(DATA / "blender_279_ball_7400_binary.fbx")
        self.assertEqual([o.name for o in objects], ["Icosphere"])
        self.assertEqual(materials, ["Red", "White"])
        np.testing.assert_allclose(objects[0].material_colors[0], [0.8, 0.0, 0.0, 1.0], atol=1e-6)  # base colors
        self.assertEqual(sorted(s.material for s in objects[0].submeshes), [0, 1])  # faces keep their material
        self.assertEqual(len(objects[0].idx), 80 * 3)
        self.assertTrue(np.allclose(np.linalg.norm(objects[0].nrm, axis=1), 1.0, atol=1e-4))
        self.assertEqual(cams, [])  # FBX 6 "Producer" cameras are viewport cameras, not the scene's

    def test_maya_units_and_uvs(self):
        objects, materials, _ = meshio.load(DATA / "maya_cube_7500_binary.fbx")
        cube = objects[0]
        self.assertEqual((cube.name, materials), ("pCube1", ["lambert1"]))
        np.testing.assert_allclose(cube.pos.max(0) - cube.pos.min(0), [0.01] * 3, rtol=1e-4)  # 1 cm in meters
        self.assertEqual(len(cube.pos), 24)  # corners welded: 4 per face (hard edges, UV seams)
        self.assertGreater(float(cube.uv.max() - cube.uv.min()), 0.5)  # the file's UVs, not box projected
        fitted, _, _ = meshio.load(DATA / "maya_cube_7500_binary.fbx")
        center, k = meshio.fit(fitted)
        self.assertAlmostEqual(float(fitted[0].pos.max() - fitted[0].pos.min()), 2.0, places=4)
        self.assertAlmostEqual(k, 200.0, places=2)  # 1 cm -> 2 units

    def test_catmull_clark_subdivision(self):
        """`subdivide` levels of Catmull-Clark on the file's own polygons: a cube's quads become 4 per level and
        its corners round off towards the inside; other formats refuse it (their loaders give triangles)."""
        cube = DATA / "maya_cube_7500_binary.fbx"
        tris = lambda objs: sum(len(o.idx) // 3 for o in objs)  # noqa: E731
        flat, one, two = (meshio.load(cube, subdivide=k)[0] for k in (0, 1, 2))
        self.assertEqual(tris(one), tris(flat) * 4)
        self.assertEqual(tris(two), tris(flat) * 16)
        corner = lambda objs: max(float(np.linalg.norm(o.pos, axis=1).max()) for o in objs)  # noqa: E731
        self.assertLess(corner(two), corner(one))
        self.assertLess(corner(one), corner(flat))
        with self.assertRaises(ValueError):
            meshio.load(Path("x.obj"), subdivide=1)

    def test_camera_axes(self):
        _, _, cams, _ = fbx.read(DATA / "maya_camera_light_axes_y_up_7700_ascii.fbx")
        cam = next(c for c in cams if c["name"] == "forwardCamera")
        np.testing.assert_allclose(np.subtract(cam["target"], cam["eye"]), [0, 0, -1], atol=1e-6)  # looks down -Z

    def test_bad_file(self):
        with tempfile.NamedTemporaryFile(suffix=".fbx") as f:
            f.write(b"not an fbx")
            f.flush()
            with self.assertRaises(ValueError):
                meshio.load(f.name)

    def test_bake_from_fbx(self):
        from trafilatura3d import bake, imageio
        with tempfile.TemporaryDirectory() as tmp:
            s = bake.Settings(size=(32, 32), use_low_as_high=True, dilation=0, apply_diffusion=False)
            path = bake.Baker().bake(str(DATA / "maya_cube_7500_binary.fbx"), {"world_space_normals": {}}, s, tmp)[0]
            img = imageio.read(path).astype(np.float64) / 65535
            covered = img[img.max(-1) > 0.01]
            self.assertGreater(len(covered), 32 * 32 * 0.2)  # the cube's UV islands are baked
            axes = np.round(covered * 2 - 1)
            self.assertTrue(np.all(np.abs(axes).sum(-1) == 1))  # each texel faces one axis

    def test_editor_filter_lists_fbx(self):
        self.assertIn("*.fbx", meshio.MESH_FILTER)


if __name__ == "__main__":
    unittest.main()
