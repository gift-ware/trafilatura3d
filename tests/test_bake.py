"""Bakers (model information into textures) and the command line."""

import contextlib
import dataclasses
import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from trafilatura3d import bake, imageio
from trafilatura3d.cli import main as cli

BUMP = (0.3, 0.3)  # x, z of the high poly's bump: u = 0.65, v = 0.65 (image row 0.35)


def grid(n, height, name=""):
    """A 2x2 plane in XZ facing +Y, UV v up along -Z (OBJ convention)."""
    u = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, u)
    x, z = U * 2 - 1, V * 2 - 1
    pos = np.stack([x, height(x, z), -z], -1).reshape(-1, 3)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    m = trimesh.Trimesh(pos, np.array(faces), process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.stack([U, V], -1).reshape(-1, 2))
    m.metadata["name"] = name
    return m


def bump(x, z):
    return 0.15 * np.exp(-((x - BUMP[0]) ** 2 + (z - BUMP[1]) ** 2) / 0.05)


def read(path):
    a = imageio.read(path).astype(np.float64)
    a = a / (65535 if a.dtype == np.uint16 or a.max() > 255 else 255)
    return a if a.ndim == 3 else a[..., None]


class BakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        grid(2, lambda x, z: 0 * x).export(cls.dir / "low.obj")
        grid(160, bump).export(cls.dir / "high.obj")
        cls.baker = bake.Baker()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_bake(self, bakers, size=64, **kw):
        s = bake.Settings(size=(size, size), high_meshes=[str(self.dir / "high.obj")], frontal=0.2, rear=0.2,
                          dilation=2, **kw)
        out = Path(tempfile.mkdtemp(dir=self.dir))
        return {p.stem.split("low_", 1)[1]: p for p in self.baker.bake(str(self.dir / "low.obj"), bakers, s, out)}

    def at(self, img, u, v):
        h, w = img.shape[:2]
        return img[min(int((1 - v) * h), h - 1), min(int(u * w), w - 1)]

    def test_normal_map_orientation(self):
        self.assertEqual(bake.baker_defaults("normal_map_from_mesh")["normal_orientation"], "DirectX")
        f = self.run_bake({"normal_map_from_mesh": {"normal_orientation": "OpenGL"}})["normal_map_from_mesh"]
        n = read(f)
        flat = self.at(n, 0.1, 0.1)
        np.testing.assert_allclose(flat, [0.5, 0.5, 1.0], atol=0.02)  # no detail: the low's normal
        beyond = self.at(n, 0.65, 0.72)  # past the bump's top along +V: the surface faces +V
        self.assertGreater(beyond[1], 0.6)  # OpenGL: green up the image
        right = self.at(n, 0.72, 0.65)
        self.assertGreater(right[0], 0.6)  # red along +U
        dx = read(self.run_bake({"normal_map_from_mesh": {}})["normal_map_from_mesh"])
        self.assertLess(self.at(dx, 0.65, 0.72)[1], 0.4)  # DirectX: green flipped

    def test_height_position_opacity(self):
        maps = self.run_bake({"height_map_from_mesh": {}, "position": {}, "opacity_mask_from_mesh": {},
                              "world_space_normals": {}})
        h = read(maps["height_map_from_mesh"])
        self.assertGreater(self.at(h, 0.65, 0.65)[0], self.at(h, 0.1, 0.1)[0] + 0.1)  # the bump stands out
        self.assertAlmostEqual(self.at(h, 0.1, 0.1)[0], 0.5, delta=0.02)  # on the low surface: 0.5
        p = read(maps["position"])
        self.assertLess(self.at(p, 0.1, 0.5)[0], self.at(p, 0.9, 0.5)[0])  # x grows with u
        self.assertTrue(np.all(read(maps["opacity_mask_from_mesh"]) > 0.99))
        np.testing.assert_allclose(self.at(read(maps["world_space_normals"]), 0.5, 0.5), [0.5, 1.0, 0.5], atol=0.01)

    def test_occlusion_and_thickness(self):
        maps = self.run_bake({"ambient_occlusion_from_mesh": {"secondary_rays": 128},
                              "thickness_map_from_mesh": {"secondary_rays": 64}}, size=96)
        ao = read(maps["ambient_occlusion_from_mesh"])[..., 0]
        self.assertAlmostEqual(float(self.at(ao[..., None], 0.05, 0.05)[0]), 1.0, delta=0.02)  # open plane
        self.assertLess(float(ao.min()), 0.99)  # the bump's foot is occluded
        th = read(maps["thickness_map_from_mesh"])[..., 0]
        self.assertLess(float(self.at(th[..., None], 0.65, 0.65)[0]), 0.95)  # the bump is a closed-ish volume

    def test_dilation_antialiasing_template_and_svg(self):
        half = self.dir / "half.obj"
        m = grid(2, lambda x, z: 0 * x)
        m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv * 0.5)  # only the bottom-left quarter of UV space
        m.export(half)
        s = bake.Settings(size=(32, 32), use_low_as_high=True, dilation=0, apply_diffusion=False,
                          antialiasing="Subsampling 2x2")
        out = self.dir / "dil"
        p0 = self.baker.bake(str(half), {"world_space_normals": {}}, s, out, "$(bakername)-of-$(mesh)")[0]
        self.assertEqual(p0.name, "world_space_normals-of-half.png")
        img = read(p0)
        self.assertEqual(img.shape[:2], (32, 32))
        self.assertLess(img[2, 30].max(), 0.01)  # outside the island: empty
        s.dilation = 4
        img = read(self.baker.bake(str(half), {"world_space_normals": {}}, s, out)[0])
        self.assertGreater(img[16 - 3, 17].max(), 0.4)  # grown past the island border
        s.apply_diffusion = True
        img = read(self.baker.bake(str(half), {"world_space_normals": {}}, s, out)[0])
        self.assertGreater(img[1, 30].max(), 0.4)  # diffusion fills everything
        svg = self.baker.bake(str(half), {"convert_uv_to_svg": {"color_mode": "Uniform color", "padding": 2}}, s,
                              out)[0].read_text(encoding="utf-8")
        self.assertEqual(svg.count("<polygon"), 2)  # one shell, two triangles
        self.assertIn('fill="#808080" stroke="#808080" stroke-width="4"', svg)  # 50% grey, padding as stroke

    def test_tiles_match_a_single_pass(self):
        s = bake.Settings(size=(64, 64), high_meshes=[str(self.dir / "high.obj")], frontal=0.2, rear=0.2,
                          dilation=0, apply_diffusion=False, antialiasing="Subsampling 2x2")
        bakers = {"normal_map_from_mesh": {}, "ambient_occlusion_from_mesh": {"secondary_rays": 16}}
        whole = [read(p) for p in self.baker.bake(str(self.dir / "low.obj"), bakers, s, self.dir / "whole")]
        old = bake.TILE
        bake.TILE = 32  # 16 output pixels per tile at 2x2: 16 tiles
        try:
            tiled = [read(p) for p in self.baker.bake(str(self.dir / "low.obj"), bakers, s, self.dir / "tiled")]
        finally:
            bake.TILE = old
        for a, b in zip(whole, tiled):
            np.testing.assert_allclose(a, b, atol=2e-4)  # same samples, same per pixel noise

    def test_denoise(self):
        from trafilatura3d.bake_params import BAKER_PARAMS
        self.assertFalse(BAKER_PARAMS["ambient_occlusion_from_mesh"]["denoise"].default)  # ours: off

        def ao(rays, denoise=False):
            return read(self.run_bake({"ambient_occlusion_from_mesh": {"secondary_rays": rays, "denoise": denoise}},
                                      size=128)["ambient_occlusion_from_mesh"])[..., 0]
        ref, noisy, clean = ao(1024), ao(8), ao(8, True)
        err = lambda a: float(np.sqrt(np.mean((a - ref) ** 2)))
        self.assertLess(err(clean), err(noisy) / 2)
        self.assertAlmostEqual(float(clean.mean()), float(ref.mean()), delta=0.005)  # no darkening or lightening
        # Tiles are baked with a margin as wide as the filter: the same map as a single pass.
        s = bake.Settings(size=(64, 64), high_meshes=[str(self.dir / "high.obj")], frontal=0.2, rear=0.2,
                          dilation=0, apply_diffusion=False, antialiasing="Subsampling 2x2")
        bakers = {"ambient_occlusion_from_mesh": {"secondary_rays": 8, "denoise": True}}
        whole = read(self.baker.bake(str(self.dir / "low.obj"), bakers, s, self.dir / "dn_whole")[0])
        old = bake.TILE
        bake.TILE = 32
        try:
            tiled = read(self.baker.bake(str(self.dir / "low.obj"), bakers, s, self.dir / "dn_tiled")[0])
        finally:
            bake.TILE = old
        np.testing.assert_allclose(whole, tiled, atol=2e-4)

    def test_large_map_with_antialiasing(self):
        """2048 at 2x2 is 4096^2 samples: more than one GPU buffer can hold, baked in tiles."""
        s = bake.Settings(size=(2048, 2048), use_low_as_high=True, antialiasing="Subsampling 2x2", dilation=1,
                          apply_diffusion=False)
        img = read(self.baker.bake(str(self.dir / "low.obj"), {"world_space_normals": {}}, s, self.dir / "big")[0])
        self.assertEqual(img.shape[:2], (2048, 2048))
        np.testing.assert_allclose(img[1024, 1024], [0.5, 1.0, 0.5], atol=0.01)

    def test_island_borders_keep_their_value(self):
        half = self.dir / "half_border.obj"
        m = grid(2, lambda x, z: 0 * x)
        m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv * 0.5 + 0.13)  # borders inside pixels
        m.export(half)
        s = bake.Settings(size=(32, 32), use_low_as_high=True, antialiasing="Subsampling 4x4", dilation=0,
                          apply_diffusion=False)
        img = read(self.baker.bake(str(half), {"world_space_normals": {}}, s, self.dir / "border")[0])
        covered = img[img.max(-1) > 0.01]
        np.testing.assert_allclose(covered, np.broadcast_to([0.5, 1.0, 0.5], covered.shape), atol=0.01)

    def test_curvature_of_the_low_mesh(self):
        sphere = self.dir / "sphere.obj"
        u, v = np.meshgrid(np.linspace(0, 1, 65), np.linspace(0, 1, 33))
        th, ph = u * 2 * np.pi, v * np.pi
        pos = np.stack([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)], -1).reshape(-1, 3)
        n = 65
        faces = []
        for j in range(32):
            for i in range(64):
                a = j * n + i
                faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]  # outward
        m = trimesh.Trimesh(pos, np.array(faces), process=False)
        m.visual = trimesh.visual.TextureVisuals(uv=np.stack([u, 1 - v], -1).reshape(-1, 2))
        m.export(sphere)
        s = bake.Settings(size=(64, 64), use_low_as_high=True, dilation=0, apply_diffusion=False)
        img = read(self.baker.bake(str(sphere), {"curvature": {}}, s, self.dir / "curv")[0])[..., 0]
        self.assertGreater(float(np.median(img[16:48, 8:56])), 0.55)  # convex everywhere
        inward = self.dir / "inward.obj"
        m.invert()
        m.export(inward)
        img = read(self.baker.bake(str(inward), {"curvature": {}}, s, self.dir / "curv")[0])[..., 0]
        self.assertLess(float(np.median(img[16:48, 8:56])), 0.45)  # seen from inside: concave
        flat = read(self.baker.bake(str(self.dir / "low.obj"), {"curvature": {}}, s, self.dir / "curv")[0])[..., 0]
        self.assertAlmostEqual(float(np.median(flat)), 0.5, delta=0.01)

    def test_udim_tiles(self):
        up = grid(2, lambda x, z: 0 * x)  # faces +Y, UVs in 1001
        side = grid(2, lambda x, z: 0 * x)
        side.vertices = side.vertices[:, [0, 2, 1]] * [1, -1, 1]  # faces +Z
        side.visual = trimesh.visual.TextureVisuals(uv=up.visual.uv + [1.0, 1.0])  # 1012
        scene = trimesh.Scene()
        scene.add_geometry(up, geom_name="up")
        scene.add_geometry(side, geom_name="side")
        scene.export(self.dir / "udim.obj")
        from trafilatura3d import meshio
        self.assertEqual(bake.udim_tiles(meshio.load(str(self.dir / "udim.obj"))[0]), [(0, 0), (1, 1)])
        s = bake.Settings(size=(32, 32), use_low_as_high=True, dilation=0, apply_diffusion=False)
        paths = self.baker.bake(str(self.dir / "udim.obj"), {"world_space_normals": {}}, s, self.dir / "udim")
        self.assertEqual([p.name for p in paths], ["udim_world_space_normals.1001.png", "udim_world_space_normals.1012.png"])
        np.testing.assert_allclose(read(paths[0])[16, 16], [0.5, 1.0, 0.5], atol=0.02)
        self.assertGreater(read(paths[1])[16, 16][2], 0.9)  # the side quad faces +Z
        named = self.baker.bake(str(self.dir / "udim.obj"), {"world_space_normals": {}}, s, self.dir / "udim",
                                "$(mesh)_$(udim)_$(bakername)")
        self.assertEqual(named[1].name, "udim_1012_world_space_normals.png")

    def test_baker_parameters_and_defaults(self):
        from trafilatura3d.bake_params import BAKER_PARAMS, applies
        ao = BAKER_PARAMS["ambient_occlusion_from_mesh"]
        self.assertEqual((ao["max_occluder_distance"].default, ao["spread_angle"].default,
                          ao["ignore_backface"].options, ao["attenuation"].default),
                         (0.1, 180.0, ("Never", "Always", "By Mesh Name"), "Linear"))
        self.assertEqual(BAKER_PARAMS["curvature_from_mesh"]["secondary_rays"].default, 32)
        values = bake.resolved("normal_map_from_mesh", {"map_type": "World Space", "unknown": 1})
        self.assertNotIn("unknown", values)
        self.assertFalse(applies("normal_map_from_mesh", "normal_orientation", values))  # only in Tangent Space

    def test_occlusion_options(self):
        def ao(**params):
            img = read(self.run_bake({"ambient_occlusion_from_mesh": {"secondary_rays": 128, **params}}, size=64)
                       ["ambient_occlusion_from_mesh"])[..., 0]
            return float(img[16:48, 16:48].mean())  # around the bump
        full = ao()
        self.assertLess(ao(attenuation="None"), full)  # every hit counts fully
        self.assertGreater(ao(spread_angle=20.0), full)  # a narrow cone sees past the bump
        self.assertLess(ao(ground_plane=True, ground_plane_offset=0.0, distribution="Uniform"), 0.999)
        self.assertGreater(ao(ignore_backface="Always"), 0.0)

    def test_self_occlusion_by_mesh_name(self):
        """Two meshes: a plane and a wall beside it. The wall occludes the plane's occlusion rays only when
        Self Occlusion is Always."""
        plane = grid(2, lambda x, z: 0 * x)
        wall = grid(2, lambda x, z: 0 * x)
        wall.vertices = wall.vertices[:, [1, 0, 2]] * [1, 0.5, 1] + [1.02, 0.5, 0]  # standing at x = 1
        wall.visual = trimesh.visual.TextureVisuals(uv=plane.visual.uv * 0.4 + 0.55)
        plane.visual = trimesh.visual.TextureVisuals(uv=plane.visual.uv * 0.5)
        lows, highs = trimesh.Scene(), trimesh.Scene()
        lows.add_geometry(plane.copy(), geom_name="floor_low")
        lows.add_geometry(wall.copy(), geom_name="wall_low")
        highs.add_geometry(plane.copy(), geom_name="floor_high")
        highs.add_geometry(wall.copy(), geom_name="wall_high")
        lows.export(self.dir / "so_low.obj")
        highs.export(self.dir / "so_high.obj")
        s = bake.Settings(size=(64, 64), high_meshes=[str(self.dir / "so_high.obj")], match="By Mesh Name",
                          dilation=0, apply_diffusion=False)

        def floor_ao(self_occlusion):
            img = read(self.baker.bake(str(self.dir / "so_low.obj"), {"ambient_occlusion_from_mesh": {
                "self_occlusion": self_occlusion, "secondary_rays": 64, "max_occluder_distance": 0.3}}, s,
                self.dir / self_occlusion.replace(" ", ""))[0])[..., 0]
            return float(img[60, 30])  # the floor next to the wall (u ~ 0.47, v ~ 0.03)
        self.assertLess(floor_ao("Always"), floor_ao("By Mesh Name") - 0.05)

    def test_color_map_sources(self):
        scene = trimesh.Scene()
        a, b = grid(2, lambda x, z: 0 * x), grid(2, lambda x, z: 0 * x + 0.001)
        b.visual = trimesh.visual.TextureVisuals(uv=a.visual.uv)
        scene.add_geometry(a, geom_name="a")
        scene.export(self.dir / "cm.obj")
        s = bake.Settings(size=(16, 16), high_meshes=[str(self.dir / "cm.obj")], frontal=0.01, rear=0.01,
                          dilation=0, apply_diffusion=False)
        img = lambda **p: read(self.baker.bake(str(self.dir / "low.obj"), {"color_map_from_mesh": p}, s,
                                               self.dir / "cm")[0])[8, 8]
        gray = img(color_source="Mesh ID", color_generator="Grayscale")
        np.testing.assert_allclose(gray, [1.0, 1.0, 1.0], atol=0.01)  # one mesh: (0 + 1) / 1
        np.testing.assert_allclose(img(color_source="Vertex Color"), [1.0, 1.0, 1.0], atol=0.01)  # no colors: white
        self.assertEqual(len(set(np.round(img(color_source="Polygroup/Submesh ID"), 3))), 3)  # a random color

    def test_world_space_direction_and_normals(self):
        s = bake.Settings(size=(16, 16), use_low_as_high=True, dilation=0, apply_diffusion=False)
        def one(baker, **p):
            return read(self.baker.bake(str(self.dir / "low.obj"), {baker: p}, s, self.dir / "ws")[0])[8, 8]
        # +Y is the plane's normal: (0, 0, 1) in tangent space; -Z is +V, up the image: green, inverted in DirectX.
        np.testing.assert_allclose(one("world_space_direction", direction=[0, 1, 0]), [0.5, 0.5, 1.0], atol=0.01)
        up = one("world_space_direction", direction=[0, 0, -1], normal_orientation="OpenGL")
        np.testing.assert_allclose(up, [0.5, 1.0, 0.5], atol=0.01)
        self.assertLess(one("world_space_direction", direction=[0, 0, -1])[1], 0.01)
        np.testing.assert_allclose(one("world_space_normals", baking_type="Tangent"), [1.0, 0.5, 0.5], atol=0.01)

    def test_position_normalizations(self):
        s = bake.Settings(size=(16, 16), use_low_as_high=True, dilation=0, apply_diffusion=False)
        def one(**p):
            return read(self.baker.bake(str(self.dir / "low.obj"), {"position": p}, s, self.dir / "pos")[0])
        box, sphere = one(), one(normalization_type="BSphere")
        self.assertLess(box[8, 1][0], 0.1)  # x from 0 at the left edge (box)
        self.assertGreater(sphere[8, 1][0], 0.1)  # the sphere is larger than the box
        axis = one(mode="One axis", axis="Z")
        np.testing.assert_allclose(axis[8, 8], [axis[8, 8][0]] * 3, atol=1e-6)  # gray

    def test_curvature_from_mesh_and_low_ao(self):
        bakers = {"curvature_from_mesh": {"automatic_tonemapping": False, "sampling_radius": 0.02,
                                          "secondary_rays": 32},
                  "ambient_occlusion": {"quality": "Low (3 pass)"}}
        maps = self.run_bake(bakers, size=96)
        curv = read(maps["curvature_from_mesh"])[..., 0]
        top = float(self.at(curv[..., None], 0.65, 0.65)[0])
        flat = float(self.at(curv[..., None], 0.1, 0.1)[0])
        self.assertAlmostEqual(flat, 0.5, delta=0.03)  # flat: 0 in [-1, 1]
        self.assertGreater(top, flat + 0.01)  # the bump's top is convex
        ao = read(maps["ambient_occlusion"])[..., 0]
        self.assertGreater(float(ao.min()), 0.99)  # the low is a flat plane: nothing occludes it

    def test_skew_correction(self):
        skew = self.dir / "white.png"
        imageio.write(skew, np.full((4, 4, 1), 255, np.uint8))
        s = bake.Settings(size=(32, 32), high_meshes=[str(self.dir / "high.obj")], frontal=0.2, rear=0.2,
                          dilation=0, apply_diffusion=False, skew_correction=True, skew_map=str(skew))
        img = read(self.baker.bake(str(self.dir / "low.obj"), {"normal_map_from_mesh": {}}, s, self.dir / "skew")[0])
        self.assertGreater(float(img[..., 2].max()), 0.9)  # bakes with the original normals as ray directions

    def test_missed_rays_give_the_neutral_value(self):
        s = bake.Settings(size=(64, 64), high_meshes=[str(self.dir / "high.obj")], frontal=0.01, rear=0.01,
                          relative_to_bounding_box=False, dilation=0, apply_diffusion=False)
        maps = {p.stem.split("low_", 1)[1]: read(p) for p in self.baker.bake(
            str(self.dir / "low.obj"), {"normal_map_from_mesh": {}, "height_map_from_mesh": {}}, s, self.dir / "miss")}
        top = self.at(maps["normal_map_from_mesh"], 0.65, 0.65)  # the bump's top is beyond the frontal distance
        np.testing.assert_allclose(top, [0.5, 0.5, 1.0], atol=0.01)
        self.assertAlmostEqual(float(self.at(maps["height_map_from_mesh"], 0.65, 0.65)[0]), 0.5, delta=0.01)

    def test_cage_visualization(self):
        from trafilatura3d import meshio
        low = meshio.load(str(self.dir / "low.obj"))[0]
        high = meshio.load(str(self.dir / "high.obj"))[0]
        s = bake.Settings(frontal=0.03, relative_to_bounding_box=False)
        cage = bake.implicit_cage(low, s)
        self.assertAlmostEqual(float(cage[0].pos[:, 1].mean()), 0.03, places=5)  # the plane pushed up
        out = bake.unreached(self.baker.device, high, low, s)[0]
        y = high[0].pos[:, 1]
        self.assertTrue(np.all(out[y > 0.035]))  # the bump's top pokes through the cage
        self.assertFalse(np.any(out[y < 0.025]))  # the rest is under it
        s.frontal = 0.2
        self.assertFalse(bake.unreached(self.baker.device, high, low, s)[0].any())
        # A high poly sunk below the rear distance is missed too.
        sunk = [dataclasses.replace(o, pos=o.pos - [0, 0.05, 0]) for o in high]
        s.rear = 0.02
        self.assertAlmostEqual(float(bake.rear_surface(low, s)[0].pos[:, 1].mean()), -0.02, places=5)
        out = bake.unreached(self.baker.device, sunk, low, s)[0]
        y = sunk[0].pos[:, 1]
        self.assertTrue(np.all(out[y < -0.025]))  # the flat part, 5 cm down
        self.assertFalse(np.any(out[(y > -0.015) & (y < 0.15)]))  # the bump's upper part is reached
        # With a cage file: reached between the cage and as far behind the low.
        s.frontal = 0.03
        out = bake.unreached(self.baker.device, high, low, s, cage=bake.implicit_cage(low, s))[0]
        self.assertTrue(np.all(out[high[0].pos[:, 1] > 0.035]))
        # Beyond the low's borders nothing is reached.
        wide = [dataclasses.replace(o, pos=o.pos * [1.5, 1.0, 1.5]) for o in high]
        out = bake.unreached(self.baker.device, wide, low, dataclasses.replace(s, frontal=0.2))[0]
        x = np.abs(wide[0].pos[:, 0])
        self.assertTrue(np.all(out[x > 1.1]))
        self.assertFalse(np.any(out[(x < 0.9) & (np.abs(wide[0].pos[:, 2]) < 0.9)]))

    def test_ignore_backface(self):
        """The common settings' "Ignore backface": the rays skip a high poly surface they meet from behind
        (a plane turned away above the low) and reach the one facing them below."""
        flipped = grid(2, lambda x, z: 0 * x + 0.05)
        flipped.faces = flipped.faces[:, ::-1]
        highs = trimesh.Scene()
        highs.add_geometry(flipped, geom_name="away")
        highs.add_geometry(grid(2, lambda x, z: 0 * x - 0.05), geom_name="facing")
        highs.export(self.dir / "bf_high.obj")

        def height(ignore):
            s = bake.Settings(size=(16, 16), high_meshes=[str(self.dir / "bf_high.obj")], frontal=0.1, rear=0.1,
                              relative_to_bounding_box=False, dilation=0, apply_diffusion=False,
                              ignore_backface=ignore)
            params = {"height_map_from_mesh": {"normalization": "Relative to ray distance"}}  # [-rear, +frontal]
            img = read(self.baker.bake(str(self.dir / "low.obj"), params, s, self.dir / f"bf{int(ignore)}")[0])
            return float(img[8, 8, 0])
        self.assertAlmostEqual(height(False), 0.75, delta=0.03)  # the first surface met, from behind
        self.assertAlmostEqual(height(True), 0.25, delta=0.03)  # the facing one below

    def test_match_by_mesh_name(self):
        def plane(y, uv_offset=None):
            m = grid(2, lambda x, z: 0 * x + y)
            if uv_offset is not None:  # a and b in separate halves of UV space
                m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv * [0.5, 1.0] + [uv_offset, 0.0])
            return m
        lows = trimesh.Scene()
        lows.add_geometry(plane(0.0, 0.0), geom_name="a_low")
        lows.add_geometry(plane(0.1, 0.5), geom_name="b_low")
        lows.export(self.dir / "pair_low.obj")
        highs = trimesh.Scene()
        highs.add_geometry(plane(0.03), geom_name="a_high")
        highs.add_geometry(plane(0.09), geom_name="b_high")  # within a_low's frontal distance, above a_high
        highs.export(self.dir / "pair_high.obj")
        from trafilatura3d import meshio
        self.assertEqual(sorted(o.name for o in meshio.load(str(self.dir / "pair_low.obj"))[0]), ["a_low", "b_low"])

        def height(match):
            s = bake.Settings(size=(32, 32), high_meshes=[str(self.dir / "pair_high.obj")], frontal=0.1, rear=0.1,
                              relative_to_bounding_box=False, match=match, dilation=0, apply_diffusion=False)
            height = {"height_map_from_mesh": {"normalization": "Relative to ray distance"}}  # [-rear, +frontal]
            img = read(self.baker.bake(str(self.dir / "pair_low.obj"), height, s,
                                       self.dir / match.replace(" ", ""))[0])
            return float(img[16, 8, 0]), float(img[16, 24, 0])  # a's half, b's half
        a, b = height("Always")
        self.assertAlmostEqual(a, 0.95, delta=0.03)  # a_low catches b_high, the first surface its ray meets
        a, b = height("By Mesh Name")
        self.assertAlmostEqual(a, 0.65, delta=0.03)  # only a_high
        self.assertAlmostEqual(b, 0.45, delta=0.03)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def quiet(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli(argv)
        return code, out.getvalue(), err.getvalue()

    def package(self):
        from trafilatura3d.package import Package
        from trafilatura3d.nodes.base import Param, ParamKind
        pkg = Package()
        g = pkg.new_graph("main", 5)
        u = g.add_node("uniform_color")
        o = g.add_node("output")
        g.connect(u, "output", o, "input")
        g.expose("tint", Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0]))
        g.link(u, "output_color", "tint")
        g.presets = [{"name": "red", "values": {"tint": [1.0, 0.0, 0.0, 1.0]}}]
        path = self.dir / "p.trafilatura3d"
        pkg.save(path)
        return path

    def test_info_and_cook_overrides(self):
        path = self.package()
        code, out, _ = self.quiet(["info", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("input  tint", out)
        self.assertIn("preset red", out)
        code, out, _ = self.quiet(["cook", str(path), "--out", str(self.dir / "a"), "--preset", "red"])
        self.assertEqual(code, 0, out)
        img = read(Path(out.splitlines()[0]))
        self.assertGreater(img[0, 0, 0], 0.9)
        self.assertLess(img[0, 0, 1], 0.1)
        code, out, _ = self.quiet(["cook", str(path), "--out", str(self.dir / "b"), "--preset", "red",
                                   "--set", "tint=[0,0,1,1]"])
        img = read(Path(out.splitlines()[0]))
        self.assertLess(img[0, 0, 0], 0.1)
        self.assertGreater(img[0, 0, 2], 0.9)
        code, _, err = self.quiet(["cook", str(path), "--out", str(self.dir / "c"), "--set", "nope=1"])
        self.assertEqual(code, 1)
        self.assertIn("no input 'nope'", err)

    def test_bake(self):
        grid(2, lambda x, z: 0 * x).export(self.dir / "low.obj")
        grid(40, bump).export(self.dir / "high.obj")
        code, out, err = self.quiet(["bake", str(self.dir / "low.obj"), "--out", str(self.dir / "maps"),
                                     "--baker", "normal_map_from_mesh", "--baker", "ambient_occlusion_from_mesh",
                                     "--high", str(self.dir / "high.obj"), "--size", "32", "--frontal", "0.2",
                                     "--rear", "0.2", "--param", "ambient_occlusion_from_mesh.secondary_rays=8"])
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(Path(p).name for p in out.splitlines()),
                         ["low_ambient_occlusion_from_mesh.png", "low_normal_map_from_mesh.png"])
        code, _, err = self.quiet(["bake", str(self.dir / "low.obj"), "--out", str(self.dir / "m"), "--baker",
                                   "position", "--param", "position.rays=3"])
        self.assertEqual(code, 1)


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "GUI test: run with QT_QPA_PLATFORM=offscreen")
class PackageFilesTest(unittest.TestCase):
    def test_baking_files_are_relative_in_the_file(self):
        """High poly, cage, output folder and file parameters: relative to the package in its file, absolute in
        memory; saving elsewhere rewrites them, saving never touches the configuration in memory."""
        import json
        from trafilatura3d.package import Package, Resource
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp).resolve()
            cfg = {"settings": {"high_meshes": [str(d / "meshes" / "high.fbx")], "cage_file": str(d / "cage.fbx")},
                   "folder": str(d / "maps"), "name": "$(mesh)_$(bakername)",
                   "bakers": [{"id": "ambient_occlusion", "params": {"normal_map": str(d / "maps" / "n.png")}}]}
            pkg = Package()
            pkg.new_graph("main", 5)
            pkg.resources["low"] = Resource("scene", "low.fbx", baking=cfg)
            pkg.save(d / "p.trafilatura3d")
            stored = json.loads((d / "p.trafilatura3d").read_text())["resources"]["low"]["baking"]
            self.assertEqual(stored["settings"], {"high_meshes": ["meshes/high.fbx"], "cage_file": "cage.fbx"})
            self.assertEqual((stored["folder"], stored["bakers"][0]["params"]["normal_map"]), ("maps", "maps/n.png"))
            self.assertEqual(pkg.resources["low"].baking["folder"], str(d / "maps"))  # memory untouched
            pkg.save(d / "sub" / "q.trafilatura3d")
            moved = json.loads((d / "sub" / "q.trafilatura3d").read_text())["resources"]["low"]["baking"]
            self.assertEqual((moved["settings"]["high_meshes"], moved["folder"]), (["../meshes/high.fbx"], "../maps"))
            again = Package.load(d / "sub" / "q.trafilatura3d").resources["low"].baking
            self.assertEqual(again["settings"]["high_meshes"], [str(d / "meshes" / "high.fbx")])
            self.assertEqual((again["folder"], again["bakers"][0]["params"]["normal_map"]), (str(d / "maps"), str(d / "maps" / "n.png")))


class DialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.setOrganizationName("Trafilatura3DTest")  # never the user's settings (recent files, layout)
        QCoreApplication.setApplicationName("Trafilatura3DTest")

    def test_bake_and_link(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from trafilatura3d.editor.bakers import BakerDialog
        from trafilatura3d.editor.document import Document
        from trafilatura3d.package import Package, Resource
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            grid(2, lambda x, z: 0 * x).export(d / "low.obj")
            grid(40, bump).export(d / "high.obj")
            pkg = Package()
            pkg.new_graph("main", 5)
            pkg.resources["low"] = Resource("scene", "low.obj")
            pkg.save(d / "p.trafilatura3d")
            doc = Document()
            pkg = doc.open(d / "p.trafilatura3d")
            dlg = BakerDialog(doc, pkg, "low")
            dlg._add("normal_map_from_mesh")
            dlg._add("opacity_mask_from_mesh")
            dlg.settings.high_meshes = [str(d / "high.obj")]
            dlg.settings.frontal = dlg.settings.rear = 0.2
            dlg.width_.setCurrentText("128")
            dlg.height_.setCurrentText("128")
            dlg.list.item(2).setCheckState(Qt.Unchecked)  # opacity off
            paths = dlg.render()
            self.assertEqual([p.name for p in paths], ["low_normal_map_from_mesh.png"])
            self.assertEqual(pkg.resources["low_normal_map_from_mesh"].kind, "bitmap")
            self.assertEqual(pkg.resources["low"].baking["bakers"][1]["enabled"], False)
            pkg.save()
            again = Package.load(d / "p.trafilatura3d")
            self.assertEqual(again.resources["low"].baking["settings"]["size"], [128, 128])
            dlg.close()
            app.processEvents()
            doc.undo.indexChanged.disconnect()  # the stack's teardown would signal a deleted Document

    def test_3d_view_shows_the_bake(self):
        """A linked mesh opens in the 3D View, its baked maps go on it, the high poly and cage show, and a
        baked bitmap dragged into a graph becomes a Bitmap node."""
        from PySide6.QtCore import QByteArray, QMimeData, QPointF
        from PySide6.QtWidgets import QApplication
        import rendercanvas.qt as rq
        app = QApplication.instance() or QApplication([])
        init = rq.QRenderWidget.__init__
        rq.QRenderWidget.__init__ = lambda self, *a, **k: init(self, *a, **{**k, "present_method": "bitmap"})
        try:
            from trafilatura3d.editor.bakers import BakerDialog
            from trafilatura3d.editor.mainwindow import MainWindow
            from trafilatura3d.package import Package, Resource
            with tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                grid(2, lambda x, z: 0 * x).export(d / "low.obj")
                grid(60, bump).export(d / "high.obj")
                pkg = Package()
                pkg.new_graph("main", 5)
                pkg.resources["low"] = Resource("scene", "low.obj")
                pkg.resources["low"].baking = {
                    "settings": {"size": [64, 64], "high_meshes": [str(d / "high.obj")], "frontal": 0.03,
                                 "rear": 0.03, "relative_to_bounding_box": False},
                    "bakers": [{"id": "normal_map_from_mesh", "enabled": True, "params": {}},
                               {"id": "ambient_occlusion_from_mesh", "enabled": True, "params": {"secondary_rays": 8}}],
                    "folder": str(d / "maps"), "link": True}
                pkg.save(d / "p.trafilatura3d")
                w = MainWindow([str(d / "p.trafilatura3d")])
                p, v = w.doc.packages[0], w.view3d
                v.load_resource(p, "low")  # no saved scene state: the mesh alone
                self.assertEqual(Path(v.scene.source).name, "low.obj")
                self.assertNotIn("normal", v.view.renderer.materials[0]["spec"])
                BakerDialog(w.doc, p, "low", w).render()
                w.doc.live.refresh_now()  # the edit's refresh, otherwise a step per turn of the event loop
                spec = v.view.renderer.materials[0]["spec"]
                self.assertIn("normal", spec)
                self.assertIn("ambientOcclusion", spec)
                v._set_bake_look("show_high", True)
                v._set_bake_look("show_cage", True)
                v._set_bake_look("show_rear", True)
                r = v.view.renderer
                self.assertEqual((len(r.bake_high), len(r.bake_cage), len(r.bake_rear)), (1, 1, 1))
                v.render_image(64, 48)  # draws the overlay
                # The Baking window is not modal, one per resource, and edits reach the 3D View live in one
                # undo step.
                from trafilatura3d.editor.bakers import open_baker
                dlg = open_baker(w.doc, p, "low", w)
                self.assertIs(open_baker(w.doc, p, "low", w), dlg)
                self.assertFalse(dlg.isModal())
                steps = w.doc.undo.count()
                for f in (0.02, 0.01):
                    dlg.settings.frontal = f
                    dlg._save()
                self.assertEqual(w.doc.undo.count(), steps + 1)
                self.assertEqual(p.resources["low"].baking["settings"]["frontal"], 0.01)
                self.assertEqual(v._overlay_key[4], 0.01)  # the cage followed
                dlg.close()
                mime = QMimeData()
                mime.setData("application/x-trafilatura3d-resource", QByteArray(b"0\nlow_normal_map_from_mesh"))
                w.open_graph(p, "main")
                self.assertTrue(w.graph._drop(mime, QPointF(0, 0)))
                node = next(n for n in p.graphs["main"].nodes.values() if n.type == "bitmap")
                self.assertEqual(node.params["pkg_resource_path"], "maps/low_normal_map_from_mesh.png")
                w.doc.undo.indexChanged.disconnect()  # no close(): it would ask to save (modal)
                w.graph.scene.selectionChanged.disconnect()  # nor signals from widgets torn down at exit
                app.processEvents()
        finally:
            rq.QRenderWidget.__init__ = init


if __name__ == "__main__":
    unittest.main()
