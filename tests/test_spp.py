""".spp import: the document's tagged tree, the container's images, the scene dump, the conversion to a Paint project
and the strokes replayed where they were painted (a quad in front of a known camera)."""

import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trafilatura3d.evaluator import Evaluator
from trafilatura3d.spp import alphas, document, import_spp, procedurals
from trafilatura3d.spp.container import TILE_LIST, decode_image, tile_list
from trafilatura3d.spp.convert import srgb
from trafilatura3d.spp.scene import parse as parse_scene

EV = None


def ev():
    global EV
    if EV is None:
        EV = Evaluator()
    return EV


# --- writers of the formats (the reader's layouts, written back) -------------------------------------------------

class Doc:
    """A tagged tree in the document layout: objects {"@": class, ...}, lists, numbers, strings, bools; a tuple
    (tag, value) forces a tag."""

    def __init__(self):
        self.b = bytearray(b"\0" * document.HEADER)

    def string(self, s: str) -> None:
        raw = s.encode("utf-8")
        self.b += struct.pack("<I", len(raw)) + raw

    def value(self, v) -> None:
        b = self.b
        if isinstance(v, tuple):
            tag, v = v
            b.append(tag)
            fmt = {0x0c: "<Q", 0x15: "<Q", 0x0e: "<16f", 0x06: "<2i", 0x0b: "<i"}[tag]
            b += struct.pack(fmt, *(v if isinstance(v, list) else [v]))
        elif isinstance(v, dict):
            b += bytes([0x12, 0])
            at = len(b)
            b += b"\0\0\0\0"
            self.string(v["@"])
            fields = [k for k in v if k != "@"]
            b += struct.pack("<H", len(fields))
            for k in fields:
                self.string(k)
                self.value(v[k])
            b[at:at + 4] = struct.pack("<I", len(b))
        elif v is None:
            b += bytes([0x12, 0xFF])
        elif isinstance(v, list) and v and all(isinstance(x, float) for x in v) and len(v) in (2, 3):
            b.append({2: 0x02, 3: 0x03}[len(v)])
            b += struct.pack(f"<{len(v)}f", *v)
        elif isinstance(v, list):
            b.append(0x13)
            at = len(b)
            b += b"\0\0\0\0" + struct.pack("<I", len(v))
            for x in v:
                self.value(x)
            b[at:at + 4] = struct.pack("<I", len(b))
        elif isinstance(v, bool):
            b += bytes([0x0A, int(v)])
        elif isinstance(v, int):
            b.append(0x09)
            b += struct.pack("<q", v)
        elif isinstance(v, float):
            b.append(0x01)
            b += struct.pack("<f", v)
        elif isinstance(v, str):
            b.append(0x10)
            self.string(v)
        else:
            raise TypeError(v)


def doc_bytes(root) -> bytes:
    d = Doc()
    d.value(root)
    return bytes(d.b)


def qt_image(pixels: np.ndarray, fmt: str = "LUM8") -> bytes:
    """An image record: the header map, then the LZ4 block."""
    import lz4.block
    h, w, _ = pixels.shape
    raw = np.ascontiguousarray(pixels).tobytes()
    block = lz4.block.compress(raw, store_size=False)

    def key(k):
        e = k.encode("utf-16-be")
        return struct.pack(">I", len(e)) + e
    entries = [("width", 3, w), ("height", 3, h), ("pitch", 3, raw.__len__() // h), ("format", 10, fmt),
               ("dataSizeRaw", 3, len(raw)), ("dataSizeLZ4", 3, len(block)), ("compression", 10, "lz4")]
    out = struct.pack(">I", len(entries))
    for k, t, v in entries:
        out += key(k) + struct.pack(">IB", t, 0)
        out += struct.pack(">I", v) if t == 3 else key(v)
    return out + block


def scene_bytes(pos, nrm, uv, tri, uid: int, center=(0.0, 0.0, 0.0), radius=1.0) -> bytes:
    """A scene dump of one mesh and one material."""
    def string(s):
        return struct.pack("<I", len(s)) + s.encode()

    def chunk(body):
        return struct.pack("<Q", len(body) + 8) + body
    keys = ["ALG_PRE_NORMALIZATION_CENTER", "ALG_PRE_NORMALIZATION_RADIUS"]
    meta = struct.pack("<HI", 1, 2) + b"".join(string(k) for k in keys) + struct.pack("<I", 2)
    meta += struct.pack("<I3f", 5, *center) + struct.pack("<If", 3, radius)
    root = chunk(string("RootNode") + np.eye(4, dtype="<f4").tobytes() + struct.pack("<3I", 0, 0, 0) + meta)
    nv = len(pos)
    body = string("mesh") + struct.pack("<4I", 4, 0, 0x103, nv) + np.asarray(pos, "<f4").tobytes()
    body += struct.pack("<I", nv) + np.asarray(nrm, "<f4").tobytes()
    uv3 = np.concatenate([np.asarray(uv, "<f4"), np.zeros((nv, 1), "<f4")], 1)
    body += struct.pack("<II", 2, nv) + uv3.tobytes()
    faces = np.concatenate([np.full((len(tri), 1), 3), np.asarray(tri), np.zeros((len(tri), 1))], 1)
    body += struct.pack("<II", len(tri), 1) + faces.astype("<u4").tobytes()
    material = chunk(string("$mat.uid") + struct.pack("<4IQ", 0, 0, 5, 8, uid))
    scene = struct.pack("<I", 8) + root + struct.pack("<I", 1) + chunk(body) + struct.pack("<I", 1) + material
    scene += struct.pack("<4I", 0, 0, 0, 0)
    head = bytearray(512)
    head[:19] = b"ASSIMP.binary-dump."
    head[44:64] = struct.pack("<4I2H", 1, 6, 0, 0, 0, 0)
    return bytes(head) + struct.pack("<Q", len(scene) + 8) + scene


# --- a project: a quad in the XY plane (normals +Z, UV 0-1) seen from z = 3 -----------------------------------

UID = 1234
F = 3.0  # the camera's focal factor: a screen point at x = 0.75 sees the quad at x = 0.5 (u = 0.75)


def gl_matrices():
    """(view, projection) of the camera, stored for row vectors."""
    view = np.eye(4)
    view[3, 2] = -3.0  # world z 0 -> camera z -3
    n, f = 0.1, 100.0
    proj = np.array([[F, 0, 0, 0], [0, F, 0, 0], [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)], [0, 0, -1, 0]])
    return [float(x) for x in view.ravel()], [float(x) for x in proj.T.ravel()]


def stroke(x, y):
    view, proj = gl_matrices()
    return {"@": "DataStroke", "viewType": 1, "viewMatrix": (0x0e, view), "projectionMatrix": (0x0e, proj),
            "points": [{"@": "DataStrokeKey", "position": [float(x), float(y)], "time": 0.0, "type": 0}],
            "pressure": [], "duplicateCount": 0, "duplicateMatrix": (0x0e, [0.0] * 16), "randomSeed": 1}


def paint_action(strokes, value=1.0, size=0.1):
    return {"@": "DataActionPaint", "enabled": True, "opacity": 1.0, "stampBlendingMode": 0,
            "brush": {"@": "DataBrush", "alignment": 0, "brushSize": size, "brushSizeSpace": 1, "flow": 1.0,
                      "brushSizeJitter": 0.0, "flowJitter": 0.0, "cullingEnabled": True, "cullingAngle": 0.5,
                      "stampsEmitter": {"@": "DataEmitterStroke", "spacing": 0.1, "rotation": 0.0,
                                        "rotationJitter": 0.0, "positionJitter": 0.0, "followPath": False,
                                        "pressureSizeEnabled": False, "pressureFlowEnabled": False}},
            "sourceColor": [{"@": "DataSourceUniform", "channelTypes": (0x0c, 0), "color": [value] * 3}],
            "sourceTransparent": [{"@": "DataSourceProcedural", "channelTypes": (0x0c, 0), "mode": 0,
                                   "procedural": {"@": "DataProcedural",
                                                  "urlToSbsRes": "resource://lib/shape/shape?version=0.sbsar",
                                                  "tweaks": [{"@": "DataTweakFloat", "identifier": "hardness",
                                                              "value": 1.0}]}}],
            "strokes": strokes}


def project_doc(color=(0.2, 0.2, 0.2), mask_actions=None, blending=None):
    fill = {"@": "DataActionFill", "projection": 0, "filtering": 2, "uvSamplingWrap": 3,
            "sources": [{"@": "DataSourceUniform", "channelTypes": (0x0c, 1), "color": [float(c) for c in color]}]}
    layer = {"@": "DataLayerColor", "label": "Painted", "enabled": True, "maskInitial": 0, "maskEnabled": True,
             "perChannelBlending": blending or [], "actions": {"@": "DataStackActions", "items": [fill]},
             "maskActions": {"@": "DataStackActions", "items": mask_actions or []}}
    stack = {"@": "DataMaterialStack",
             "channels": [{"@": "DataChannel", "format": 0, "type": (0x15, 1)},
                          {"@": "DataChannel", "format": 1, "type": (0x15, 128)}],
             "stack": {"@": "DataStackLayers", "items": [layer]}}
    material = {"@": "DataMaterial", "sceneMaterialName": "Quad", "sceneMaterialUID": (0x0c, UID),
                "resolutionLog2": (0x06, [7, 7]), "useUVTiles": False, "uvTiles": [], "additionalMaps": [],
                "normalMapBitmap": None, "stacks": [stack]}
    return {"@": "DataDocument", "materials": [material], "normalMapSyscoord": 0}


def write_spp(folder: Path, root) -> Path:
    import h5py
    pos = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    uv = [[0, 0], [1, 0], [1, 1], [0, 1]]
    path = folder / "quad.spp"
    with h5py.File(path, "w") as f:
        f["paint/document.bin"] = np.frombuffer(doc_bytes(root), np.uint8)
        f["paint/scene.assbin"] = np.frombuffer(
            scene_bytes(pos, [[0, 0, 1]] * 4, uv, [[0, 1, 2], [0, 2, 3]], UID), np.uint8)
    return path


# --- tests --------------------------------------------------------------------------------------------------------

class TestFormats(unittest.TestCase):
    def test_document_round_trip(self):
        root = {"@": "Root", "a": 1, "f": 0.5, "s": "text", "b": True, "none": None, "v": [1.0, 2.0, 3.0],
                "bits": (0x0c, (1 << 64) - 1), "items": [{"@": "Item", "x": 2}, {"@": "Item", "x": 3}]}
        got = document.parse(doc_bytes(root))
        self.assertEqual(got["@"], "Root")
        self.assertEqual((got["a"], got["f"], got["s"], got["b"], got["none"]), (1, 0.5, "text", True, None))
        self.assertEqual(got["v"], [1.0, 2.0, 3.0])
        self.assertEqual(got["bits"], (1 << 64) - 1)
        self.assertEqual([i["x"] for i in document.walk(got, "Item")], [2, 3])

    def test_document_checks_its_end_offsets(self):
        data = bytearray(doc_bytes({"@": "Root", "a": 1}))
        data[document.HEADER + 2] += 1  # the root's end offset
        with self.assertRaises(document.DocumentError):
            document.parse(bytes(data))

    def test_image_record(self):
        px = (np.arange(16 * 8, dtype=np.uint8).reshape(8, 16, 1) * 2)
        a, head = decode_image(qt_image(px))
        self.assertEqual(a.shape, (8, 16, 1))
        np.testing.assert_allclose(a[..., 0], px[..., 0] / 255.0)
        bgra = np.zeros((2, 2, 4), np.uint8)
        bgra[..., 0], bgra[..., 2], bgra[..., 3] = 10, 200, 255  # blue, red
        a, _ = decode_image(qt_image(bgra, "BGRA8"))
        np.testing.assert_allclose(a[0, 0], [200 / 255, 0, 10 / 255, 1.0])

    def test_tile_list(self):
        names = {1001: "a" * 40 + ".image", 1012: "b" * 40 + ".image"}
        data = TILE_LIST + struct.pack(">3I", 1, 0, 2)
        for udim, n in names.items():
            u, v = (udim - 1001) % 10, (udim - 1001) // 10
            data += struct.pack(">3I", u, v, len(n)) + n.encode()
        self.assertEqual(tile_list(data), names)

    def test_scene(self):
        pos = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        s = parse_scene(scene_bytes(pos, [[0, 0, 1]] * 3, [[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 77,
                                    center=(1, 2, 3), radius=4.0))
        self.assertEqual(s.materials, [77])
        self.assertEqual((s.center.tolist(), s.radius), ([1, 2, 3], 4.0))
        self.assertEqual(len(s.meshes), 1)
        np.testing.assert_allclose(s.meshes[0].pos, pos)
        np.testing.assert_array_equal(s.meshes[0].tri, [[0, 1, 2]])


class TestMapping(unittest.TestCase):
    def test_filter_parameters(self):
        res = procedurals.resolve({"urlToSbsRes": "resource://lib/blur/blur?version=0.sbsar",
                                   "tweaks": [{"identifier": "intensity", "value": 2.5},
                                              {"identifier": "$outputsize", "value": [8, 8]}]})
        self.assertEqual(res.type, "paint_blur")
        self.assertEqual(res.values["blur_intensity"], 2.5)
        self.assertEqual(res.unmatched, [])

    def test_unknown_graph(self):
        self.assertIsNone(procedurals.resolve({"urlToSbsRes": "resource://lib/nothing/nothing?version=0.sbsar"}))

    def test_srgb(self):
        np.testing.assert_allclose(srgb([0.0, 0.2140, 1.0]), [0.0, 0.5, 1.0], atol=1e-3)


class TestAlphas(unittest.TestCase):
    def test_hardness_levels(self):
        b = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(alphas.levels(b, 0.5), b)  # one half: the base as it is
        np.testing.assert_allclose(alphas.levels(b, 0.25), [0, 0, 0, 0.5, 1])  # input black at 0.5
        np.testing.assert_allclose(alphas.levels(b, 0.75), [0, 0.5, 1, 1, 1])  # input white at 0.5

    def test_shape_falloff(self):
        n = 64
        np.testing.assert_array_equal(alphas.falloff(n, 0.0), 1.0)
        m = alphas.falloff(n, 0.5)
        self.assertEqual(m[n // 2, n // 2], 1.0)  # inside 1 - shape: untouched
        self.assertLess(m[0, 0], 0.01)  # the corners fade out
        np.testing.assert_allclose(m, m.T)
        base = np.full((n, n), 0.8)
        a = alphas.shaped(base, 0.5, 0.5, False)
        np.testing.assert_allclose(a, np.clip(0.8 + m - 1, 0, 1))
        np.testing.assert_allclose(alphas.shaped(base, 0.5, 0.0, True), 0.2)  # inverted

    def test_disc_and_dots(self):
        d = alphas.disc(1.0, 128)
        self.assertEqual(d[64, 64], 1.0)
        self.assertEqual(d[64, 10], 0.0)  # radius: half the frame
        self.assertEqual(d[64, 40], 1.0)
        soft = alphas.disc(0.0, 128)
        self.assertAlmostEqual(float(soft[64, 32]), 0.5, delta=0.05)  # the edge centered on the radius
        tweaks = {"layer-01_amount": 60, "layer-01_size": 0.17, "spawn": 0.5}
        a = alphas.dots(tweaks, 3, 256)
        self.assertGreater(a.max(), 0.9)
        t = (np.arange(256) + 0.5) / 128 - 1
        far = np.hypot(t[:, None], t[None, :]) > 0.5 + 0.06
        self.assertEqual(a[far].max(), 0.0)  # within `spawn`
        np.testing.assert_array_equal(a, alphas.dots(tweaks, 3, 256))  # deterministic

    def test_bases_from_our_generators(self):
        from trafilatura3d.nodes import NODES
        for name, spec in alphas.BASES.items():
            for node in (spec.node, spec.modulator):
                self.assertTrue(node is None or node in NODES, (name, node))
        a = alphas.alpha(Evaluator(), "alpha_dirt_06", {"hardness": 0.5, "shape": 0.75, "invert": 0})
        self.assertEqual(a.shape, (512, 512))
        self.assertEqual(a[0, 0], 0.0)  # the falloff empties the border
        self.assertGreater(a.max(), 0.5)


class TestImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, root, replay=True):
        return import_spp(write_spp(self.folder, root), evaluator=ev(), replay=replay)

    def test_project(self):
        s = self.load(project_doc(blending=[{"@": "DataBlending", "blendingMode": 4, "opacity": 0.5,
                                             "channelTypes": (0x0c, 1)}]), replay=False)
        ts = s.project.texture_sets[0]
        self.assertEqual((ts.name, ts.size), ("Quad", [128, 128]))
        self.assertEqual([(c.id, c.format) for c in ts.channels], [("basecolor", "sRGB8"), ("roughness", "L8")])
        layer = ts.layers[0]
        self.assertEqual((layer.kind, layer.name), ("fill", "Painted"))
        np.testing.assert_allclose(layer.material["basecolor"].color, srgb([0.2] * 3))
        self.assertEqual((layer.blend["basecolor"].mode, layer.blend["basecolor"].opacity), ("multiply", 0.5))
        self.assertEqual(layer.mask.background, 0.0)

    def test_strokes_land_where_they_were_painted(self):
        group = {"@": "DataActionGroup", "enabled": True, "perChannelBlending": [],
                 "subStack": {"@": "DataStackActions", "items": [paint_action([stroke(0.75, 0.5)]),
                                                                 paint_action([stroke(0.5, 0.75)])]}}
        s = self.load(project_doc(mask_actions=[group]))
        ts = s.project.texture_sets[0]
        fx = ts.layers[0].mask.effects[0]
        self.assertEqual(fx.kind, "paint")
        a = s.store.read(fx.uid, "mask", 1001)[..., 0]
        h, w = a.shape

        def at(u, v):
            return float(a[int((1 - v) * h), int(u * w)])
        self.assertGreater(at(0.75, 0.5), 0.9)  # right of the center
        self.assertGreater(at(0.5, 0.75), 0.9)  # above it
        for u, v in ((0.25, 0.5), (0.5, 0.25), (0.5, 0.5)):
            self.assertEqual(at(u, v), 0.0, (u, v))
