"""Painted effects made again: every stroke of a .spp paint group replayed with our brush into the effect's pixels.

A stroke holds its pointer samples in the viewport's normalized coordinates (x right, y up, 0-1), the view and
projection matrices of its camera (row vectors, clip space z in -1..1) in the project's normalized space (the
scene's normalization: position = (file position - center) / radius), the pressure over time and, with symmetry,
the matrix of the mirrored copy. Strokes painted in the 2D view have no camera: their samples are UVs.

The brush's numbers: size a fraction (viewport space: of the viewport's height; object space: of the bounding box
diagonal; texture space: of the UV square), spacing a fraction of the size, jitters 0-1 (rotation in radians),
alignment 0 camera, 1 tangent wrap, 2 tangent planar, 3 UV; size space 0 object, 1 viewport, 2 texture.
Polygon fill hits: a click (screen position, inverse view projection) or a rectangle, granularity 0 triangle,
1 polygon, 2 mesh, 3 UV chunk.
"""

from __future__ import annotations

import math
import re

import numpy as np
import wgpu

from trafilatura3d.paint import alphas
from trafilatura3d.paint.brush import BrushSettings, Stroke, Target, TileTargets, View
from trafilatura3d.paint.model import MASK_KEY
from trafilatura3d.paint.paint import storage_for
from trafilatura3d.spp import alphas as spp_alphas
from trafilatura3d.spp import procedurals
from trafilatura3d.spp.container import Container, decode_image
from trafilatura3d.paint.channels import CHANNELS
from trafilatura3d.spp.convert import ALL_BITS, CHANNEL_BITS, srgb
from trafilatura3d.spp.document import CLASS

VIEW = 2048  # the camera buffer's side: the viewport the strokes are replayed in
ALIGNMENTS = {0: "Camera", 1: "Tangent | Wrap", 2: "Tangent | Planar"}
SIZE_SPACES = {0: "Object", 1: "Viewport"}
STAMPS_BLENDING = {0: "Normal", 1: "Lighten"}  # 1: only seen with brushes made from another app's presets
FILL_MODES = {0: "Triangle Fill", 1: "Polygon Fill", 2: "Mesh Fill", 3: "UV Chunk Fill"}
# The brush's size: our Size (percent; Object: of half the bounding box diagonal, Viewport: of half the view's
# height, both as a diameter) per unit of the file's size.
SIZE_SCALE = 200.0
# Z of OpenGL clip space (-w..w) to ours (0..w)
GL_TO_OURS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 1]], np.float64)


def _matrix(values) -> np.ndarray:
    """A 4x4 stored for row vectors as the matrix for column vectors."""
    return np.asarray(values, np.float64).reshape(4, 4).T


class Replayer:
    def __init__(self, session, conv, scene, path, say=None):
        self.s, self.conv, self.say = session, conv, say or (lambda text: None)
        self.path = path
        # ours -> normalized: p_n = A p + b
        center, k = session.fit
        r = scene.radius
        self.to_norm = np.identity(4)
        self.to_norm[:3, :3] /= k * r
        self.to_norm[:3, 3] = (np.asarray(center, np.float64) - scene.center) / r
        self.from_norm = np.linalg.inv(self.to_norm)
        self.engine = session.brush_engine()
        self._meshes = None
        self._cameras: dict = {}
        self._alpha_cache: dict = {}
        self.stamps = self.strokes = 0

    def note(self, where, text):
        self.conv.note(where, text)

    # --- GPU meshes and cameras ----------------------------------------------------------------------------------

    def meshes(self):
        if self._meshes is None:
            d = self.s.ev.device
            mk = lambda a, u: d.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(), usage=u)  # noqa: E731
            V, I = wgpu.BufferUsage.VERTEX, wgpu.BufferUsage.INDEX
            self._meshes = [([mk(o.pos, V), mk(o.nrm, V), mk(o.tan, V), mk(o.uv, V)], mk(o.idx, I),
                             [(x.start, x.count, x.material) for x in o.submeshes]) for o in self.s.objects]
        return self._meshes

    def view(self, view_m, proj_m) -> View:
        """Our View of a stroke's camera (VIEW pixels a side)."""
        V, P = _matrix(view_m), _matrix(proj_m)
        vp = GL_TO_OURS @ P @ V @ self.to_norm
        inv = np.linalg.inv(V)
        ortho = abs(P[3, 3] - 1.0) < 1e-6 and abs(P[3, 2]) < 1e-9
        if ortho:  # an orthographic view: a far eye with the field of view that gives the same pixels per unit
            half = 1.0 / max(abs(P[1, 1]), 1e-9)  # normalized units
            forward = -inv[:3, 2] / max(np.linalg.norm(inv[:3, 2]), 1e-12)
            eye_n = inv[:3, 3] - forward * half * 1000.0
            fov = 2 * math.atan(1 / 1000.0)
        else:
            eye_n = inv[:3, 3]
            fov = 2 * math.atan(1.0 / max(abs(P[1, 1]), 1e-9))
        eye = (self.from_norm @ np.array([*eye_n, 1.0]))[:3]
        return View(vp.astype(np.float32), eye, fov, VIEW, VIEW)

    def camera(self, view: View):
        key = view.view_proj.tobytes()
        if key not in self._cameras:
            if len(self._cameras) > 4:
                self.release()
            self._cameras[key] = self.engine.camera_buffer(self.meshes(), view)
        return self._cameras[key]

    def release(self) -> None:
        for cam in self._cameras.values():
            for t in (cam.texture, cam.normals, cam.uvs):
                if t is not None:
                    t.destroy()
        self._cameras.clear()

    def symmetry(self, stroke) -> list | None:
        """The stroke's mirrored copy as `brush.symmetry_copies` gives it, in our space."""
        if not stroke.get("duplicateCount"):
            return None
        out = []
        D = _matrix(stroke["duplicateMatrix"])
        m4 = self.from_norm @ D @ self.to_norm
        m, t = m4[:3, :3], m4[:3, 3]
        out.append((m, t, bool(np.linalg.det(m) < 0), 0))
        if stroke["duplicateCount"] > 1:
            self.note("", f"strokes with {stroke['duplicateCount']} symmetric copies: one copy replayed")
        return out

    # --- alphas and values ---------------------------------------------------------------------------------------

    def alpha(self, action, where) -> str:
        """The brush alpha of a paint action, ready in the brush engine."""
        srcs = action.get("sourceTransparent") or []
        if not srcs:
            return "Round hard"
        s = srcs[0]
        if s[CLASS] == "DataSourceUniform":
            return "Square"
        if s[CLASS] == "DataSourceBitmap":
            return self._bitmap_alpha(s["bitmap"]["urlToBitmapRes"], where)
        if s[CLASS] != "DataSourceProcedural":
            return alphas.DEFAULT_ALPHA
        proc = s["procedural"]
        package, graph = procedurals.graph_of(proc["urlToSbsRes"])
        tweaks = {t["identifier"]: t["value"] for t in proc.get("tweaks", [])}
        if graph in ("shape", "RoundBrush"):
            hardness = float(np.clip(tweaks.get("hardness", 0.5), 0.0, 1.0))
            key = f"round:{hardness:.3f}"
            if key not in self._alpha_cache:
                self.engine.alpha(key, alphas.round_(hardness)(256))
                self._alpha_cache[key] = key
            return key
        if graph in spp_alphas.NAMES:
            values = {k: v for k, v in tweaks.items() if not k.startswith("$")}
            seed = int(tweaks.get("$randomseed", 0))
            key = f"{graph}:{seed}:" + ":".join(f"{k}={v}" for k, v in sorted(values.items()))
            if key not in self._alpha_cache:
                self.engine.alpha(key, spp_alphas.alpha(self.s.ev, graph, values, seed))
                self._alpha_cache[key] = key
            if graph == "shape_inky":
                self.note(where, "brush alpha shape_inky: one shape for every stamp (the original varies it)")
            return key
        for si in proc.get("sourcesInputs", []):  # an alpha made from an image (brush presets): that image
            src = si.get("source") or {}
            if src.get(CLASS) == "DataSourceBitmap":
                return self._bitmap_alpha(src["bitmap"]["urlToBitmapRes"], where)
        name = "Grunge 1" if "dirt" in graph else "Cracks 1" if "crack" in graph else "Splatter 1"
        self.note(where, f"brush alpha {graph}: no counterpart, {name!r} used")
        return name

    def _bitmap_alpha(self, url: str, where: str) -> str:
        if url in self._alpha_cache:
            return self._alpha_cache[url]
        with Container(self.path) as c:
            ds = c.resource(url)
            if ds is None:
                self.note(where, f"brush alpha {url}: not in the file, Round soft used")
                return alphas.DEFAULT_ALPHA
            pixels, _ = decode_image(c.bytes(ds))
        gray = pixels[..., 0] if pixels.shape[2] < 3 else pixels[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        key = "image:" + re.sub(r"\?.*", "", url)
        self.engine.alpha(key, np.asarray(gray, np.float32))
        self._alpha_cache[url] = key
        return key

    def values(self, action, mask: bool, ts, where) -> dict:
        """{key: RGB} a paint action lays: on the mask its gray, else per channel."""
        out = {}
        for s in action.get("sourceColor") or []:
            if s[CLASS] != "DataSourceUniform":
                self.note(where, f"painting with a {s[CLASS][10:].lower()} source: its uniform value is used")
                c = s.get("color", [1.0, 1.0, 1.0])
            else:
                c = s["color"]
            c = [float(v) for v in c][:3]
            if mask:
                out[MASK_KEY] = (c[0],) * 3
                continue
            bits = s["channelTypes"]
            chans = ts.channel_ids() if bits in (0, ALL_BITS) else [
                CHANNEL_BITS[b] for b in CHANNEL_BITS if bits & b and CHANNEL_BITS[b] in ts.channel_ids()]
            for ch in chans:
                out[ch] = tuple(srgb(c) if CHANNELS[ch].color else c)
        if mask and not out:
            out[MASK_KEY] = (1.0, 1.0, 1.0)
        return out

    def settings(self, action, where) -> tuple[BrushSettings, int]:
        b = action["brush"]
        e = b["stampsEmitter"]
        s = BrushSettings()
        s.size = float(b["brushSize"]) * SIZE_SCALE
        s.flow = float(b.get("flow", 1.0)) * 100
        s.stroke_opacity = float(action.get("opacity", 1.0)) * 100
        s.spacing = max(float(e["spacing"]) * 100, 0.1)
        s.angle = math.degrees(float(e.get("rotation", 0.0)))
        s.follow_path = bool(e.get("followPath", False))
        s.size_jitter = float(b.get("brushSizeJitter", 0.0))
        s.flow_jitter = float(b.get("flowJitter", 0.0))
        s.angle_jitter = min(float(e.get("rotationJitter", 0.0)) / math.pi, 1.0)
        s.position_jitter = float(e.get("positionJitter", 0.0))
        align = b.get("alignment", 1)
        if align not in ALIGNMENTS:
            self.note(where, "brush alignment UV: Tangent | Wrap used")
        s.alignment = ALIGNMENTS.get(align, "Tangent | Wrap")
        s.backface_culling = float(b.get("cullingAngle", 0.5)) * 180 if b.get("cullingEnabled", True) else 180.0
        space = b.get("brushSizeSpace", 0)
        s.size_space = SIZE_SPACES.get(space, "Object")
        s.size_pressure = bool(e.get("pressureSizeEnabled", False))
        s.flow_pressure = bool(e.get("pressureFlowEnabled", False))
        return s, space

    # --- replay --------------------------------------------------------------------------------------------------

    def run(self) -> None:
        groups = self.conv.paint
        total = sum(len(a.get("strokes", [])) for g in groups for a in g.actions)
        self.say(f"Replaying {total} strokes")
        for gi, g in enumerate(groups):
            ts = self.s.project.texture_set(g.ts)
            _layer, fx, _in_mask = ts.owner_of(g.effect)
            where = f"{ts.name} > {_layer.name}" + (" (mask)" if g.mask else "")
            for action in g.actions:
                cls = action[CLASS]
                if not action.get("enabled", True):
                    continue
                if cls in ("DataActionPaint", "DataActionEraser", "DataActionPaintProj"):
                    self.paint(ts, g, action, where)
                elif cls == "DataActionDecal":
                    self.decal(ts, g, action, where)
                else:
                    self.note(where, f"paint action {cls} is not replayed")
            if gi % 10 == 0:
                self.say(f"Replayed {self.strokes} of {total} strokes")
        self.release()
        self.say(f"Replayed {self.strokes} strokes, {self.stamps} stamps")

    def targets(self, ts, g, values: dict, erase: bool) -> list:
        store = self.s.store
        out = []
        for udim in ts.tiles:
            tg = []
            for key, value in values.items():
                if erase and store.get(g.effect, key, udim) is None:
                    continue
                fmt = None if key == MASK_KEY else ts.format_of(key)
                e = store.ensure(g.effect, key, udim, ts.size_of(udim), storage_for(fmt))
                tg.append(Target((g.effect, key, udim), e.pages, value, erase))
            if tg:
                out.append(TileTargets(udim, self.s.uv_maps(ts, udim), tg))
        return out

    def paint(self, ts, g, action, where) -> None:
        erase = action[CLASS] == "DataActionEraser"
        values = ({MASK_KEY: (1.0,) * 3} if g.mask else {c: (1.0,) * 3 for c in ts.channel_ids()}) if erase \
            else self.values(action, g.mask, ts, where)
        if not values:
            return
        settings, space = self.settings(action, where)
        blending = action.get("stampBlendingMode", 0)
        if blending in STAMPS_BLENDING:
            settings.stamps_blending = STAMPS_BLENDING[blending]
        else:
            self.note(where, f"stamp blending mode {blending} replayed as the default")
        alpha = self.alpha(action, where)
        material = self.s.material_index(ts)
        diag = self.s.bbox_diagonal()
        image = self.projected_image(action, where) if action[CLASS] == "DataActionPaintProj" else None
        for stroke in action.get("strokes", []):
            tiles = self.targets(ts, g, values, erase)
            if not tiles:
                continue
            pts = self.samples(stroke)
            if not pts:
                continue
            seed = int(stroke.get("randomSeed", 0)) & 0x7FFFFFFF
            if stroke["viewType"] == 0 or not any(stroke["viewMatrix"]):
                st = Stroke(self.engine, tiles, None, [], material, settings, 1.0, alpha, seed=seed)
                w, h = ts.size
                points = [(x * w, (1.0 - y) * h, p) for x, y, p in pts]
                radius = float(action["brush"]["brushSize"]) / 2 * w
                if space != 2:
                    self.note(where, "strokes painted in the 2D view with a viewport size: sized on the UV square")
                self.stamps += st.add_texture(points, radius, (w, h))
            else:
                view = self.view(stroke["viewMatrix"], stroke["projectionMatrix"])
                if image is not None:
                    self.project(tiles, view, self.camera(view), image, action["projMatrix"], g.mask, ts, diag)
                st = Stroke(self.engine, tiles, view, self.meshes(), material, settings, diag, alpha, seed=seed,
                            camera=self.camera(view), symmetry=self.symmetry(stroke))
                points = [(x * VIEW, (1.0 - y) * VIEW, p) for x, y, p in pts]
                self.stamps += st.add(points)
            st.end().destroy()
            for tile in tiles:
                for t in tile.targets:
                    self.s.store.touched(*t.key)
                    if t.source is not None:
                        t.source.destroy()
            self.strokes += 1

    def projected_image(self, action, where):
        """The image a projection action lays (pixels, rows from the top), or None (its uniform value then)."""
        src = (action.get("sourceColor") or [{}])[0]
        if src.get(CLASS) != "DataSourceBitmap":
            return None
        with Container(self.path) as c:
            ds = c.resource(src["bitmap"]["urlToBitmapRes"])
            if ds is None:
                self.note(where, "the projected image is not in the file: its uniform value is used")
                return None
            pixels, _ = decode_image(c.bytes(ds))
        return np.asarray(pixels, np.float32)

    def project(self, tiles, view, cam, image, matrix, mask: bool, ts, diag) -> None:
        """Each target's values seen through a projection (the Target sources, as our Projection tool makes them):
        the image placed on the screen by `matrix` (a 3x3 for row vectors from the screen's normalized device
        coordinates, y up, to the image's, v up), on the surface the view sees and faces; a mask takes the image's
        luminance."""
        from trafilatura3d.paint.brush import _read
        d = self.s.ev.device
        M = np.asarray(matrix, np.float64).reshape(3, 3)
        ih, iw = image.shape[:2]
        rgb = image[..., :3] if image.shape[2] >= 3 else np.repeat(image[..., :1], 3, 2)
        alpha = image[..., 3] if image.shape[2] in (2, 4) else np.ones((ih, iw), np.float32)
        hits = _read(d, cam.texture)  # the visible surface per pixel of the view (A = material + 1)
        vp = np.asarray(view.view_proj, np.float64)
        for tile in tiles:
            pos = _read(d, tile.maps.position).astype(np.float64)
            nrm = _read(d, tile.maps.normal)[..., :3].astype(np.float64)
            h, w = pos.shape[:2]
            p = pos[..., :3].reshape(-1, 3)
            clip = np.c_[p, np.ones(len(p))] @ vp.T
            wc = np.maximum(clip[:, 3], 1e-9)
            x, y = clip[:, 0] / wc, clip[:, 1] / wc
            px = np.clip(((x * 0.5 + 0.5) * view.width).astype(int), 0, view.width - 1)
            py = np.clip(((0.5 - y * 0.5) * view.height).astype(int), 0, view.height - 1)
            seen = hits[py, px]
            visible = (seen[:, 3] > 0) & (np.linalg.norm(seen[:, :3] - p, axis=1) < diag * 4e-3)
            facing = np.einsum("ij,ij->i", nrm.reshape(-1, 3), np.asarray(view.eye) - p) > 0
            u = x * M[0, 0] + y * M[1, 0] + M[2, 0]
            v = x * M[0, 1] + y * M[1, 1] + M[2, 1]
            inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & (clip[:, 3] > 0)
            col = np.clip((u * iw).astype(int), 0, iw - 1)
            row = np.clip(((1.0 - v) * ih).astype(int), 0, ih - 1)
            value = rgb[row, col]
            if mask:
                value = np.repeat((value @ np.array([0.2126, 0.7152, 0.0722], np.float32))[:, None], 3, 1)
            out = np.zeros((len(p), 4), np.float32)
            out[:, :3] = value
            out[:, 3] = alpha[row, col] * (inside & visible & facing & (pos[..., 3].reshape(-1) > 0))
            tex = d.create_texture(size=(w, h, 1), format="rgba32float",
                                   usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            d.queue.write_texture({"texture": tex}, out.reshape(h, w, 4).tobytes(), {"bytes_per_row": w * 16},
                                  (w, h, 1))
            for t in tile.targets:
                t.source = tex

    @staticmethod
    def samples(stroke) -> list:
        """(x, y, pressure) of the stroke's samples; pressure from its timeline (1 without)."""
        keys = stroke.get("points", [])
        times = np.array([k.get("time", 0.0) for k in keys], np.float64)
        press = stroke.get("pressure") or []
        if press:
            px = np.array([k["x"] for k in press], np.float64)
            pv = np.array([k["value"] for k in press], np.float64)
            order = np.argsort(px, kind="stable")
            pressure = np.interp(times, px[order], pv[order])
        else:
            pressure = np.ones(len(keys))
        return [(float(k["position"][0]), float(k["position"][1]), float(p)) for k, p in zip(keys, pressure)]

    def decal(self, ts, g, action, where) -> None:
        """Polygon fill hits: the triangles under each click (or inside each rectangle) filled with its value."""
        tris = self.s.triangles(ts)
        material = self.s.material_index(ts)
        for hit in action.get("hits", []):
            gran = hit.get("granularity", 0)
            mode = FILL_MODES.get(gran, "Triangle Fill")
            value = float(hit.get("color", [1.0])[0])
            values = {MASK_KEY: (value,) * 3} if g.mask else {c: tuple(hit.get("color", [1, 1, 1])[:3])
                                                              for c in ts.channel_ids()}
            tiles = self.targets(ts, g, values, False)
            if not tiles:
                continue
            if hit[CLASS] == "DataDecalHitRect":
                (x0, y0), (x1, y1) = hit["position0"], hit["position1"]
                if abs(x1 - x0) * VIEW > 2 or abs(y1 - y0) * VIEW > 2:
                    self.note(where, "a rectangle polygon fill: its center is filled")
                pos = ((x0 + x1) / 2, (y0 + y1) / 2)
            else:
                pos = hit["position"]
            M = _matrix(hit["projViewInvMatrix"])  # clip -> normalized space
            if hit.get("viewType") == 0 or not M.any():  # in the 2D view: a UV
                t = _triangle_at(tris.uv, (pos[0], 1.0 - pos[1]))
                view = None
            else:
                vp = GL_TO_OURS @ np.linalg.inv(M) @ self.to_norm
                view = View(vp.astype(np.float32), np.zeros(3), math.radians(45), VIEW, VIEW)  # sizes nothing
                cam = self.camera(view)
                under = cam.hit(pos[0] * VIEW, (1.0 - pos[1]) * VIEW)
                if under is None or under[0][3] <= 0 or int(round(under[0][3])) - 1 != material:
                    continue
                t = tris.pick(under[0][:3].astype(np.float64))
            if t is None:
                continue
            sel = tris.select(t, mode)
            st = Stroke(self.engine, tiles, None, [], material, BrushSettings(), 1.0)
            st.fill(tris.uv[sel])
            st.end().destroy()
            for tile in tiles:
                for tg in tile.targets:
                    self.s.store.touched(*tg.key)


def _triangle_at(uv: np.ndarray, p) -> int | None:
    """The triangle ((N, 3, 2) UVs) containing the UV point `p`, or None."""
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    p = np.asarray(p, np.float64)

    def side(u, v):
        return (v[:, 0] - u[:, 0]) * (p[1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (p[0] - u[:, 0])
    s0, s1, s2 = side(a, b), side(b, c), side(c, a)
    inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    hits = np.flatnonzero(inside)
    return int(hits[0]) if len(hits) else None
