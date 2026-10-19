"""The Path tool's pixels (Path, Filled path and Ribbon path pages): a path's points on the surface become a curve,
and the curve paints along itself (Paint / Erase / Smudge along path: stamps of the path's brush every Spacing),
fills its inside (Filled path) or lays an image along itself (Ribbon path). A path is non-destructive: its pixels
are made again from its data whenever they change (the session caches them by content) and composited over the
paint layer's painting (stack.py).

Ours where the documentation is silent: the curve (cubic Bezier between the points; their handles automatic,
Catmull-Rom, unless custom; a corner point has none), Projection depth in stamp radii, the Filled path's inside (the curve projected on its best fitting
plane, even-odd rule, within Projection depth of the plane, faces toward the plane's normal), the Ribbon's
mapping (nearest point of the curve: length along it and side distance) and the values of its enums the
documentation does not list (Image orientation). Symmetry paints the mirror image of the path too. Not offered:
Corner joins other than Round,
the Ribbon's self-overlap Channel blending (each texel takes the nearest part of the curve: a ribbon does not
overlap itself), dynamic strokes, path presets.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, fields

import numpy as np
import wgpu

from trafilatura3d.nodes.base import INT_MATH_WGSL
from trafilatura3d.paint.brush import WGSL, BrushSettings, Stroke, Target, TileTargets

UP_AXES = ("X", "Y", "Z")
RIBBON_ORIENTATIONS = ("Horizontal", "Vertical")  # ours: the documentation lists none
CORNERS = ("Round join",)  # Miter, Bevel and Cut joins are not offered
STRETCH_MODES = ("Stretch along path", "Keep aspect ratio")
TILING_MODES = ("None", "Auto", "Custom")
RATIOS = ("Fit to path width", "Fit to path length")
ALIGNMENTS = ("Align at start", "Align at center", "Align at end")


@dataclass
class RibbonSettings:
    """Ribbon path parameters (the page's labels in the Properties). Stroke width in % of half the mesh's bounding
    box diagonal like the brush Size (ours); offsets as fractions of the image."""
    stroke_width: float = 5.0
    stroke_opacity: float = 100.0
    image_orientation: str = "Horizontal"
    flip_image: bool = False
    corner: str = "Round join"
    omit_ends_when_closed: bool = False
    stretching_mode: str = "Stretch along path"
    stretch_between_offsets: bool = False
    start_offset: float = 0.1
    end_offset: float = 0.1
    tiling_mode: str = "Auto"
    tiling_amount: int = 4
    mirror_every_2nd_tile: bool = False
    aspect_ratio_factor: float = 1.0
    ratio: str = "Fit to path width"
    remove_clipped_tiles: bool = False
    alignment: str = "Align at start"

    @staticmethod
    def of(d: dict) -> RibbonSettings:
        names = {f.name for f in fields(RibbonSettings)}
        return RibbonSettings(**{k: v for k, v in d.items() if k in names})


def brush_of(d: dict) -> BrushSettings:
    names = {f.name for f in fields(BrushSettings)}
    return BrushSettings(**{k: v for k, v in d.items() if k in names})


# --- the curve ---------------------------------------------------------------------------------------------

@dataclass
class Curve:
    pos: np.ndarray  # (M, 3) samples
    nrm: np.ndarray  # (M, 3)
    pressure: np.ndarray  # (M,)
    s: np.ndarray  # (M,) arc length from the start
    tangent: np.ndarray  # (M, 3)

    @property
    def length(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0


# A path point: [x, y, z, nx, ny, nz, pressure, corner (0/1)] and, with custom tangents, + [in handle xyz, out
# handle xyz, 1]: the Bezier control points' offsets from the point (the in handle points backward).
CUSTOM = 14  # the index of the custom tangents flag


def transformed(points: list, m, offset) -> list:
    """The points through a symmetry copy (brush.symmetry_copies): positions, normals and handles."""
    m = np.asarray(m, float)
    out = []
    for pt in points:
        q = list(pt)
        q[0:3] = list(m @ np.asarray(pt[0:3], float) + offset)
        q[3:6] = list(m @ np.asarray(pt[3:6], float))
        if len(q) > CUSTOM:
            q[8:11] = list(m @ np.asarray(pt[8:11], float))
            q[11:14] = list(m @ np.asarray(pt[11:14], float))
        out.append(q)
    return out


def symmetric(path) -> list:
    """(points, flip) of the path's symmetry copies (its Symmetry: the type, axis, position, copies and Flip copy of
    the scene's symmetry when it was turned on), [] when off."""
    from trafilatura3d.paint.brush import symmetry_copies
    s = path.symmetry
    if not s:
        return []
    copies = symmetry_copies(s["type"], s["axis"], s["position"], s.get("copies", 2), s.get("flip", "None"))
    return [(transformed(path.points, m, t), flip) for m, t, _reflect, flip in copies]


def handles(points: list, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """(in, out) handle offsets of every point: its custom ones, or Catmull-Rom's (a third of the tangent each
    way), zero at a corner (Toggle corner / smooth)."""
    a = np.asarray([list(pt[:8]) for pt in points], np.float64)
    p, corner = a[:, :3], a[:, 7] > 0.5
    k = len(p)

    def at(i):
        return i % k if closed else min(max(i, 0), k - 1)
    h_in, h_out = np.zeros_like(p), np.zeros_like(p)
    for i in range(k):
        pt = points[i]
        if len(pt) > CUSTOM and pt[CUSTOM] > 0.5:
            h_in[i], h_out[i] = pt[8:11], pt[11:14]
        elif not corner[i]:
            t = (p[at(i + 1)] - p[at(i - 1)]) * (0.5 if closed or 0 < i < k - 1 else 1.0)
            h_in[i], h_out[i] = -t / 3, t / 3
    return h_in, h_out


def curve(points: list, closed: bool, step: float, reverse: bool = False) -> Curve | None:
    """The path's curve sampled every `step` (world units): cubic Bezier segments between the points with their
    handles (`handles`); normals and pressure interpolated."""
    if len(points) < 2:
        return None
    a = np.asarray([list(pt[:8]) for pt in points], np.float64)
    h_in, h_out = handles(points, closed)
    if reverse:
        a, h_in, h_out = a[::-1], h_out[::-1], h_in[::-1]
    p, n, pr = a[:, :3], a[:, 3:6], a[:, 6]
    k = len(p)
    seg = k if closed else k - 1

    def at(i):
        return i % k if closed else min(max(i, 0), k - 1)
    out_p, out_n, out_pr = [], [], []
    for i in range(seg):
        j = at(i + 1)
        c0, c1, c2, c3 = p[i], p[i] + h_out[i], p[j] + h_in[j], p[j]
        length = float(np.linalg.norm(c1 - c0) + np.linalg.norm(c2 - c1) + np.linalg.norm(c3 - c2))
        m = max(2, int(math.ceil(length / max(step, 1e-9))) + 1)
        t = np.linspace(0, 1, m)[:-1 if i < seg - 1 else None, None]
        u = 1 - t
        out_p.append(u**3 * c0 + 3 * u**2 * t * c1 + 3 * u * t**2 * c2 + t**3 * c3)
        out_n.append((1 - t) * n[i] + t * n[j])
        out_pr.append((1 - t[:, 0]) * pr[i] + t[:, 0] * pr[j])
    pos = np.concatenate(out_p)
    nrm = np.concatenate(out_n)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    tg = np.gradient(pos, axis=0) if len(pos) > 1 else np.zeros_like(pos)
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    return Curve(pos, nrm, np.concatenate(out_pr), s, tg)


# --- along the path ------------------------------------------------------------------------------------------

def stamps(path, c: Curve, settings: BrushSettings, diag: float, uv_of=None, seed: int = 0,
           flip: int = 0) -> list[bytes]:
    """The stamps of Paint / Erase / Smudge along path: one every Spacing of the stamp size along the curve, on
    the surface plane of the curve's normal, turned along the curve (Follow Path) or toward the Up axis. `uv_of`:
    world -> UV of the stamp centers (Smudge drags in texture space); `flip`: a symmetry copy's Flip copy."""
    from trafilatura3d.paint.brush import FLIP_CODE
    rng = np.random.default_rng(seed)
    up = np.eye(3)[UP_AXES.index(path.up_axis)]
    out, next_s = [], 0.0
    s = settings
    for i in range(len(c.pos)):
        if c.s[i] + 1e-12 < next_s:
            continue
        pressure = float(c.pressure[i])
        size = s.size * (pressure if s.size_pressure else 1.0)
        rw = size / 100 * diag / 2 / 2
        rw *= 1.0 - s.size_jitter * rng.random()
        next_s = c.s[i] + max(2 * rw * s.spacing / 100, 1e-6)
        n = c.nrm[i]
        ref = c.tangent[i] if s.follow_path else up - n * float(np.dot(up, n))
        if np.linalg.norm(ref) < 1e-6:
            ref = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
        u = ref - n * float(np.dot(ref, n))
        u /= max(np.linalg.norm(u), 1e-12)
        if not s.follow_path:  # Up axis: the stamp's up (its -V) toward the axis, so its U is across
            u = np.cross(u, n)
        angle = math.radians(s.angle) + (rng.random() * 2 - 1) * math.pi * s.angle_jitter
        flow = s.flow / 100 * (pressure if s.flow_pressure else 1.0) * (1.0 - s.flow_jitter * rng.random())
        p = c.pos[i]
        if s.position_jitter > 0:
            v = np.cross(n, u)
            p = p + (u * (rng.random() * 2 - 1) + v * (rng.random() * 2 - 1)) * 2 * rw * s.position_jitter
        su, sv = (uv_of(p) if uv_of is not None else (0.0, 0.0))
        out.append(struct.pack("16f", *p, rw, *n, flow, *u, angle + FLIP_CODE * flip, su, sv, 0.0,
                               max(path.projection_depth, 1e-3)))
    return out


# --- GPU passes of Filled and Ribbon --------------------------------------------------------------------

class PathPasses:
    def __init__(self, device):
        self.device = device

        def pipe(name):
            return device.create_compute_pipeline(layout="auto", compute={
                "module": device.create_shader_module(code=INT_MATH_WGSL + (WGSL / name).read_text(encoding="utf-8")),
                "entry_point": "main"})
        self.filled_pipe = pipe("path_filled.wgsl")
        self.ribbon_pipe = pipe("path_ribbon.wgsl")
        self.no_image = device.create_texture(size=(1, 1, 1), format="rgba32float",
                                              usage=wgpu.TextureUsage.TEXTURE_BINDING)

    def _run(self, pipe, entries, size):
        d = self.device
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=entries)
        enc = d.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((size[0] + 7) // 8, (size[1] + 7) // 8)
        cp.end()
        d.queue.submit([enc.finish()])

    def fill(self, maps, c: Curve, depth: float, coverage) -> None:
        """Filled path: coverage 1 inside the curve projected on its best fitting plane (even-odd), within `depth`
        (world) of the plane, on faces toward its normal."""
        d = self.device
        pts = c.pos
        center = pts.mean(0)
        _u, _s, vt = np.linalg.svd(pts - center)
        ax, ay, nz = vt[0], vt[1], vt[2]
        if float(np.dot(nz, c.nrm.mean(0))) < 0:
            nz, ay = -nz, -ay
        poly = np.stack([(pts - center) @ ax, (pts - center) @ ay], 1)
        lo, hi = poly.min(0), poly.max(0)
        buf = d.create_buffer_with_data(data=np.ascontiguousarray(
            np.concatenate([poly, np.zeros((len(poly), 2))], 1), np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE)
        uni = struct.pack("20f", *center, float(len(poly)), *ax, depth, *ay, 0.0, *nz, 0.0, *lo, *hi)
        self._run(self.filled_pipe, [
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": {"buffer": buf}},
            {"binding": 2, "resource": maps.position.create_view()},
            {"binding": 3, "resource": maps.normal.create_view()},
            {"binding": 4, "resource": coverage.create_view()}], maps.size)

    def ribbon(self, maps, c: Curve, r: RibbonSettings, diag: float, closed: bool, color, image, single, dst) -> None:
        """Ribbon path: each texel near the curve (within half the width, facing the curve's normal) takes the
        image at its length along the curve and its side distance, through the stretching and tiling settings."""
        d = self.device
        width = r.stroke_width / 100 * diag / 2
        length = max(c.length, 1e-9)
        iw, ih = (image.width, image.height) if image is not None else (1, 1)
        if r.image_orientation == "Vertical":
            iw, ih = ih, iw
        tile = width * iw / max(ih, 1) * max(r.aspect_ratio_factor, 1e-3)  # one image along the path
        data = np.concatenate([c.pos, c.s[:, None], c.nrm, np.zeros((len(c.pos), 1))], 1)
        buf = d.create_buffer_with_data(data=np.ascontiguousarray(data, np.float32).tobytes(),
                                        usage=wgpu.BufferUsage.STORAGE)
        lo, hi = c.pos.min(0) - width, c.pos.max(0) + width
        flags = (int(r.flip_image) | int(r.image_orientation == "Vertical") << 1 | int(r.mirror_every_2nd_tile) << 2
                 | int(r.remove_clipped_tiles) << 3 | int(r.stretch_between_offsets) << 4
                 | int(closed and r.omit_ends_when_closed) << 5 | int(single) << 6 | int(image is not None) << 7)
        uni = struct.pack("4f 4f 4f 4f 4i", *lo, width, *hi, length, tile, float(r.tiling_amount),
                          r.start_offset, r.end_offset, *color[:3], 0.0,
                          STRETCH_MODES.index(r.stretching_mode), TILING_MODES.index(r.tiling_mode),
                          RATIOS.index(r.ratio) | ALIGNMENTS.index(r.alignment) << 2 | flags << 4, len(c.pos))
        self._run(self.ribbon_pipe, [
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": {"buffer": buf}},
            {"binding": 2, "resource": maps.position.create_view()},
            {"binding": 3, "resource": maps.normal.create_view()},
            {"binding": 4, "resource": (image or self.no_image).create_view()},
            {"binding": 5, "resource": dst.create_view()}], maps.size)


# --- rendering a path ------------------------------------------------------------------------------------

def render(session, ts, owner_uid, path, textures: dict, udim: int) -> None:
    """Paint `path` into `textures` ({channel or "mask": paint.Pages}, new, the path's pixels in the UDIM tile `udim`):
    its mode's pixels for every channel it paints. The painting it belongs to (`session.store` entries of
    `owner_uid`: a paint layer, a layer's mask, a paint effect) is the Smudge's content."""
    from trafilatura3d.paint.model import local_uv
    from trafilatura3d.paint.paint import MASK
    s = session
    engine, maps = s.brush_engine(), s.uv_maps(ts, udim)
    diag = s.bbox_diagonal()
    settings = brush_of(path.brush)
    tris = s.triangles(ts)
    step = diag * 1e-3
    c = curve(path.points, path.closed, step, path.reversed)
    if c is None or not textures:
        return
    curves = [(c, 0)]  # (curve, Flip copy)
    for pts, flip in symmetric(path):  # the path's Symmetry: its copies painted with it
        sc = curve(pts, path.closed, step, path.reversed)
        if sc is not None:
            curves.append((sc, flip))
    material = s.material_index(ts)
    targets = []
    for ch, pages in textures.items():
        value = (path.mask_value,) * 3 if ch == MASK else tuple(path.values.get(ch, (1.0, 1.0, 1.0)))
        targets.append(Target((path.uid, ch), pages, value))
    if path.mode == "Smudge along path":  # the layer's painting copied, then dragged along the curve
        for tg in targets:
            e = s.store.get(owner_uid, tg.key[1], udim)
            if e is not None and (e.pages.format, e.pages.size) == (tg.pages.format, tg.pages.size):
                tg.pages.copy_from(e.pages)

    def uv_of(p):  # in the tile's texture
        uv = local_uv(tris.uv_at(tris.pick(p), p), udim)
        return float(uv[0]), float(uv[1])

    if path.mode in ("Paint along path", "Erase along path", "Smudge along path"):
        smudge = {tg.key: None for tg in targets} if path.mode == "Smudge along path" else None
        alpha = s.alpha_name(path.alpha or "Round soft")
        stroke = Stroke(engine, [TileTargets(udim, maps, targets)], None, [], material, settings, diag, alpha,
                        seed=hash(path.uid) & 0xFFFF, smudge=smudge)
        mode = 2 if settings.alignment == "Tangent | Planar" else 1
        for cv, flip in curves:
            st = stamps(path, cv, settings, diag, uv_of if smudge is not None else None, seed=hash(path.uid) & 0xFFFF,
                        flip=flip)
            if smudge is not None:
                stroke._smudge_prev = None  # each curve drags from its own start
                stroke.smudge_stamps(st, mode)
            elif st:
                stroke._dispatch(st, mode=mode)
        stroke.end().destroy()
        return
    if path.mode == "Filled path":
        settings.stroke_opacity = path.opacity * 100
        stroke = Stroke(engine, [TileTargets(udim, maps, targets)], None, [], material, settings, diag)
        rw = settings.size / 100 * diag / 4  # the brush's stamp radius (Object size space)
        for cv, _flip in curves:  # each one adds its inside to the coverage
            passes(engine.device).fill(maps, cv, max(path.projection_depth * rw, diag * 1e-3), stroke.stroke_texs[0])
        stroke.commit()
        stroke.end().destroy()
        return
    # Ribbon: per channel, the image (or the uniform value) along the curve as per texel values, laid at full
    # coverage with the ribbon's Stroke opacity.
    r = RibbonSettings.of(path.ribbon)
    settings.stroke_opacity = r.stroke_opacity
    from trafilatura3d.paint.model import ChannelSource
    stroke = Stroke(engine, [TileTargets(udim, maps, targets)], None, [], material, settings, diag)
    pp = passes(engine.device)
    U = wgpu.TextureUsage
    for tg in targets:
        ch = tg.key[1]
        src = ChannelSource(**path.sources[ch]) if ch in path.sources else None
        got = s.source_texture(src, ts, udim) if src is not None else None
        tg.source = engine.device.create_texture(size=(*maps.size, 1), format="rgba32float",
                                                 usage=U.STORAGE_BINDING | U.TEXTURE_BINDING)
        for cv, _flip in curves:  # each one writes only where its ribbon is
            pp.ribbon(maps, cv, r, diag, path.closed, tg.value, got[0] if got else None, got[1] if got else False,
                      tg.source)
    stroke.cover_all()
    stroke.commit()
    for tg in targets:
        tg.source.destroy()
    stroke.end().destroy()


_PASSES = {}


def passes(device) -> PathPasses:
    """The Filled and Ribbon pipelines of a device, made once."""
    if device not in _PASSES:
        _PASSES[device] = PathPasses(device)
    return _PASSES[device]
