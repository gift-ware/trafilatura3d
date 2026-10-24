"""A Texture Set's mesh in texture space: per texel the world position, normal and tangent of the surface
it covers, and the UV coverage mask. The brush reads them to find which texels a stamp touches; the padding
reads the mask. One UDIM tile at a time: its UVs moved into its own 0-1 square (model.local_uv)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wgpu

from trafilatura3d.paint.model import local_uv

SHADER = """
struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f, @location(1) normal: vec3f, @location(2) tangent: vec4f,
};
@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) t: vec4f, @location(3) uv: vec2f) -> VOut {
    var o: VOut;
    o.pos = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);  // UV v grows down the image
    o.world = p; o.normal = n; o.tangent = t;
    return o;
}
struct F0 { @location(0) a: vec4f, @location(1) b: vec4f };
@fragment
fn fs0(i: VOut) -> F0 { return F0(vec4f(i.world, 1.0), vec4f(normalize(i.normal), 1.0)); }
@fragment
fn fs1(i: VOut) -> @location(0) vec4f { return vec4f(1.0); }
"""
VB = [{"array_stride": s, "attributes": [{"shader_location": i, "offset": 0, "format": f}]}
      for i, (s, f) in enumerate([(12, "float32x3"), (12, "float32x3"), (16, "float32x4"), (8, "float32x2")])]

# A value per triangle (corners not shared), rasterized flat in texture space.
FLAT = """
struct VOut { @builtin(position) pos: vec4f, @location(0) v: vec4f };
@vertex
fn vs(@location(0) uv: vec2f, @location(1) v: vec4f) -> VOut {
    return VOut(vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0), v);
}
@fragment
fn fs(i: VOut) -> @location(0) vec4f { return i.v; }
"""
FLAT_VB = [{"array_stride": 8, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x2"}]},
           {"array_stride": 16, "attributes": [{"shader_location": 1, "offset": 0, "format": "float32x4"}]}]
COMPUTED = ("uv_islands", "texel_density")  # the mesh maps made from the mesh, not baked (nodes/paint/_lib.py)


def _triangles(objects: list, material: int, names=None, udim: int | None = None):
    """(uv, world position) of every triangle corner of the submeshes on `material`: (N, 3, 2), (N, 3, 3); with
    `names` a list, the object index of each triangle is appended to it; with `udim`, UVs in that tile's square."""
    uvs, pos = [], []
    for k, o in enumerate(objects):
        for s in o.submeshes:
            if s.material != material:
                continue
            idx = np.asarray(o.idx, np.int64)[s.start:s.start + s.count].reshape(-1, 3)
            uvs.append(np.asarray(o.uv, np.float64)[idx])
            pos.append(np.asarray(o.pos, np.float64)[idx])
            if names is not None:
                names.extend([k] * len(idx))
    if not uvs:
        return np.zeros((0, 3, 2)), np.zeros((0, 3, 3))
    uv = np.concatenate(uvs)
    return (uv if udim is None else local_uv(uv, udim)), np.concatenate(pos)


def mesh_names(objects: list, material: int) -> list[str]:
    """The names of the meshes (the file's objects) with a part on `material`, sorted: the Geometry mask's Mesh
    names."""
    return sorted({o.name for o in objects if any(s.material == material for s in o.submeshes)})


def components(corners: np.ndarray) -> np.ndarray:
    """Per triangle the index of its connected component, triangles connected when they share a corner value
    (`corners`: (N, 3, k) coordinates, equal after rounding = shared)."""
    n = len(corners)
    if n == 0:
        return np.zeros(0, np.int64)
    keys = np.round(corners.reshape(n * 3, -1) * 2 ** 20).astype(np.int64)
    _u, corner = np.unique(keys, axis=0, return_inverse=True)
    tri = corner.reshape(n, 3)
    # Every corner takes the smallest label of its triangles, then jumps to its label's label, until nothing
    # changes.
    lab = np.arange(len(_u))
    while True:
        new = lab.copy()
        np.minimum.at(new, tri.ravel(), np.repeat(lab[tri].min(1), 3))
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    return np.unique(lab[tri[:, 0]], return_inverse=True)[1]


def island_values(uv: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per triangle: RG two random numbers of its UV island (triangles sharing a UV corner, ours: the
    documentation does not define an island), B 0, A 1. (N, 4) float32."""
    n = len(uv)
    if n == 0:
        return np.zeros((0, 4), np.float32)
    island = components(uv)
    _r = np.arange(island.max() + 1)
    rng = np.random.default_rng(seed)
    rand = rng.random((len(_r), 2))
    out = np.zeros((n, 4), np.float32)
    out[:, :2] = rand[island]
    out[:, 3] = 1.0
    return out


class Triangles:
    """A Texture Set's triangles for the Polygon Fill tool: picking under a world position and the four selection
    modes (Polygon Fill page). Polygons are rebuilt as triangle pairs sharing their longest edge in one plane
    (the quads of the file: our mesh import triangulates, the documentation's Polygon Fill equals Triangle Fill
    on triangulated meshes)."""

    def __init__(self, objects: list, material: int):
        owners: list = []
        self.uv, self.pos = _triangles(objects, material, owners)
        self.object = np.asarray(owners, np.int64)  # per triangle: the index of its object (mesh)
        n = len(self.uv)
        self.mesh = components(self.pos)  # connected through shared positions
        self.chunk = components(self.uv)  # connected in UV space
        self.partner = np.full(n, -1, np.int64)
        if n:
            e = self.pos[:, [1, 2, 0]] - self.pos  # edge k from corner k to k+1
            longest = np.argmax(np.linalg.norm(e, axis=2), 1)
            a = self.pos[np.arange(n), longest]
            b = self.pos[np.arange(n), (longest + 1) % 3]
            key = np.round(np.sort(np.stack([a, b], 1), axis=1).reshape(n, 6) * 2 ** 20).astype(np.int64)
            order = np.lexsort(key.T[::-1])
            same = np.all(key[order[1:]] == key[order[:-1]], 1)
            nrm = np.cross(e[:, 0], -e[:, 2])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
            for i in np.where(same)[0]:
                t0, t1 = order[i], order[i + 1]
                if abs(float(np.dot(nrm[t0], nrm[t1]))) > 0.999:
                    self.partner[t0], self.partner[t1] = t1, t0

    def pick(self, p) -> int | None:
        """The triangle nearest to the world position `p` (a hit of the camera buffer)."""
        if not len(self.pos):
            return None
        a, b, c = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        # Distance to each triangle: to its plane when the projection falls inside, else to its corners.
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        d = np.sum((np.asarray(p) - a) * n, 1)
        q = np.asarray(p) - d[:, None] * n
        inside = np.ones(len(a), bool)
        for u, v in ((a, b), (b, c), (c, a)):
            inside &= np.sum(np.cross(v - u, q - u) * n, 1) >= -1e-9
        corner = np.min(np.linalg.norm(self.pos - np.asarray(p)[None, None], axis=2), 1)
        dist = np.where(inside, np.abs(d), corner)
        return int(np.argmin(dist))

    def uv_at(self, t: int, p) -> np.ndarray:
        """The UV of world position `p` on triangle `t` (barycentric, clamped inside the triangle)."""
        a, b, c = self.pos[t]
        v0, v1, v2 = b - a, c - a, np.asarray(p) - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01 or 1e-20
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        bary = np.clip([1 - v - w, v, w], 0, 1)
        bary /= bary.sum() or 1.0
        return bary @ self.uv[t]

    def select(self, t: int, mode: str) -> np.ndarray:
        """The triangles a click on `t` fills: mode Triangle Fill, Polygon Fill, Mesh Fill or UV Chunk Fill."""
        return self.expand([t], mode)

    def expand(self, tris, mode: str) -> np.ndarray:
        """The triangles `tris` take in a fill mode: themselves, their polygons, their meshes or their UV chunks."""
        tris = np.unique(np.asarray(tris, np.int64))
        if not len(tris):
            return tris
        if mode == "Mesh Fill":
            return np.where(np.isin(self.mesh, self.mesh[tris]))[0]
        if mode == "UV Chunk Fill":
            return np.where(np.isin(self.chunk, self.chunk[tris]))[0]
        if mode == "Polygon Fill":
            p = self.partner[tris]
            return np.unique(np.concatenate([tris, p[p >= 0]]))
        return tris

    def in_rect(self, view_proj, size, rect, hit_test: str, visible=None) -> np.ndarray:
        """The triangles a screen rectangle takes: `rect` (x0, y0, x1, y1) in pixels of a view of `size` (w, h);
        `hit_test` Intersection (any part of the triangle in it) or Contained center (its center in it);
        `visible(points (M, 3)) -> bool (M,)`, when given, keeps only the triangles seen (a point of each: its
        center and its corners pulled a fifth toward it). Triangles crossing the camera plane are left out."""
        n = len(self.pos)
        if not n:
            return np.zeros(0, np.int64)
        x0, x1 = sorted(rect[0::2])
        y0, y1 = sorted(rect[1::2])
        m = np.asarray(view_proj, np.float64)
        clip = np.concatenate([self.pos, np.ones((n, 3, 1))], 2) @ m.T
        front = np.all(clip[..., 3] > 1e-9, 1)
        w = np.where(clip[..., 3:] > 1e-9, clip[..., 3:], 1.0)
        ndc = clip[..., :2] / w
        s = np.stack([(ndc[..., 0] * 0.5 + 0.5) * size[0], (0.5 - ndc[..., 1] * 0.5) * size[1]], -1)  # (n, 3, 2)
        if hit_test == "Contained center":
            c = s.mean(1)
            hit = (c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1)
        else:
            hit = _triangles_meet_rect(s, x0, y0, x1, y1)
        tris = np.where(hit & front)[0]
        if visible is not None and len(tris):
            p = self.pos[tris]
            c = p.mean(1, keepdims=True)
            points = np.concatenate([c, p + (c - p) * 0.2], 1)  # (k, 4, 3)
            seen = visible(points.reshape(-1, 3)).reshape(len(tris), 4).any(1)
            tris = tris[seen]
        return tris


def _triangles_meet_rect(s: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    """Per screen triangle ((n, 3, 2)): does any part of it lie in the rectangle? A corner inside it, a corner of the
    rectangle inside the triangle, or an edge crossing one of its sides."""
    inside = np.any((s[..., 0] >= x0) & (s[..., 0] <= x1) & (s[..., 1] >= y0) & (s[..., 1] <= y1), 1)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    def cross(o, p, q):
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    corners = [np.array([x, y], np.float64) for x in (x0, x1) for y in (y0, y1)]
    for q in corners:
        d1, d2, d3 = cross(a, b, q), cross(b, c, q), cross(c, a, q)
        inside |= ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    sides = [(corners[0], corners[1]), (corners[1], corners[3]), (corners[3], corners[2]), (corners[2], corners[0])]
    for p, q in ((a, b), (b, c), (c, a)):
        for r, t in sides:
            r, t = np.broadcast_to(r, p.shape), np.broadcast_to(t, p.shape)
            inside |= ((cross(p, q, r) * cross(p, q, t) < 0) & (cross(r, t, p) * cross(r, t, q) < 0))
    return inside


def density_values(uv: np.ndarray, pos: np.ndarray, size) -> np.ndarray:
    """Per triangle: log2 of its texel density (texels per surface unit) relative to the area weighted mean of
    the Texture Set; A 1. (N, 4) float32."""
    n = len(uv)
    out = np.zeros((n, 4), np.float32)
    if n == 0:
        return out
    duv = (uv[:, 1:] - uv[:, :1]) * np.asarray(size, np.float64)
    texels = 0.5 * np.abs(duv[:, 0, 0] * duv[:, 1, 1] - duv[:, 0, 1] * duv[:, 1, 0])
    area = 0.5 * np.linalg.norm(np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]), axis=1)
    ok = (area > 1e-20) & (texels > 1e-20)
    if ok.any():
        dens = np.sqrt(texels[ok] / area[ok])  # texels per unit of length
        mean = np.sqrt(texels[ok].sum() / area[ok].sum())
        out[ok, 0] = np.log2(dens / mean)
    out[:, 1] = out[:, 2] = out[:, 0]
    out[:, 3] = 1.0
    return out


@dataclass
class UVMaps:
    position: object  # rgba32float: world position, A = covered
    normal: object  # rgba16float: world normal (a direction: half floats are enough, a quarter less memory)
    mask: object  # rgba8unorm: 1 inside the UV islands (UVRasterizer.mask)
    size: tuple
    bounds: tuple | None = None  # (min, max) world corners of the triangles in the tile, None: none there


class UVRasterizer:
    def __init__(self, device):
        self.device = device
        mod = device.create_shader_module(code=SHADER)
        self.pipes = [device.create_render_pipeline(
            layout="auto", vertex={"module": mod, "entry_point": "vs", "buffers": VB},
            fragment={"module": mod, "entry_point": entry, "targets": [{"format": f} for f in fmts]},
            primitive={"topology": "triangle-list", "cull_mode": "none"})
            for entry, fmts in (("fs0", ("rgba32float", "rgba16float")), ("fs1", ("rgba8unorm",)))]
        flat = device.create_shader_module(code=FLAT)
        self.flat_pipe = device.create_render_pipeline(
            layout="auto", vertex={"module": flat, "entry_point": "vs", "buffers": FLAT_VB},
            fragment={"module": flat, "entry_point": "fs", "targets": [{"format": "rgba32float"}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"})

    def coverage(self, objects: list, material: int, size, excluded, udim: int):
        """The Geometry mask (Mesh names) in a tile's texture space: 1 on the triangles of the meshes not
        `excluded`, 0 elsewhere (rgba32float, opaque)."""
        owners: list = []
        uv, _pos = _triangles(objects, material, owners, udim)
        keep = np.array([objects[k].name not in excluded for k in owners], bool) if owners else np.zeros(0, bool)
        uv = uv[keep] if len(uv) else uv
        values = np.ones((len(uv), 4), np.float32)
        return self._flat(uv, values, size, clear=(0.0, 0.0, 0.0, 1.0))

    def _flat(self, uv, values, size, clear=(0.0, 0.0, 0.0, 0.0)):
        """Triangles ((N, 3, 2) UVs) rasterized with a value each ((N, 4)) into a new rgba32float texture."""
        d = self.device
        w, h = size
        U = wgpu.TextureUsage
        target = d.create_texture(size=(w, h, 1), format="rgba32float",
                                  usage=U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC)
        enc = d.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{"view": target.create_view(), "load_op": "clear",
                                                       "store_op": "store", "clear_value": clear}])
        if len(uv):
            vb_uv = d.create_buffer_with_data(data=np.ascontiguousarray(uv.reshape(-1, 2), np.float32).tobytes(),
                                              usage=wgpu.BufferUsage.VERTEX)
            vb_v = d.create_buffer_with_data(data=np.ascontiguousarray(np.repeat(values, 3, 0), np.float32).tobytes(),
                                             usage=wgpu.BufferUsage.VERTEX)
            rp.set_pipeline(self.flat_pipe)
            rp.set_vertex_buffer(0, vb_uv)
            rp.set_vertex_buffer(1, vb_v)
            rp.draw(len(uv) * 3, 1, 0, 0)
        rp.end()
        d.queue.submit([enc.finish()])
        return target

    def computed(self, objects: list, material: int, size, slot: str, udim: int):
        """A mesh map made from the mesh (`COMPUTED`) in a tile: rgba32float texture, 0 outside the UV islands."""
        uv, pos = _triangles(objects, material, udim=udim)
        values = island_values(uv) if slot == "uv_islands" else density_values(uv, pos, size)
        return self._flat(uv, values, size)

    def mask(self, objects: list, material: int, size, udim: int):
        """The UV coverage of the submeshes on `material` in a tile (rgba8unorm, 1 inside the islands): all the
        padding needs (64 MB at 4096, the full maps 576)."""
        w, h = size
        t = self.device.create_texture(size=(w, h, 1), format="rgba8unorm", usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                                       | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC)
        self._draw(objects, material, udim, self.pipes[1], [t])
        return t

    def maps(self, objects: list, material: int, size, udim: int, mask) -> UVMaps:
        """The maps of the submeshes of `objects` (meshio.MeshObject) on `material`, in the UDIM tile `udim`, with
        its `mask`: what the brush and the 3D projections read."""
        w, h = size
        U = wgpu.TextureUsage
        usage = U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC
        targets = [self.device.create_texture(size=(w, h, 1), format=f, usage=usage)
                   for f in ("rgba32float", "rgba16float")]  # position, normal
        self._draw(objects, material, udim, self.pipes[0], targets)
        uv, pos = _triangles(objects, material, udim=udim)
        inside = np.all((uv.mean(1) >= 0.0) & (uv.mean(1) <= 1.0), 1) if len(uv) else np.zeros(0, bool)
        bounds = (pos[inside].reshape(-1, 3).min(0), pos[inside].reshape(-1, 3).max(0)) if inside.any() else None
        return UVMaps(*targets, mask, size=(w, h), bounds=bounds)

    def _draw(self, objects, material, udim, pipe, outs) -> None:
        """The submeshes on `material` rasterized in the tile's texture space into `outs`."""
        d = self.device
        buffers = []
        for o in objects:
            subs = [s for s in o.submeshes if s.material == material]
            if not subs:
                continue
            vbs = [d.create_buffer_with_data(data=np.ascontiguousarray(a, np.float32).tobytes(),
                                             usage=wgpu.BufferUsage.VERTEX)
                   for a in (o.pos, o.nrm, o.tan, local_uv(o.uv, udim))]
            ib = d.create_buffer_with_data(data=np.ascontiguousarray(o.idx, np.uint32).tobytes(),
                                           usage=wgpu.BufferUsage.INDEX)
            buffers.append((vbs, ib, subs))
        enc = d.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{"view": t.create_view(), "load_op": "clear",
                                                       "store_op": "store", "clear_value": (0, 0, 0, 0)} for t in outs])
        rp.set_pipeline(pipe)
        for vbs, ib, subs in buffers:
            for i, b in enumerate(vbs):
                rp.set_vertex_buffer(i, b)
            rp.set_index_buffer(ib, "uint32")
            for s in subs:
                rp.draw_indexed(s.count, 1, s.start, 0, 0)
        rp.end()
        d.queue.submit([enc.finish()])


def read_mask(device, maps: UVMaps) -> np.ndarray:
    """The coverage mask as a bool HxW array (tests, exports)."""
    t = maps.mask
    row = (t.width * 4 + 255) // 256 * 256
    data = device.queue.read_texture({"texture": t, "origin": (0, 0, 0)}, {"bytes_per_row": row}, (t.width, t.height, 1))
    a = np.frombuffer(data, np.uint8).reshape(t.height, row)[:, :t.width * 4].reshape(t.height, t.width, 4)
    return a[..., 0] > 127
