"""Bakers: model information baked into textures.

The low poly mesh is rasterized in UV space into a G-buffer (position, ray direction, normal, tangent,
match group); rays go from the low surface to the high poly meshes through a BVH on the GPU, within
the Frontal / Rear distances (or from a cage); secondary rays give ambient occlusion, bent normals and
thickness. After baking, Dilation extends UV islands and Apply Diffusion fills the rest.

The map is baked in tiles of at most TILE x TILE supersampled pixels (any size and antialiasing fit the
GPU's buffer limits); each tile is reduced to output pixels weighting samples by coverage, so UV borders
don't darken.

Bakers: Ambient Occlusion, Ambient Occlusion from Mesh, Bent Normals from Mesh, Color
Map from Mesh, Convert UV to SVG, Curvature, Curvature from Mesh, Height Map from Mesh, Normal Map from
Mesh, Opacity Mask from Mesh, Position, Position Map from Mesh, Thickness Map from Mesh, Transferred
Texture from Mesh, World Space Direction, World Space Normals.
"""

from __future__ import annotations

import functools
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wgpu

from trafilatura3d import bvh, imageio, meshio
from trafilatura3d.bake_params import BAKER_PARAMS, BAKERS, applies, baker_defaults, names_match, resolved  # noqa: F401
from trafilatura3d.bake_wgsl import DENOISE, DILATE, GBUFFER, PRIMARY, SECONDARY

ANTIALIASING = {"None": 1, "Subsampling 2x2": 2, "Subsampling 4x4": 4, "Subsampling 8x8": 8}
MATCH = ("Always", "By Mesh Name")
FORMATS = ("png", "tga", "tif", "exr")
SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)  # Common parameters > Size (non-square allowed)
NAME_TEMPLATE = "$(mesh)_$(bakername)"  # the default output name
TILE = 2048  # supersampled pixels per tile side: 64 MB per rgba32float target
DENOISE_STEPS = (1, 2, 4, 8)  # the denoise's a-trous steps (after its step 0): it reaches 2 * 15 samples

# Primary pass modes (bake_wgsl.PRIMARY) and the bakers that go on to the secondary pass.
PRIMARY_MODES = {"normal_map_from_mesh": 1, "height_map_from_mesh": 5, "position_map_from_mesh": 6,
                 "color_map_from_mesh": 7, "transferred_texture_from_mesh": 8, "opacity_mask_from_mesh": 9,
                 "position": 10, "world_space_direction": 11, "world_space_normals": 14, "_low_normal": 16,
                 "_low_tangent": 17, "_low_position": 18, "_low_bitangent": 19, "_low_curvature": 20}
SECONDARY_MODES = {"ambient_occlusion": 2, "ambient_occlusion_from_mesh": 2, "bent_normals_from_mesh": 3,
                   "thickness_map_from_mesh": 4, "curvature_from_mesh": 12}
LOW_ONLY = {"position", "world_space_direction", "world_space_normals", "ambient_occlusion", "curvature",
            "_low_normal", "_low_tangent", "_low_position", "_low_bitangent", "_low_curvature"}
QUALITY_PASSES = {"Low (3 pass)": 3, "Medium (5 pass)": 5, "High (10 pass)": 10, "Very High (16 pass)": 16}
RAYS_PER_PASS = 16  # ours: the page doesn't say what a pass is


@dataclass
class Settings:
    """The common baker parameters (`ignore_backface`: the Paint workspace's common setting "Ignore backface", the
    primary rays skip backface hits, default ours)."""
    size: tuple = (2048, 2048)
    format: str = "png"
    antialiasing: str = "None"
    uv_set: int = 0  # only the first UV set is read by the mesh loaders for now
    dilation: int = 16
    apply_diffusion: bool = True
    average_normals: bool = True
    high_meshes: list = field(default_factory=list)
    use_low_as_high: bool = False
    use_cage: bool = False
    cage_file: str = ""
    frontal: float = 0.01
    rear: float = 0.01
    relative_to_bounding_box: bool = True
    match: str = "Always"
    low_suffix: str = "_low"
    high_suffix: str = "_high"
    ignore_backface_suffix: str = "_ignorebf"
    ignore_backface: bool = False
    skew_correction: bool = False
    skew_map: str = ""
    invert_skew: bool = False


def _load(path: str) -> list:
    objects, _materials, _cams = meshio.load(path)
    return objects


def udim_tiles(objects: list, material: int | None = None) -> list[tuple[int, int]]:
    """The UDIM tiles (U, V offsets) the meshes' triangles are in, by their UV centroid; (0, 0) is 1001.
    `material`: only the triangles of the submeshes with that material index."""
    tiles = set()
    for o in objects:
        idx = o.idx if material is None else np.concatenate(
            [o.idx[s.start:s.start + s.count] for s in o.submeshes if s.material == material] or [o.idx[:0]])
        c = o.uv[idx.reshape(-1, 3)].mean(1)
        u, v = np.floor(c[:, 0]).astype(int), np.floor(1.0 - c[:, 1]).astype(int)  # stored y = 1 - V
        ok = (u >= 0) & (u < 10) & (v >= 0)
        tiles |= set(zip(u[ok].tolist(), v[ok].tolist()))
    return sorted(tiles, key=lambda t: (t[1], t[0])) or [(0, 0)]


def spanning_triangles(objects: list) -> list[tuple[str, int, list[int]]]:
    """The meshes with triangles whose UVs reach more than one UDIM tile: (mesh name, how many, the tiles they
    reach). Such a triangle takes the tile of its centroid (`udim_tiles`, `meshio.split_by_tile`): beyond that tile
    it shows that tile's texture repeated. A corner on a tile's border counts in the triangle's tile (pulled a hair
    toward the centroid)."""
    out = []
    for o in objects:
        uv = o.uv[o.idx.reshape(-1, 3)].astype(np.float64)  # (n, 3, 2), stored y = 1 - V
        uv = uv + (uv.mean(1, keepdims=True) - uv) * 1e-5
        u, v = np.floor(uv[..., 0]).astype(int), np.floor(1.0 - uv[..., 1]).astype(int)
        span = (u.min(1) != u.max(1)) | (v.min(1) != v.max(1))
        if span.any():
            tiles = sorted({udim(t) for t in zip(u[span].ravel().tolist(), v[span].ravel().tolist())})
            out.append((o.name, int(span.sum()), tiles))
    return out


def udim(tile) -> int:
    return 1001 + tile[0] + 10 * tile[1]


def output_path(low_path, baker: str, settings: Settings, out_dir, template: str = NAME_TEMPLATE, tile=(0, 0),
                multi_tile: bool = False) -> Path:
    """Where a baker writes: `template` with $(mesh), $(bakername) and $(udim); with several UDIM tiles and
    no $(udim) in the template, ".<udim>" is appended (Mari's convention: name.1001.png)."""
    name = template.replace("$(mesh)", Path(low_path).stem).replace("$(bakername)", baker)
    if "$(udim)" in name:
        name = name.replace("$(udim)", str(udim(tile)))
    elif multi_tile:
        name = f"{name}.{udim(tile)}"
    return Path(out_dir) / f"{name}.{'svg' if baker == 'convert_uv_to_svg' else settings.format}"


def _welded(pos: np.ndarray) -> np.ndarray:
    """Per vertex, the index of its position among the distinct ones (vertices split by seams share one)."""
    key = np.round(pos / max(float(np.abs(pos).max()), 1e-9) * 1e5).astype(np.int64)
    return np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)


def elements(o) -> np.ndarray:
    """Per vertex, the connected piece of the mesh it belongs to (Color Map's Polygroup / Submesh ID)."""
    w = _welded(o.pos)
    parent = np.arange(w.max() + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in w[o.idx.reshape(-1, 3)]:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    roots = np.array([find(a) for a in range(len(parent))])
    return np.unique(roots, return_inverse=True)[1].reshape(-1)[w]


def vertex_curvature(o) -> np.ndarray:
    """Per vertex mean curvature of the welded surface (seam free): the normal's turn per unit length along
    the edges around it, times 2 (both principal directions), as the image-space estimate measures."""
    w = _welded(o.pos)
    n = np.zeros((w.max() + 1, 3))
    np.add.at(n, w, o.nrm)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    p = np.zeros((w.max() + 1, 3))
    p[w] = o.pos
    tri = w[o.idx.reshape(-1, 3)]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    d = p[edges[:, 1]] - p[edges[:, 0]]
    k = np.sum((n[edges[:, 1]] - n[edges[:, 0]]) * d, 1) / np.maximum(np.sum(d * d, 1), 1e-18)
    acc, cnt = np.zeros(len(p)), np.zeros(len(p))
    np.add.at(acc, edges[:, 0], k)
    np.add.at(cnt, edges[:, 0], 1)
    return (2.0 * acc / np.maximum(cnt, 1))[w].astype(np.float32)


class Baker:
    def __init__(self, device=None):
        if device is None:
            from trafilatura3d.evaluator import Evaluator
            device = Evaluator().device
        self.device = device
        self._pipes = {}

    def _pipe(self, name, code):
        if name not in self._pipes:
            self._pipes[name] = self.device.create_compute_pipeline(
                layout="auto", compute={"module": self.device.create_shader_module(code=code), "entry_point": "main"})
        return self._pipes[name]

    # --- low poly G-buffer --------------------------------------------------------------------------------

    def _low_buffers(self, low: list, cage: list | None, settings: Settings, groups: dict):
        """Vertex buffers of the low meshes: position, normal, tangent, UV, ray direction, cage position, match
        group, vertex curvature; drawn per submesh with its material as the instance index."""
        d = self.device
        meshes = []
        for oi, o in enumerate(low):
            ray = _averaged_normals(o.pos, o.nrm) if settings.average_normals else o.nrm
            cp = cage[oi].pos if cage and oi < len(cage) and len(cage[oi].pos) == len(o.pos) else o.pos
            g = np.full((len(o.pos), 1), groups.get(oi, 0), np.float32)
            arrays = (o.pos, o.nrm, o.tan, o.uv, ray, cp, g, vertex_curvature(o))
            bufs = [d.create_buffer_with_data(data=np.ascontiguousarray(a, np.float32).tobytes(),
                                              usage=wgpu.BufferUsage.VERTEX) for a in arrays]
            ib = d.create_buffer_with_data(data=np.ascontiguousarray(o.idx, np.uint32).tobytes(),
                                           usage=wgpu.BufferUsage.INDEX)
            meshes.append((bufs, ib, [(s.start, s.count, s.material) for s in o.submeshes]))
        return meshes

    def _gbuffer(self, meshes, size, tile, ss, udim=(0, 0)):
        """The G-buffer of one tile: `tile` = (x0, y0, w, h) in output pixels of a map of `size`, rendered
        at `ss` samples per pixel side, of the UDIM tile `udim` (its U and V offsets)."""
        d = self.device
        (W, H), (x0, y0, tw, th) = size, tile
        R = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
        targets = [d.create_texture(size=(tw * ss, th * ss, 1), format="rgba32float", usage=R) for _ in range(5)]
        sx, sy = 2.0 * W / tw, -2.0 * H / th
        # UVs are stored with rows down (y = 1 - V): tile (u, v) covers x in [u, u+1], y in [-v, 1-v].
        view = struct.pack("4f", sx, sy, -2.0 * x0 / tw - 1.0 - udim[0] * sx, 1.0 + 2.0 * y0 / th + udim[1] * sy)
        ub = d.create_buffer_with_data(data=view, usage=wgpu.BufferUsage.UNIFORM)
        if "gbuffer" not in self._pipes:
            mod = d.create_shader_module(code=GBUFFER)
            attrs = [(12, "float32x3"), (12, "float32x3"), (16, "float32x4"), (8, "float32x2"), (12, "float32x3"),
                     (12, "float32x3"), (4, "float32"), (4, "float32")]
            vb = [{"array_stride": s, "attributes": [{"shader_location": i, "offset": 0, "format": f}]}
                  for i, (s, f) in enumerate(attrs)]
            self._pipes["gbuffer"] = [d.create_render_pipeline(
                layout="auto", vertex={"module": mod, "entry_point": "vs", "buffers": vb},
                fragment={"module": mod, "entry_point": entry, "targets": [{"format": "rgba32float"}] * n},
                primitive={"topology": "triangle-list", "cull_mode": "none"}) for entry, n in (("fs0", 2), ("fs1", 2),
                                                                                                ("fs2", 1))]
        enc = d.create_command_encoder()
        for pipe, outs in zip(self._pipes["gbuffer"], (targets[0:2], targets[2:4], targets[4:5])):
            bg = d.create_bind_group(layout=pipe.get_bind_group_layout(0),
                                     entries=[{"binding": 0, "resource": {"buffer": ub}}])
            rp = enc.begin_render_pass(color_attachments=[{"view": t.create_view(), "load_op": "clear",
                                                           "store_op": "store", "clear_value": (0, 0, 0, 0)} for t in outs])
            rp.set_pipeline(pipe)
            rp.set_bind_group(0, bg)
            for bufs, ib, submeshes in meshes:
                for i, b in enumerate(bufs):
                    rp.set_vertex_buffer(i, b)
                rp.set_index_buffer(ib, "uint32")
                for start, count, material in submeshes:
                    rp.draw_indexed(count, 1, start, 0, material)
            rp.end()
        d.queue.submit([enc.finish()])
        return targets

    # --- high poly BVH ---------------------------------------------------------------------------------------

    def _high(self, objects: list, ids: list, settings: Settings, element_base: list, material_base: list):
        """GPU buffers of high meshes: vertices (position, normal, tangent), triangles (bit 0: the mesh's name
        ends with the Ignore Backfaces suffix), BVH nodes, and per vertex attributes: (UV, mesh ID, material),
        vertex color, (material color, element)."""
        verts, attrs, tris, base = [], [], [], 0
        for o, mesh_id in zip(objects, ids):
            v = np.zeros((len(o.pos), 3, 4), np.float32)
            v[:, 0, :3], v[:, 1, :3], v[:, 2] = o.pos, o.nrm, o.tan
            verts.append(v.reshape(-1, 4))
            a = np.zeros((len(o.pos), 3, 4), np.float32)
            a[:, 0, :2] = o.uv
            a[:, 0, 2] = mesh_id
            a[:, 1] = o.colors if o.colors is not None else 1.0
            a[:, 2, 3] = elements(o) + element_base[mesh_id]
            colors = o.material_colors or [[1.0, 1.0, 1.0, 1.0]]
            for s in o.submeshes:
                rows = o.idx[s.start:s.start + s.count]
                a[rows, 0, 3] = material_base[mesh_id] + s.material
                a[rows, 2, :3] = colors[min(s.material, len(colors) - 1)][:3]
            attrs.append(a.reshape(-1, 4))
            t = o.idx.reshape(-1, 3).astype(np.uint32) + base
            flag = int(bool(settings.ignore_backface_suffix) and o.name.endswith(settings.ignore_backface_suffix))
            tris.append(np.concatenate([t, np.full((len(t), 1), flag, np.uint32)], 1))
            base += len(o.pos)
        verts, attrs, tris = np.concatenate(verts), np.concatenate(attrs), np.concatenate(tris)
        order, nodes, leaf_base = bvh.build(verts.reshape(-1, 3, 4)[:, 0, :3][tris[:, :3]])
        tris = tris[order]
        mk = lambda a: self.device.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(),
                                                           usage=wgpu.BufferUsage.STORAGE)
        return mk(verts), mk(tris), mk(nodes), mk(attrs), len(tris), leaf_base

    # --- bake --------------------------------------------------------------------------------------------------

    def bake(self, low_path: str, bakers: dict, settings: Settings, out_dir: str, template: str = NAME_TEMPLATE,
             progress=None) -> list[Path]:
        """Bake `bakers` ({baker id: parameters}) for the low poly mesh; returns the written files, named
        by `template` (see `output_path`), one per baker and UDIM tile the low mesh's UVs use.
        `progress(done, total, baker)` is called before each baker and once at the end; returning False
        stops the bake. Plugins hear before_bake (and may refuse it with Veto) and bake_finished."""
        from trafilatura3d import plugins
        out_dir = Path(out_dir)
        plugins.emit("before_bake", low_path=low_path, bakers=bakers, settings=settings, out_dir=out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        low = _load(low_path)
        tiles = udim_tiles(low)
        if settings.use_low_as_high or not settings.high_meshes:
            highs = list(low)
        else:
            highs = [o for hp in settings.high_meshes for o in _load(hp)]
        cage = _load(settings.cage_file) if settings.use_cage and settings.cage_file else None
        job = self.prepare(low, highs, cage, settings)
        written = []
        for i, (baker, params) in enumerate(bakers.items()):
            if progress is not None and progress(i, len(bakers), baker) is False:
                return written
            params = resolved(baker, params)
            for tile in tiles:
                path = output_path(low_path, baker, settings, out_dir, template, tile, len(tiles) > 1)
                if baker == "convert_uv_to_svg":
                    _uv_svg(low, path, settings.size, params, tile)
                    written.append(path)
                    continue
                img = self.finished(job, baker, params, tile)
                _write(path, img, settings.format, baker)
                written.append(path)
        if progress is not None:
            progress(len(bakers), len(bakers), "")
        plugins.emit("bake_finished", low_path=low_path, bakers=bakers, settings=settings, paths=written)
        return written

    def prepare(self, low: list, highs: list, cage: list | None, settings: Settings) -> _Job:
        """A bake of meshes already loaded (the low poly, the high poly ones, the cage or None): their buffers and
        BVHs, ready for `finished`. Match: Always pairs every low mesh with every high mesh; By Mesh Name with the
        high meshes whose names match (bake_params.names_match). Low meshes with the same matches share a group."""
        sets, groups = {}, {}
        for i, o in enumerate(low):
            if settings.match == "By Mesh Name" and not settings.use_low_as_high:
                hit = frozenset(k for k, h in enumerate(highs)
                                if names_match(o.name, h.name, settings.low_suffix, settings.high_suffix))
            else:
                hit = frozenset(range(len(highs)))
            groups[i] = sets.setdefault(hit, len(sets))
        return _Job(self, settings, low, cage, highs, {g: s for s, g in sets.items()}, groups)

    def finished(self, job: _Job, baker: str, params: dict, tile=(0, 0)) -> np.ndarray:
        """A baker's map of a prepared bake for one UDIM tile, with Dilation and Apply Diffusion (RGBA float,
        alpha = covered)."""
        img = job.result(baker, resolved(baker, params), tile)
        img = self._dilate(img, job.settings.dilation)
        if job.settings.apply_diffusion:
            img = _diffuse(img)
        return img

    def _dilate(self, img: np.ndarray, px: int) -> np.ndarray:
        """Dilation: UV islands grown by `px` pixels (each step averages filled neighbors)."""
        if px <= 0:
            return img
        d = self.device
        h, w = img.shape[:2]
        U = wgpu.TextureUsage
        tex = [d.create_texture(size=(w, h, 1), format="rgba32float",
                                usage=U.TEXTURE_BINDING | U.STORAGE_BINDING | U.COPY_DST | U.COPY_SRC) for _ in range(2)]
        d.queue.write_texture({"texture": tex[0]}, np.ascontiguousarray(img, np.float32).tobytes(),
                              {"bytes_per_row": w * 16}, (w, h, 1))
        pipe = self._pipe("dilate", DILATE)
        groups = [d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": tex[k].create_view()}, {"binding": 1, "resource": tex[1 - k].create_view()}])
            for k in (0, 1)]
        enc = d.create_command_encoder()
        for step in range(px):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, groups[step % 2])
            cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
            cp.end()
        d.queue.submit([enc.finish()])
        data = d.queue.read_texture({"texture": tex[px % 2]}, {"bytes_per_row": w * 16}, (w, h, 1))
        return np.frombuffer(data, np.float32).reshape(h, w, 4).copy()


class _Job:
    """One bake: the low mesh's buffers, the high meshes' BVHs (per match group, all of them, the low's own)
    built once, and the maps baked tile by tile in two passes."""

    def __init__(self, baker, settings, low, cage, highs, group_sets, groups):
        self.baker, self.settings, self.low, self.highs = baker, settings, low, highs
        self.group_sets = group_sets  # group -> high mesh indices
        self.groups = groups  # low mesh -> group
        self.buffers = baker._low_buffers(low, cage, settings, groups)
        self.scale = bounds_scale(low, settings)
        self.diag = bounds_scale(low, Settings(relative_to_bounding_box=True))
        counts = [int(elements(o).max()) + 1 for o in highs]
        self.element_base = list(np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int))
        self.elements = int(sum(counts))
        mats = [len(o.material_colors or [0]) for o in highs]
        self.material_base = list(np.concatenate([[0], np.cumsum(mats)[:-1]]).astype(int))
        self._bvh = {}

    def bvh(self, key):
        """("group", g), "all" (every high mesh) or "low" (the low itself, for Ambient Occlusion)."""
        if key not in self._bvh:
            if key == "low":
                objects, ids = self.low, list(range(len(self.low)))
                ebase, mbase = [0] * len(ids), [0] * len(ids)
            else:
                ids = sorted(self.group_sets[key[1]]) if key != "all" else list(range(len(self.highs)))
                if not ids:
                    self._bvh[key] = None
                    return None
                objects, ebase, mbase = [self.highs[k] for k in ids], self.element_base, self.material_base
            self._bvh[key] = self.baker._high(objects, ids, self.settings, ebase, mbase)
        return self._bvh[key]

    def result(self, baker: str, params: dict, tile) -> np.ndarray:
        """A baker's map for one UDIM tile, finished (normalized as its parameters say)."""
        if baker == "curvature":
            maps = [self.map(m, {}, tile) for m in ("_low_normal", "_low_tangent", "_low_bitangent", "_low_position",
                                                     "_low_curvature")]
            return _curvature_low(*maps, self.diag, params)
        img = self.map(baker, params, tile)
        if baker == "height_map_from_mesh":
            return _height(img, params, self.settings, self.scale)
        if baker == "curvature_from_mesh":
            return _tonemap(img, params)
        if baker == "thickness_map_from_mesh" and params["automatic_normalization"]:
            covered = img[..., 3] > 0
            top = float(img[covered, 0].max()) if covered.any() else 1.0
            img[covered, :3] /= max(top, 1e-9)
        return img

    # --- one map ---------------------------------------------------------------------------------------------

    def _boxes(self, baker, params):
        """Bounding boxes for the position bakers: per material, then the whole at the end."""
        objects = self.highs if baker == "position_map_from_mesh" else self.low
        bases = self.material_base if baker == "position_map_from_mesh" else [0] * len(objects)
        n = max(b + len(o.material_colors or [0]) for o, b in zip(objects, bases))
        n = max(n, max(s.material for o in objects for s in o.submeshes) + 1)
        lo, hi = np.full((n + 1, 3), np.inf), np.full((n + 1, 3), -np.inf)
        for o, b in zip(objects, bases):
            for s in o.submeshes:
                pts = o.pos[o.idx[s.start:s.start + s.count]]
                m = b + s.material if baker == "position_map_from_mesh" else s.material
                lo[m], hi[m] = np.minimum(lo[m], pts.min(0)), np.maximum(hi[m], pts.max(0))
        lo[n] = np.min([o.pos.min(0) for o in objects], 0)
        hi[n] = np.max([o.pos.max(0) for o in objects], 0)
        empty = ~np.isfinite(lo).all(1)
        lo[empty], hi[empty] = lo[n], hi[n]
        out = np.zeros((2 * (n + 1), 4), np.float32)
        out[0::2, :3], out[1::2, :3] = lo, hi
        return out, n

    def _primary_options(self, baker, params):
        """(mode, o0, o1, o2, o3, v0, input texture) of a baker's primary pass."""
        dx = lambda: int(params.get("normal_orientation", "DirectX") == "DirectX")
        axis = {"All axes": 0}.get(params.get("mode", "All axes"), " XYZ".find(params.get("axis", "X")))
        if baker in ("position", "position_map_from_mesh"):
            return (PRIMARY_MODES[baker], axis, int(params["normalization_type"] == "BSphere"),
                    int(params["normalization_scale"] == "Per Material"), 0, (0, 0, 0, 0), "")
        if baker == "normal_map_from_mesh":
            return 1, int(params["map_type"] == "World Space"), dx(), 0, 0, (0, 0, 0, 0), ""
        if baker == "color_map_from_mesh":
            source = ("Vertex Color", "Material Color", "Mesh ID", "Polygroup/Submesh ID").index(params["color_source"])
            gen = ("Random", "Hue Shift", "Grayscale").index(params["color_generator"])
            return 7, source, gen, 0, 0, (len(self.highs), self.elements, 0, 0), ""
        if baker == "transferred_texture_from_mesh":
            return (8, int(params["filtering_mode"] == "Nearest"), int(params["normal_map"]),
                    int(params["map_type"] == "World Space"), dx(), (0, 0, 0, 0), params["texture_file"])
        if baker == "world_space_direction":
            tex = params["input_direction"] == "From Texture"
            return 11, int(tex), dx(), 0, 0, (*params["direction"], 0.0), params["direction_file"] if tex else ""
        if baker == "world_space_normals":
            kind = ("Normal", "Tangent", "Binormal").index(params["baking_type"])
            return 14, kind, int(bool(params["normal_map"])), dx(), 0, (0, 0, 0, 0), params["normal_map"]
        if baker in SECONDARY_MODES:  # the hits, with the normal map that replaces the hit's normal
            nm = params.get("normal_map", "")
            low = baker == "ambient_occlusion"
            return (31 if low else 30, int(bool(nm)), int(params.get("world_space", False)), dx(),
                    int(params.get("invert_normal", False)), (0, 0, 0, 0), nm)
        return PRIMARY_MODES[baker], 0, 0, 0, 0, (0, 0, 0, 0), ""

    def _secondary(self, baker, params):
        """(uniform values, BVH keys per group or None for all at once) of a baker's secondary pass."""
        rel = self.diag if params.get("relative_to_bounding_box", True) else 1.0
        cull = {"Never": 0, "Always": 1, "By Mesh Name": 2, False: 0, True: 1}.get(params.get("ignore_backface"), 0)
        cos_spread = float(np.cos(np.radians(params.get("spread_angle", 180.0) / 2)))
        cosine = int(params.get("distribution", "Cosine") == "Cosine")
        if baker == "ambient_occlusion":
            rays = QUALITY_PASSES[params["quality"]] * RAYS_PER_PASS
            values = dict(mode=2, rays=rays, cosine=1, cull=0, attenuation=1, ground=0, world=0, directx=0,
                          min_occ=params["precision_bias"] * self.diag, max_occ=params["distance_fade"] * self.diag,
                          cos_spread=0.0, ground_y=0.0, radius=0.0)
            values["denoise"] = bool(params.get("denoise", False))
            return values, "low"
        by_name = params.get("self_occlusion", params.get("self_intersection", "Always")) == "By Mesh Name"
        values = dict(mode=SECONDARY_MODES[baker], rays=int(params["secondary_rays"]), cosine=cosine, cull=cull,
                      attenuation=("None", "Linear", "Smooth").index(params.get("attenuation", "None")),
                      ground=int(params.get("ground_plane", False)),
                      world=int(params.get("map_type", "") == "World space"),
                      directx=int(params.get("normal_orientation", "DirectX") == "DirectX"),
                      min_occ=params.get("min_occluder_distance", 0.0) * rel,
                      max_occ=params.get("max_occluder_distance", 0.1) * rel, cos_spread=cos_spread,
                      ground_y=float(np.min([o.pos[:, 1].min() for o in self.highs])) - params.get("ground_plane_offset", 0.0),
                      radius=params.get("sampling_radius", 0.01) * rel, denoise=bool(params.get("denoise", False)))
        return values, ("group" if by_name else "all")

    def map(self, baker: str, params: dict, udim=(0, 0)) -> np.ndarray:
        """(H, W, 4) float32 at the output size for one UDIM tile; alpha is the coverage (0 where nothing was
        baked). Raw values for the bakers finished in `result`."""
        s, d = self.settings, self.baker.device
        W, H = s.size
        ss = ANTIALIASING.get(s.antialiasing, 1)
        step = max(1, TILE // ss)
        out = np.zeros((H, W, 4), np.float32)
        mode, o0, o1, o2, o3, v0, tex_path = self._primary_options(baker, params)
        secondary = self._secondary(baker, params) if baker in SECONDARY_MODES else None
        low_only = baker in LOW_ONLY
        boxes, nmat = self._boxes(baker, params) if baker in ("position", "position_map_from_mesh") else \
            (np.zeros((2, 4), np.float32), 0)
        S, R = wgpu.BufferUsage.STORAGE, wgpu.BufferUsage.COPY_SRC
        boxes_buf = d.create_buffer_with_data(data=boxes.tobytes(), usage=S)
        texture = _texture(d, tex_path)
        skew = (2 if s.invert_skew else 1) if s.skew_correction and s.skew_map else 0
        skew_tex = _texture(d, s.skew_map if skew else "")
        sampler = d.create_sampler(mag_filter="linear", min_filter="linear", address_mode_u="repeat",
                                   address_mode_v="repeat")
        primary = self.baker._pipe("primary", bvh.WGSL + PRIMARY)
        second = self.baker._pipe("secondary", bvh.WGSL + SECONDARY) if secondary else None
        blue = d.create_buffer_with_data(data=blue_noise().tobytes(), usage=S) if secondary else None
        denoise = self.baker._pipe("denoise", DENOISE) if secondary and secondary[0]["denoise"] else None
        # A denoised tile is baked with a margin as wide as the filter's reach, then cropped: no seams.
        margin = -(-2 * sum(DENOISE_STEPS) // ss) if denoise else 0
        # The primary rays meet the high meshes of their low mesh's group (any BVH will do for low-only bakers).
        if low_only:
            passes = [(0, 1, self.bvh("low") if baker == "ambient_occlusion" else (self.bvh("all") or self.bvh("low")))]
        else:
            passes = [(g, 0, self.bvh(("group", g))) for g in self.group_sets]
            passes = [p for p in passes if p[2] is not None]
        udim_index = udim[0] + 10 * udim[1]
        for iy in range(0, H, step):
            for ix in range(0, W, step):
                iw, ih = min(step, W - ix), min(step, H - iy)  # the tile
                x0, y0 = max(0, ix - margin), max(0, iy - margin)  # the baked area: the tile and the margin
                tw, th = min(W, ix + iw + margin) - x0, min(H, iy + ih + margin) - y0
                g = self.baker._gbuffer(self.buffers, (W, H), (x0, y0, tw, th), ss, udim)
                n = tw * ss * th * ss
                acc = d.create_buffer(size=n * 16, usage=S | R)
                aux = d.create_buffer(size=n * 16, usage=S | R) if secondary else acc
                aux2 = d.create_buffer(size=n * 16, usage=S | R)
                origin = (x0 * ss, y0 * ss, udim_index, 0)
                dims = (W * ss, H * ss, int(s.ignore_backface), 0)
                enc = d.create_command_encoder()
                for group, any_group, (verts, tris, nodes, attrs, ntri, leaf_base) in passes:
                    u = struct.pack("4I 4I 4I 4f 4f 4I 4I", mode, group, int(s.use_cage and s.cage_file != ""),
                                    leaf_base, ntri, o0, o1, o2, o3, nmat, any_group, skew,
                                    s.frontal * self.scale, s.rear * self.scale, 0.0, 0.0, *v0, *origin, *dims)
                    bg = d.create_bind_group(layout=primary.get_bind_group_layout(0), entries=[
                        {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(
                            data=u, usage=wgpu.BufferUsage.UNIFORM)}},
                        *[{"binding": 1 + i, "resource": g[i].create_view()} for i in range(5)],
                        {"binding": 6, "resource": {"buffer": verts}}, {"binding": 7, "resource": {"buffer": tris}},
                        {"binding": 8, "resource": {"buffer": nodes}}, {"binding": 9, "resource": {"buffer": attrs}},
                        {"binding": 10, "resource": {"buffer": aux}},
                        {"binding": 11, "resource": texture.create_view()}, {"binding": 12, "resource": sampler},
                        {"binding": 13, "resource": {"buffer": aux2}}, {"binding": 14, "resource": {"buffer": boxes_buf}},
                        {"binding": 15, "resource": skew_tex.create_view()}])
                    cp = enc.begin_compute_pass()
                    cp.set_pipeline(primary)
                    cp.set_bind_group(0, bg)
                    cp.dispatch_workgroups((tw * ss + 7) // 8, (th * ss + 7) // 8)
                    cp.end()
                if secondary:
                    values, which = secondary
                    if which == "group":
                        runs = [(grp, 0, self.bvh(("group", grp))) for grp in self.group_sets]
                    else:
                        runs = [(0, 1, self.bvh(which))]
                    for group, any_group, b in runs:
                        if b is None:
                            continue
                        verts, tris, nodes, _attrs, ntri, leaf_base = b
                        v = values
                        u = struct.pack("4I 4I 4I 4f 4f 4I 4I", v["mode"], v["rays"], v["cosine"], leaf_base,
                                        ntri, v["cull"], v["attenuation"], v["ground"], group, any_group, v["world"],
                                        v["directx"], v["min_occ"], v["max_occ"], v["cos_spread"], v["ground_y"],
                                        v["radius"], 0.0, 0.0, 0.0, *origin, *dims)
                        bg = d.create_bind_group(layout=second.get_bind_group_layout(0), entries=[
                            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(
                                data=u, usage=wgpu.BufferUsage.UNIFORM)}},
                            {"binding": 1, "resource": {"buffer": aux}}, {"binding": 2, "resource": {"buffer": aux2}},
                            {"binding": 3, "resource": {"buffer": verts}}, {"binding": 4, "resource": {"buffer": tris}},
                            {"binding": 5, "resource": {"buffer": nodes}}, {"binding": 6, "resource": {"buffer": acc}},
                            {"binding": 7, "resource": g[2].create_view()}, {"binding": 8, "resource": g[3].create_view()},
                            {"binding": 9, "resource": {"buffer": blue}}])
                        cp = enc.begin_compute_pass()
                        cp.set_pipeline(second)
                        cp.set_bind_group(0, bg)
                        cp.dispatch_workgroups((tw * ss + 7) // 8, (th * ss + 7) // 8)
                        cp.end()
                    if denoise:
                        tmp = d.create_buffer(size=n * 16, usage=S)
                        var = [d.create_buffer(size=n * 4, usage=S) for _ in range(2)]
                        # Step 0 writes the variance only; then an even count of steps: the result ends in acc.
                        for k, st in enumerate((0, *DENOISE_STEPS)):
                            src, dst = (tmp, acc) if k % 2 == 0 else (acc, tmp)
                            bg = d.create_bind_group(layout=denoise.get_bind_group_layout(0), entries=[
                                {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(
                                    data=struct.pack("4I", tw * ss, th * ss, st, 0), usage=wgpu.BufferUsage.UNIFORM)}},
                                {"binding": 1, "resource": {"buffer": aux}}, {"binding": 2, "resource": {"buffer": aux2}},
                                {"binding": 3, "resource": {"buffer": acc if k == 0 else src}},
                                {"binding": 4, "resource": {"buffer": tmp if k == 0 else dst}},
                                {"binding": 5, "resource": {"buffer": var[(k + 1) % 2]}},
                                {"binding": 6, "resource": {"buffer": var[k % 2]}}])
                            cp = enc.begin_compute_pass()
                            cp.set_pipeline(denoise)
                            cp.set_bind_group(0, bg)
                            cp.dispatch_workgroups((tw * ss + 7) // 8, (th * ss + 7) // 8)
                            cp.end()
                d.queue.submit([enc.finish()])
                tile = np.frombuffer(d.queue.read_buffer(acc), np.float32).reshape(th * ss, tw * ss, 4)
                tile = tile[(iy - y0) * ss:(iy - y0 + ih) * ss, (ix - x0) * ss:(ix - x0 + iw) * ss]
                out[iy:iy + ih, ix:ix + iw] = _reduce(tile, ss)
        return out

# --- baking visualization (high poly, cage, high poly outside the cage) --------

def bounds_scale(low: list, settings: Settings) -> float:
    """What Frontal / Rear distances are multiplied by: the low's bounding box diagonal when they are
    relative to it."""
    if not settings.relative_to_bounding_box:
        return 1.0
    lo = np.min([o.pos.min(0) for o in low], 0)
    hi = np.max([o.pos.max(0) for o in low], 0)
    return float(np.linalg.norm(hi - lo)) or 1.0


def _offset_surface(low: list, settings: Settings, distance: float) -> list:
    """The low surface moved by `distance` (times the bounds scale) along the bake's ray directions
    (averaged normals, or the vertex normals when Average Normals is off)."""
    out = []
    push = distance * bounds_scale(low, settings)
    for o in low:
        ray = _averaged_normals(o.pos, o.nrm) if settings.average_normals else o.nrm
        out.append(meshio.MeshObject(o.name, (o.pos + ray * push).astype(np.float32), o.nrm, o.tan, o.uv, o.idx,
                                     o.submeshes))
    return out


def implicit_cage(low: list, settings: Settings) -> list:
    """The cage the bake uses without a cage file: the low surface pushed out by Max Frontal Distance."""
    return _offset_surface(low, settings, settings.frontal)


def rear_surface(low: list, settings: Settings) -> list:
    """How deep the bake's rays reach without a cage file: the low surface pushed in by Max Rear Distance."""
    return _offset_surface(low, settings, -settings.rear)


# Which high poly vertices the bake's rays reach: the nearest point of the low (BVH nearest query), the
# ray direction there; reached when the vertex lies on that ray's segment: within [-rear, +frontal] along
# it, or from the cage file's point to as far behind the low (what `main` traces).
UNREACHED = """
struct C { count: u32, leaf_base: u32, tri_count: u32, use_cage: u32, frontal: f32, rear: f32, eps: f32, pad: f32 };
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;  // per low vertex: position, ray direction, cage
@group(0) @binding(2) var<storage, read> tris: array<vec4u>;
@group(0) @binding(3) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(4) var<storage, read> points: array<vec4f>;  // high vertices
@group(0) @binding(5) var<storage, read_write> outside: array<u32>;

fn box_dist2(p: vec3f, lo: vec3f, hi: vec3f) -> f32 {
    let d = max(max(lo - p, p - hi), vec3f(0.0));
    return dot(d, d);
}
// Closest point of triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5): (u, v) weights of b
// and c, and the squared distance.
fn closest(p: vec3f, a: vec3f, b: vec3f, cc: vec3f) -> vec3f {
    let ab = b - a; let ac = cc - a; let ap = p - a;
    let d1 = dot(ab, ap); let d2 = dot(ac, ap);
    var uv = vec2f(0.0);
    if (d1 <= 0.0 && d2 <= 0.0) { uv = vec2f(0.0, 0.0); }
    else {
        let bp = p - b; let d3 = dot(ab, bp); let d4 = dot(ac, bp);
        let cp = p - cc; let d5 = dot(ab, cp); let d6 = dot(ac, cp);
        let vc = d1 * d4 - d3 * d2; let vb = d5 * d2 - d1 * d6; let va = d3 * d6 - d5 * d4;
        if (d3 >= 0.0 && d4 <= d3) { uv = vec2f(1.0, 0.0); }
        else if (d6 >= 0.0 && d5 <= d6) { uv = vec2f(0.0, 1.0); }
        else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { uv = vec2f(d1 / (d1 - d3), 0.0); }
        else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { uv = vec2f(0.0, d2 / (d2 - d6)); }
        else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            uv = vec2f(1.0 - w, w);
        } else {
            let den = 1.0 / (va + vb + vc);
            uv = vec2f(vb * den, vc * den);
        }
    }
    let q = a + ab * uv.x + ac * uv.y;
    return vec3f(uv, dot(p - q, p - q));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= c.count) { return; }
    let p = points[id.x].xyz;
    var best = 1e30;
    var best_tri = 0u;
    var best_uv = vec2f(0.0);
    var stack: array<u32, 40>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp -= 1u;
        let n = stack[sp];
        if (box_dist2(p, nodes[n * 2u].xyz, nodes[n * 2u + 1u].xyz) >= best) { continue; }
        if (n >= c.leaf_base) {
            let first = (n - c.leaf_base) * 4u;
            for (var k = 0u; k < 4u; k++) {
                let ti = first + k;
                if (ti >= c.tri_count) { break; }
                let t = tris[ti];
                let r = closest(p, verts[t.x * 3u].xyz, verts[t.y * 3u].xyz, verts[t.z * 3u].xyz);
                if (r.z < best) { best = r.z; best_tri = ti; best_uv = r.xy; }
            }
        } else if (sp + 2u <= 40u) {
            stack[sp] = n * 2u + 2u;
            stack[sp + 1u] = n * 2u + 1u;
            sp += 2u;
        }
    }
    let t = tris[best_tri];
    let w = vec3f(1.0 - best_uv.x - best_uv.y, best_uv.x, best_uv.y);
    let q = verts[t.x * 3u].xyz * w.x + verts[t.y * 3u].xyz * w.y + verts[t.z * 3u].xyz * w.z;
    var dir = normalize(verts[t.x * 3u + 1u].xyz * w.x + verts[t.y * 3u + 1u].xyz * w.y + verts[t.z * 3u + 1u].xyz * w.z);
    var front = c.frontal;
    var back = c.rear;
    if (c.use_cage == 1u) {
        let cage = verts[t.x * 3u + 2u].xyz * w.x + verts[t.y * 3u + 2u].xyz * w.y + verts[t.z * 3u + 2u].xyz * w.z;
        front = length(cage - q);
        back = front;
        dir = (cage - q) / max(front, 1e-12);
    }
    let rel = p - q;
    let along = dot(rel, dir);
    let side = length(rel - dir * along);  // off the ray: beyond the low's borders, nothing reaches it
    let reached = along <= front + c.eps && along >= -back - c.eps && side <= 0.25 * (front + back) + c.eps;
    outside[id.x] = select(1u, 0u, reached);
}
"""


def unreached(device, highs: list, low: list, settings: Settings, cage: list | None = None) -> list[np.ndarray]:
    """Per high mesh, a bool per vertex: no bake ray reaches it (outside the cage or beyond the rear
    distance; shown red). `cage`: the cage file's meshes (same topology as the
    low), else the implicit cage from the frontal and rear distances."""
    scale = bounds_scale(low, settings)
    verts, tris, base = [], [], 0
    for i, o in enumerate(low):
        v = np.zeros((len(o.pos), 3, 4), np.float32)
        v[:, 0, :3] = o.pos
        v[:, 1, :3] = _averaged_normals(o.pos, o.nrm) if settings.average_normals else o.nrm
        v[:, 2, :3] = cage[i].pos if cage and i < len(cage) and len(cage[i].pos) == len(o.pos) else o.pos
        verts.append(v.reshape(-1, 4))
        t = o.idx.reshape(-1, 3).astype(np.uint32) + base
        tris.append(np.concatenate([t, np.zeros((len(t), 1), np.uint32)], 1))
        base += len(o.pos)
    verts, tris = np.concatenate(verts), np.concatenate(tris)
    order, nodes, leaf_base = bvh.build(verts.reshape(-1, 3, 4)[:, 0, :3][tris[:, :3]])
    tris = tris[order]
    S = wgpu.BufferUsage.STORAGE
    mk = lambda a: device.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(), usage=S)
    pts = np.zeros((sum(len(o.pos) for o in highs), 4), np.float32)
    pts[:, :3] = np.concatenate([o.pos for o in highs])
    n = len(pts)
    flags = device.create_buffer(size=max(n, 1) * 4, usage=S | wgpu.BufferUsage.COPY_SRC)
    pipe = device.create_compute_pipeline(layout="auto", compute={
        "module": device.create_shader_module(code=UNREACHED), "entry_point": "main"})
    counts = device.create_buffer_with_data(
        data=struct.pack("4I 4f", n, leaf_base, len(tris), int(bool(cage)), settings.frontal * scale,
                         settings.rear * scale, 1e-5 * float(np.abs(verts).max() or 1.0), 0.0),
        usage=wgpu.BufferUsage.UNIFORM)
    bg = device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": counts}}, {"binding": 1, "resource": {"buffer": mk(verts)}},
        {"binding": 2, "resource": {"buffer": mk(tris)}}, {"binding": 3, "resource": {"buffer": mk(nodes)}},
        {"binding": 4, "resource": {"buffer": mk(pts)}}, {"binding": 5, "resource": {"buffer": flags}}])
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((n + 63) // 64)
    cp.end()
    device.queue.submit([enc.finish()])
    out = np.frombuffer(device.queue.read_buffer(flags), np.uint32)[:n].astype(bool)
    return np.split(out, np.cumsum([len(o.pos) for o in highs])[:-1])


def _reduce(tile: np.ndarray, ss: int) -> np.ndarray:
    """Supersampled samples to output pixels: color averaged over the covered samples, alpha = coverage."""
    if ss == 1:
        return tile
    h, w = tile.shape[0] // ss, tile.shape[1] // ss
    t = tile.reshape(h, ss, w, ss, 4)
    a = t[..., 3:]
    cov = a.sum((1, 3))
    rgb = (t[..., :3] * a).sum((1, 3)) / np.maximum(cov, 1e-9)
    return np.concatenate([rgb, cov / (ss * ss)], -1)


@functools.cache
def blue_noise(size: int = 64, sigma: float = 1.5) -> np.ndarray:
    """(size, size, 2) float32: two blue noise masks (void and cluster, Ulichney), values (rank + 0.5) /
    size^2, uniform and without low frequencies. The secondary rays' rotation per texel."""
    d = np.minimum(np.arange(size), size - np.arange(size))
    kernel = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma * sigma))
    n = size * size

    def splat(energy, p, sign):
        energy += sign * np.roll(kernel, (p // size, p % size), (0, 1)).ravel()

    def mask(seed):
        on = np.zeros(n, bool)
        on[np.random.default_rng(seed).choice(n, n // 10, replace=False)] = True
        energy = np.zeros(n)
        for p in np.flatnonzero(on):
            splat(energy, p, 1)
        while True:  # the initial pattern: the tightest cluster's point to the largest void
            c = np.flatnonzero(on)[np.argmax(energy[on])]
            on[c] = False
            splat(energy, c, -1)
            v = np.flatnonzero(~on)[np.argmin(energy[~on])]
            on[v] = True
            splat(energy, v, 1)
            if v == c:
                break
        rank = np.zeros(n)
        ones, e1 = on.copy(), energy.copy()
        for r in range(int(on.sum()) - 1, -1, -1):  # the initial points, tightest cluster last
            c = np.flatnonzero(ones)[np.argmax(e1[ones])]
            ones[c] = False
            splat(e1, c, -1)
            rank[c] = r
        for r in range(int(on.sum()), n):  # the others, largest void first
            v = np.flatnonzero(~on)[np.argmin(energy[~on])]
            on[v] = True
            splat(energy, v, 1)
            rank[v] = r
        return ((rank + 0.5) / n).reshape(size, size)

    return np.stack([mask(1), mask(2)], -1).astype(np.float32)


def _averaged_normals(pos, nrm):
    """Average Normals: the ray direction of vertices sharing a position (split for hard
    edges or UV seams) averaged."""
    key = np.round(pos / max(float(np.abs(pos).max()), 1e-9) * 1e5).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    acc = np.zeros((inverse.max() + 1, 3))
    np.add.at(acc, inverse, nrm)
    out = acc[inverse]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _diffuse(img: np.ndarray) -> np.ndarray:
    """Apply Diffusion: the empty texels get the smoothed colors of the UV borders (push-pull)."""
    levels = [img]
    while min(levels[-1].shape[:2]) > 1:
        a = levels[-1]
        h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        a = a[:h, :w].reshape(h // 2, 2, w // 2, 2, 4)
        wsum = a[..., 3:].sum((1, 3))
        rgb = (a[..., :3] * a[..., 3:]).sum((1, 3)) / np.maximum(wsum, 1e-9)
        levels.append(np.concatenate([rgb, (wsum > 0).astype(np.float32)], -1))
    for i in range(len(levels) - 2, -1, -1):
        a, coarse = levels[i], levels[i + 1]
        up = np.repeat(np.repeat(coarse, 2, 0), 2, 1)
        up = np.pad(up, ((0, a.shape[0] - up.shape[0]), (0, a.shape[1] - up.shape[1]), (0, 0)), mode="edge")
        empty = a[..., 3] <= 0
        a[empty] = up[empty]
        levels[i] = a
    out = levels[0]
    out[..., 3] = 1.0
    return out


def _gray(v: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.stack([v, v, v, alpha], -1).astype(np.float32)
    out[alpha <= 0] = 0.0
    return out


def _curvature_low(normal, tangent, bitangent, position, vertex, scale, params) -> np.ndarray:
    """Curvature: how fast the low's normal turns per unit of surface along the image's x and y (measured on
    the tangent and the bitangent, which follow them), relative to the bounding box, times Details; black
    concave, white convex, grey flat. Texels at UV island borders have no neighbors across the seam: with
    Enable Seams they take the seam free per vertex curvature (`vertex_curvature`), by Seams Intensity."""
    n, t, b, p, alpha = normal[..., :3], tangent[..., :3], bitangent[..., :3], position[..., :3], normal[..., 3]
    dx = np.roll(n, -1, 1) - np.roll(n, 1, 1)
    dy = np.roll(n, -1, 0) - np.roll(n, 1, 0)  # down the image, like the bitangent
    lx = np.linalg.norm(np.roll(p, -1, 1) - np.roll(p, 1, 1), axis=-1)
    ly = np.linalg.norm(np.roll(p, -1, 0) - np.roll(p, 1, 0), axis=-1)
    valid = (np.roll(alpha, -1, 1) > 0) & (np.roll(alpha, 1, 1) > 0) & (np.roll(alpha, -1, 0) > 0) & \
        (np.roll(alpha, 1, 0) > 0)
    k = np.where(valid, np.sum(dx * t, -1) / np.maximum(lx, 1e-12) + np.sum(dy * b, -1) / np.maximum(ly, 1e-12), 0.0)
    if params["enable_seams"]:
        border = (alpha > 0) & ~valid
        k = np.where(border, (1 - params["seams_intensity"]) * k + params["seams_intensity"] * vertex[..., 0], k)
    v = 0.5 + k * scale * params["details"] / 32.0
    return _gray(np.clip(v, 0.0, 1.0), alpha)


def _height(img: np.ndarray, params: dict, settings: Settings, scale: float) -> np.ndarray:
    """Height Map from Mesh's Normalization of the signed distances (high above the low positive), per UV
    tile: Relative to ray distance maps [-rear, +frontal] to [0, 1]; Relative to low poly mesh keeps the low
    surface at 0.5 with the largest distance at 0 or 1 (ours: the page gives no detail); Relative to Min/Max
    maps the smallest to 0 and the largest to 1; Manual is 0.5 + distance / Scaling divisor."""
    d, covered = img[..., 0], img[..., 3] > 0
    norm = params["normalization"]
    if norm == "Relative to ray distance":
        f, r = settings.frontal * scale, settings.rear * scale
        v = (d + r) / max(f + r, 1e-12)
    elif norm == "Relative to Min/Max (per UV Tile)":
        lo, hi = (float(d[covered].min()), float(d[covered].max())) if covered.any() else (0.0, 1.0)
        v = (d - lo) / max(hi - lo, 1e-12)
    elif norm == "Manual":
        v = 0.5 + d / max(params["scaling_divisor"], 1e-12)
    else:
        top = float(np.abs(d[covered]).max()) if covered.any() else 1.0
        v = 0.5 + d / max(2.0 * top, 1e-12)
    return _gray(np.clip(v, 0.0, 1.0), img[..., 3])


def _tonemap(img: np.ndarray, params: dict) -> np.ndarray:
    """Curvature from Mesh's tonemapping: the range found (Automatic Tonemapping bounds) or Tonemapping Min /
    Max to [0, 1]."""
    k, covered = img[..., 0], img[..., 3] > 0
    if params["automatic_tonemapping"]:
        lo, hi = (float(k[covered].min()), float(k[covered].max())) if covered.any() else (-1.0, 1.0)
    else:
        lo, hi = params["tonemapping_min"], params["tonemapping_max"]
    return _gray(np.clip((k - lo) / max(hi - lo, 1e-12), 0.0, 1.0), img[..., 3])


def _texture(device, path):
    if path and Path(path).exists():
        from trafilatura3d.nodes.atomic.raster import _read_image
        pixels = _read_image(path)
    else:
        pixels = np.ones((1, 1, 4), np.float32)
    h, w = pixels.shape[:2]
    t = device.create_texture(size=(w, h, 1), format="rgba16float",
                              usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    device.queue.write_texture({"texture": t}, np.ascontiguousarray(pixels, np.float16).tobytes(),
                               {"bytes_per_row": w * 8}, (w, h, 1))
    return t


def _write(path: Path, img: np.ndarray, fmt: str, baker: str):
    gray = baker in ("ambient_occlusion", "ambient_occlusion_from_mesh", "thickness_map_from_mesh", "height_map_from_mesh",
                     "opacity_mask_from_mesh", "curvature", "curvature_from_mesh")
    data = img[..., :1] if gray else img[..., :3]
    if fmt == "exr":
        imageio.write(path, data.astype(np.float32))
    elif fmt == "tga":
        imageio.write(path, (np.clip(data, 0, 1) * 255 + 0.5).astype(np.uint8))
    else:
        imageio.write(path, (np.clip(data, 0, 1) * 65535 + 0.5).astype(np.uint16))


def uv_shells(o) -> np.ndarray:
    """Per triangle, its UV shell: triangles connected through shared vertices (UV seams split vertices)."""
    tri = o.idx.reshape(-1, 3)
    parent = np.arange(len(o.uv))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in tri:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    roots = np.array([find(a) for a in tri[:, 0]])
    return np.unique(roots, return_inverse=True)[1].reshape(-1)


def _svg_color(mode: str, shell: int, count: int, material_color) -> str:
    import colorsys
    if mode == "Uniform color":
        rgb = (0.5, 0.5, 0.5)
    elif mode == "Material ID Color":
        rgb = tuple(material_color[:3])
    elif mode == "HUE Shift":
        rgb = colorsys.hsv_to_rgb(shell / max(count, 1), 1.0, 1.0)
    elif mode == "Grayscale":
        rgb = ((shell + 1) / max(count, 1),) * 3
    else:  # Random
        s = (shell * 2654435761 + 12345) & 0xffffffff
        s = ((s ^ (s >> 16)) * 2246822519) & 0xffffffff
        s ^= s >> 13
        rgb = ((s & 255) / 255, ((s >> 8) & 255) / 255, ((s >> 16) & 255) / 255)
    return "#" + "".join(f"{int(round(min(max(c, 0.0), 1.0) * 255)):02x}" for c in rgb)


def _uv_svg(objects, path: Path, size, params: dict, tile=(0, 0)):
    """Convert UV to SVG: the UV shells of one UDIM tile as filled shapes, colored per shell by Color Mode,
    grown by Padding (pixels, as a stroke of the same color)."""
    w, h = size
    shift = np.array([-tile[0], tile[1]], np.float64)  # stored y = 1 - V
    pad = float(params["padding"])
    shells_total = sum(int(uv_shells(o).max()) + 1 for o in objects)
    groups, base = [], 0
    for o in objects:
        tri = o.idx.reshape(-1, 3)
        shells = uv_shells(o)
        material = np.zeros(len(tri), int)
        for s in o.submeshes:
            material[s.start // 3:(s.start + s.count) // 3] = s.material
        colors = o.material_colors or [[1.0, 1.0, 1.0, 1.0]]
        for sh in range(int(shells.max()) + 1):
            rows = np.nonzero(shells == sh)[0]
            fill = _svg_color(params["color_mode"], base + sh, shells_total,
                              colors[min(int(material[rows[0]]), len(colors) - 1)])
            polys = []
            for t in tri[rows]:
                pts = (o.uv[t] + shift) * (w, h)
                polys.append('<polygon points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in pts) + '"/>')
            stroke = f' stroke="{fill}" stroke-width="{2 * pad:g}" stroke-linejoin="round"' if pad > 0 else ""
            groups.append(f'<g fill="{fill}"{stroke}>' + "".join(polys) + "</g>")
        base += int(shells.max()) + 1
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
                    + "".join(groups) + "</svg>\n", encoding="utf-8")
