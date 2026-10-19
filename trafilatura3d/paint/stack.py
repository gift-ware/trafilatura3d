"""The layer stack compiled to evaluator jobs, per Texture Set and channel.

A channel's result is RGBA: RGB the value, A how much of the texture the layers cover (a grayscale channel stores
its value and coverage in two channels, PortType.GRAY_ALPHA, read as (v, v, v, a)) (unpainted texels keep the channel's default value with alpha 0, what the export's alpha and
padding use). Order from the documentation: bottom layer first; a layer is computed, blended over the result
below with its channel's mode and opacity, its mask scaling the blend. Folders composite their content on an
empty layer, then blend it like a layer (ours: the documentation does not say).

Every job's key hashes what decides its pixels (the evaluator recooks only new keys): painted entries by their
revision, images by their resource name and content stamp.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from trafilatura3d.evaluator import HALO, PREVIEW, Evaluator, Job
from trafilatura3d.nodes.base import PAGE_GRID, PortType
from trafilatura3d.nodes.paint._lib import MESH_MAPS as PAINT_MESH_MAPS
from trafilatura3d.paint import blending
from trafilatura3d.paint.channels import CHANNELS, FORMATS, default_rgba, depth, is_hdr
from trafilatura3d.paint.model import (FILTERING, LEVELS_COLORS, MASK_KEY, MATCH_PER_TILE, PROJECTIONS, SHAPE_CROP,
                                   UV_WRAP, Layer, Project, TextureSet)
from trafilatura3d.paint.nodes import BLEND, CONVERT, COPY, DILATE, ERASE, FILL, LEVELS, NEAREST, NORMAL
from trafilatura3d.paint.paint import MASK, PaintStore

MASK_DEPTH = "16f"  # a mask's computations
MASK_RESULT_DEPTH = "8"  # a layer's mask as kept: 256 levels of how much the layer shows
# The mesh maps holding one value (the Paint filters and generators read them as grayscale), kept in one channel.
GRAY_MESH_MAPS = {m[0] for m in PAINT_MESH_MAPS.values() if m[1] is PortType.GRAY} | {"opacity"}


class _Reached(Exception):
    """Compiling a channel up to an effect or a layer (`StackCompiler.channel_at`): the stack below it."""

    def __init__(self, job):
        super().__init__()
        self.job = job


def _gray(a: np.ndarray) -> bool:
    """R = G = B (and alpha 1): a colour image is told apart on a sample first (whole comparisons of every image
    of a project took half a second at opening)."""
    return np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]) \
        and (a.shape[2] == 3 or bool(np.all(a[..., 3] == 1)))


class Packed:
    """An image kept compressed (LZ4 of its half floats): what a project holds in memory between the cooks that read
    it (the baked mesh maps of a tiled project are gigabytes)."""

    __slots__ = ("data", "shape", "dtype")

    def __init__(self, image: np.ndarray | None = None):
        import lz4.block
        if image is None:  # `of`
            return
        a = np.ascontiguousarray(image)
        # read in place (no copy of the image: every large temporary of a load stays reserved by the allocator)
        self.data = lz4.block.compress(memoryview(a).cast("B"), store_size=False)
        self.shape, self.dtype = a.shape, a.dtype

    @classmethod
    def of(cls, data: bytes, shape, dtype) -> Packed:
        """An image packed already (a project file keeps them so)."""
        p = cls()
        p.data, p.shape, p.dtype = data, tuple(int(v) for v in shape), np.dtype(dtype)
        return p

    def stamp(self) -> str:
        """What identifies the content (of the packed bytes: no unpacking)."""
        return hashlib.sha1(self.data).hexdigest()[:16]

    def unpack(self) -> np.ndarray:
        import lz4.block
        n = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        return np.frombuffer(lz4.block.decompress(self.data, uncompressed_size=n), self.dtype).reshape(self.shape)


class Images(dict):
    """name -> image of a project's resources: kept Packed, unpacked when read (`[name]`, `items()`)."""

    def __getitem__(self, name):
        return super().__getitem__(name).unpack()

    def __setitem__(self, name, image):
        super().__setitem__(name, image if isinstance(image, Packed) else Packed(image))

    def get(self, name, default=None):
        return self[name] if name in self else default

    def items(self):
        return ((name, self[name]) for name in self)

    def values(self):
        return (self[name] for name in self)

    def pop(self, name, *default):
        if name not in self:
            if default:
                return default[0]
            raise KeyError(name)
        return super().pop(name).unpack()

    def packed(self, name) -> Packed:
        return super().__getitem__(name)


class Resources:
    """Images a project uses (fill layers, imported mesh maps): name -> float16 HxWx4, or HxWx1 for a grayscale
    image (a quarter of the memory: AO, curvature, thickness, height maps), with a stamp that changes with the
    content. Kept compressed in memory (`Images`), unpacked when read."""

    def __init__(self):
        self.images = Images()
        self.stamps: dict[str, str] = {}

    def add(self, name: str, pixels: np.ndarray) -> None:
        self.put(name, *self.prepare(pixels))

    def put(self, name: str, image, stamp: str) -> None:
        """An image made by `prepare` (which threads can run: opening a project prepares its images in parallel)."""
        self.images[name], self.stamps[name] = image, stamp

    def shape(self, name: str) -> tuple:
        """(height, width, channels) of an image, without unpacking it."""
        return self.images.packed(name).shape

    @staticmethod
    def prepare(pixels: np.ndarray) -> tuple[Packed, str]:
        """(image, stamp): kept in half floats, as the project file stores them (a 4096 image is 128 MB, not 256); a
        grayscale image (one or two channels, or R = G = B with alpha 1) as one channel (32 MB); then compressed."""
        a = np.asarray(pixels, np.float16)
        if a.ndim == 2:
            a = a[..., None]
        if a.shape[2] == 2:  # gray and alpha
            a = a[..., :1] if np.all(a[..., 1] == 1) else np.concatenate([a[..., :1].repeat(3, 2), a[..., 1:]], 2)
        if a.shape[2] >= 3 and _gray(a[::61, ::61]) and _gray(a):
            a = a[..., :1]
        elif a.shape[2] == 3:  # RGB, alpha 1
            a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float16)], 2)
        packed = Packed(np.ascontiguousarray(a))
        return packed, packed.stamp()

    def unique(self, base: str) -> str:
        name, k = base, 1
        while name in self.images or self.sequence(name):
            k += 1
            name = f"{base}_{k}"
        return name

    # Image sequences (Image sequence page): the images named <name>.<UDIM> are one resource, <name>; the baked
    # mesh maps of a UV Tile project are one too.

    def sequence(self, name: str) -> dict[int, str]:
        """{UDIM: image name} of the sequence `name` ({} when there is none)."""
        out = {}
        for n in self.images:
            base, dot, tile = n.rpartition(".")
            if dot and base == name and tile.isdigit() and len(tile) == 4 and int(tile) >= 1001:
                out[int(tile)] = n
        return dict(sorted(out.items()))

    def image_of(self, name: str, udim: int | None = None) -> str | None:
        """The image a resource reference reads: an image itself; of a sequence, the tile's (`udim`) or, without a
        tile, its first (UV projection: the first image of a sequence on every tile); None when there is none."""
        if name in self.images:
            return name
        seq = self.sequence(name)
        if not seq:
            return None
        return seq.get(udim) if udim is not None else next(iter(seq.values()))

    def listing(self) -> list[tuple[str, str]]:
        """The resources as the Assets show them: (name, image of its preview), a sequence once under its name."""
        out, seen = [], set()
        for n in sorted(self.images):
            base, dot, tile = n.rpartition(".")
            if dot and tile.isdigit() and len(tile) == 4 and base not in self.images and self.sequence(base):
                if base not in seen:
                    seen.add(base)
                    out.append((base, self.image_of(base)))
                continue
            out.append((n, n))
        return sorted(out)


def _value(gray: bool) -> PortType:
    """What a channel's job holds: a grayscale channel its value and coverage in two channels, else RGBA."""
    return PortType.GRAY_ALPHA if gray else PortType.COLOR


def _stored(job: Job, depth: str) -> Job:
    """The last job of a chain writing its result at `depth` (computed as before, kept in less memory): the masks'
    final results, computed in half floats and kept in 8 bits."""
    if job.depths.get("output") == depth:
        return job
    return dataclasses.replace(job, key=_key(job.key, "stored", depth), depths={**job.depths, "output": depth})


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class StackCompiler:
    """The jobs of a Texture Set's stack in one UDIM tile (`udim`: every tile is its own texture, 1001 without
    the UV Tile workflow): its painted entries, its mesh maps, its mesh in texture space; the tile is part of
    every job's key. `limit`: the largest side the jobs compute at (the viewports' resolution: the stack's
    paintings, resources and mesh maps are read resampled), None the tile's own resolution (export, bake)."""

    def __init__(self, project: Project, store: PaintStore, resources: Resources, evaluator: Evaluator, udim: int,
                 uv_masks: dict | None = None, materials=None, log=None, computed=None, geometry=None,
                 path_pixels=None, coverage=None, limit: int | None = None):
        self.project, self.store, self.resources, self.ev, self.udim = project, store, resources, evaluator, udim
        self.limit = limit
        self.materials = materials  # paint.materials.Materials (Material mode), None: not available
        self.log = log or (lambda text: None)
        # (Texture Set name, tile) -> GPU texture of the UV coverage (padding)
        self.uv_masks = uv_masks if uv_masks is not None else {}
        self._pages = {}  # _pages_on of the tile's UV mask
        # The session's callbacks, each also given the tile; None: not available.
        self.computed = computed  # (Texture Set, slot) -> GPU texture of uvmaps.COMPUTED
        self.geometry = geometry  # Texture Set -> (uvmaps.UVMaps, bbox min, bbox max)
        self.path_pixels = path_pixels  # (Texture Set, layer, path, channel) -> (texture, revision) of a path
        self.coverage = coverage  # (Texture Set, excluded mesh names) -> GPU texture of the Geometry mask
        self._stop = None  # the effect or layer uid where `channel_at` stops compiling
        self._reached: dict = {}  # `channel_at` results of the current outermost `channel` call
        self._depth = 0
        self._reduced: set[tuple] = set()  # the sizes `limit` brought down (their jobs: the cache's preview partition)
        # Within a planning (Session.planning: the project does not change): a channel and a layer's mask compiled
        # once, not for every slot, tile pass and restart point that asks (a 63 layer stack: its masks compiled 20
        # times each per refresh). None outside.
        self.memo: dict | None = None

    # --- jobs -------------------------------------------------------------------------------------

    def _job(self, node, values: dict, inputs: dict, size, dep: str, extra=None, key_extra=None, path="",
             out: PortType = PortType.COLOR) -> Job:
        """`out`: what the result holds, stored as such: COLOR (RGBA), GRAY_ALPHA (a grayscale channel's value and
        coverage), GRAY (a mask: one channel, no coverage)."""
        full = {name: values.get(name, p.default) for name, p in node.params.items()}
        ins = {port: inputs.get(port) for port in node.inputs}
        key = _key(self.udim, node.type, full, {p: (j[0].key if j else None) for p, j in ins.items()}, list(size),
                   dep, key_extra, out.value)
        return Job(key=key, path=path or node.type, node=node, values=full, inputs=ins,
                   output_types={"output": out}, size=tuple(size),
                   depths={"output": dep}, tiling=3, seed=0.0, extra=extra or {}, keep=False,
                   partition=self.partition(size))

    def _size(self, ts: TextureSet) -> list:
        """The size the jobs compute at: the tile's, scaled down to `limit` (aspect kept)."""
        w, h = ts.size_of(self.udim)
        k = 1 if self.limit is None else max(max(w, h) // self.limit, 1)
        size = [max(w // k, 1), max(h // k, 1)]
        if k > 1:
            self._reduced.add(tuple(size))
        return size

    def partition(self, size) -> str:
        """The cache partition of a job of this size: the preview's when `limit` reduced it (Job.partition)."""
        return PREVIEW if tuple(size) in self._reduced else ""

    @staticmethod
    def _checkpoint(job: Job | None) -> Job | None:
        """A result an edit restarts from (a layer's, an effect's, a mask, a mesh map): the cache keeps it within
        its budget; the stack's other jobs are intermediates, freed while cooking (Job.keep)."""
        if job is not None:
            job.keep = True
        return job

    def _texture_job(self, texture, key_extra, size, dep, gray=False) -> Job:
        return self._job(COPY, {"gray": gray}, {}, size, dep, extra={"texture": texture}, key_extra=key_extra,
                         out=_value(gray))

    def _pages_job(self, pages, key_extra, size, dep, gray=False) -> Job:
        """Painted pixels (paint.Pages) in the stack: expanded again when needed, never kept (Job.cheap)."""
        job = self._job(COPY, {"gray": gray}, {}, size, dep, extra={"pages": pages}, key_extra=key_extra,
                        out=_value(gray))
        job.cheap = True
        return job

    def image_job(self, name: str, size, dep: str, gray: bool = False) -> Job:
        """A project resource in the stack: uploaded only when its job is cooked (Copy.encode). `gray`: kept as one
        channel (a grayscale mesh map: a quarter of the memory)."""
        images = self.resources.images
        return self._job(COPY, {"gray": gray}, {}, size, dep, extra={"image": lambda: images[name]},
                         key_extra=("res", name, self.resources.stamps[name]),
                         out=PortType.GRAY if gray else PortType.COLOR)

    def constant(self, rgba, size, dep: str) -> Job:
        """A uniform value: a 1x1 job (the nodes reading it resample, a full size texture would only cost
        memory: 128 MB at 4096 in half float)."""
        return self._job(FILL, {"color": [*rgba[:3], 1.0], "alpha": rgba[3]}, {}, (1, 1), dep)

    # --- the stack ----------------------------------------------------------------------------------

    def _memoized(self, key, make, layer: Layer | None = None):
        """`make()` once per planning (`memo`). While a partial stack is compiled (`channel_at`, `reaching`: `_stop`)
        only a layer's mask the stop is not in (its layer or one of its mask's effects) is the same."""
        if self.memo is None:
            return make()
        if self._stop is not None and (layer is None or self._stop == layer.uid or any(
                e.uid == self._stop for e in (layer.mask.effects if layer.mask else []))):
            return make()
        if key not in self.memo:
            self.memo[key] = make()
        return self.memo[key]

    def channel(self, ts: TextureSet, channel: str) -> Job:
        """The Texture Set's channel with every layer applied."""
        if self._depth == 0:
            return self._memoized(("channel", ts.name, channel), lambda: self._channel(ts, channel))
        return self._channel(ts, channel)

    def _channel(self, ts: TextureSet, channel: str) -> Job:
        fmt = ts.format_of(channel)
        size, dep = self._size(ts), depth(fmt)
        base = self.constant(default_rgba(channel), size, dep)
        if self._depth == 0:
            self._reached = {}
        self._depth += 1
        try:
            return self._layers(ts, ts.layers, channel, base, fmt)
        finally:
            self._depth -= 1

    def _layers(self, ts, layers, channel, below: Job, fmt) -> Job:
        record = self.memo is not None and self._stop is None
        for layer in layers:
            if record:  # what `channel_at` would compile again for this layer
                self.memo[("below", ts.name, channel, layer.uid)] = below
            below = self._apply(ts, layer, channel, below, fmt)
        return below

    def _apply(self, ts: TextureSet, layer: Layer, channel: str, below: Job, fmt: str) -> Job:
        if layer.uid == self._stop:  # `channel_at` a layer: the stack below it
            raise _Reached(below)
        if not layer.visible or layer.excludes_tile(self.udim):  # Geometry mask UV Tiles: the tile skipped whole
            return below
        b = layer.blend_of(channel)
        if b.mode == "disable":
            return below
        if b.mode == "passthrough":  # the stack below flattened into the layer: its content and effects over it
            top = self.content(ts, layer, channel, fmt, below=below)
            if top is None or top is below:
                return below
            mask = self.layer_mask(ts, layer)
            return self._checkpoint(self._job(
                BLEND, {"mode": 0, "opacity": b.opacity, "linear": CHANNELS[channel].color,
                        "clamp01": not is_hdr(fmt), "normal_": channel == "normal"},
                {"below": (below, "output"), "top": (top, "output"), "mask": (mask, "output") if mask else None},
                self._size(ts), depth(fmt), path=f"{layer.name}/{channel}", out=_value(FORMATS[fmt][0])))
        top = self.content(ts, layer, channel, fmt)
        if top is None:
            return below
        mask = self.layer_mask(ts, layer)
        values = {"mode": blending.INDEX[b.mode], "opacity": b.opacity, "linear": CHANNELS[channel].color,
                  "clamp01": not is_hdr(fmt), "normal_": channel == "normal"}
        return self._checkpoint(self._job(BLEND, values, {"below": (below, "output"), "top": (top, "output"),
                                                          "mask": (mask, "output") if mask else None},
                                          self._size(ts), depth(fmt), path=f"{layer.name}/{channel}",
                                          out=_value(FORMATS[fmt][0])))

    def content(self, ts: TextureSet, layer: Layer, channel: str, fmt: str, below: Job | None = None) -> Job | None:
        """A layer's own pixels for a channel (None: it has none): its painting or fill, then its effects.
        `below`: Passthrough, the content starts from the stack below."""
        size, dep, gray = self._size(ts), depth(fmt), FORMATS[fmt][0]
        base = None
        if layer.kind == "paint":
            e = self.store.get(layer.uid, channel, self.udim)
            if e is not None:
                base = self._pages_job(e.pages, ("paint", layer.uid, channel, e.revision), size, dep, gray)
            base = self._paths(ts, layer.uid, layer.paths, channel, base, size, dep, gray)
        elif layer.kind == "fill":
            src = layer.material.get(channel)
            if src is not None:
                base = self._fill(ts, src, layer.projection, gray, dep, self._material(ts, layer.resource, channel),
                                  signed=channel == "height", owner=layer.uid, key=channel)
        elif layer.kind == "folder":
            if not layer.children:
                return None
            empty = self.constant(default_rgba(channel), size, dep)
            return self._layers(ts, layer.children, channel, empty, fmt)
        if below is not None:  # Passthrough: the layer's own content over the stack below, then the effects
            if base is not None:
                base = self._job(BLEND, {"mode": 0, "opacity": 1.0, "linear": CHANNELS[channel].color,
                                         "clamp01": not is_hdr(fmt), "normal_": channel == "normal"},
                                 {"below": (below, "output"), "top": (base, "output")}, size, dep, out=_value(gray))
            else:
                base = below
        return self._effects(ts, layer.effects, channel, base, default_rgba(channel), dep, gray, not is_hdr(fmt),
                             CHANNELS[channel].color, channel == "normal", layer.uid)

    def _paths(self, ts, owner_uid, paths, key, base, size, dep, gray):
        """The paths of a painting (a paint layer's content, a mask, a paint effect: `owner_uid` its PaintStore
        uid, `key` the channel or "mask") over it, in order: Paint / Filled / Ribbon laid over, Erase removing
        coverage, Smudge replacing (its pixels are the dragged painting)."""
        if not paths or self.path_pixels is None:
            return base
        for path in paths:
            if not path.visible or len(path.points) < 2 or key in path.disabled:
                continue
            got = self.path_pixels(ts, owner_uid, path, key, self.udim)
            if got is None:
                continue
            pages, rev = got
            top = self._pages_job(pages, ("path", path.uid, key, rev), size, dep, gray)
            empty = (0.0, 0.0, 0.0, 0.0) if key == MASK_KEY else default_rgba(key)[:3] + (0.0,)
            below = base if base is not None else self.constant(empty, size, dep)
            if path.mode == "Erase along path":
                base = self._job(ERASE, {}, {"below": (below, "output"), "top": (top, "output")}, size, dep,
                                 out=_value(gray))
            elif path.mode == "Smudge along path":
                base = top
            else:
                color = key != MASK_KEY and CHANNELS[key].color
                base = self._job(BLEND, {"mode": 0, "opacity": 1.0, "linear": color, "clamp01": True,
                                         "normal_": key == "normal"},
                                 {"below": (below, "output"), "top": (top, "output")}, size, dep, out=_value(gray))
        return base

    def _material(self, ts, ref, channel):
        """(job, output, flip green) of a Material mode resource for a channel, or None."""
        if not ref or self.materials is None:
            return None
        try:
            key, textures = self.materials.textures(ref, self._size(ts), self.ev)
        except Exception as e:  # noqa: BLE001 - a missing or broken material: the channel's own source, logged
            self.log(f"Material {Path(ref.get('package', '')).name}#{ref.get('graph')}: {e}")
            return None
        if channel not in textures:
            return None
        from trafilatura3d.paint.materials import normal_is_directx
        flip = channel == "normal" and normal_is_directx() != self.project.directx
        texture, gray = textures[channel]
        return texture, gray, (key, channel), flip

    def _fill(self, ts, src, pr, gray, dep, material=None, signed=False, owner="", key="") -> Job:
        """A fill's channel: the Material mode resource's output if it has one for the channel, else the
        channel's image or uniform value, through the UV projection. `signed`: the Height channel, whose image
        sources are remapped from [0, 1] to [-1, 1] (Height map painting page: mid grey = 0). Fill (match per UV
        Tile): each tile takes its image of the sequence as it is (no deformation; a tile without one takes the
        uniform value, ours: the page does not say), every other projection the sequence's first image. An anchor
        point (referenced from the layer `owner`, in the channel or mask `key`) is the stack where it is, its
        coverage kept: nothing where the anchor holds nothing."""
        match = pr.mode == MATCH_PER_TILE
        per_tile = self.udim if match else None
        values = {"color": [*src.color[:3], 1.0], "alpha": 1.0, "tiling": 1.0 if match else pr.tiling,
                  "rotation": 0.0 if match else pr.rotation, "offset": [0.0, 0.0] if match else list(pr.offset),
                  "wrap": UV_WRAP.index(pr.wrap), "filtering": FILTERING.index(pr.filtering), "gray": gray,
                  "signed_": signed}
        extra, geo_key = self._projection(ts, pr, values)
        image = None
        if material is not None:  # the material's output texture, bound by Fill.encode
            texture, image_gray, key, flip = material
            values["flip_green"] = flip
            return self._job(FILL, values, {}, self._size(ts), dep,
                             extra={**extra, "image_texture": (texture, image_gray)},
                             key_extra=("material", key, geo_key), out=_value(gray))
        if src.anchor:
            image = self.reference(ts, src, owner, key)
            if image is None:
                return self._job(FILL, {**values, "mode": 0, "alpha": 0.0}, {}, (1, 1), dep, out=_value(gray))
            values["signed_"] = False  # the stack's height is signed already
        elif src.procedural:
            from trafilatura3d.paint import graph_nodes as dn
            node = dn.node_of(src.procedural)
            if node is not None:
                image = dn.job(self, node, src.values, {}, self._size(ts), "16f")
            else:
                self.log(f"Unknown procedural {src.procedural!r}")
        elif src.image and self.resources.image_of(src.image, per_tile) is not None:
            name = self.resources.image_of(src.image, per_tile)
            h, w, c = self.resources.shape(name)
            # matched per tile the image covers the tile as it is: read at the tile's size; else at its own (tiled,
            # rotated: its texels matter); half floats as the resources are; one channel for a gray image
            image = self.image_job(name, self._size(ts) if match else (w, h), "16f", gray=c == 1)
        if image is None:  # a uniform value: 1x1, every projection gives the same
            return self._job(FILL, {**values, "mode": 0}, {}, (1, 1), dep, out=_value(gray))
        return self._job(FILL, values, {"image": (image, "output")}, self._size(ts), dep, extra=extra,
                         key_extra=geo_key, out=_value(gray))

    def _projection(self, ts, pr, values: dict) -> tuple[dict, tuple | None]:
        """The 3D projections' values (into `values`), the textures of the mesh in texture space they read
        (Fill.encode binds them) and the key part for them. Without the mesh (no `geometry`) a 3D projection
        falls back to the UV projection."""
        mode = 0 if pr.mode == MATCH_PER_TILE else PROJECTIONS.index(pr.mode)
        if mode == 0 or self.geometry is None:
            values["mode"] = 0
            return {}, None
        maps, lo, hi = self.geometry(ts, self.udim)
        unit = float(np.max(hi - lo)) or 1.0
        rx, ry, rz = np.radians(pr.rotation3d)
        cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
        rot = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
               @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
        origin = (lo + hi) / 2 + np.asarray(pr.offset3d) * unit
        m = np.diag(1.0 / (np.maximum(np.abs(pr.scale3d), 1e-6) * unit)) @ rot.T  # world -> projection space
        for i in range(3):
            values[f"m{i}"] = [*m[i], float(-m[i] @ origin)]
        values.update({"mode": mode, "crop": pr.shape_crop == SHAPE_CROP[0], "hardness": pr.hardness,
                       "depth_culling": pr.depth_culling, "depth_hardness": pr.depth_hardness,
                       "backface_culling": pr.backface_culling,
                       "backface_cos": float(np.cos(np.radians(pr.backface_angle))),
                       "backface_hardness": pr.backface_hardness, "angle": float(np.radians(max(pr.angle, 1e-3))),
                       "warp_depth": pr.warp_depth})
        extra = {"position_texture": (maps.position, False), "normal_texture": (maps.normal, False)}
        key = ("geometry", ts.name, list(self._size(ts)))
        if pr.mode == "Warp projection":
            from trafilatura3d.paint import warp
            wkey = _key(pr.warp_rows, pr.warp_cols, pr.warp_points, pr.warp_tangents)
            surf = self.ev.upload(f"paint-warp:{wkey}", lambda: np.concatenate(
                [s := warp.surface(pr), np.ones(s.shape[:2] + (1,), np.float32)], 2))
            extra["warp_texture"] = (surf, False)
            key = (*key, wkey)
        if pr.symmetry:  # the copies of the fill (Symmetry page: fill layers with a 3D projection)
            from trafilatura3d.paint.brush import symmetry_copies
            s = pr.symmetry
            copies = symmetry_copies(s["type"], s["axis"], s["position"], s.get("copies", 2), s.get("flip", "None"))
            rows = []
            for m, t, _reflect, flip in copies:
                mt = np.asarray(m).T
                rows += [[*mt[i], float(-(mt @ t)[i])] for i in range(3)] + [[float(flip), 0.0, 0.0, 0.0]]
            skey = _key(s)
            data = np.asarray(rows, np.float32).reshape(1, -1, 4)
            extra["sym_texture"] = (self.ev.upload(f"paint-sym:{skey}", lambda: data), False)
            key = (*key, skey)
        return extra, key

    def _effects(self, ts, effects, key, cur, empty_rgba, dep, gray, clamp01, linear, normal, owner) -> Job | None:
        """An effect stack (bottom first) of the layer `owner` over `cur` (None: nothing yet). `key` is the
        channel, or "mask". An anchor point changes nothing: `anchor` stops here to read `cur`."""
        out = PortType.GRAY if key == MASK_KEY else _value(gray)
        for fx in effects:
            if fx.uid == self._stop:
                raise _Reached(cur)
            if not fx.visible or fx.kind == "anchor":
                continue
            if fx.kind == "levels":
                if cur is None or (key != MASK_KEY and fx.levels.channel != key):
                    continue
                lv = fx.levels
                cur = self._checkpoint(self._job(
                    LEVELS, {"in_low": lv.in_low, "in_mid": lv.in_mid, "in_high": lv.in_high, "out_low": lv.out_low,
                             "out_high": lv.out_high, "component": LEVELS_COLORS.index(lv.colors),
                             "clamp01": lv.clamp},
                    {"image": (cur, "output")}, self._size(ts), dep, out=out))
                continue
            if fx.kind in ("filter", "generator"):
                cur = self._checkpoint(self._procedural(ts, fx, key, cur, empty_rgba, dep, clamp01, linear, normal,
                                                        out, owner))
                continue
            b = fx.blend_of(key)
            if b.mode in ("disable", "passthrough"):
                continue
            if fx.kind == "fill":
                src = fx.material.get(key)
                top = None if src is None else self._fill(ts, src, fx.projection, gray, dep,
                                                          self._material(ts, fx.resource, key), signed=key == "height",
                                                          owner=owner, key=key)
            else:  # paint, and its paths
                e = self.store.get(fx.uid, key, self.udim)
                top = None if e is None else self._pages_job(e.pages, ("paint", fx.uid, key, e.revision),
                                                                self._size(ts), dep, gray)
                top = self._paths(ts, fx.uid, fx.paths, key, top, self._size(ts), dep, gray)
            if top is None:
                continue
            below = cur if cur is not None else self.constant(empty_rgba, self._size(ts), dep)
            cur = self._checkpoint(self._job(BLEND, {"mode": blending.INDEX[b.mode], "opacity": b.opacity,
                                                     "linear": linear, "clamp01": clamp01, "normal_": normal},
                                             {"below": (below, "output"), "top": (top, "output")},
                                             self._size(ts), dep, out=out))
        return cur

    def _procedural(self, ts, fx, key, cur, empty_rgba, dep, clamp01, linear, normal, kind, owner) -> Job | None:
        """A library node (paint/graph_nodes.py) as a filter (its main input takes the stack below, the
        coverage is kept) or as a generator (its mesh map inputs take the Texture Set's baked maps), blended with
        the effect's mode and opacity."""
        from trafilatura3d.paint import graph_nodes as dn
        b = fx.blend_of(key)
        node = dn.node_of(fx.procedural)
        if b.mode in ("disable", "passthrough") or node is None:
            return cur
        output = dn.output_for(node, key)
        if output is None:  # a material filter that does not change this channel
            return cur
        if fx.kind == "filter" and cur is None:
            return None
        blend = {"mode": blending.INDEX[b.mode], "opacity": b.opacity, "linear": linear, "clamp01": clamp01,
                 "normal_": normal}
        inputs = {}
        for port in node.inputs:
            what, arg = dn.role(node, port, fx.kind)
            if what == "stack":
                src = cur
            elif what == "channel":  # in a mask there is no other channel to read
                src = cur if arg == key else None if key == MASK_KEY else self.channel_at(ts, arg, fx.uid)
                src = None if src is None else self.to_material(ts, src, arg if key != MASK_KEY else "", False)[0]
            elif what == "mesh":
                src = self.mesh_map(ts, arg)
                if src is None:
                    self.log(f"{node.label}: the {arg.replace('_', ' ')} mesh map is missing "
                             "(bake it in the Baking mode)")
            else:
                s = fx.inputs.get(port)
                src = None if s is None else self._source(ts, s, owner)
            inputs[port] = (src, "output") if src is not None else None
        out = dn.job(self, node, fx.values, inputs, self._size(ts), self._material_depths(ts, node) or dep)
        top = (out, output)
        if output in dn.CHANNEL_PORTS:  # a material output: back to the Paint convention
            top = self.to_material(ts, out, key, True, output)
        if fx.kind == "filter":
            return self._job(BLEND, {**blend, "keep_alpha": True}, {"below": (cur, "output"), "top": top},
                             self._size(ts), dep, out=kind)
        below = cur if cur is not None else self.constant(empty_rgba, self._size(ts), dep)
        return self._job(BLEND, blend, {"below": (below, "output"), "top": top}, self._size(ts), dep, out=kind)

    @staticmethod
    def _material_depths(ts, node) -> dict | None:
        """A material filter (channel outputs) is one job for all the channels: each output at the depth of the
        channel taking it, the deepest for "output" (the channels without their own port); None for other nodes
        (their job is per channel)."""
        from trafilatura3d.paint import graph_nodes as dn
        if not any(o in dn.CHANNEL_PORTS for o in node.outputs):
            return None
        rank = ["8", "16f", "16", "32f"]  # by precision ("16" is stored as 32 bit floats)
        taking = {o: [] for o in node.outputs}
        for ch in ts.channel_ids():
            o = dn.output_for(node, ch)
            if o is not None:
                taking[o].append(depth(ts.format_of(ch)))
        return {o: max(d, key=rank.index) if d else "8" for o, d in taking.items()}

    def to_material(self, ts, job: Job, channel: str, back: bool, output: str = "output") -> tuple[Job, str]:
        """(job, output) of a channel's value in the Graph material convention (`back`: from it): the height is
        signed in Paint and centered on 0.5 in Graph; normals are OpenGL in Graph, the project's format in
        Paint."""
        flip = channel == "normal" and self.project.directx
        if channel != "height" and not flip:
            return job, output
        scale, offset = ((2.0, -1.0) if back else (0.5, 0.5)) if channel == "height" else (1.0, 0.0)
        return self._job(CONVERT, {"scale": scale, "offset": offset, "flip_green": flip}, {"image": (job, output)},
                         self._size(ts), job.depths[output], out=_value(job.is_gray(output))), "output"

    def _source(self, ts, src, owner) -> Job | None:
        """An effect's resource slot (ChannelSource) of the layer `owner`: an anchor point (in the Graph material
        convention, as the channels a filter reads), a procedural, a project texture, or a uniform value."""
        if src.anchor:
            job = self.reference(ts, src, owner, "")
            return None if job is None else self.to_material(ts, job, src.anchor_channel, False)[0]
        if src.procedural:
            from trafilatura3d.paint import graph_nodes as dn
            node = dn.node_of(src.procedural)
            return dn.job(self, node, src.values, {}, self._size(ts), "16f") if node is not None else None
        name = self.resources.image_of(src.image) if src.image else None
        if name is not None:
            return self.image_job(name, self.resources.shape(name)[1::-1], "16f")  # the resources are half floats
        return self.constant([*src.color[:3], 1.0], self._size(ts), "16f")

    def reference(self, ts, src, owner: str, key: str) -> Job | None:
        """The anchor point a channel source references, read from the layer `owner` in the channel or mask `key`
        (Anchor Point page: only the anchors under the layer; one moved above breaks the reference, logged): what
        its stack holds, None when nothing."""
        if not any(e.uid == src.anchor for _l, e, _m in ts.anchors(below=owner)):
            fx = ts.owner_of(src.anchor)[1]
            self.log(f"Broken reference to {fx.name if fx else 'a deleted anchor point'}: an anchor point must be "
                     "under the layer using it")
            return None
        return self.anchor(ts, src.anchor, src.anchor_channel or key)

    def anchor(self, ts, uid: str, channel: str) -> Job | None:
        """What an anchor point holds (live: its job is keyed by everything under it): in a layer's content, that
        layer's `channel` up to the anchor (a Passthrough layer's content starts from the stack below); in a
        mask, the mask up to it. None when the stack holds nothing there."""
        fx = ts.owner_of(uid)[1]
        return self.reaching(ts, uid, channel) if fx is not None and fx.kind == "anchor" else None

    def reaching(self, ts, uid: str, channel: str) -> Job | None:
        """What reaches the effect `uid` (what it filters, what an anchor point holds): in a layer's content, that
        layer's `channel` up to the effect; in a mask, the mask up to it. None when nothing is there."""
        layer, fx, in_mask = ts.owner_of(uid)
        if fx is None or (not in_mask and channel not in ts.channel_ids()):
            return None
        memo = ("reaching", uid, MASK_KEY if in_mask else channel, ts.name)
        if self._depth == 0:
            self._reached = {}
        if memo in self._reached:
            return self._reached[memo]
        self._depth += 1
        saved, self._stop = self._stop, uid
        try:
            if in_mask:
                self.mask(ts, layer)
            else:
                fmt = ts.format_of(channel)
                below = self.channel_at(ts, channel, layer.uid) if layer.blend_of(channel).mode == "passthrough" \
                    else None
                self.content(ts, layer, channel, fmt, below=below)
            found = None
        except _Reached as r:
            found = r.job
        finally:
            self._stop = saved
            self._depth -= 1
        self._reached[memo] = found
        return found

    def channel_at(self, ts, channel: str, effect_uid: str) -> Job | None:
        """The stack below of `channel` where the effect or layer `effect_uid` is (what that effect would filter
        in that channel; what a Passthrough layer is laid over), or None when the channel does not reach it (not in
        the Texture Set, the layer off for it)."""
        if channel not in ts.channel_ids():
            return None
        if self.memo is not None and self._stop is None:  # the channel compiled whole once, its layers' below kept
            self.channel(ts, channel)
            known = self.memo.get(("below", ts.name, channel, effect_uid))
            if known is not None:
                return known
        memo = (channel, effect_uid, ts.name)
        if self._depth == 0:  # reached outside `channel` (a mask compiled alone): nothing to reuse
            self._reached = {}
        if memo in self._reached:
            return self._reached[memo]
        saved, self._stop = self._stop, effect_uid
        try:
            self.channel(ts, channel)
            found = None
        except _Reached as r:
            found = r.job
        finally:
            self._stop = saved
        self._reached[memo] = found
        return found

    def layer_mask(self, ts: TextureSet, layer: Layer) -> Job | None:
        """What limits a layer (or folder): its mask times its Geometry mask (the excluded meshes at 0); None when
        it has neither."""
        return self._memoized(("layer_mask", ts.name, layer.uid), lambda: self._layer_mask(ts, layer), layer)

    def _layer_mask(self, ts: TextureSet, layer: Layer) -> Job | None:
        mask = self.mask(ts, layer)
        excluded = layer.excluded_meshes()
        if not excluded or self.coverage is None:
            return mask
        tex = self.coverage(ts, excluded, self.udim)
        geo = self._texture_job(tex, ("geometry", ts.name, sorted(excluded)), self._size(ts), MASK_DEPTH,
                                True)
        if mask is None:
            return geo
        return self._checkpoint(self._job(BLEND, {"mode": blending.INDEX["multiply"], "opacity": 1.0},
                                          {"below": (mask, "output"), "top": (geo, "output")}, self._size(ts),
                                          MASK_RESULT_DEPTH, out=PortType.GRAY))

    def mask(self, ts: TextureSet, layer: Layer) -> Job | None:
        """The layer's mask (one channel), or None when it has none or it is turned off: the background, what is
        painted on it, then its effects. Every result of the chain is one channel (a quarter of an RGBA texture,
        and the filters on it take their grayscale variant); what is laid over it keeps its coverage in alpha."""
        return self._memoized(("mask", ts.name, layer.uid), lambda: self._mask(ts, layer), layer)

    def _mask(self, ts: TextureSet, layer: Layer) -> Job | None:
        m = layer.mask
        if m is None or not m.enabled:
            return None
        cur = self._job(FILL, {"color": [m.background] * 3 + [1.0], "alpha": 1.0}, {}, (1, 1), MASK_DEPTH,
                        out=PortType.GRAY)
        e = self.store.get(layer.uid, MASK, self.udim)
        painted = None
        if e is not None:
            painted = self._pages_job(e.pages, ("mask", layer.uid, e.revision), self._size(ts), MASK_DEPTH,
                                        True)
        painted = self._paths(ts, layer.uid, m.paths, MASK_KEY, painted, self._size(ts), MASK_DEPTH, True)
        if painted is not None:
            cur = self._job(BLEND, {"mode": 0, "opacity": 1.0}, {"below": (cur, "output"), "top": (painted, "output")},
                            self._size(ts), MASK_DEPTH, out=PortType.GRAY)
        return self._checkpoint(_stored(self._effects(ts, m.effects, MASK_KEY, cur, (0, 0, 0, 1), MASK_DEPTH, True,
                                                      True, False, False, layer.uid), MASK_RESULT_DEPTH))

    # --- what the viewport and the export read --------------------------------------------------------

    def mesh_map(self, ts: TextureSet, slot: str) -> Job | None:
        """A mesh map slot: the baked or imported map, or the map made from the mesh (uvmaps.COMPUTED)."""
        from trafilatura3d.paint.uvmaps import COMPUTED
        gray = slot in GRAY_MESH_MAPS
        if slot in COMPUTED:
            if self.computed is None:
                return None
            return self._checkpoint(self._job(COPY, {"gray": gray}, {}, self._size(ts), "32f",
                                              extra={"texture": self.computed(ts, slot, self.udim)},
                                              key_extra=("computed", ts.name, list(self._size(ts)), slot),
                                              out=PortType.GRAY if gray else PortType.COLOR))
        name = ts.mesh_map(slot, self.udim)
        if not name or name not in self.resources.images:
            return None
        return self._checkpoint(self.image_job(name, self._size(ts), "16f", gray))

    def normal(self, ts: TextureSet, directx_out: bool = False, with_height: bool = True) -> Job:
        """The normal the shader reads: baked normal + Normal channel + Height (Texture Set settings)."""
        ids = ts.channel_ids()
        baked = self.mesh_map(ts, "normal")
        nrm = self.channel(ts, "normal") if "normal" in ids else None
        hgt = self.channel(ts, "height") if "height" in ids and with_height else None
        values = {"mixing": 1 if ts.normal_mixing == "Combine" else 0, "sobel": ts.height_to_normal != "Sharp",
                  "dx_in": self.project.directx, "dx_out": directx_out}
        return self._job(NORMAL, values, {"baked": (baked, "output") if baked else None,
                                          "normal": (nrm, "output") if nrm else None,
                                          "height": (hgt, "output") if hgt else None}, self._size(ts), "16f")

    def ambient_occlusion(self, ts: TextureSet) -> Job | None:
        """Baked AO and the Ambient occlusion channel (Ambient occlusion mixing)."""
        baked = self.mesh_map(ts, "ambient_occlusion")
        ch = self.channel(ts, "ambientocclusion") if "ambientocclusion" in ts.channel_ids() else None
        if baked is None:
            return ch
        if ch is None:
            return baked
        mode = blending.INDEX["multiply" if ts.ao_mixing == "Multiply" else "normal"]
        return self._job(BLEND, {"mode": mode, "opacity": 1.0}, {"below": (baked, "output"), "top": (ch, "output")},
                         self._size(ts), "16f")

    def uv_only(self, ts: TextureSet, job: Job) -> Job:
        """`job` for a viewport: its pointwise steps at the tile's size, from the result down to the first one that is
        not (a filter reading far, which reads its inputs whole), computed only on the pages near the UV islands
        (`_pages_on`: 0 elsewhere). What a viewport shows of them is the islands (padded from inside them, or drawn
        on the mesh); the export, whose No padding keeps the texels outside, reads the jobs whole."""
        on = self._pages_on(ts)
        if on is None:
            return job
        memo = {}

        def rewrite(j):
            if j.key in memo:
                return memo[j.key]
            out = j
            if j.node.pointwise and tuple(j.size) == tuple(self._size(ts)) and j.size != (1, 1):
                ins = {port: (rewrite(src[0]), src[1]) if src else None for port, src in j.inputs.items()}
                key = hashlib.sha1(f"{j.key}|pages{on}".encode()).hexdigest()[:24]
                out = dataclasses.replace(j, key=key, inputs=ins, extra={**j.extra, "pages_on": on})
            memo[j.key] = out
            return out
        return rewrite(job)

    def _pages_on(self, ts: TextureSet):
        """The pages (nodes.base.PAGE_GRID a side) a viewport computes: those within HALO texels of the UV islands
        (the normal reads around a texel, the renderer filters around the islands), as the 8 words of Variant.pages;
        None without a UV mask or when every page is needed."""
        tex = self.uv_masks.get((ts.name, self.udim))
        if tex is None:
            return None
        key = (ts.name, id(tex))
        if self._pages.get("key") != key:
            row = (tex.width * 4 + 255) // 256 * 256
            data = self.ev.device.queue.read_texture({"texture": tex}, {"bytes_per_row": row}, (tex.width, tex.height, 1))
            m = np.frombuffer(data, np.uint8).reshape(tex.height, row)[:, :tex.width * 4:4] > 127
            g = PAGE_GRID
            if tex.width % g or tex.height % g:
                self._pages = {"key": key, "on": None}
                return None
            near = m.copy()  # the islands grown by HALO texels
            for axis in (0, 1):
                grown = near.copy()
                for k in range(1, HALO + 1):
                    grown[(slice(k, None),) if axis == 0 else (slice(None), slice(k, None))] |= \
                        near[(slice(None, -k),) if axis == 0 else (slice(None), slice(None, -k))]
                    grown[(slice(None, -k),) if axis == 0 else (slice(None), slice(None, -k))] |= \
                        near[(slice(k, None),) if axis == 0 else (slice(None), slice(k, None))]
                near = grown
            on = near.reshape(g, tex.height // g, g, tex.width // g).any((1, 3))
            bits = np.packbits(on.reshape(-1), bitorder="little").view(np.uint32)
            words = tuple(int(np.int32(np.uint32(w))) for w in bits)
            self._pages = {"key": key, "on": None if on.all() else words}
        return self._pages["on"]

    def padded(self, ts: TextureSet, job: Job, distance: float = -1.0, outside: int = 0, color=(0, 0, 0)) -> Job:
        """`job` with the texels outside the UV islands filled from the nearest island (Padding)."""
        tex = self.uv_masks.get((ts.name, self.udim))
        if tex is None:
            return job
        mask = self._checkpoint(self._job(COPY, {"gray": True}, {}, self._size(ts), "8", extra={"texture": tex},
                                          key_extra=("uvmask", ts.name, id(tex)), out=PortType.GRAY))
        nearest = self._checkpoint(self._job(NEAREST, {}, {"mask": (mask, "output")}, self._size(ts), "32f",
                                             out=PortType.GRAY_ALPHA))  # a coordinate: two channels
        return self._job(DILATE, {"distance": float(distance), "outside": outside, "color": [*color[:3], 0.0]},
                         {"image": (job, "output"), "mask": (mask, "output"), "seeds": (nearest, "output")},
                         self._size(ts), job.depths["output"], out=job.output_types["output"])
