"""An open Paint project: the document, its mesh (the arrays the viewport and the brush use), the painted
pixels, the resources, the UV maps of each Texture Set and the stack compiler.

The project file (`.trafilatura3dpaint`) is a zip: project.json, mesh.npz (the project mesh as imported, so it
opens without the source file), paint/<uid>__<key>__<udim>__<storage>.npz (the painted pages only, as the
GPU keeps them: `paint.Pages.raw`) and resources/<name>.npz (packed as kept in memory: `stack.Packed`).
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from trafilatura3d import meshio, shaders
from trafilatura3d.evaluator import Evaluator, one_channel
from trafilatura3d.paint.model import (FIRST_TILE, MESH_MAPS, PROJECT_SUFFIX, UV_TILE_MODES, Project, ProjectError,
                                   drop_unknown_nodes, drop_unknown_shader, new_texture_set)
from trafilatura3d.paint.paint import MASK, PaintStore, storage_for
from trafilatura3d.paint.stack import Packed, Resources, StackCompiler
from trafilatura3d.paint.uvmaps import UVRasterizer

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tga *.tif *.tiff *.exr *.bmp *.webp *.hdr *.psd)"
_PATH_REVISIONS = itertools.count(1)  # the paths' pixels revisions, unique in the process
UDIM_FILE = re.compile(r"^(.+?)([._-])(\d{4})$")  # an image sequence's file name (its stem): base, separator, UDIM


class Session:
    def __init__(self, project: Project, objects: list, materials: list[str], evaluator: Evaluator | None = None):
        self.project = project
        self.objects = objects  # meshio.MeshObject, fitted to the viewport (meshio.fit)
        self.materials = materials  # mesh material names, by submesh material index
        self.ev = evaluator or Evaluator()
        self.store = PaintStore(self.ev.device)
        self.resources = Resources()
        self.path: Path | None = None
        self.source = ""  # the mesh file the project was made from
        self.load_warnings: list[str] = []  # what opening the file had to leave out (drop_unknown_nodes)
        self._uv = UVRasterizer(self.ev.device)
        self._maps: dict = {}
        self._computed: dict = {}
        self._triangles: dict = {}
        self._coverage: dict = {}
        self._paths: dict = {}  # (path uid, tile) -> (key, {channel: texture}, revision)
        self._brush = None
        self.fit = (np.zeros(3), 1.0)  # the import's transform into the viewport (meshio.fit): high poly meshes follow
        self.uv_masks: dict = {}
        from trafilatura3d.paint.materials import Materials
        self.material_library = Materials()
        self.log_lines: list[str] = []  # what the stack could not do (missing materials…), for the Log
        self.warnings: list[str] = []  # what the mesh's import found (`_check_tiles`), for the Log
        self._compilers: dict[tuple, StackCompiler] = {}
        self._bounds = None  # (objects, min, max): `bounds`
        self._planning = 0  # `planning` depth

    @contextlib.contextmanager
    def planning(self):
        """Within it the project does not change: the compilers compile a channel or a mask once (StackCompiler.memo),
        whoever asks for it."""
        self._planning += 1
        try:
            for sc in self._compilers.values():
                if sc.memo is None:
                    sc.memo = {}
            yield
        finally:
            self._planning -= 1
            if not self._planning:
                for sc in self._compilers.values():
                    sc.memo = None

    def compiler(self, udim: int, limit: int | None = None) -> StackCompiler:
        """The stack compiler of a UDIM tile (every tile is computed on its own texture), at the tile's resolution
        or at most `limit` on a side (the viewports)."""
        key = (udim, limit)
        if key not in self._compilers:
            self._compilers[key] = StackCompiler(
                self.project, self.store, self.resources, self.ev, udim, self.uv_masks, self.material_library,
                self._log, self.computed_map, self.geometry, self.path_pixels, self.geometry_coverage, limit)
            if self._planning:
                self._compilers[key].memo = {}
        return self._compilers[key]

    def _log(self, text: str) -> None:
        if not self.log_lines or self.log_lines[-1] != text:
            self.log_lines.append(text)

    # --- creation -------------------------------------------------------------------------------------

    @classmethod
    def new(cls, mesh_path, resolution: int = 2048, normal_format: str | None = None, baked_maps=(),
            evaluator: Evaluator | None = None, uv_tiles: str = "", template: str = shaders.DEFAULT_TEMPLATE) -> Session:
        """New Project: one Texture Set per material of the mesh, with the channels of the template
        (shaders.TEMPLATES), whose shader the project takes; baked maps named TextureSetName_suffix are
        assigned to their Texture Set's mesh map slots (New Project > Import baked maps). `uv_tiles`: the UV Tile
        mode (Use UV Tile workflow): each Texture Set keeps the UDIM tiles of its material."""
        from trafilatura3d.bake import udim, udim_tiles
        if uv_tiles == UV_TILE_MODES[1]:
            raise ProjectError("the legacy UV Tile mode (UV Tiles into Texture Sets) is not offered")
        objects, materials, _cameras = meshio.load(mesh_path)
        fit = meshio.fit(objects)
        used = sorted({s.material for o in objects for s in o.submeshes})
        project = Project(mesh=Path(mesh_path).name, resolution=resolution, uv_tiles=uv_tiles)
        if template not in shaders.TEMPLATES:
            raise ProjectError(f"unknown template {template!r}")
        project.shader.shader = template
        if normal_format:
            project.normal_format = normal_format
        for m in used:
            ts = new_texture_set(materials[m], resolution, template)
            if uv_tiles:
                ts.tiles = [udim(t) for t in udim_tiles(objects, m)]
            project.texture_sets.append(ts)
        project.validate()
        s = cls(project, objects, materials, evaluator)
        s.source = str(mesh_path)
        s.fit = fit
        s._check_tiles()
        for path in baked_maps:
            s.import_baked_map(path)
        return s

    def import_baked_map(self, path) -> str | None:
        """Assign an image named TextureSetName_<suffix> to that Texture Set's mesh map slot (TextureSetName_<suffix>
        .<UDIM> to that tile's); returns the slot."""
        stem = Path(path).stem
        base, _dot, tile = stem.rpartition(".")
        udim = int(tile) if base and tile.isdigit() else FIRST_TILE
        name = base if base and tile.isdigit() else stem
        for ts in self.project.texture_sets:
            for slot, _label, suffix in MESH_MAPS:
                if name == f"{ts.name}_{suffix}" and udim in ts.tiles:
                    ts.set_mesh_map(slot, udim, self._import_one(Path(path), self.resources.unique(stem)))
                    return slot
        return None

    def reimport(self, mesh_path) -> list[str]:
        """Project configuration > Reimport mesh: the new mesh replaces the old one, the rest of the project is kept.
        Texture Sets follow their material by name; a new material gets a new Texture Set (returned). Painted
        pixels stay in texture space (Preserve strokes positions on mesh is not offered). A Texture Set whose
        material is gone refuses the reimport (the Texture Set Reassignment window is not offered)."""
        objects, materials, _cameras = meshio.load(mesh_path)
        fit = meshio.fit(objects)
        used = [materials[m] for m in sorted({s.material for o in objects for s in o.submeshes})]
        gone = [ts.name for ts in self.project.texture_sets if ts.material not in used]
        if gone:
            raise ProjectError(f"the new mesh has no material for the Texture Sets {', '.join(gone)}")
        have = {ts.material for ts in self.project.texture_sets}
        added = [m for m in used if m not in have]
        for m in added:
            self.project.texture_sets.append(new_texture_set(m, self.project.resolution, self.project.shader.shader))
        if self.project.uv_tiles:  # the UDIM tiles follow the new UVs
            from trafilatura3d.bake import udim, udim_tiles
            for ts in self.project.texture_sets:
                ts.tiles = [udim(t) for t in udim_tiles(objects, materials.index(ts.material))]
        self.project.validate()
        self.objects, self.materials, self.fit = objects, materials, fit
        self.source = str(mesh_path)
        self.project.mesh = Path(mesh_path).name
        self._maps, self._computed = {}, {}
        self.uv_masks.clear()
        self._brush = None
        self._check_tiles()
        return added

    def _check_tiles(self) -> None:
        """With the UV Tile workflow, the meshes whose triangles cross a tile's border, for the Log: those triangles
        show only the texture of the tile of their center (`bake.spanning_triangles`)."""
        if not self.project.uv_tiles:
            return
        from trafilatura3d.bake import spanning_triangles
        for name, count, tiles in spanning_triangles(self.objects):
            self.warnings.append(f"Mesh {name}: {count} triangle(s) span several UV tiles "
                                 f"({', '.join(map(str, tiles))}); each shows only the tile of its center")

    def mesh_in_file_space(self) -> list:
        """The project's meshes in their file's coordinates (the viewport fit undone)."""
        import dataclasses
        center, k = self.fit
        return [dataclasses.replace(o, pos=(o.pos / k + center).astype(np.float32), visible=True)
                for o in self.objects]

    # --- mesh and UV maps -------------------------------------------------------------------------------

    def material_index(self, ts) -> int:
        return self.materials.index(ts.material)

    def uv_mask(self, ts, udim: int):
        """A tile's UV coverage (what the padding reads, in `uv_masks`), made again when its size changes."""
        t = self.uv_masks.get((ts.name, udim))
        if t is None or [t.width, t.height] != ts.size_of(udim):
            t = self.uv_masks[(ts.name, udim)] = self._uv.mask(self.objects, self.material_index(ts),
                                                                ts.size_of(udim), udim)
        return t

    def uv_maps(self, ts, udim: int):
        """The Texture Set's mesh in a tile's texture space (uvmaps.UVMaps: the brush, the paths, the 3D
        projections), made on first use and again when its size changes."""
        size = ts.size_of(udim)
        key = (ts.name, udim, tuple(size))
        if key not in self._maps:
            self._maps = {k: v for k, v in self._maps.items() if k[:2] != (ts.name, udim)}
            self._maps[key] = self._uv.maps(self.objects, self.material_index(ts), size, udim, self.uv_mask(ts, udim))
        return self._maps[key]

    def path_pixels(self, ts, owner_uid, path, channel, udim: int):
        """(paint.Pages, revision) of a path's pixels for a channel or "mask" (paint/paths.py) of the painting
        `owner_uid` (a paint layer, a layer's mask, a paint effect), made again only when what decides them changes:
        the path's data, the Texture Set's size and formats, the mesh, and for Smudge along path the painting."""
        from trafilatura3d.paint import paths
        from trafilatura3d.paint.paint import MASK, Pages, storage_for
        chans = [MASK] if channel == MASK else [c for c in ts.channel_ids() if c not in path.disabled]
        if channel not in chans:
            return None
        fmt = {c: (None if c == MASK else ts.format_of(c)) for c in chans}
        deps = [(c, self.store.get(owner_uid, c, udim).revision if self.store.get(owner_uid, c, udim) else None)
                for c in chans] if path.mode == "Smudge along path" else []
        key = json.dumps([path.to_dict(), ts.size_of(udim), fmt, id(self.objects), deps,
                          {c: self.resources.stamps.get((path.sources.get(c) or {}).get("image", "")) for c in chans},
                          self.resources.stamps.get(path.alpha.split(":", 1)[-1])],  # a texture alpha's content
                         sort_keys=True, default=str)
        cached = self._paths.get((path.uid, udim))
        if cached is None or cached[0] != key:
            if cached is not None:
                for pages in cached[1].values():
                    pages.destroy()
            textures = {c: Pages(self.ev.device, ts.size_of(udim), storage_for(fmt[c])) for c in chans}
            paths.render(self, ts, owner_uid, path, textures, udim)
            # a revision unique in the process: sessions share the evaluator (a reopened project's paths have the
            # same uids)
            cached = self._paths[(path.uid, udim)] = (key, textures, next(_PATH_REVISIONS))
        return cached[1][channel], cached[2]

    def alpha_name(self, name: str) -> str:
        """A brush alpha ready in the brush engine: a built-in one as is, a project texture ("texture:<name>") by its
        luminance under a key that changes with the image; an unknown texture falls back to Round soft."""
        from trafilatura3d.paint.brush import TEXTURE_ALPHA
        if not name.startswith(TEXTURE_ALPHA):
            return name
        tex = name[len(TEXTURE_ALPHA):]
        if tex not in self.resources.images:
            return "Round soft"
        key = f"{name}:{self.resources.stamps[tex]}"
        img = self.resources.images[tex]
        gray = img[..., 0] if img.shape[2] == 1 else img[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        self.brush_engine().alpha(key, gray)
        return key

    def source_texture(self, src, ts, udim: int):
        """(GPU texture, single channel) of a tool's or a path's channel resource (a project texture or a
        procedural, paint/model.ChannelSource), or None for a uniform value."""
        if src.procedural:
            from trafilatura3d.paint import graph_nodes as dn
            node = dn.node_of(src.procedural)
            if node is None:
                return None
            job = dn.job(self.compiler(udim), node, src.values, {}, ts.size, "16f")
            self.ev.cook([job])
            return self.ev.texture(job, "output"), job.is_gray("output")
        name = self.resources.image_of(src.image) if src.image else None  # a sequence: its first image
        if name is not None:
            tex = self.ev.upload(f"paint-res:{name}:{self.resources.stamps[name]}", lambda: self.resources.images[name])
            return tex, one_channel(tex)
        return None

    def triangles(self, ts):
        """The Texture Set's triangles for picking and the Polygon Fill modes (uvmaps.Triangles), made once."""
        key = (ts.name, id(self.objects))
        if key not in self._triangles:
            from trafilatura3d.paint.uvmaps import Triangles
            self._triangles = {k: v for k, v in self._triangles.items() if k[1] == id(self.objects)}
            self._triangles[key] = Triangles(self.objects, self.material_index(ts))
        return self._triangles[key]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(min, max) corners of the mesh's bounding box, computed once per mesh (the stack asks for it at every
        fill with a projection it compiles: over a mesh of 376 000 vertices that was most of a refresh)."""
        if self._bounds is None or self._bounds[0] is not self.objects:
            lo = np.min([o.pos.min(0) for o in self.objects], 0)
            hi = np.max([o.pos.max(0) for o in self.objects], 0)
            self._bounds = (self.objects, lo, hi)
        return self._bounds[1], self._bounds[2]

    def geometry(self, ts, udim: int):
        """(uv maps of the tile, scene bounding box min, max) for the 3D projections of the fills."""
        return (self.uv_maps(ts, udim), *self.bounds())

    def mesh_names(self, ts) -> list[str]:
        """The Texture Set's meshes by name (the Geometry mask's Mesh names)."""
        from trafilatura3d.paint.uvmaps import mesh_names
        return mesh_names(self.objects, self.material_index(ts))

    def geometry_coverage(self, ts, excluded, udim: int):
        """The Geometry mask in a tile's texture space (1 on the meshes not excluded), made once per exclusion
        list."""
        key = (ts.name, udim, tuple(ts.size_of(udim)), tuple(sorted(excluded)), id(self.objects))
        if key not in self._coverage:
            if len(self._coverage) > 16:  # a few lists at a time (editing a mask makes many)
                for tex in self._coverage.values():
                    tex.destroy()
                self._coverage = {}
            self._coverage[key] = self._uv.coverage(self.objects, self.material_index(ts), ts.size_of(udim),
                                                    set(excluded), udim)
        return self._coverage[key]

    def computed_map(self, ts, slot: str, udim: int):
        """A mesh map made from the mesh (uvmaps.COMPUTED: UV islands, texel density) in a tile at the Texture
        Set's size, made once."""
        key = (ts.name, udim, tuple(ts.size_of(udim)), slot)
        if key not in self._computed:
            self._computed = {k: v for k, v in self._computed.items() if k[:2] != (ts.name, udim) or k[3] != slot}
            self._computed[key] = self._uv.computed(self.objects, self.material_index(ts), ts.size_of(udim), slot,
                                                    udim)
        return self._computed[key]

    def brush_engine(self):
        if self._brush is None:
            from trafilatura3d.paint.brush import BrushEngine
            self._brush = BrushEngine(self.ev.device)
        return self._brush

    def bbox_diagonal(self) -> float:
        lo, hi = self.bounds()
        return float(np.linalg.norm(hi - lo))

    def used_resources(self) -> set:
        """Images the project uses: mesh maps, fill sources (of layers and of fill effects) and the resource slots
        of filters and generators."""
        used = {n for ts in self.project.texture_sets for tiles in ts.mesh_maps.values() for n in tiles.values()}
        for ts in self.project.texture_sets:
            for layer in ts.walk():
                effects = layer.effects + (layer.mask.effects if layer.mask is not None else [])
                sources = [*layer.material.values(), *(s for fx in effects for s in fx.material.values()),
                           *(s for fx in effects for s in fx.inputs.values())]
                used |= {s.image for s in sources if s.image}
        return used | {n for name in used for n in self.resources.sequence(name).values()}  # a sequence's images

    def clean(self) -> list[str]:
        """File > Clean: the resources nothing uses leave the project (gone from the file at the next save)."""
        unused = [n for n in self.resources.images if n not in self.used_resources()]
        for n in unused:
            self.resources.images.pop(n)
            self.resources.stamps.pop(n)
        return unused

    def import_image(self, path) -> str:
        """File > Import resources (to the project): an image file as a project resource; returns its name. A file
        named like a UDIM sequence (Image sequence page: the UDIM number last, after ".", "-" or "_") brings the
        whole sequence next to it, one resource (Resources.sequence)."""
        path = Path(path)
        m = UDIM_FILE.match(path.stem)
        if m is None or int(m.group(3)) < 1001:
            return self._import_one(path, self.resources.unique(path.stem))
        base, sep = m.group(1), m.group(2)
        name = self.resources.unique(base)
        for f in sorted(path.parent.glob(f"*{path.suffix}")):
            g = UDIM_FILE.match(f.stem)
            if g is not None and g.group(1) == base and g.group(2) == sep and int(g.group(3)) >= 1001:
                self._import_one(f, f"{name}.{g.group(3)}")
        return name

    def _import_one(self, path: Path, name: str) -> str:
        from trafilatura3d import imageio
        img = imageio.read(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        self.resources.add(name, img)
        return name

    # --- files ----------------------------------------------------------------------------------------

    def save(self, path=None) -> None:
        path = Path(path or self.path)
        if path.suffix != PROJECT_SUFFIX:
            path = path.with_suffix(PROJECT_SUFFIX)
        uids = {u for ts in self.project.texture_sets for u in ts.all_uids()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            data = self.project.to_dict()
            data["source"] = self.source
            z.writestr("project.json", json.dumps(data, indent=1))
            z.writestr("mesh.npz", _mesh_bytes(self.objects, self.materials, self.fit))
            for (uid, key, udim), e in self.store.entries.items():
                if uid in uids:
                    table, pages = e.pages.raw()
                    b = io.BytesIO()
                    np.savez(b, table=table, pages=pages, size=np.array(e.pages.size))
                    z.writestr(f"paint/{uid}__{key}__{udim}__{e.pages.format}.npz", b.getvalue())
            for name in self.resources.images:  # imported into the project: kept until Clean
                p = self.resources.images.packed(name)  # as kept: packed already, stored as it is
                b = io.BytesIO()
                np.savez(b, lz4=np.frombuffer(p.data, np.uint8), shape=np.array(p.shape), dtype=np.array(p.dtype.str))
                z.writestr(f"resources/{name}.npz", b.getvalue(), compress_type=zipfile.ZIP_STORED)
        tmp.replace(path)
        self.path = path

    @classmethod
    def open(cls, path, evaluator: Evaluator | None = None) -> Session:
        path = Path(path)
        try:
            z = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as e:
            raise ProjectError(f"{path.name}: not a project ({e})") from None
        with z:
            data = json.loads(z.read("project.json"))
            warnings = drop_unknown_nodes(data)  # nodes gone from the library: emptied, reported
            warnings += drop_unknown_shader(data)
            project = Project.from_dict(data)
            objects, materials, fit = _mesh_from(z.read("mesh.npz"))
            s = cls(project, objects, materials, evaluator)
            s.fit = fit
            s.source = data.get("source", "")
            s.load_warnings = warnings
            res = [n for n in z.namelist() if n.startswith("resources/")]
            for name, (image, stamp) in zip(res, _load_resources(path, res)):
                s.resources.put(Path(name).stem, image, stamp)
            for name in z.namelist():
                if name.startswith("paint/"):
                    parts = Path(name).stem.split("__")
                    if len(parts) != 4 or not parts[2].isdigit() or not name.endswith(".npz"):
                        raise ProjectError(f"{path.name}: painting {name} in an older format (whole textures)")
                    uid, key, udim, fmt = parts
                    d = np.load(io.BytesIO(z.read(name)))
                    storage = storage_for(None) if key == MASK else fmt  # a mask in the storage masks have now
                    s.store.write_raw(uid, key, int(udim), tuple(int(v) for v in d["size"]), storage, d["table"],
                                      d["pages"], fmt)
        s.path = path
        return s


def _load_resources(path: Path, names: list[str]) -> list:
    """(image, stamp) of the project's images, kept packed as the file stores them (Packed): read in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    def load(name):
        if not name.endswith(".npz"):
            raise ProjectError(f"{path.name}: resource {name} in an older format (unpacked)")
        with zipfile.ZipFile(path) as z:  # one handle per read: a ZipFile is not shared between threads
            d = np.load(io.BytesIO(z.read(name)))
            p = Packed.of(d["lz4"].tobytes(), d["shape"], str(d["dtype"]))
            return p, p.stamp()

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(names)))) as pool:
        return list(pool.map(load, names))


def _mesh_bytes(objects, materials, fit) -> bytes:
    arrays = {"materials": np.array(materials), "names": np.array([o.name for o in objects]),
              "fit_center": np.asarray(fit[0], np.float64), "fit_scale": np.array(float(fit[1]))}
    for i, o in enumerate(objects):
        for k in ("pos", "nrm", "tan", "uv", "idx"):
            arrays[f"{i}_{k}"] = getattr(o, k)
        arrays[f"{i}_sub"] = np.array([[s.start, s.count, s.material] for s in o.submeshes], np.int64)
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()


def _mesh_from(data: bytes):
    z = np.load(io.BytesIO(data))
    objects = []
    for i, name in enumerate(z["names"]):
        subs = [meshio.Submesh(int(a), int(b), int(c)) for a, b, c in z[f"{i}_sub"]]
        objects.append(meshio.MeshObject(str(name), z[f"{i}_pos"], z[f"{i}_nrm"], z[f"{i}_tan"], z[f"{i}_uv"],
                                         z[f"{i}_idx"], subs))
    return objects, [str(m) for m in z["materials"]], (z["fit_center"], float(z["fit_scale"]))
