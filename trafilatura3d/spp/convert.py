"""A .spp project as a Paint project: the mesh, the Texture Sets with their channels, mesh maps and layer stacks.

The document's numbers, as read from projects (`document.py` for the layout):
- channel types are bits: 1 base color, 2 height, 128 roughness, 8192 metallic, 4194304 normal; 1 << 27 a mask;
  all bits set: every channel. A bit not in this table is reported, never guessed.
- channel formats index `channels.FORMATS` in its order (sRGB8, L8, RGB8, L16, RGB16, L16F, RGB16F, L32F, RGB32F).
- blending modes are the shader API's BlendingMode_* values (BLEND_MODES).
- fill projections: 0 UV, 2 tri-planar, 6 fill matched per UV tile; filtering 0 nearest, 1 bilinear sharp,
  2 bilinear HQ; UV wrap 0 none, 1 horizontal, 2 vertical, 3 both.
- mesh maps by usage: 0 ID (color map from mesh), 1 ambient occlusion, 2 curvature, 3 world space normals,
  4 position, 5 thickness; the material's normal map bitmap is the Normal slot.
- a layer's stack and its mask's are bottom first, as ours; a layer's content starting with a fill is a fill layer.
- every group of paint actions is a paint effect, its pixels made by replaying its strokes (`replay.py`).
- uniform values of color managed channels are linear (ours: sRGB encoded).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trafilatura3d import meshio
from trafilatura3d.paint import graph_nodes as gn
from trafilatura3d.paint.channels import CHANNELS, FORMATS
from trafilatura3d.paint.model import (FIRST_TILE, MASK_KEY, MATCH_PER_TILE, UV_TILE_MODES, ChannelBlend, ChannelSource,
                                       Effect, Layer, Levels, Mask, Project, TextureSet, Channel, UVProjection, new_uid)
from trafilatura3d.spp import procedurals
from trafilatura3d.spp.document import CLASS
from trafilatura3d.spp.scene import Scene

CHANNEL_BITS = {1: "basecolor", 2: "height", 128: "roughness", 8192: "metallic", 4194304: "normal"}
MASK_BIT = 1 << 27
ALL_BITS = (1 << 64) - 1
BLEND_MODES = ("disable", "replace", "normal", "darken", "multiply", "colorburn", "linearburn", "lighten", "screen",
               "colordodge", "lineardodge", "overlay", "softlight", "hardlight", "vividlight", "linearlight",
               "pinlight", "difference", "exclusion", "tint", "saturation", "color", "value", "divide",
               "inversedivide", "passthrough", "normalcombine", "normaldetail", "normalinversedetail", "subtract",
               "inversesubtract", "signedaddition")
PROJECTIONS = {0: "UV projection", 2: "Tri-planar projection", 6: MATCH_PER_TILE}
FILTERING = {0: "Nearest", 1: "Bilinear | Sharp", 2: "Bilinear | HQ"}
UV_WRAP = ("None", "Repeat horizontally", "Repeat vertically", "Repeat")
MESH_MAP_USAGE = {0: "id", 1: "ambient_occlusion", 2: "curvature", 3: "world_space_normals", 4: "position",
                  5: "thickness"}
FORMAT_ORDER = tuple(FORMATS)
NORMAL_SYSCOORD = {0: "OpenGL (X+, Y+, Z+)", 1: "DirectX (X+, Y-, Z+)"}


@dataclass
class PaintGroup:
    """A group of paint actions to replay into a paint effect's pixels: on "mask", or on the channels."""
    ts: str
    effect: str  # our effect uid
    mask: bool
    actions: list = field(default_factory=list)


class Converter:
    """Builds the Project from the document and the scene; `resources` collects the images to load ({our name:
    (dataset, tile list name or None)}), `paint` the groups to replay, `notes` what could not be carried over."""

    def __init__(self, doc: dict, scene: Scene, container):
        self.doc, self.scene, self.c = doc, scene, container
        self.notes: list[str] = []
        self.paint: list[PaintGroup] = []
        self.images: dict[str, dict] = {}  # our resource name -> {UDIM or 0: dataset}
        self._by_url: dict[str, str] = {}
        self._unknown_bits: set = set()

    # --- report --------------------------------------------------------------------------------------------------

    def note(self, where: str, text: str) -> None:
        line = f"{where}: {text}" if where else text
        if line not in self.notes:
            self.notes.append(line)

    # --- mesh ----------------------------------------------------------------------------------------------------

    def meshes(self, names: list[str]):
        """meshio objects in the file's units, the materials named after their Texture Sets (`names`, by the scene's
        material index)."""
        objects = []
        for m in self.scene.meshes:
            pos = m.pos * self.scene.radius + self.scene.center
            uv = np.column_stack([m.uv[:, 0], 1.0 - m.uv[:, 1]])  # V up -> rows down
            groups = np.full(len(m.tri), m.material)
            objects.append(meshio.finish(m.name, pos, m.nrm, uv, m.tri, groups, len(names)))
        return objects

    # --- project -------------------------------------------------------------------------------------------------

    def project(self, source: str) -> tuple[Project, list[str]]:
        """The Project and the mesh material names (by the scene's material index)."""
        doc = self.doc
        mats = doc["materials"]
        by_uid = {m["sceneMaterialUID"]: m for m in mats}
        names = []
        for i, uid in enumerate(self.scene.materials):
            m = by_uid.get(uid)
            names.append(m["sceneMaterialName"] if m else f"material {i + 1}")
            if m is None:
                self.note("", f"the mesh material {i + 1} has no Texture Set")
        tiles = any(m.get("useUVTiles") for m in mats)
        log2 = max(int(m["resolutionLog2"][0]) for m in mats)
        syscoord = int(doc.get("normalMapSyscoord", 0))
        if syscoord not in NORMAL_SYSCOORD:
            self.note("", f"unknown normal map format {syscoord}: OpenGL used")
        project = Project(mesh=Path(source).name, normal_format=NORMAL_SYSCOORD.get(syscoord, NORMAL_SYSCOORD[0]),
                          resolution=2 ** log2, uv_tiles=UV_TILE_MODES[0] if tiles else "")
        for m in mats:
            project.texture_sets.append(self.texture_set(m, tiles))
        return project, names

    def texture_set(self, m: dict, uv_tiles: bool) -> TextureSet:
        name = m["sceneMaterialName"]
        size = [2 ** int(m["resolutionLog2"][0]), 2 ** int(m["resolutionLog2"][1])]
        ts = TextureSet(name=name, material=name, size=size, description=m.get("description", ""))
        if uv_tiles:
            ts.tiles = sorted(FIRST_TILE + t["indices"][0] + 10 * t["indices"][1] for t in m["uvTiles"] if t["active"])
            for t in m["uvTiles"]:
                if t["active"] and t.get("resolutionOverridden"):
                    ts.tile_sizes[str(FIRST_TILE + t["indices"][0] + 10 * t["indices"][1])] = [
                        2 ** int(t["resolutionLog2"][0]), 2 ** int(t["resolutionLog2"][1])]
        stack = m["stacks"][0]
        if len(m["stacks"]) > 1:
            self.note(name, f"{len(m['stacks'])} stacks: only the first is read")
        for ch in stack["channels"]:
            cid = CHANNEL_BITS.get(ch["type"])
            if cid is None:
                self.note(name, f"channel type {ch['type']} is not known: left out")
                continue
            fmt = FORMAT_ORDER[ch["format"]] if 0 <= ch["format"] < len(FORMAT_ORDER) else CHANNELS[cid].format
            ts.channels.append(Channel(cid, fmt))
        self.ts = ts
        for a in m.get("additionalMaps", []):
            slot = MESH_MAP_USAGE.get(a["usage"])
            if slot is None:
                self.note(name, f"mesh map of usage {a['usage']} ({_label(a['urlToBitmapRes'])}) left out")
                continue
            self._mesh_map(ts, slot, a["urlToBitmapRes"])
        if m.get("normalMapBitmap"):
            self._mesh_map(ts, "normal", m["normalMapBitmap"]["urlToBitmapRes"])
        ts.layers = self.layers(stack["stack"]["items"])
        return ts

    def _mesh_map(self, ts, slot: str, url: str) -> None:
        res = self.resource(url)
        if res is None:
            return
        tiles = self.images[res]
        for udim in ts.tiles:
            if udim in tiles:
                ts.set_mesh_map(slot, udim, f"{res}.{udim}")
            elif 0 in tiles and len(ts.tiles) == 1:
                ts.set_mesh_map(slot, udim, res)

    # --- resources -----------------------------------------------------------------------------------------------

    def resource(self, url: str) -> str | None:
        """Our resource name for an image url (a UDIM image: the name of its sequence), registered once."""
        if url in self._by_url:
            return self._by_url[url]
        dataset = self.c.resource(url)
        if dataset is None:
            self.note("", f"the image {_label(url)!r} is not in the file")
            return None
        from trafilatura3d.spp.container import TILE_LIST, tile_list
        data = self.c.bytes(dataset)
        tiles = ({u: f"resources/{n}" for u, n in tile_list(data).items()} if data.startswith(TILE_LIST)
                 else {0: dataset})
        base = re.sub(r"[^\w .-]+", "_", _label(url)).strip() or "image"
        name, k = base, 1
        while name in self.images:
            k += 1
            name = f"{base}_{k}"
        self.images[name] = tiles
        self._by_url[url] = name
        return name

    # --- channels and blending -----------------------------------------------------------------------------------

    def channels_of(self, bits: int, where: str) -> list[str]:
        """Our channels of a channel bit mask, those of the current Texture Set."""
        have = self.ts.channel_ids()
        if bits == ALL_BITS:
            return list(have)
        out = []
        for bit in _bits(bits):
            cid = CHANNEL_BITS.get(bit)
            if cid is None:
                if bit != MASK_BIT and bit not in self._unknown_bits:
                    self._unknown_bits.add(bit)
                    self.note(where, f"channel type bit {bit} is not known: left out")
                continue
            if cid in have:
                out.append(cid)
        return out

    def blend(self, entries: list, where: str, mask: bool = False, effect: bool = False) -> dict:
        out = {}
        for b in entries:
            mode = b["blendingMode"]
            name = BLEND_MODES[mode] if 0 <= mode < len(BLEND_MODES) else None
            if name is None:
                self.note(where, f"unknown blending mode {mode}: Normal used")
                name = "normal"
            if effect and name == "passthrough":  # a group of paint actions lays over what is below
                name = "normal"
            opacity = float(min(max(b["opacity"], 0.0), 1.0))
            if mask:
                if b["channelTypes"] in (0, MASK_BIT, ALL_BITS):
                    out[MASK_KEY] = ChannelBlend(name, opacity)
                continue
            for ch in self.channels_of(b["channelTypes"], where):
                if name in ("normalcombine", "normaldetail", "normalinversedetail") and ch != "normal":
                    continue
                out[ch] = ChannelBlend(name, opacity)
        return out

    # --- layers --------------------------------------------------------------------------------------------------

    def layers(self, items: list) -> list:
        return [self.layer(x) for x in items]

    def layer(self, d: dict) -> Layer:
        label = d.get("label") or "Layer"
        where = f"{self.ts.name} > {label}"
        if d[CLASS] == "DataLayerGroup":
            layer = Layer(uid=new_uid(), kind="folder", name=label, visible=bool(d.get("enabled", True)))
            layer.children = self.layers(d["subStack"]["items"])
        else:
            actions = (d.get("actions") or {}).get("items", [])
            first = actions[0] if actions else None
            kind = "fill" if first is not None and first[CLASS] == "DataActionFill" else "paint"
            layer = Layer(uid=new_uid(), kind=kind, name=label, visible=bool(d.get("enabled", True)))
            if kind == "fill":
                layer.projection = self.projection(first, where)
                layer.material = self.fill_sources(first, where, mask=False)
                actions = actions[1:]
            layer.effects = [e for a in actions if (e := self.effect(a, where, mask=False)) is not None]
        layer.blend = self.blend(d.get("perChannelBlending", []), where)
        if d.get("maskActions") is not None:
            items = d["maskActions"].get("items", [])
            layer.mask = Mask(background=float(d.get("maskInitial", 0)), enabled=bool(d.get("maskEnabled", True)),
                              effects=[e for a in items if (e := self.effect(a, where + " (mask)", mask=True))])
        self.geometry(d, layer, where)
        return layer

    def geometry(self, d: dict, layer: Layer, where: str) -> None:
        """The Geometry mask: the UV tiles (and meshes) left out, the exceptions to the default."""
        listed = [FIRST_TILE + b["value"][0] + 10 * b["value"][1] for b in d.get("enabledUVTileList", [])]
        if listed:
            excluded = listed if d.get("enabledUVTileDefault", True) else [t for t in self.ts.tiles if t not in listed]
            layer.geometry_type = "UV Tiles"
            layer.geometry_tiles = sorted(t for t in excluded if t in self.ts.tiles)
        if d.get("enabledMeshList"):
            self.note(where, "the Geometry mask by mesh names is not read")

    # --- effects -------------------------------------------------------------------------------------------------

    def effect(self, a: dict, where: str, mask: bool) -> Effect | None:
        cls = a[CLASS]
        name = a.get("label") or ""
        visible = bool(a.get("enabled", True))
        if cls == "DataActionGroup":
            fx = Effect(uid=new_uid(), kind="paint", name=name or "Paint", visible=visible,
                        blend=self.blend(a.get("perChannelBlending", []), where, mask, effect=True))
            group = PaintGroup(self.ts.name, fx.uid, mask, list(a["subStack"]["items"]))
            if group.actions:
                self.paint.append(group)
            return fx
        if cls == "DataActionFill":
            gen = self._generator(a, where, mask)
            if gen is not None:
                gen.visible = visible
                gen.blend = self.blend(a.get("perChannelBlending", []), where, mask)
                return gen
            return Effect(uid=new_uid(), kind="fill", name=name or "Fill", visible=visible,
                          blend=self.blend(a.get("perChannelBlending", []), where, mask),
                          projection=self.projection(a, where), material=self.fill_sources(a, where, mask))
        if cls == "DataActionFilterLevels":
            return self.levels(a, where, mask)
        if cls == "DataActionFilterProcedural":
            res = procedurals.resolve(a["procedural"])
            graph = "/".join(procedurals.graph_of(a["procedural"]["urlToSbsRes"]))
            if res is None:
                self.note(where, f"the filter {graph} has no counterpart: left out")
                return None
            if res.unmatched:
                self.note(where, f"filter {graph}: parameters {', '.join(sorted(set(res.unmatched)))} not carried over")
            return Effect(uid=new_uid(), kind="filter", name=name or gn.NODES[res.type].label, visible=visible,
                          procedural=gn.PREFIX + res.type, values=res.values)
        self.note(where, f"effect {cls} is not read")
        return None

    def levels(self, a: dict, where: str, mask: bool) -> Effect | None:
        lv = a["levels"]
        chans = [MASK_KEY] if mask else self.channels_of(a["channelTypes"], where)
        if not chans:
            return None
        lo, hi = lv["inputMinimum"], lv["inputMaximum"]
        olo, ohi, gamma = lv["outputMinimum"], lv["outputMaximum"], lv["gamma"]
        if len({tuple(np.round(v, 6)) for v in zip(lo, hi, olo, ohi, gamma)}) > 1:
            self.note(where, "Levels with different values per component: the red one is used for all")
        g = max(float(gamma[0]), 1e-3)
        levels = Levels(channel="basecolor" if mask else chans[0], in_low=float(lo[0]), in_mid=0.5 ** g,
                        in_high=float(hi[0]), out_low=float(olo[0]), out_high=float(ohi[0]), clamp=bool(lv["clamp"]))
        if len(chans) > 1:
            self.note(where, f"Levels on {', '.join(chans)}: only {chans[0]}")
        return Effect(uid=new_uid(), kind="levels", name=a.get("label") or "Levels", visible=bool(a.get("enabled", True)),
                      levels=levels)

    def _generator(self, a: dict, where: str, mask: bool) -> Effect | None:
        """A fill whose only source is a generator (a node reading mesh maps): a generator effect."""
        procs = [s for s in a["sources"] if s[CLASS] == "DataSourceProcedural"]
        if len(procs) != 1 or len(a["sources"]) != 1:
            return None
        res = procedurals.resolve(procs[0]["procedural"])
        if res is None or not procedurals.is_generator(res.type):
            return None
        graph = "/".join(procedurals.graph_of(procs[0]["procedural"]["urlToSbsRes"]))
        if res.unmatched:
            self.note(where, f"generator {graph}: parameters {', '.join(sorted(set(res.unmatched)))} not carried over")
        self._generator_inputs(procs[0]["procedural"], where, graph)
        if not mask:
            self.note(where, f"generator {graph} in a layer's content: read as a mask generator on every channel")
        return Effect(uid=new_uid(), kind="generator", name=a.get("label") or gn.NODES[res.type].label,
                      procedural=gn.PREFIX + res.type, values=res.values)

    def _generator_inputs(self, proc: dict, where: str, graph: str) -> None:
        """A generator reads the Texture Set's mesh maps; an input fed with another image is reported."""
        maps = {n for tiles in self.ts.mesh_maps.values() for n in tiles.values()}
        for si in proc.get("sourcesInputs", []):
            src = si.get("source") or {}
            if src.get(CLASS) == "DataSourceBitmap":
                res = self.resource(src["bitmap"]["urlToBitmapRes"])
                if res is not None and not any(n == res or n.startswith(res + ".") for n in maps):
                    self.note(where, f"generator {graph}: input {si['input']['identifier']} reads {res}, "
                                     "not a mesh map: the mesh map is used")

    # --- fill ----------------------------------------------------------------------------------------------------

    def projection(self, a: dict, where: str) -> UVProjection:
        p = UVProjection()
        mode = a.get("projection", 0)
        if mode not in PROJECTIONS:
            self.note(where, f"fill projection {mode} is not known: UV projection used")
        p.mode = PROJECTIONS.get(mode, "UV projection")
        p.filtering = FILTERING.get(a.get("filtering", 2), "Bilinear | HQ")
        wrap = a.get("uvSamplingWrap", 3)
        p.wrap = UV_WRAP[wrap] if 0 <= wrap < len(UV_WRAP) else "Repeat"
        t = a.get("uvTransformation0") or {}
        sx, sy = (t.get("scaleFactors") or [1.0, 1.0])[:2]
        if abs(abs(sx) - abs(sy)) > 1e-6:
            self.note(where, f"fill tiling {sx:g} x {sy:g}: {abs(sx):g} on both axes")
        if sx < 0 or sy < 0:
            self.note(where, "a mirrored fill tiling: not mirrored")
        p.tiling = max(abs(float(sx)), 1e-3)
        p.rotation = math.degrees(float(t.get("rotation", 0.0)))
        p.offset = [float(v) for v in (t.get("translation") or [0.0, 0.0])[:2]]
        p.hardness = float(a.get("sideHardness", 0.5))
        p.offset3d = [float(v) for v in a.get("dddPosition", [0, 0, 0])]
        p.rotation3d = [math.degrees(float(v)) for v in a.get("dddEulerAngles", [0, 0, 0])]
        p.scale3d = [float(v) for v in a.get("dddSize", [1, 1, 1])]
        p.depth_culling = bool(a.get("depthCulling", False))
        p.depth_hardness = float(a.get("depthCullingHardness", 0.5))
        p.backface_culling = bool(a.get("backfaceCulling", True))
        p.backface_angle = float(a.get("backfaceCullingAngle", 0.5)) * 180.0
        p.backface_hardness = float(a.get("backfaceCullingHardness", 0.5))
        return p

    def fill_sources(self, a: dict, where: str, mask: bool) -> dict:
        out = {}
        for s in a["sources"]:
            keys = [MASK_KEY] if mask else self.channels_of(s["channelTypes"], where)
            for k in keys:
                src = self.source(s, where, color=k != MASK_KEY and CHANNELS[k].color)
                if src is not None:
                    out[k] = src
        return out

    def source(self, s: dict, where: str, color: bool = False) -> ChannelSource | None:
        """A channel source; `color`: a color managed channel, whose uniform values the file keeps linear and ours
        encoded."""
        cls = s[CLASS]
        if cls == "DataSourceUniform":
            c = [float(v) for v in s["color"]][:3]
            c = c + [c[-1]] * (3 - len(c))
            return ChannelSource(color=srgb(c) if color else c)
        if cls == "DataSourceBitmap":
            res = self.resource(s["bitmap"]["urlToBitmapRes"])
            return None if res is None else ChannelSource(image=res)
        if cls == "DataSourceProcedural":
            res = procedurals.resolve(s["procedural"])
            graph = "/".join(procedurals.graph_of(s["procedural"]["urlToSbsRes"]))
            if res is None:
                self.note(where, f"the procedural {graph} has no counterpart: its uniform value is used")
                return ChannelSource()
            if res.unmatched:
                self.note(where, f"procedural {graph}: parameters {', '.join(sorted(set(res.unmatched)))} not carried over")
            return ChannelSource(procedural=gn.PREFIX + res.type, values=res.values)
        self.note(where, f"source {cls} is not read")
        return None


def srgb(c) -> list:
    """Linear values encoded as sRGB."""
    c = np.clip(np.asarray(c, np.float64), 0.0, None)
    return [float(v) for v in np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)]


def _bits(mask: int):
    bit = 1
    while bit <= mask:
        if mask & bit:
            yield bit
        bit <<= 1


def _label(url: str) -> str:
    """A readable name of a resource url: its last path element without the version."""
    return re.sub(r"\?.*$", "", url).rstrip("/").split("/")[-1]
