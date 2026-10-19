"""The Paint project document: the mesh, the project settings and one Texture Set per material of the mesh,
each with its channels, its settings and its layer stack (New Project window, Texture Set settings and Layer
stack pages of the documentation).

Layers are data: a Paint layer's pixels and a mask's painted pixels live in the PaintStore (paint.py), keyed
by the layer's uid; the project file (`.trafilatura3dpaint`, a zip: project.json, the mesh, the paint data and the
resources) keeps both. Ours: the file format, painting stored as pixels (not as strokes replayed), folders
compositing their content on an empty layer before blending it.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from trafilatura3d.paint import blending
from trafilatura3d import shaders
from trafilatura3d.paint.channels import CHANNELS, DEFAULT_BLEND, FORMATS, channel_id

PROJECT_SUFFIX = ".trafilatura3dpaint"
FORMAT_VERSION = 1
NORMAL_FORMATS = ("DirectX (X+, Y-, Z+)", "OpenGL (X+, Y+, Z+)")  # New Project > Normal Map Format
# New Project > UV Tile Settings (UDIMs) > UV Tile mode; "" = Use UV Tile workflow off (only the 0-1 range)
UV_TILE_MODES = ("Preserve UV Tile layout per materials and enable painting across tiles",
                 "Convert UV Tiles into individual Textures Sets (legacy)")
GEOMETRY_TYPES = ("UV Tiles", "Mesh names")  # Geometry mask > Masking type
FIRST_TILE = 1001  # the 0-1 UV range: the only tile without the UV Tile workflow


def tile_uv(udim: int) -> tuple[int, int]:
    """The UV offsets (U to the right, V up) of a UDIM tile: 1001 is (0, 0), 1002 (1, 0), 1011 (0, 1)."""
    return (udim - FIRST_TILE) % 10, (udim - FIRST_TILE) // 10


def udim_at(u: float, y: float) -> int:
    """The UDIM tile of a stored UV (y = 1 - V, rows down: the convention of the loaded meshes)."""
    return FIRST_TILE + int(math.floor(u)) + 10 * int(math.floor(1.0 - y))


def local_uv(uv, udim: int):
    """Stored UVs ((..., 2), y = 1 - V) inside a tile's own 0-1 square (its texture's)."""
    du, dv = tile_uv(udim)
    return np.asarray(uv, np.float64) + (-du, dv)


SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)  # Texture Set size: up to 4096 in the application (documentation);
# the list itself is ours
NORMAL_MIXING = ("Replace", "Combine")
HEIGHT_TO_NORMAL = ("Sharp", "Smooth (Sobel)")
AO_MIXING = ("Replace", "Multiply")
# UV padding: only 2D Space Neighbor (the nearest texel of the island in texture space) is implemented;
# 3D Space Neighbor (across the UV seams) is not offered yet.
UV_PADDING = ("2D Space Neighbor",)
LAYER_KINDS = ("paint", "fill", "folder")
# Mesh map slots of Texture Set settings: (slot, label, file name suffix of New Project > Import baked maps).
MESH_MAPS = [("normal", "Normal", "normal_base"), ("world_space_normals", "World space normal", "world_space_normals"),
             ("id", "ID", "id"), ("ambient_occlusion", "Ambient occlusion", "ambient_occlusion"),
             ("curvature", "Curvature", "curvature"), ("position", "Position", "position"),
             ("thickness", "Thickness", "thickness"),
             # Bakeable in the Baking mode, not listed by Import baked maps: suffixes of the custom effects'
             # mesh map inputs (identifiers height_base, bent_normals_base, opacity_base).
             ("height", "Height", "height_base"), ("bent_normals", "Bent normals", "bent_normals_base"),
             ("opacity", "Opacity", "opacity_base")]
# Fill properties (Fill projections pages).
# Fill projections (Properties > Projection): Fill (match per UV Tile), UV Tile projects only, then PROJECTIONS,
# whose index is the fill shader's mode.
MATCH_PER_TILE = "Fill (match per UV Tile)"
PROJECTIONS = ("UV projection", "Tri-planar projection", "Planar projection", "Spherical projection",
               "Cylindrical projection", "Warp projection")


def warp_grid(rows: int, cols: int) -> list:
    """The flat warp grid's points, row by row from the top: (cols + 1) x (rows + 1) points on XY."""
    return [[c / cols - 0.5, 0.5 - r / rows, 0.0] for r in range(rows + 1) for c in range(cols + 1)]
SHAPE_CROP = ("Project cropped to shape", "Projection extends outside shape")
FILTERING = ("Bilinear | HQ", "Bilinear | Sharp", "Nearest")
UV_WRAP = ("None", "Repeat horizontally", "Repeat vertically", "Repeat")


class ProjectError(ValueError):
    pass


def drop_unknown_nodes(data, where: str = "") -> list[str]:
    """A saved project's (or smart material's) JSON made loadable when it names library nodes that no longer
    exist ("node:<type>" of a filter or generator effect, or of a channel source's procedural): the reference is
    emptied in place, the effect stays with no node chosen, the source falls back to its uniform value; one
    warning per reference. The strict checks of `from_dict` stay for everything else (and for edits)."""
    from trafilatura3d.paint.graph_nodes import PREFIX, node_of
    warnings = []
    if isinstance(data, list):
        for item in data:
            warnings += drop_unknown_nodes(item, where)
        return warnings
    if not isinstance(data, dict):
        return warnings
    here = f"{where} > {data['name']}" if isinstance(data.get("name"), str) else where
    proc = data.get("procedural")
    if isinstance(proc, str) and proc.startswith(PREFIX) and node_of(proc) is None:
        data["procedural"] = ""
        for key in ("values", "inputs"):
            if key in data:
                data[key] = {}
        what = f"the {data['kind']} is left empty" if data.get("kind") in ("filter", "generator") else \
            "the source is its uniform value"
        warnings.append(f"{here.lstrip(' >') or 'project'}: the node {proc[len(PREFIX):]!r} no longer exists, {what}")
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            warnings += drop_unknown_nodes(value, here if key not in ("material", "inputs") else f"{here} > {key}")
    return warnings


def new_uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ChannelBlend:
    """A layer's blending mode and opacity (0-1) for one channel."""
    mode: str = "normal"
    opacity: float = 1.0


@dataclass
class ChannelSource:
    """What a fill layer (or the brush) puts in a channel: a uniform value, a bitmap resource, a procedural
    (a library texture generator, "node:<type>", with its parameter values) or an anchor point (Anchor Point page:
    a live reference to the stack where the anchor is; `anchor_channel` the channel it reads of an anchor in a
    layer's content, an anchor in a mask has one)."""
    color: list = field(default_factory=lambda: [0.5, 0.5, 0.5])  # RGB; gray channels use the first
    image: str = ""  # resource name in the project ("" = uniform)
    procedural: str = ""
    values: dict = field(default_factory=dict)
    anchor: str = ""  # the anchor point effect's uid
    anchor_channel: str = ""


@dataclass
class UVProjection:
    """Projection parameters of a fill (Fill properties, Fill projections pages): the UV transformation of every
    mode; the 3D transform (offset measured in sizes of the scene's bounding box from its center; rotation in degrees; scale 1 =
    the bounding box's largest side), the planes' Hardness (Tri-planar), Shape crop and the cullings (Planar).
    Defaults the documentation does not give are ours."""
    mode: str = "UV projection"
    filtering: str = "Bilinear | HQ"
    wrap: str = "Repeat"
    tiling: float = 1.0
    rotation: float = 0.0  # degrees
    offset: list = field(default_factory=lambda: [0.0, 0.0])
    shape_crop: str = "Projection extends outside shape"
    hardness: float = 0.5
    depth_culling: bool = False
    depth_hardness: float = 0.5
    backface_culling: bool = True
    backface_angle: float = 90.0  # degrees
    backface_hardness: float = 0.5
    offset3d: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation3d: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale3d: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    angle: float = 360.0  # Cylindrical: the projection's size on the perimeter, degrees (ours: the page gives none)
    # Warp: the grid's rows and columns, its points in the projection's space ([] = the flat grid: XY in -0.5..0.5,
    # row 0 at the top), Projection depth (projection units) and Automatic tangents (smooth: Catmull-Rom through
    # the points; off: straight between them).
    warp_rows: int = 3
    warp_cols: int = 3
    warp_points: list = field(default_factory=list)
    warp_depth: float = 0.25
    warp_tangents: bool = True
    # Symmetry of a fill with a 3D projection (Symmetry page): the scene's symmetry when it was turned on, as a
    # path's (paint/paths.symmetric); {} = off
    symmetry: dict = field(default_factory=dict)


MASK_KEY = "mask"  # the "channel" of mask effects (blending, fill value)
# Effects (Effects page) offered so far: Fill (as a fill layer), Paint (painting over the effects below), Levels,
# Filter and Generator (library nodes, paint/graph_nodes.py), Anchor point (Anchor Point page: what the stack
# holds where it is, the layer's content (a Passthrough layer's starts from the stack below) or the mask, up to
# the anchor; referenced by the channel sources of the layers above).
EFFECT_KINDS = ("fill", "paint", "levels", "filter", "generator", "anchor")
LEVELS_COLORS = ("RGB", "Red", "Green", "Blue")


@dataclass
class Levels:
    """The Levels effect: one Affected Channel; input levels (shadows, midtones, highlights), output levels
    (black, white points), Colors, Clamp (Levels page). Midtones is a relative position between shadows and
    highlights (0.5 = linear), as Trafilatura 3D's Levels node (decision 7)."""
    channel: str = "basecolor"
    in_low: float = 0.0
    in_mid: float = 0.5
    in_high: float = 1.0
    out_low: float = 0.0
    out_high: float = 1.0
    colors: str = "RGB"
    clamp: bool = True


@dataclass
class Effect:
    """An effect of a layer's content stack or of its mask's stack, bottom first. Fill and Paint effects
    blend like layers (per channel; "mask" in a mask); a Paint effect's pixels are the PaintStore entries
    (effect uid, channel or "mask")."""
    uid: str
    kind: str
    name: str
    visible: bool = True
    blend: dict = field(default_factory=dict)
    projection: UVProjection = field(default_factory=UVProjection)
    material: dict = field(default_factory=dict)  # fill: channel (or "mask") -> ChannelSource
    levels: Levels = field(default_factory=Levels)
    resource: dict | None = None  # fill: Material mode ({"package", "graph", "values"}, paint/materials.py)
    procedural: str = ""  # filter / generator: "node:<type>" of the library (graph_nodes; "" = none chosen yet)
    values: dict = field(default_factory=dict)  # filter / generator parameters (node parameter name -> value)
    inputs: dict = field(default_factory=dict)  # filter / generator resource slots: input port -> ChannelSource
    paths: list = field(default_factory=list)  # paint: PaintPath drawn over the effect's painting

    def blend_of(self, channel: str) -> ChannelBlend:
        return self.blend.get(channel) or ChannelBlend("normal")

    def to_dict(self) -> dict:
        d = {"uid": self.uid, "kind": self.kind, "name": self.name, "visible": self.visible,
             "blend": {k: asdict(v) for k, v in self.blend.items()}}
        if self.kind == "fill":
            d["projection"] = asdict(self.projection)
            d["material"] = {k: asdict(v) for k, v in self.material.items()}
            if self.resource:
                d["resource"] = self.resource
        if self.kind == "levels":
            d["levels"] = asdict(self.levels)
        if self.kind in ("filter", "generator"):
            d["procedural"] = self.procedural
            d["values"] = dict(self.values)
            d["inputs"] = {k: asdict(v) for k, v in self.inputs.items()}
        if self.paths:
            d["paths"] = [p.to_dict() for p in self.paths]
        return d

    @staticmethod
    def from_dict(d: dict) -> Effect:
        try:
            e = Effect(uid=str(d["uid"]), kind=d["kind"], name=str(d["name"]), visible=bool(d.get("visible", True)),
                       blend={k: ChannelBlend(**v) for k, v in d.get("blend", {}).items()},
                       projection=UVProjection(**d.get("projection", {})),
                       material={k: ChannelSource(**v) for k, v in d.get("material", {}).items()},
                       levels=Levels(**d.get("levels", {})), resource=d.get("resource"),
                       procedural=str(d.get("procedural", "")), values=dict(d.get("values", {})),
                       inputs={k: ChannelSource(**v) for k, v in d.get("inputs", {}).items()},
                       paths=[PaintPath.from_dict(p) for p in d.get("paths", [])])
        except (KeyError, TypeError) as err:
            raise ProjectError(f"invalid effect: {err}") from None
        if e.kind not in EFFECT_KINDS:
            raise ProjectError(f"unknown effect {e.kind!r}")
        if e.paths and e.kind != "paint":
            raise ProjectError(f"{e.name}: only paint effects have paths")
        if e.kind in ("filter", "generator") and e.procedural:
            from trafilatura3d.paint.graph_nodes import node_of
            if node_of(e.procedural) is None:
                raise ProjectError(f"{e.name}: unknown {e.kind} {e.procedural!r}")
        if e.levels.colors not in LEVELS_COLORS:
            raise ProjectError(f"{e.name}: invalid Levels colors")
        for b in e.blend.values():
            if b.mode not in blending.INDEX or not 0.0 <= b.opacity <= 1.0:
                raise ProjectError(f"{e.name}: invalid blending")
        return e


@dataclass
class Mask:
    """A layer mask: a white or black background, what is painted on it (the PaintStore entry
    (layer uid, "mask")) and its effects. `enabled` is the Shift+click toggle."""
    background: float = 1.0
    enabled: bool = True
    effects: list = field(default_factory=list)  # Effect, bottom first
    paths: list = field(default_factory=list)  # PaintPath drawn over the mask's painting

    def to_dict(self) -> dict:
        d = {"background": self.background, "enabled": self.enabled, "effects": [e.to_dict() for e in self.effects]}
        if self.paths:
            d["paths"] = [p.to_dict() for p in self.paths]
        return d

    @staticmethod
    def from_dict(d: dict) -> Mask:
        return Mask(float(d.get("background", 1.0)), bool(d.get("enabled", True)),
                    [Effect.from_dict(e) for e in d.get("effects", [])],
                    [PaintPath.from_dict(p) for p in d.get("paths", [])])


PATH_MODES = ("Paint along path", "Ribbon path", "Filled path", "Erase along path", "Smudge along path")


@dataclass
class PaintPath:
    """A path of a painting: a paint layer's content, a layer's mask or a paint effect (Path tool page: "available
    on paint layers and paint effects"; masks are paintings too). Points on the surface (world position, normal, pressure, corner),
    open or closed, and what it paints, kept with it (the tool's properties when it was drawn: brush, alpha,
    channel values or resources). Non-destructive: its pixels are made again from these data (session
    `path_textures`), never saved."""
    uid: str
    name: str
    mode: str = "Paint along path"
    visible: bool = True
    points: list = field(default_factory=list)  # [x, y, z, nx, ny, nz, pressure, corner (0/1)]
    closed: bool = False
    reversed: bool = False
    brush: dict = field(default_factory=dict)  # BrushSettings fields
    alpha: str = ""
    values: dict = field(default_factory=dict)  # channel -> RGB
    sources: dict = field(default_factory=dict)  # channel -> ChannelSource dict (Ribbon images)
    disabled: list = field(default_factory=list)  # channels off
    mask_value: float = 1.0  # the grayscale when the path is on a mask
    projection_depth: float = 1.0  # stamp radii from the surface where stamps land (ours: the doc gives no unit)
    up_axis: str = "Y"  # stamps' up when Follow Path is off
    opacity: float = 1.0  # Filled path
    ribbon: dict = field(default_factory=dict)  # Ribbon path parameters (paint/paths.RibbonSettings fields)
    # Symmetry: {"type": Mirror | Radial, "axis", "position" (Mirror: a number, Radial: x y z), "copies", "flip"}
    # (paint/paths.symmetric); {} = off
    symmetry: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> PaintPath:
        try:
            p = PaintPath(**d)
        except TypeError as e:
            raise ProjectError(f"invalid path: {e}") from None
        if p.mode not in PATH_MODES:
            raise ProjectError(f"{p.name}: unknown path mode {p.mode!r}")
        return p


@dataclass
class Layer:
    uid: str
    kind: str  # paint | fill | folder
    name: str
    visible: bool = True
    blend: dict = field(default_factory=dict)  # channel -> ChannelBlend (missing: the channel's default)
    mask: Mask | None = None
    # fill layers: the projection and the Material section (the channels it fills and their sources)
    projection: UVProjection = field(default_factory=UVProjection)
    material: dict = field(default_factory=dict)  # channel -> ChannelSource (missing: channel off)
    resource: dict | None = None  # Material mode: {"package", "graph", "values"} (paint/materials.py)
    children: list = field(default_factory=list)  # folders: layers, bottom first
    effects: list = field(default_factory=list)  # Effect of the content, bottom first (not on folders)
    paths: list = field(default_factory=list)  # paint layers: PaintPath, drawn over the painting in order
    # Geometry mask (Geometry mask page): the Masking type (GEOMETRY_TYPES) and what it excludes (the layer or
    # folder does nothing there): Mesh names, the meshes; UV Tiles (UV Tile workflow), the UDIM tiles, whose
    # textures the layer skips. [] = nothing excluded.
    geometry_type: str = "Mesh names"
    geometry_excluded: list = field(default_factory=list)
    geometry_tiles: list = field(default_factory=list)

    def excludes_tile(self, udim: int) -> bool:
        """The Geometry mask (UV Tiles) leaves this tile out."""
        return self.geometry_type == "UV Tiles" and udim in self.geometry_tiles

    def excluded_meshes(self) -> list:
        """The meshes the Geometry mask (Mesh names) leaves out."""
        return self.geometry_excluded if self.geometry_type == "Mesh names" else []

    def effect(self, uid: str):
        """(effect, in the mask) of this layer, or (None, False)."""
        for e in self.effects:
            if e.uid == uid:
                return e, False
        for e in (self.mask.effects if self.mask else []):
            if e.uid == uid:
                return e, True
        return None, False

    def blend_of(self, channel: str) -> ChannelBlend:
        return self.blend.get(channel) or ChannelBlend(DEFAULT_BLEND.get(channel, "normal"))

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def to_dict(self) -> dict:
        d = {"uid": self.uid, "kind": self.kind, "name": self.name, "visible": self.visible,
             "blend": {k: asdict(v) for k, v in self.blend.items()}}
        if self.mask is not None:
            d["mask"] = self.mask.to_dict()
        if self.effects:
            d["effects"] = [e.to_dict() for e in self.effects]
        if self.kind == "fill":
            d["projection"] = asdict(self.projection)
            d["material"] = {k: asdict(v) for k, v in self.material.items()}
            if self.resource:
                d["resource"] = self.resource
        if self.kind == "folder":
            d["children"] = [c.to_dict() for c in self.children]
        if self.paths:
            d["paths"] = [p.to_dict() for p in self.paths]
        if self.geometry_type != "Mesh names":
            d["geometry_type"] = self.geometry_type
        if self.geometry_excluded:
            d["geometry_excluded"] = list(self.geometry_excluded)
        if self.geometry_tiles:
            d["geometry_tiles"] = list(self.geometry_tiles)
        return d

    @staticmethod
    def from_dict(d: dict) -> Layer:
        try:
            layer = Layer(uid=str(d["uid"]), kind=d["kind"], name=str(d["name"]), visible=bool(d.get("visible", True)),
                          blend={k: ChannelBlend(**v) for k, v in d.get("blend", {}).items()},
                          mask=Mask.from_dict(d["mask"]) if d.get("mask") is not None else None,
                          projection=UVProjection(**d.get("projection", {})),
                          material={k: ChannelSource(**v) for k, v in d.get("material", {}).items()},
                          children=[Layer.from_dict(c) for c in d.get("children", [])],
                          effects=[Effect.from_dict(e) for e in d.get("effects", [])],
                          resource=d.get("resource"), paths=[PaintPath.from_dict(p) for p in d.get("paths", [])],
                          geometry_type=d.get("geometry_type", "Mesh names"),
                          geometry_excluded=[str(n) for n in d.get("geometry_excluded", [])],
                          geometry_tiles=[int(t) for t in d.get("geometry_tiles", [])])
        except (KeyError, TypeError) as e:
            raise ProjectError(f"invalid layer: {e}") from None
        layer.validate()
        return layer

    def validate(self) -> None:
        if self.kind not in LAYER_KINDS:
            raise ProjectError(f"unknown layer kind {self.kind!r}")
        for ch, b in self.blend.items():
            if b.mode not in blending.INDEX:
                raise ProjectError(f"{self.name}: unknown blending mode {b.mode!r}")
            if b.mode in blending.NORMAL_ONLY and ch != "normal":
                raise ProjectError(f"{self.name}: {blending.LABEL[b.mode]} is for the Normal channel")
            if not 0.0 <= b.opacity <= 1.0:
                raise ProjectError(f"{self.name}: opacity out of 0-1")
        if self.kind != "folder" and self.children:
            raise ProjectError(f"{self.name}: only folders hold layers")
        if self.geometry_type not in GEOMETRY_TYPES:
            raise ProjectError(f"{self.name}: unknown Geometry mask type {self.geometry_type!r}")
        if self.kind == "folder" and self.effects:
            raise ProjectError(f"{self.name}: a folder has no content effects")
        if self.kind == "folder" and self.material:
            raise ProjectError(f"{self.name}: a folder has no material")
        if self.paths and self.kind != "paint":
            raise ProjectError(f"{self.name}: only paint layers have paths")
        p = self.projection
        if (p.filtering not in FILTERING or p.wrap not in UV_WRAP or p.mode not in (MATCH_PER_TILE, *PROJECTIONS)
                or p.shape_crop not in SHAPE_CROP or any(len(v) != 3 for v in (p.offset3d, p.rotation3d, p.scale3d))
                or not (1 <= p.warp_rows <= 32 and 1 <= p.warp_cols <= 32)
                or (p.warp_points and len(p.warp_points) != (p.warp_rows + 1) * (p.warp_cols + 1))):
            raise ProjectError(f"{self.name}: invalid fill properties")
        for ch in self.material:
            if ch not in CHANNELS:
                raise ProjectError(f"{self.name}: unknown channel {ch!r}")
        for c in self.children:
            c.validate()


@dataclass
class Channel:
    id: str
    format: str


@dataclass
class TextureSet:
    name: str
    material: str  # the mesh material it textures (Reset name to the original name)
    size: list = field(default_factory=lambda: [2048, 2048])
    description: str = ""
    visible: bool = True
    channels: list = field(default_factory=list)  # Channel, in the Texture Set settings order
    normal_mixing: str = "Combine"
    height_to_normal: str = "Smooth (Sobel)"
    ao_mixing: str = "Multiply"
    uv_padding: str = "2D Space Neighbor"  # the documentation's default is 3D Space Neighbor (not offered)
    # the UDIM tiles it textures, sorted, a texture each: with the UV Tile workflow those of its triangles, else 1001
    tiles: list = field(default_factory=lambda: [FIRST_TILE])
    tile_sizes: dict = field(default_factory=dict)  # UV tile (str) -> [w, h]: its own resolution (default `size`)
    mesh_maps: dict = field(default_factory=dict)  # slot -> {UDIM tile (str): resource name} (baked or imported)
    baking: dict = field(default_factory=dict)  # Baking mode configuration (paint/baking.py)
    layers: list = field(default_factory=list)  # Layer, bottom first

    def channel_ids(self) -> list[str]:
        return [c.id for c in self.channels]

    def format_of(self, channel: str) -> str:
        return next(c.format for c in self.channels if c.id == channel)

    def walk(self):
        for layer in self.layers:
            yield from layer.walk()

    def all_uids(self) -> list[str]:
        """Every layer and effect uid (the PaintStore keys their pixels by it)."""
        out = []
        for layer in self.walk():
            out.append(layer.uid)
            out += [e.uid for e in layer.effects]
            out += [e.uid for e in (layer.mask.effects if layer.mask else [])]
        return out

    def owner_of(self, uid: str):
        """(layer, effect or None, in the mask) for a layer or effect uid."""
        for layer in self.walk():
            if layer.uid == uid:
                return layer, None, False
            e, in_mask = layer.effect(uid)
            if e is not None:
                return layer, e, in_mask
        return None, None, False

    def find(self, uid: str) -> Layer | None:
        return next((layer for layer in self.walk() if layer.uid == uid), None)

    def stack_order(self) -> dict:
        """Layer uid -> its place in the order the stack computes (bottom first, a folder after its content)."""
        order = []

        def visit(layers):
            for layer in layers:
                visit(layer.children)
                order.append(layer.uid)
        visit(self.layers)
        return {uid: i for i, uid in enumerate(order)}

    def anchors(self, below: str | None = None) -> list:
        """(layer, anchor effect, in the mask) of the anchor points, bottom first; `below`: only those a layer
        (or its effects) can reference, the anchors of the layers under it (Anchor Point page)."""
        order = self.stack_order()
        out = []
        for layer in sorted(self.walk(), key=lambda l: order[l.uid]):
            if below is not None and order[layer.uid] >= order.get(below, -1):
                continue
            out += [(layer, e, False) for e in layer.effects if e.kind == "anchor"]
            out += [(layer, e, True) for e in (layer.mask.effects if layer.mask else []) if e.kind == "anchor"]
        return out

    def references(self, anchor_uid: str) -> list:
        """(layer, effect or None, slot) of the channel sources referencing an anchor point (its Properties
        list them)."""
        out = []
        for layer in self.walk():
            owners = [(None, layer.material)] + [(e, {**e.material, **e.inputs}) for e in layer.effects]
            owners += [(e, {**e.material, **e.inputs}) for e in (layer.mask.effects if layer.mask else [])]
            for fx, sources in owners:
                out += [(layer, fx, slot) for slot, src in sources.items() if src.anchor == anchor_uid]
        return out

    def parent_list(self, uid: str) -> list | None:
        """The list holding the layer `uid` (the stack or a folder's children)."""
        if any(layer.uid == uid for layer in self.layers):
            return self.layers
        for layer in self.walk():
            if any(c.uid == uid for c in layer.children):
                return layer.children
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = [asdict(c) for c in self.channels]
        d["layers"] = [layer.to_dict() for layer in self.layers]
        return d

    @staticmethod
    def from_dict(d: dict) -> TextureSet:
        try:
            ts = TextureSet(**{k: v for k, v in d.items() if k not in ("channels", "layers")},
                            channels=[Channel(**c) for c in d.get("channels", [])],
                            layers=[Layer.from_dict(x) for x in d.get("layers", [])])
        except TypeError as e:
            raise ProjectError(f"invalid Texture Set: {e}") from None
        ts.validate()
        return ts

    def validate(self) -> None:
        ids = self.channel_ids()
        if len(set(ids)) != len(ids):
            raise ProjectError(f"{self.name}: a channel is listed twice")
        for c in self.channels:
            if c.id not in CHANNELS:
                raise ProjectError(f"{self.name}: unknown channel {c.id!r}")
            if c.format not in FORMATS:
                raise ProjectError(f"{self.name}: unknown format {c.format!r}")
        if any(s not in SIZES for s in self.size) or len(self.size) != 2:
            raise ProjectError(f"{self.name}: invalid size {self.size}")
        if (not self.tiles or self.tiles != sorted(set(self.tiles))
                or any(not 1001 <= t <= 9999 or t % 10 == 0 for t in self.tiles)):
            raise ProjectError(f"{self.name}: invalid UV Tiles {self.tiles}")
        if any(not isinstance(m, dict) or any(int(t) not in self.tiles for t in m) for m in self.mesh_maps.values()):
            raise ProjectError(f"{self.name}: invalid mesh maps")
        if any(int(t) not in self.tiles or len(v) != 2 or any(x not in SIZES for x in v)
               for t, v in self.tile_sizes.items()):
            raise ProjectError(f"{self.name}: invalid UV tile sizes {self.tile_sizes}")

    def size_of(self, udim: int) -> list:
        """A tile's resolution: its own (Texture Set settings, UV Tile), else the Texture Set's."""
        return list(self.tile_sizes.get(str(udim), self.size))

    def mesh_map(self, slot: str, udim: int) -> str | None:
        """The resource of a mesh map slot in a tile, or None."""
        return self.mesh_maps.get(slot, {}).get(str(udim))

    def set_mesh_map(self, slot: str, udim: int, name: str | None) -> None:
        tiles = self.mesh_maps.setdefault(slot, {})
        if name:
            tiles[str(udim)] = name
        else:
            tiles.pop(str(udim), None)
        if not tiles:
            del self.mesh_maps[slot]
        if self.normal_mixing not in NORMAL_MIXING or self.height_to_normal not in HEIGHT_TO_NORMAL \
                or self.ao_mixing not in AO_MIXING or self.uv_padding not in UV_PADDING:
            raise ProjectError(f"{self.name}: invalid mixing settings")
        uids = self.all_uids()
        if len(set(uids)) != len(uids):
            raise ProjectError(f"{self.name}: two layers share a uid")
        for layer in self.layers:
            layer.validate()


@dataclass
class ShaderSettings:
    """Shader settings window: the shader, one of shaders.TEMPLATES (a renderer model with its blending mode; the
    New Project's template), the Common and Parallax Occlusion Mapping parameters it supports,
    displacement and tessellation. Defaults from the shader libraries' page (AO Intensity 0.75, Emissive Intensity 1,
    Alpha threshold 0.33, POM off, Strength 1); the displacement and tessellation ones are ours (none)."""
    shader: str = shaders.DEFAULT_TEMPLATE
    ao_intensity: float = 0.75
    emissive_intensity: float = 1.0
    alpha_threshold: float = 0.33  # lib-alpha, the alpha test shaders'
    pom_enable: bool = False
    pom_strength: float = 1.0
    displacement_source: str = "Height"
    scale_unit: str = "Normalized"
    scale_amount: float = 0.0  # Normalized: fraction of the bounding box's largest side for a height of 1
    subdivision_mode: str = "Uniform"
    subdivision_count: int = 0  # levels: every triangle into 4 per level


@dataclass
class Project:
    mesh: str = ""  # file name of the mesh inside the project
    normal_format: str = "OpenGL (X+, Y+, Z+)"
    resolution: int = 2048  # New Project > Resolution: the default Texture Set size
    uv_tiles: str = ""  # New Project > UV Tile mode (UV_TILE_MODES), "" without the UV Tile workflow; set at creation
    texture_sets: list = field(default_factory=list)
    export: dict = field(default_factory=dict)  # Export textures > Save settings
    shader: ShaderSettings = field(default_factory=ShaderSettings)

    @property
    def directx(self) -> bool:
        return self.normal_format.startswith("DirectX")

    def texture_set(self, name: str) -> TextureSet:
        return next(ts for ts in self.texture_sets if ts.name == name)

    def to_dict(self) -> dict:
        return {"version": FORMAT_VERSION, "mesh": self.mesh, "normal_format": self.normal_format,
                "resolution": self.resolution, "uv_tiles": self.uv_tiles,
                "texture_sets": [ts.to_dict() for ts in self.texture_sets],
                "export": copy.deepcopy(self.export), "shader": asdict(self.shader)}

    @staticmethod
    def from_dict(d: dict) -> Project:
        if d.get("version") != FORMAT_VERSION:
            raise ProjectError(f"unsupported project version {d.get('version')!r}")
        p = Project(mesh=d.get("mesh", ""), normal_format=d.get("normal_format", NORMAL_FORMATS[1]),
                    resolution=int(d.get("resolution", 2048)), uv_tiles=d.get("uv_tiles", ""),
                    texture_sets=[TextureSet.from_dict(t) for t in d.get("texture_sets", [])],
                    export=d.get("export", {}))
        try:
            p.shader = ShaderSettings(**d.get("shader", {}))
        except TypeError as e:
            raise ProjectError(f"invalid shader settings: {e}") from None
        p.validate()
        return p

    def validate(self) -> None:
        if self.normal_format not in NORMAL_FORMATS:
            raise ProjectError(f"unknown normal format {self.normal_format!r}")
        if self.resolution not in SIZES:
            raise ProjectError(f"invalid resolution {self.resolution}")
        if self.uv_tiles not in ("", *UV_TILE_MODES):
            raise ProjectError(f"unknown UV Tile mode {self.uv_tiles!r}")
        if not self.uv_tiles and any(ts.tiles != [FIRST_TILE] for ts in self.texture_sets):
            raise ProjectError("UV Tiles on a Texture Set without the UV Tile workflow")
        sh = self.shader
        if sh.shader not in shaders.TEMPLATES:
            raise ProjectError(f"unknown shader {sh.shader!r}")
        if (sh.displacement_source != "Height" or sh.scale_unit != "Normalized" or sh.subdivision_mode != "Uniform"
                or not 0 <= sh.subdivision_count <= 4 or not 0.0 <= sh.alpha_threshold <= 1.0):
            raise ProjectError("invalid shader settings")
        names = [ts.name for ts in self.texture_sets]
        if len(set(names)) != len(names):
            raise ProjectError("two Texture Sets share a name")
        for ts in self.texture_sets:
            ts.validate()


def new_texture_set(material: str, resolution: int, template: str = shaders.DEFAULT_TEMPLATE) -> TextureSet:
    """A Texture Set for a mesh material with the channels of the project's template (shaders.TEMPLATES: what its
    shader reads), as a new project shows it."""
    ids = [channel_id(u) for u in shaders.TEMPLATES[template].channels]
    return TextureSet(name=material, material=material, size=[resolution, resolution],
                      channels=[Channel(c, CHANNELS[c].format) for c in ids])


def drop_unknown_shader(data: dict) -> list[str]:
    """A saved project's JSON made loadable when its shader is not one of shaders.TEMPLATES (a renamed or removed
    one): the default's instead, reported."""
    sh = data.get("shader")
    if isinstance(sh, dict) and sh.get("shader", shaders.DEFAULT_TEMPLATE) not in shaders.TEMPLATES:
        old, sh["shader"] = sh["shader"], shaders.DEFAULT_TEMPLATE
        return [f"project: the shader {old!r} no longer exists, {shaders.DEFAULT_TEMPLATE!r} instead"]
    return []


def remap_anchors(items: list, remap: dict) -> None:
    """Copies with new uids (`remap`: old uid -> new) of layers or effects: their channel sources referencing an
    anchor point copied with them follow the copy; the others keep their anchor."""
    layers = [x for item in items if isinstance(item, Layer) for x in item.walk()]
    effects = [e for item in items if isinstance(item, Effect) for e in [item]]
    effects += [e for x in layers for e in x.effects + (x.mask.effects if x.mask else [])]
    sources = [s for x in layers for s in x.material.values()]
    sources += [s for e in effects for s in (*e.material.values(), *e.inputs.values())]
    for s in sources:
        s.anchor = remap.get(s.anchor, s.anchor)


def new_layer(ts: TextureSet, kind: str, name: str) -> Layer:
    """A layer for the Texture Set's channels: Paint layers and folders blend every channel; a Fill layer's
    Material starts with every channel at its default value."""
    layer = Layer(uid=new_uid(), kind=kind, name=name)
    if kind == "fill":
        for ch in ts.channel_ids():
            d = CHANNELS[ch].default
            layer.material[ch] = ChannelSource(color=list(d) * 3 if len(d) == 1 else list(d))
    return layer


def unique_name(ts: TextureSet, base: str) -> str:
    """New layers are numbered: "Layer 1", "Fill layer 2"…"""
    names = {layer.name for layer in ts.walk()}
    k = 1
    while f"{base} {k}" in names:
        k += 1
    return f"{base} {k}"
