"""Baking mode: mesh maps baked per Texture Set with Trafilatura 3D's bakers (trafilatura3d/bake.py,
bake_params.py). The baked maps become project resources assigned to the Texture Set's mesh
map slots, which the stack (normal and AO mixing) and the export (Mesh maps) read.

The mesh maps and the bakers behind them: Normal (Normal Map from Mesh, always tangent space, in the
project's normal format), World space normal, ID (Color Map from Mesh), Ambient occlusion, Curvature (from
mesh), Position, Thickness, Height, Bent normals, Opacity. Common settings as the Common settings page lists
them (Output size, Dilation width, Apply diffusion, Use Low poly mesh as high poly mesh, High definition
meshes, Cage, distances, Average normals, Match and suffixes, Antialiasing). Not offered: the Automatic
cage, skew correction painting, Auto-rebake, the settings synchronization between Texture Sets, UV Tiles.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, fields

import numpy as np

from trafilatura3d import bake as bk
from trafilatura3d import meshio
from trafilatura3d.bake_params import BAKER_PARAMS, baker_defaults

# (mesh map slot, label, baker, fixed parameters)
BAKERS = [
    ("normal", "Normal", "normal_map_from_mesh", {"map_type": "Tangent Space"}),
    ("world_space_normals", "World space normal", "world_space_normals", {}),
    ("id", "ID", "color_map_from_mesh", {}),
    ("ambient_occlusion", "Ambient occlusion", "ambient_occlusion_from_mesh", {}),
    ("curvature", "Curvature", "curvature_from_mesh", {}),
    ("position", "Position", "position", {}),
    ("thickness", "Thickness", "thickness_map_from_mesh", {}),
    ("height", "Height", "height_map_from_mesh", {}),
    ("bent_normals", "Bent normals", "bent_normals_from_mesh", {}),
    ("opacity", "Opacity", "opacity_mask_from_mesh", {}),
]
BAKER_OF = {slot: baker for slot, _l, baker, _f in BAKERS}
FIXED = {slot: f for slot, _l, _b, f in BAKERS}
CAGES = ("Distance based", "Custom file")  # Automatic (experimental) is not offered
# Checked by default (ours: the maps the default filters use).
DEFAULT_CHECKED = ("normal", "world_space_normals", "id", "ambient_occlusion", "curvature", "position", "thickness")


def default_config(ts) -> dict:
    """A Texture Set's baking configuration: common settings and the bakers' check state and parameters."""
    common = asdict(bk.Settings(size=tuple(ts.size)))
    common["size"] = list(ts.size)
    common["cage"] = "Distance based"
    return {"common": common,
            "bakers": {slot: {"enabled": slot in DEFAULT_CHECKED, "params": {}} for slot, *_ in BAKERS}}


def config(ts) -> dict:
    """The Texture Set's saved configuration completed with the defaults."""
    cfg = default_config(ts)
    saved = ts.baking or {}
    cfg["common"].update({k: v for k, v in saved.get("common", {}).items() if k in cfg["common"]})
    for slot, b in saved.get("bakers", {}).items():
        if slot in cfg["bakers"]:
            cfg["bakers"][slot].update(b)
    return cfg


def editable_params(slot: str) -> dict:
    """The baker's parameters shown for a mesh map: the normal's map type and orientation are
    fixed, and the input maps are not offered."""
    baker = BAKER_OF[slot]
    hidden = set(FIXED[slot]) | {"normal_orientation"} if slot == "normal" else set(FIXED[slot])
    return {k: p for k, p in BAKER_PARAMS.get(baker, {}).items()
            if k not in hidden and getattr(p, "kind", "") != "file" and k != "world_space"}


def settings_of(common: dict, low_count: int) -> bk.Settings:
    names = {f.name for f in fields(bk.Settings)}
    s = bk.Settings(**{k: (tuple(v) if k == "size" else v) for k, v in common.items() if k in names})
    s.use_cage = common.get("cage") == "Custom file" and bool(s.cage_file)
    return s


def low_of(session, ts) -> list:
    """The project mesh restricted to the Texture Set's material (the bake's low poly)."""
    m = session.material_index(ts)
    out = []
    for o in session.objects:
        subs = [s for s in o.submeshes if s.material == m]
        if subs:
            out.append(meshio.MeshObject(o.name, o.pos, o.nrm, o.tan, o.uv, o.idx, subs, o.visible, o.colors,
                                         o.material_colors))
    return out


def _loaded(session, path) -> list:
    objects, _m, _c = meshio.load(path)
    meshio.apply_fit(objects, session.fit)  # in the project mesh's space
    return objects


def bake_texture_set(session, ts, slots=None, progress=None, baker=None, tiles=None) -> dict:
    """Bake the checked mesh maps of a Texture Set (or `slots`) in its UDIM tiles (or `tiles`); returns {slot:
    {tile (str): resource name}} and assigns them to its mesh map slots. A tile's resource is named
    TextureSetName_slot, .<UDIM> after it with the UV Tile workflow. `progress(done, total, label)` returning False
    stops."""
    from trafilatura3d.paint.model import tile_uv
    cfg = config(ts)
    common = cfg["common"]
    slots = [s for s, *_ in BAKERS if cfg["bakers"][s]["enabled"]] if slots is None else list(slots)
    low = low_of(session, ts)
    if not low:
        raise ValueError(f"{ts.name}: no geometry")
    s = settings_of(common, len(low))
    if s.use_low_as_high or not s.high_meshes:
        highs = list(session.objects)
        s.use_low_as_high = True
    else:
        highs = [o for p in s.high_meshes for o in _loaded(session, p)]
    cage = _loaded(session, s.cage_file) if s.use_cage else None
    baker = baker or bk.Baker(session.ev.device)
    job = baker.prepare(low, highs, cage, s)
    tiles = list(ts.tiles if tiles is None else tiles)
    work = [(slot, udim) for slot in slots for udim in tiles]
    out = {}
    for i, (slot, udim) in enumerate(work):
        label = next(lbl for sl, lbl, *_ in BAKERS if sl == slot)
        if progress is not None and progress(i, len(work), label if len(tiles) == 1 else f"{label} {udim}") is False:
            break
        params = {**cfg["bakers"][slot]["params"], **FIXED[slot]}
        if slot == "normal":
            params["normal_orientation"] = "DirectX" if session.project.directx else "OpenGL"
        symmetric = slot == "curvature" and params.get("automatic_tonemapping", True)
        if symmetric:  # baked in a fixed wide range, then normalized around flat (below)
            params.update(automatic_tonemapping=False, tonemapping_min=-CURV_RANGE, tonemapping_max=CURV_RANGE)
        img = baker.finished(job, BAKER_OF[slot], params, tile_uv(udim))
        if symmetric:
            img = _symmetric_curvature(np.asarray(img, np.float32))
        name = f"{ts.name}_{slot}" + (f".{udim}" if session.project.uv_tiles else "")
        session.resources.images.pop(name, None)
        session.resources.add(name, np.asarray(img, np.float32))
        ts.set_mesh_map(slot, udim, name)
        out.setdefault(slot, {})[str(udim)] = name
    if progress is not None:
        progress(len(work), len(work), "")
    return out


CURV_RANGE = 100.0


def _symmetric_curvature(img: np.ndarray) -> np.ndarray:
    """Auto tonemapping of the Curvature map scaled symmetrically: flat stays 0.5 gray, the strongest curvature
    found (convex or concave) reaches white or black. Ours: the documentation says the bounds come from the
    min/max found, and its maps show flat areas gray, which a min/max stretch loses on convex only meshes."""
    covered = img[..., 3] > 0
    k = img[..., 0] * (2 * CURV_RANGE) - CURV_RANGE
    top = float(np.abs(k[covered]).max()) if covered.any() else 1.0
    v = 0.5 + 0.5 * k / max(top, 1e-9)
    out = img.copy()
    out[..., 0] = out[..., 1] = out[..., 2] = np.clip(v, 0.0, 1.0)
    return out


def set_config(ts, cfg: dict) -> None:
    ts.baking = copy.deepcopy(cfg)


def defaults_of(slot: str) -> dict:
    return baker_defaults(BAKER_OF[slot])
