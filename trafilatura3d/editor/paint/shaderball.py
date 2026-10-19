"""Thumbnails of the Paint filters and generators (shown by the Graph Library and
the Paint Assets alike): the Paint/ categories, each node at work on the shader ball, rendered
by the Paint workspace itself (project, baked mesh maps, layer stack, 3D view), so a thumbnail shows what the
node does in use. The Graph nodes offered as effects show their cooked image or their initials
(trafilatura3d/thumbnails.py `cooked`).

The shader ball (shaderball/Shader-Ball.fbx, CC BY 4.0: shaderball/CREDITS.txt) is the low poly of the bake; its
high poly is the same file Catmull-Clark subdivided (meshio.load subdivide), so the maps carry no facets.
Recipes (ours), in black and white: a generator's mask is shown as it is (a white fill masked by it over black);
a filter is a Passthrough layer (as when dropped into the stack) over a two-gray material with a cellular height;
nodes that make color show it. The camera looks from
slightly above; the ball is unlit (its textures alone, the effect without any shading) and no environment is
seen: two renders over black and over white give the alpha."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

FBX = Path(__file__).resolve().parent / "shaderball" / "Shader-Ball.fbx"
RENDER, SIZE = 384, 96  # rendered, then reduced (antialiasing)
SUBDIVIDE = 2  # Catmull-Clark levels of the high poly
ELEVATION, AZIMUTH = 12.0, 25.0  # degrees
LIT = True  # shaded by the studio environment (False: the textures alone, flat)
FILL = 0.9  # the camera's distance from the workspace's framing (the whole ball and pedestal in view)
# Black and white: a generator's mask is seen as it is (white over black), a filter works on a gray
# material; a node that makes color shows it by itself.
GENERATOR_COLOR = [1.0, 1.0, 1.0]  # what a generator's mask reveals
BASE = {"basecolor": [0.0, 0.0, 0.0], "roughness": [0.45] * 3, "metallic": [0.0] * 3}
WARM, COOL = [0.28, 0.28, 0.28], [0.78, 0.78, 0.78]  # the filters' material, mixed by a noise
COLOR_NOISE, HEIGHT_NOISE = "node:clouds_soft", "node:cells_domes"
TILING = 3.0  # the filters' material repeated on the ball (finer, so a filter's work reads at thumbnail size)
# A node whose thumbnail with its defaults visibly changes less than SUBTLE of the ball's pixels (a change above
# CHANGE, 0-1) is shown stronger: each numeric parameter tried alone half way and all the way to its slider's
# ends; kept, the one changing the most pixels while still leaving a readable picture (not the whole ball
# changed, the material's contrast not flattened). Thumbnail only, not the node's defaults; values ours.
SUBTLE, CHANGE, MOST, KEEP_CONTRAST = 0.03, 0.06, 0.9, 0.5
MESH_MAPS = ("normal", "world_space_normals", "ambient_occlusion", "curvature", "position", "thickness", "id")


def effect_nodes() -> dict[str, str]:
    """{node type: "filter" or "generator"} of the nodes these thumbnails are for: the Paint filters and
    generators the Assets offer as effects (graph_nodes, so the two lists never part)."""
    from trafilatura3d.paint import graph_nodes as dn
    out = {n.type: "generator" for n in dn.generators() if _paint(n.type)}
    out.update({n.type: "filter" for n in dn.filters() if n.type not in out and _paint(n.type)})
    return out


def _high_poly(folder: Path) -> Path:
    """The shader ball subdivided, as an OBJ for the bakers (no UVs: the high needs none)."""
    from trafilatura3d import meshio
    lines, base = [], 0
    for o in meshio.load(FBX, subdivide=SUBDIVIDE)[0]:
        lines.append(f"o {o.name}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in o.pos]
        lines += ["f " + " ".join(str(base + k + 1) for k in tri) for tri in o.idx.reshape(-1, 3)]
        base += len(o.pos)
    path = folder / "shaderball_high.obj"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _project(window, folder: Path):
    """The Paint workspace on a shader ball project with its mesh maps baked from the subdivided high poly."""
    from trafilatura3d.editor.paint.document import PaintDocument
    from trafilatura3d.paint import baking
    from trafilatura3d.paint.session import Session
    ws = window.paint
    session = Session.new(FBX, 1024, evaluator=window.doc.ev)
    ws.set_document(PaintDocument(session))
    ts = ws.doc.ts
    cfg = baking.config(ts)
    cfg["common"].update(high_meshes=[str(_high_poly(folder))], use_low_as_high=False)
    for slot in cfg["bakers"]:
        cfg["bakers"][slot]["enabled"] = slot in MESH_MAPS
    baking.set_config(ts, cfg)
    baking.bake_texture_set(session, ts)
    return ws


def _stack(ts, node_type: str, kind: str):
    """(the layers showing a node: its category's recipe, the node's effect in them)."""
    from trafilatura3d.paint.model import ChannelBlend, ChannelSource, Effect, Mask, new_layer, new_uid
    proc = "node:" + node_type
    base = new_layer(ts, "fill", "Base")
    if kind == "generator":
        for ch, value in BASE.items():
            if ch in base.material:
                base.material[ch] = ChannelSource(color=list(value))
        top = new_layer(ts, "fill", "Generated")
        for ch in list(top.material):
            if ch not in ("basecolor", "roughness"):
                del top.material[ch]
        top.material["basecolor"] = ChannelSource(color=list(GENERATOR_COLOR))
        top.material["roughness"] = ChannelSource(color=[0.3] * 3)
        top.mask = Mask(background=0.0, effects=[Effect(new_uid(), "generator", node_type, procedural=proc)])
        return [base, top], top.mask.effects[0]
    # a filter: on a two-color material with a cellular height, as a Passthrough layer above it
    base.material["basecolor"] = ChannelSource(color=list(WARM))
    if "height" in base.material:
        base.material["height"] = ChannelSource(procedural=HEIGHT_NOISE)
    if "roughness" in base.material:
        base.material["roughness"] = ChannelSource(color=[0.5] * 3)
    base.projection.tiling = TILING
    cool = new_layer(ts, "fill", "Cool")
    for ch in list(cool.material):
        if ch != "basecolor":
            del cool.material[ch]
    cool.material["basecolor"] = ChannelSource(color=list(COOL))
    noise = Effect(new_uid(), "fill", "Noise")
    noise.material["mask"] = ChannelSource(procedural=COLOR_NOISE)
    noise.projection.tiling = TILING
    cool.mask = Mask(background=0.0, effects=[noise])
    flt = new_layer(ts, "fill", "Filter")
    flt.material.clear()
    for ch in ts.channel_ids():
        flt.blend[ch] = ChannelBlend("passthrough", 1.0)
    flt.effects.append(Effect(new_uid(), "filter", node_type, procedural=proc))
    return [base, cool, flt], flt.effects[0]


def _render(ws) -> np.ndarray:
    """The ball alone, RGBA uint8 SIZE x SIZE: over black and over white, the difference is the alpha."""
    look = ws.view3d.look
    shots = []
    for bg in (0.0, 1.0):
        look.background = ((bg,) * 3, (bg,) * 3)
        shots.append(ws.view3d.render_image(RENDER, RENDER)[..., :3].astype(np.float64) / 255.0)
    black, white = shots
    alpha = np.clip(1.0 - (white - black).mean(-1, keepdims=True), 0.0, 1.0)
    rgba = np.concatenate([black, alpha], -1)  # premultiplied over black
    k = RENDER // SIZE
    small = rgba.reshape(SIZE, k, SIZE, k, 4).mean((1, 3))
    a = small[..., 3:4]
    color = np.where(a > 1e-4, small[..., :3] / np.maximum(a, 1e-4), 0.0)
    return (np.concatenate([np.clip(color, 0, 1), a], -1) * 255 + 0.5).astype(np.uint8)


def _ball(a: np.ndarray) -> np.ndarray:
    return a[..., 3] > 127


def _changed(img: np.ndarray, plain: np.ndarray) -> float:
    """The fraction of the ball's pixels visibly changed by the node."""
    m = _ball(img) & _ball(plain)
    d = np.abs(img[..., :3].astype(float) - plain[..., :3].astype(float)).mean(-1) / 255.0
    return float((d[m] > CHANGE).mean()) if m.any() else 0.0


def _contrast(img: np.ndarray) -> float:
    m = _ball(img)
    return float(img[..., :3].astype(float).mean(-1)[m].std() / 255.0) if m.any() else 0.0


def _readable(img: np.ndarray, plain: np.ndarray) -> bool:
    """Not the whole ball changed, and the material's contrast (when it had some) not flattened."""
    if _changed(img, plain) > MOST:
        return False
    base = _contrast(plain)
    return base < 0.03 or _contrast(img) >= KEEP_CONTRAST * base


def _stronger(ws, node_type, effect, plain, current):
    """The readable thumbnail with one numeric parameter half way or all the way to a slider end that changes
    the most pixels, or None when none beats `current`. Returns (image, "param=value")."""
    from trafilatura3d.nodes import NODES
    from trafilatura3d.nodes.base import ParamKind
    best, best_score, label = None, _changed(current, plain), ""
    for name, p in NODES[node_type].params.items():
        if p.kind not in (ParamKind.FLOAT, ParamKind.INT) or p.min is None or p.max is None:
            continue
        if "seed" in name.lower() or p.editor == "Random seed":
            continue
        tried = set()
        for end in (p.max, p.min):
            for f in (0.5, 1.0):
                value = p.default + (end - p.default) * f
                value = int(round(value)) if p.kind is ParamKind.INT else float(value)
                if value == p.default or value in tried:
                    continue
                tried.add(value)
                effect.values = {name: value}
                ws.refresh_materials()
                img = _render(ws)
                score = _changed(img, plain)
                if score > best_score and _readable(img, plain):
                    best, best_score, label = img, score, f"{name}={value:g}"
    effect.values = {}
    return best, label


def rebuild(only: list[str] | None = None, out: Path | None = None, log=print) -> tuple[int, list[str]]:
    """Make the thumbnails of the Paint nodes (all, or `only`) into the Library's folder (or `out`);
    (made, failures "type: error"). Needs a QApplication: the work happens in a hidden window of its own (never
    shown nor closed, so it saves no settings), the open projects stay untouched."""
    import OpenImageIO as oiio
    from trafilatura3d.editor.mainwindow import MainWindow
    from trafilatura3d.editor.paint import color_picker
    from trafilatura3d.thumbnails import OUT
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    kinds = effect_nodes()
    types = [t for t in (only or sorted(kinds)) if t in kinds]
    failed = []
    sampler = color_picker.SAMPLER  # the hidden workspace registers itself as the eyedropper's: given back below
    window = MainWindow([])
    try:
        made = _rebuild(window, {t: kinds[t] for t in types}, out, failed, log, oiio)
    finally:
        color_picker.set_sampler(sampler)
        window.deleteLater()
    return made, failed


def _rebuild(window, types, out, failed, log, oiio) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        ws = _project(window, Path(tmp))
        v = ws.view3d
        v.free_rotation = False
        v.yaw, v.pitch = math.radians(AZIMUTH), math.radians(ELEVATION)
        ws.frame()
        v.distance *= FILL
        look = v.look
        look.env_opacity, look.ground, look.show_env, look.world_axes = 0.0, False, True, False
        look.preview = not LIT
        ts = ws.doc.ts
        for t in types:
            try:
                layers, effect = _stack(ts, t, types[t])
                ts.layers = layers
                ws.refresh_materials()
                img = _render(ws)
                effect.visible = False  # the stack without the node
                ws.refresh_materials()
                plain = _render(ws)
                effect.visible = True
                if _changed(img, plain) < SUBTLE:
                    stronger, label = _stronger(ws, t, effect, plain, img)
                    if stronger is not None:
                        img = stronger
                        log(f"{t}: shown with {label}")
                    else:
                        log(f"{t}: no parameter shows it more")
            except Exception as e:  # noqa: BLE001 - report and go on
                failed.append(f"{t}: {e}")
                continue
            _write(oiio, out, t, img)
        ws.doc.dirty = False
    return len(types) - len(failed)


def _write(oiio, out: Path, node_type: str, img: np.ndarray) -> None:
    buf = oiio.ImageBuf(oiio.ImageSpec(SIZE, SIZE, 4, oiio.UINT8))
    buf.set_pixels(oiio.ROI(), np.ascontiguousarray(img))
    buf.write(str(out / f"{node_type}.png"))


def _paint(node_type: str) -> bool:
    from trafilatura3d.nodes import NODES
    return NODES[node_type].category.startswith("Paint/")
