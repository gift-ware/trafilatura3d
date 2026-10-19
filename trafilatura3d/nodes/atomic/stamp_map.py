"""Stamp Map (the FX-Map node): draws the patterns of its graph (trafilatura3d/fxmap.py). The tree is walked on the CPU;
the patterns are drawn on the GPU as instanced quads with hardware blending (Add Sub, Max, Alpha blend)
over the Background (input, or color), each with its opacity times Global opacity times Roughness's weight
for its level (fxmap.level_weights). Along a tiled axis a pattern is placed modulo the
tiling period and drawn again on the other side of every edge it crosses, so the result tiles. The layer is 16
bit float: the sum is clamped once, when stored."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wgpu

from trafilatura3d.fxmap import BLENDING, DEFAULT_FXMAP, graph_variables, level_weights, levels, walk
from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

_DIR = Path(__file__).parent
MAX_INPUTS = 8


class FxMap(Node):
    type = "stamp_map"
    label = "Stamp Map"
    category = "Atomic nodes"
    catalog = "fx_map"
    color_mode = "color_mode"
    inputs = {"background": Input(label="Background")}
    main_input = "background"
    outputs = {"output": Output()}
    params = {
        "color_mode": Param(ParamKind.ENUM, "Grayscale", "Color mode", choices=("Grayscale", "Color")),
        "background": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Background"),
        # x0 x1 y0 y1: the part of the FX-Map's space the image shows (it can be given in cells,
        # e.g. -15.5 16.5 for 32 cells; see _instances)
        "render_region": Param(ParamKind.FLOAT4, [0.0, 1.0, 0.0, 1.0], "Render region", 0.0, 1.0),
        "tiling_region": Param(ParamKind.FLOAT4, [0.0, 1.0, 0.0, 1.0], "Tiling region", 0.0, 1.0),
        "cull_outside": Param(ParamKind.BOOL, False, "Cull outside"),
        "roughness": Param(ParamKind.FLOAT, 1.0, "Roughness", 0.0, 1.0),  # fxmap.level_weights
        "global_opacity": Param(ParamKind.FLOAT, 1.0, "Global opacity", 0.0, 1.0),
        "fxmap": Param(ParamKind.FXMAP, DEFAULT_FXMAP, "Stamp Map graph"),
        "input_count": Param(ParamKind.INT, 0, "Input count", 0, MAX_INPUTS),
    }
    # The patterns are drawn over the background with each Quadrant's blending.
    unsupported = {"Blending mode": "a static parameter the documentation names without its choices"}

    def instantiate(self, values):
        n = values.get("input_count", 0)
        n = max(0, min(MAX_INPUTS, n)) if isinstance(n, int) else 0
        if n == len(self.inputs) - 1:
            return self
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.inputs = {"background": Input(label="Background"),
                        **{f"input_image_{i}": Input(label=f"Input image {i}") for i in range(n)}}
        return clone

    def encode(self, ctx, job):
        v = job.values
        values, types = graph_variables(job.extra.get("graph_params", {}))
        images = []
        for i in range(MAX_INPUTS):
            src = job.inputs.get(f"input_image_{i}")
            images.append(ctx.ev.texture(*src) if src else ctx.ev.constant((0.0, 0.0, 0.0, 0.0)))
        from trafilatura3d.compiler import system_variables
        context = job.extra.get("context")
        rows = rows_of(walk(v["fxmap"], values, types, job.size, int(job.seed), job.extra.get("node_variables"),
                            system_variables(context) if context is not None else None))
        if v["color_mode"] == "Grayscale":
            rows[:, 8:11] = gray_luminosity(rows[:, 8:12])[:, None]
        draw_patterns(ctx, job, rows, v, images, levels=levels(v["fxmap"]))


def draw_patterns(ctx, job, rows: np.ndarray, v: dict, images: list | None = None, dst=None,
                  levels: int | None = None, size: tuple[int, int] | None = None, clamp: bool = False,
                  cull=None) -> None:
    """The FX-Map engine: the patterns `rows` (`rows_of`: one row per pattern) drawn over the background (instances
    with the render and tiling regions, Global opacity, Roughness over `levels` quad-tree levels (default: the
    deepest row's), culling, tiling copies; `v`: the FX-Map node's values) into the job's output (or `dst`, of
    `size`: default the job's), clamped to 0-1 with `clamp` (for 8 and 16 bit FX-Maps; job outputs in those
    formats are clamped when stored anyway); `cull`: (texture, threshold), the patterns whose centre reads less there
    are left out. The FX-Map node and the noises made of patterns (noises_a/stamped.py) draw
    through it."""
    if levels is None:
        levels = int(rows[:, 15].max()) + 1 if len(rows) else 1
    size = size or job.size
    data, copies = _instances(rows, v, job.tiling, levels)
    renderer = _renderer(ctx.ev.device)
    # The result minus the background, in 32 bit floats where the device blends them: 16 bit integers step by
    # 1/65535, half floats by 1/2048 near 1 (quantised slopes on smooth domes: Cells 2 (Legacy)'s Mosaic).
    # A grayscale result's layer has one channel (the final pass reads its red only): a quarter of the memory.
    from trafilatura3d.evaluator import one_channel
    gray = one_channel(dst) if dst is not None else job.is_gray("output")
    layer = ctx.scratch(job, gray, renderer.depth, size)
    src = job.inputs.get("background")
    background = (ctx.ev.texture(*src), src[0].is_gray(src[1])) if src else (ctx.ev.constant(tuple(v["background"])),
                                                                              False)
    images = images or [ctx.ev.constant((0.0, 0.0, 0.0, 0.0))] * MAX_INPUTS
    cull = cull or (ctx.ev.constant((1.0, 1.0, 1.0, 1.0)), 0.0)

    def record(encoder):
        renderer.draw(encoder, layer, data, copies, images, background, size, cull)
    ctx.render(record, uses=[layer, *images, background[0], cull[0]])
    ctx.dispatch(FINAL, job, {**v, "clamped": int(clamp)}, srcs={"layer": (layer, gray)}, dsts={"output": dst} if dst is not None else None,
                 size=size if size != job.size else None)


def gray_luminosity(rgba: np.ndarray) -> np.ndarray:
    """A grayscale FX-Map's Luminosity given as an RGBA colour (k, 4): the RGB average (the atomic Grayscale
    Conversion's even split) bounded by the alpha. A gray value v reaches here as (v, v, v, 1) (constants) or
    (v, v, v, v) (functions) and stays v (the average times alpha would give v^2); a colour such as (0, 1, 1, rand)
    keeps its random coverage (the average alone is a constant 2/3, R alone draws nothing)."""
    return np.minimum(rgba[:, :3].mean(1), rgba[:, 3])


# A pattern's row: center (2), size (2), rotation, variation, pattern, image, color (4), filtering, alpha, blend,
# depth (the draw shader's 4 vec4; in the instances the depth is replaced by the pattern's opacity weight).
ROW = 16


def rows_of(draws) -> np.ndarray:
    """The rows of fxmap.Draw patterns, in order."""
    return np.array([[*d.center, *d.size, d.rotation, d.variation, d.pattern, d.image, *d.color, d.filtering,
                      d.alpha, d.blend, d.depth] for d in draws], np.float32).reshape(-1, ROW)


def _instances(rows: np.ndarray, v, tiling: int, levels: int) -> tuple[np.ndarray, Copies]:
    """Per-pattern data, with render region, opacity weight (Global opacity x Roughness) and the tiling's wrap, and
    its copies shifted by the tiling period (`Copies`: the draw makes them, each pattern followed by its copies
    that reach the image, the order Alpha blend needs; expanding them here took most of a large noise's time)."""
    # The render region is the part of the FX-Map's space the image shows; the tiling region repeats in that space.
    rx0, rx1, ry0, ry1 = v["render_region"]
    tx0, tx1, ty0, ty1 = v["tiling_region"]
    wx, wy = max(rx1 - rx0, 1e-6), max(ry1 - ry0, 1e-6)
    px, py = max((tx1 - tx0) / wx, 1e-3), max((ty1 - ty0) / wy, 1e-3)
    tx0, ty0 = (tx0 - rx0) / wx, (ty0 - ry0) / wy
    out = rows.astype(np.float32, copy=True)
    out[:, 0] = (out[:, 0] - rx0) / wx
    out[:, 1] = (out[:, 1] - ry0) / wy
    out[:, 2] /= wx
    out[:, 3] /= wy
    out[:, 15] = v["global_opacity"] * level_weights(v["roughness"], out[:, 15], levels)
    # Along a tiled axis a pattern lands where it falls modulo the tiling period, however far its offsets took it
    # (Crystal 2's disorder moves patterns several images away).
    if tiling & 1:
        out[:, 0] = tx0 + np.mod(out[:, 0] - tx0, px)
    if tiling & 2:
        out[:, 1] = ty0 + np.mod(out[:, 1] - ty0, py)
    # As many periods as the largest pattern spans (Mesh 2's capsules are 6 tiles long): the copies themselves are
    # the draw's (stamp_map_draw.wgsl: the instances are the patterns times their shifts, culled there).
    most = float(0.5 * np.hypot(out[:, 2], out[:, 3]).max()) if len(out) else 0.0
    kx = int(np.ceil(most / px)) + 1 if tiling & 1 else 0
    ky = int(np.ceil(most / py)) + 1 if tiling & 2 else 0
    return out, Copies(kx, ky, px, py, bool(v["cull_outside"]))


@dataclass(frozen=True)
class Copies:
    """A pattern's copies along the tiled axes (`_instances`): up to `kx` / `ky` periods `px` / `py` either way, in
    the order 0, +1, -1, +2, -2... (x, then y); those missing the image are culled, but with `cull_outside` off the
    pattern itself, drawn first."""
    kx: int
    ky: int
    px: float
    py: float
    cull_outside: bool

    def count(self) -> int:
        return (2 * self.kx + 1) * (2 * self.ky + 1)


class _Renderer:
    """Instanced pattern quads; one pipeline per blending mode and layer format (a grayscale result's layer has one
    channel: the final pass reads its red only)."""

    def __init__(self, device):
        self.device = device
        self.depth = "32f" if "float32-blendable" in device.features else "16f"  # the layer's (evaluator.py asks it)
        code = (_DIR.parent / "shapes.wgsl").read_text(encoding="utf-8") + (_DIR / "stamp_map_draw.wgsl").read_text(encoding="utf-8")
        self.module = device.create_shader_module(label="stamp_map_draw", code=code)
        self.bgl = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "read-only-storage"}}] + [
            {"binding": 1 + i, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float"}} for i in range(MAX_INPUTS + 1)] + [
            {"binding": 2 + MAX_INPUTS, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
            {"binding": 3 + MAX_INPUTS, "visibility": wgpu.ShaderStage.VERTEX,
             "texture": {"sample_type": "unfilterable-float"}}])
        self.layout = device.create_pipeline_layout(bind_group_layouts=[self.bgl])
        self._pipelines: dict[tuple[int, str], object] = {}

    def pipeline(self, blend: int, fmt: str):
        """The pipeline of a blending mode (BLENDING index) into a layer of format `fmt`."""
        if (blend, fmt) not in self._pipelines:
            one = "one"
            states = {
                "Add Sub": ({"src_factor": one, "dst_factor": one, "operation": "add"},) * 2,  # signed: the shader
                "Max": ({"src_factor": one, "dst_factor": one, "operation": "max"},) * 2,
                "Alpha blend": ({"src_factor": one, "dst_factor": "one-minus-src-alpha", "operation": "add"},) * 2,
            }
            color, alpha = states[BLENDING[blend]]
            self._pipelines[blend, fmt] = self.device.create_render_pipeline(
                layout=self.layout, vertex={"module": self.module, "entry_point": "vs"},
                fragment={"module": self.module, "entry_point": "fs",
                          "targets": [{"format": fmt, "blend": {"color": color, "alpha": alpha}}]},
                primitive={"topology": "triangle-list"})
        return self._pipelines[blend, fmt]

    def draw(self, encoder, target, data: np.ndarray, copies: Copies, images, background, size, cull):
        """`data`: a row per pattern (`_instances`), drawn with its `copies` (an instance each, the shader culls those
        missing the image); `background`: (texture, is gray), read by Max and Alpha blend (the layer holds the result
        minus it); `cull`: (texture, threshold), a pattern whose centre reads less is not drawn."""
        rp = encoder.begin_render_pass(color_attachments=[{
            "view": target.create_view(), "load_op": "clear", "store_op": "store", "clear_value": (0, 0, 0, 0)}])
        if len(data):
            n = copies.count()
            buf = self.device.create_buffer_with_data(data=data.tobytes(), usage=wgpu.BufferUsage.STORAGE)
            flags = self.device.create_buffer_with_data(data=np.array(
                [background[1], *size, cull[1], copies.px, copies.py, copies.kx, copies.ky,
                 float(copies.cull_outside), 0, 0, 0], np.float32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
            group = self.device.create_bind_group(layout=self.bgl, entries=[
                {"binding": 0, "resource": {"buffer": buf}}] + [
                {"binding": 1 + i, "resource": t.create_view()} for i, t in enumerate([*images, background[0]])] + [
                {"binding": 2 + MAX_INPUTS, "resource": {"buffer": flags}},
                {"binding": 3 + MAX_INPUTS, "resource": cull[0].create_view()}])
            rp.set_bind_group(0, group)
            blends = data[:, 14].astype(int)
            # runs of the same blending mode, in order (found at once: a loop over 800 000 patterns took 50 ms)
            starts = np.concatenate(([0], np.flatnonzero(np.diff(blends)) + 1, [len(blends)]))
            for start, end in zip(starts[:-1], starts[1:]):
                rp.set_pipeline(self.pipeline(int(blends[start]), target.format))
                rp.draw(6, int(end - start) * n, 0, int(start) * n)
        rp.end()


_RENDERERS: dict[int, _Renderer] = {}


def _renderer(device) -> _Renderer:
    if id(device) not in _RENDERERS:
        _RENDERERS[id(device)] = _Renderer(device)
    return _RENDERERS[id(device)]


class _Final(Node):
    """The node's output: the Background (input or color) plus the drawn layer."""
    type = "stamp_map_final"
    registered = False
    inputs = {"background": Input(), "layer": Input(PortType.COLOR)}
    outputs = {"output": Output()}
    params = {"background": FxMap.params["background"]}
    internal_uniforms = {"clamped": ParamKind.INT}

    def wgsl(self):
        return (_DIR / "stamp_map_final.wgsl").read_text(encoding="utf-8")


FINAL = _Final()
