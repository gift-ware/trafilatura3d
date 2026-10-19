"""Shared pieces of the generators of this folder: the base class (shader = paint lib + includes + templates +
own file), parameter groups repeated across generators (global adjustments, custom texture, micro details,
directional gradients), and the multi-pass machinery (stages, soft blurs, curvature bands, jump flooding).

Algorithms and defaults are ours (the documentation gives neither, decision 6)."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

from trafilatura3d.nodes.base import Input, Node, Output, ParamKind, PortType
from trafilatura3d.nodes.paint._lib import DIR, LIB, MAIN, B, E, F, Generator, helper, soft_blur

HERE = Path(__file__).parent

# The grayscale conversion methods (the Grayscale Conversion generator's documented list).
GRAY_METHODS = ("Desaturation", "Luma", "Average", "Max", "Min", "Red channel", "Green channel", "Blue channel")
# The documentation does not list the mask layers' blending modes: ours are the Blend modes without Switch.
BLEND_MODES = ("Copy", "Add", "Subtract", "Multiply", "Add Sub", "Max", "Min", "Divide", "Overlay", "Screen",
               "Soft Light")
UNSET = "not used by the generator"


def group(params: dict, name: str) -> dict:
    """The parameters with their Properties group set."""
    return {k: dataclasses.replace(p, group=name) for k, p in params.items()}


class GenNode(Generator):
    """A generator of this folder. Shader: paint lib.wgsl, `includes` (paths relative to nodes/paint),
    `templates` ((file of this folder, port) with $port/$PORT replaced), the node's own .wgsl and MAIN."""

    includes: tuple = ("generators/common",)
    templates: tuple = ()

    def wgsl(self) -> str:
        own = Path(inspect.getfile(type(self))).with_suffix(".wgsl").read_text(encoding="utf-8")
        parts = [LIB, *((DIR / f"{i}.wgsl").read_text(encoding="utf-8") for i in self.includes)]
        for file, port in self.templates:
            parts.append((HERE / f"{file}.wgsl").read_text(encoding="utf-8").replace("$PORT", port.upper()).replace("$port", port))
        return "\n".join(parts + [own, MAIN])


# --- inputs: custom textures (Paint: resource slots) ----------------------------------------------------------

def custom(label: str, gray: bool = False) -> Input:
    """A custom texture input; unconnected, the part that uses it is left out (shaders test X_CONNECTED)."""
    return Input(PortType.GRAY if gray else PortType.COLOR, label, (0.0, 0.0, 0.0, 1.0))


MICRO_INPUTS = {"micro_normal": Input(PortType.COLOR, "Micro Normal", (0.5, 0.5, 1.0, 1.0)),
                "micro_height": Input(PortType.GRAY, "Micro Height", (0.5, 0.5, 0.5, 1.0))}


# --- parameter groups ---------------------------------------------------------------------------------------

def global_params() -> dict:
    return {"global_invert": B("Global Invert", False), "global_blur": F("Global Blur", 0.0, 0.0, 32.0),
            "global_balance": F("Global Balance", 0.5, 0.0, 1.0),
            "global_contrast": F("Global Contrast", 0.0, -1.0, 1.0)}


def texture_params(prefix: str, name: str, mode: str = "Multiply", non_square: bool = False) -> dict:
    """The custom texture group (`prefix` for the parameter names): Invert, Grayscale Conversion, Blending
    Mode, Scale, Contrast, Brightness, Triplanar, Triplanar Contrast (+ Non-Square Tiling)."""
    p = {f"{prefix}invert": B("Invert", False),
         f"{prefix}grayscale_conversion": E("Grayscale Conversion", GRAY_METHODS, "Luma"),
         f"{prefix}blending_mode": E("Blending Mode", BLEND_MODES, mode),
         f"{prefix}scale": F("Scale", 1.0, 0.1, 16.0),
         f"{prefix}contrast": F("Contrast", 0.0, -1.0, 1.0),
         f"{prefix}brightness": F("Brightness", 0.0, -1.0, 1.0),
         f"{prefix}triplanar": B("Triplanar", False),
         f"{prefix}triplanar_contrast": F("Triplanar Contrast", 0.5, 0.0, 1.0)}
    if non_square:
        p[f"{prefix}non_square_tiling"] = B("Non-Square Tiling", False)
    return group(p, name)


def micro_params(ao: bool = True, on: bool = False) -> dict:
    """Micro details. `on`: the Micro Height / Micro Normal toggles' default (True where a Use Micro Details
    switch already guards them)."""
    p = {"micro_height": B("Micro Height", on), "micro_normal": B("Micro Normal", on),
         "curvature_type": E("Curvature Type", ("Standard", "Sobel", "Smooth")),
         "curvature_intensity": F("Curvature Intensity", 1.0, 0.0, 4.0),
         "height_details_intensity": F("Height Details Intensity", 1.0, 0.0, 4.0)}
    if ao:
        p |= {"ao_radius": F("AO Radius", 0.3, 0.0, 1.0), "ao_depth": F("AO Depth", 0.5, 0.0, 1.0)}
    return group(p, "Micro Details")


def three_params(prefix: str, name: str, sub: str = "/") -> dict:
    """Right to Left, Top To Bottom, Front to Back amounts, then each one's Invert and Blending Mode (sub
    groups `name + sub + direction`)."""
    p = group({f"{prefix}right_to_left": F("Right to Left", 0.0, 0.0, 1.0),
               f"{prefix}top_to_bottom": F("Top To Bottom", 1.0, 0.0, 1.0),
               f"{prefix}front_to_back": F("Front to Back", 0.0, 0.0, 1.0)}, name)
    for short, label in (("rl", "Right to Left"), ("tb", "Top To Bottom"), ("fb", "Front to Back")):
        p |= group({f"{prefix}{short}_invert": B("Invert", False),
                    f"{prefix}{short}_blending_mode": E("Blending Mode", BLEND_MODES, "Max")}, name + sub + label)
    return p


# --- passes -------------------------------------------------------------------------------------------------

def source(ctx, job, port: str):
    """(texture, is_gray) of what is connected to `port`, or of the input's default value."""
    s = job.inputs.get(port)
    if s:
        return ctx.ev.texture(s[0], s[1]), s[0].is_gray(s[1])
    spec = job.node.inputs[port]
    return ctx.ev.constant(spec.default), spec.type is PortType.GRAY


BAND_RADII = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
BANDS = helper("gen_bands", "generators/bands.wgsl",
               {"curvature": Input(PortType.GRAY, default=(0.5, 0.5, 0.5, 1.0)),
                **{f"b{k}": Input(PortType.GRAY) for k in range(1, 7)}},
               {"output": Output(PortType.GRAY)}, internal={f"w{k}": ParamKind.FLOAT for k in range(7)}, main=True)


def curvature_bands(ctx, job, weights) -> tuple:
    """The job's `curvature` input with its seven bands of detail weighted (bands.wgsl): (texture, True)."""
    src = source(ctx, job, "curvature")
    srcs = {f"b{k + 1}": (soft_blur(ctx, job, *src, r), src[1]) for k, r in enumerate(BAND_RADII)}
    out = ctx.scratch(job, True)
    ctx.dispatch(BANDS, job, {f"w{k}": float(w) for k, w in enumerate(weights)}, srcs={"curvature": src, **srcs},
                 dsts={"output": out})
    return out, True


FLOOD = helper("gen_jfa_flood", "generators/jfa_flood.wgsl", {"seeds": Input(PortType.COLOR)},
               {"output": Output(PortType.COLOR)}, internal={"step_": ParamKind.INT})
SEED_ISLANDS = helper("gen_jfa_islands", "generators/jfa_islands.wgsl",
                      {"uv_islands": Input(PortType.COLOR, default=(0.0, 0.0, 0.0, 1.0))},
                      {"output": Output(PortType.COLOR)}, main=True)
SEED_MASK = helper("gen_jfa_mask", "generators/jfa_mask.wgsl", {"src": Input(PortType.GRAY)},
                   {"output": Output(PortType.COLOR)}, main=True)


def flood(ctx, job, seeds):
    """Jump flooding of a seeds texture (xy and zw sets, -1 = none) at the job's size: every pixel gets its
    nearest seed of each set (exact pixel coordinates). No wrap-around (UV space)."""
    w, h = job.size
    a, b = seeds, ctx.scratch(job, False)
    step = max(w, h) // 2
    while step >= 1:
        ctx.dispatch(FLOOD, job, {"step_": step}, srcs={"seeds": (a, False)}, dsts={"output": b})
        a, b, step = b, a, step // 2
    ctx.dispatch(FLOOD, job, {"step_": 1}, srcs={"seeds": (a, False)}, dsts={"output": b})  # fixes rare misses
    return b


def island_borders(ctx, job):
    """Per pixel, the nearest pixel on a UV island border (xy; -1 when there is none), from the job's
    `uv_islands` input: rgba32f texture."""
    seeds = ctx.scratch(job, False)
    ctx.dispatch(SEED_ISLANDS, job, {}, dsts={"output": seeds})
    return flood(ctx, job, seeds)


def mask_sides(ctx, job, src):
    """Per pixel, the nearest pixel of `src` (texture, is_gray) at 0.5 or more (xy) and below (zw)."""
    seeds = ctx.scratch(job, False)
    ctx.dispatch(SEED_MASK, job, {}, srcs={"src": src}, dsts={"output": seeds})
    return flood(ctx, job, seeds)


def extended(node, key: str, extra: dict, internal: dict | None = None, drop: tuple = ()) -> Node:
    """The node's shader as a helper pass with more inputs (results of earlier passes) and internal uniforms,
    `drop`: inputs only earlier passes read (an unused binding breaks the pipeline). Cached per class and key."""
    cls = type(node)
    cache = vars(cls).get("_extended")
    if cache is None:
        cache = cls._extended = {}
    if key not in cache:
        own = {k: i for k, i in cls.inputs.items() if k not in drop}
        cache[key] = type(f"_{cls.type}_{key}", (Node,), {
            "type": f"{cls.type}_{key}", "registered": False, "inputs": {**own, **extra},
            "outputs": dict(cls.outputs), "params": cls.params, "internal_uniforms": internal or {},
            "wgsl": lambda _self: node.wgsl()})()
    return cache[key]


SEEDS_INPUT = {"seeds_": Input(PortType.COLOR, default=(-1.0, -1.0, -1.0, -1.0))}


class BorderGenerator(GenNode):
    """A generator reading, besides its inputs, `seeds_`: per pixel the nearest UV island border pixel (xy, -1 when
    none), from its `uv_islands` input."""

    def encode(self, ctx, job):
        seeds = island_borders(ctx, job)
        ctx.dispatch(extended(self, "final", SEEDS_INPUT), job, job.values, srcs={"seeds_": (seeds, False)})


# --- staged generators ---------------------------------------------------------------------------------------

class Staged(GenNode):
    """A mask generator computed in three stages of its own shader (`p.stage_`):
    0 - the layers that have a blur, raw (R, G, B: the `channel_blurs` parameters) and the curvature layer (A);
    then each of R, G, B is blurred by its parameter (inputs `blur_r_`, `blur_g_`, `blur_b_`, all read with
    sample_; `layers_` is the unblurred texture);
    1 - the layers combined (the mask before the global adjustments);
    then blurred by Global Blur (input `mix_`);
    2 - Global Balance, Contrast and Invert.
    With `bands` (seven parameter names), the input `curv_` is the curvature map with its bands weighted. Only the
    stage inputs a node uses are declared (an unused binding breaks the pipeline): `curv_` with bands, `blur_x_`
    for the channels with a blur parameter, `layers_` with `reads_layers`."""

    channel_blurs: tuple = (None, None, None)
    bands: tuple = ()
    reads_layers: bool = False

    def stage_inputs(self) -> dict:
        ins = {"mix_": Input(PortType.GRAY)}
        if self.bands:
            ins["curv_"] = Input(PortType.GRAY, default=(0.5, 0.5, 0.5, 1.0))
        if self.reads_layers:
            ins["layers_"] = Input(PortType.COLOR)
        for ch, name in zip("rgb", self.channel_blurs):
            if name:
                ins[f"blur_{ch}_"] = Input(PortType.COLOR)
        return ins

    def encode(self, ctx, job):
        # With bands the curvature map is read by the bands pass only (its weighted copy is `curv_`).
        st = extended(self, "stages", self.stage_inputs(), {"stage_": ParamKind.INT},
                      drop=("curvature",) if self.bands else ())
        v = job.values
        srcs = {}
        if self.bands:
            srcs["curv_"] = curvature_bands(ctx, job, [v[n] for n in self.bands])
        if self.reads_layers or any(self.channel_blurs):
            layers = ctx.scratch(job, False)
            ctx.dispatch(st, job, {**v, "stage_": 0}, srcs=srcs, dsts={"output": layers})
            srcs["layers_"] = (layers, False)
            for ch, name in zip("rgb", self.channel_blurs):
                if name:
                    srcs[f"blur_{ch}_"] = (soft_blur(ctx, job, layers, False, v[name]), False)
        mixed = ctx.scratch(job, True)
        ctx.dispatch(st, job, {**v, "stage_": 1}, srcs=srcs, dsts={"output": mixed})
        srcs["mix_"] = (soft_blur(ctx, job, mixed, True, v["global_blur"]), True)
        ctx.dispatch(st, job, {**v, "stage_": 2}, srcs=srcs)
