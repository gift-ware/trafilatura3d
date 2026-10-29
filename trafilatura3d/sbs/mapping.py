"""How Designer's nodes map to ours: the atomic filters (written here from the documentation's parameter lists and the
identifiers the .sbs files use) and the library graphs (library.json, derived once from the library graphs' own
interfaces: identifiers, labels, drop-down items and defaults; no Designer graph is copied into Trafilatura 3D).

Both are `NodeMap`s: our node type, and for every Designer identifier of a parameter, input and output its
counterpart in our node. Enumerations are integers in a .sbs; `ParamMap.values` gives our choice for each."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trafilatura3d.nodes.base import Param, ParamKind

COLOR_MODE = {0: "Grayscale", 1: "Color"}  # Designer's colorswitch


@dataclass(frozen=True)
class ParamMap:
    key: str
    values: dict[int, str] | None = None  # an enumeration (or a boolean as a choice): Designer's value -> ours
    by_order: bool = False  # `values` assumed in the documented order (no Designer item list to check them)
    scale: float = 1.0  # our value = Designer's × scale (an older graph's 0-100 percentage of our 0-1)


@dataclass(frozen=True)
class NodeMap:
    type: str
    params: dict[str, ParamMap] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)  # Designer's default where ours differs (our values)
    numbered: tuple[str, str] = ("", "")  # numbered inputs "id", "id:1"...: (Designer id, our port prefix)
    depth: str = ""  # the Output Format a library graph gives whatever it inherits (depths.json), "" = inherited


def _p(key, values=None):
    return ParamMap(key, values)


def _index(choices) -> dict[int, str]:
    return dict(enumerate(choices))


# Designer's atomic filters (<filter v="...">) by the identifiers the files use. Enumerations follow the order of
# the documentation's lists, which is our choices' order.
ATOMIC = {
    "blend": NodeMap("blend", {
        "opacitymult": _p("opacity"),
        "blendingmode": _p("blending_mode", _index(("Copy", "Add", "Subtract", "Multiply", "Add sub", "Max", "Min",
                                                     "Switch", "Divide", "Overlay", "Screen", "Soft light"))),
        "colorblending": _p("alpha_blending", _index(("Use source alpha", "Ignore alpha", "Straight alpha blending",
                                                      "Premultiplied alpha blending"))),
        # the older boolean "Alpha blending" (blend.sbs drives it from a bool input of that label): off ignores alpha
        "alphablending": _p("alpha_blending", {0: "Ignore alpha", 1: "Use source alpha"}),
        "maskrectangle": _p("cropping_area")},
        {"source": "foreground", "destination": "background", "opacity": "opacity"}),
    "transformation": NodeMap("transformation_2d", {
        "matrix22": _p("transformation_matrix"), "offset": _p("offset"), "mattecolor": _p("matte_color"),
        "filtering": _p("filtering", _index(("Bilinear | Mipmap", "Bilinear", "Nearest"))),
        "mipmapmode": _p("mipmap_mode", _index(("Auto", "Manual"))), "manualmiplevel": _p("mipmap_level")},
        {"input1": "input"}),
    "levels": NodeMap("levels", {
        "levelinlow": _p("level_in_low"), "levelinhigh": _p("level_in_high"), "levelinmid": _p("level_in_mid"),
        "leveloutlow": _p("level_out_low"), "levelouthigh": _p("level_out_high"),
        "clampinterm": _p("intermediary_clamp")}, {"input1": "input"}),
    "directionalwarp": NodeMap("directional_warp", {"intensity": _p("intensity"), "warpangle": _p("warp_angle")},
                               {"input1": "input", "inputintensity": "intensity_input"}),
    "warp": NodeMap("warp", {"intensity": _p("intensity")}, {"input1": "input", "inputgradient": "gradient_input"}),
    "uniform": NodeMap("uniform_color", {"outputcolor": _p("output_color"), "colorswitch": _p("color_mode", COLOR_MODE)}),
    "hsl": NodeMap("hsl", {"hue": _p("hue"), "saturation": _p("saturation"), "luminosity": _p("lightness")},
                   {"input1": "input"}),
    "dirmotionblur": NodeMap("directional_blur", {"intensity": _p("intensity"), "mblurangle": _p("angle")},
                             {"input1": "input"}),
    "blur": NodeMap("blur", {"intensity": _p("intensity")}, {"input1": "input"}),
    "sharpen": NodeMap("sharpen", {"intensity": _p("intensity")}, {"input1": "input"}),
    # inversedy is the Normal format switch (the library graphs drive it with their DirectX 0 / OpenGL 1 input);
    # unset it is DirectX, the documentation's first choice. Unset intensity is 1 (measured: a file's Normal left at
    # its default is our Intensity 1, at every size; ours defaults to 10).
    "normal": NodeMap("normal", {
        "intensity": _p("intensity"), "inversedy": _p("normal_format", {0: "DirectX", 1: "OpenGL"}),
        "input2alpha": _p("alpha_channel_content", {0: "Force Alpha To 1", 1: "Fill Alpha With Input"})},
        {"input1": "input"}, defaults={"normal_format": "DirectX", "intensity": 1.0}),
    "gradient": NodeMap("gradient_map", {"colorswitch": _p("color_mode", COLOR_MODE), "gradientrgba": _p("gradient")},
                        {"input1": "input"}),
    "dyngradient": NodeMap("gradient_dynamic", {
        "coordinate": _p("gradient_input_position"),
        "uvselector": _p("gradient_orientation", _index(("Horizontal", "Vertical")))},
        {"input1": "grayscale_input", "input2": "gradient_input"}),
    "distance": NodeMap("distance", {
        "distance": _p("maximum_distance"), "colorswitch": _p("color_mode", COLOR_MODE),
        "combinedistance": _p("combine_source_distance", {0: "Only Source", 1: "Combine"})},
        {"mask": "mask_input", "source": "source_input"}),
    "grayscaleconversion": NodeMap("grayscale_conversion", {
        "channelsweights": _p("channel_weights"), "alphamult": _p("flatten_alpha")}, {"input1": "input"}),
    "shuffle": NodeMap("channel_shuffle", {
        f"channel{c}": _p(f"{name}_channel", _index(("Input 1 Red", "Input 1 Green", "Input 1 Blue", "Input 1 Alpha",
                                                      "Input 2 Red", "Input 2 Green", "Input 2 Blue", "Input 2 Alpha")))
        for c, name in (("red", "red"), ("green", "green"), ("blue", "blue"), ("alpha", "alpha"))},
        {"input1": "input_1", "input2": "input_2"}),
    "emboss": NodeMap("emboss", {"intensity": _p("intensity"), "lightangle": _p("light_angle"),
                                 "highlightcolor": _p("highlight_color"), "shadowcolor": _p("shadow_color")},
                      {"input1": "input", "inputgradient": "intensity_input"}),
    "bitmap": NodeMap("bitmap", {"colorswitch": _p("color_mode", COLOR_MODE), "bitmapresourcepath": _p("pkg_resource_path")}),
    "svg": NodeMap("svg", {"colorswitch": _p("color_mode", COLOR_MODE), "vectorresourcepath": _p("pkg_resource_path")}),
    "text": NodeMap("text", {"text": _p("text"), "colorswitch": _p("color_mode", COLOR_MODE), "fontdata": _p("font"),
                             "fontsize": _p("font_size"), "position": _p("position"),
                             "matrix22": _p("transformation"),
                             "alignment": _p("alignment", _index(("Left", "Center", "Right")))}),
    "curve": NodeMap("curve", {"curveluminance": _p("curve")}, {"input1": "input"}),
    "passthrough": NodeMap("dot", inputs={"input": "input"}),
    "pixelprocessor": NodeMap("pixel_processor", {"perpixel": _p("per_pixel_function"),
                                                  "colorswitch": _p("color_mode", COLOR_MODE)},
                              numbered=("input", "input_image_")),
    "fxmaps": NodeMap("stamp_map", {
        "background": _p("background"), "colorswitch": _p("color_mode", COLOR_MODE), "hurst": _p("roughness"),
        "opacitymult": _p("global_opacity"), "renderroi": _p("render_region"), "tilingroi": _p("tiling_region"),
        "culling": _p("cull_outside")}, {"background": "background"}, numbered=("inputpattern", "input_image_")),
}

# Base parameters, on every node (Designer identifier -> ours). Output Format values in the documentation's list
# order: the library graphs agree (3 where only 32 bits will do: summed area tables, spline point lists, flood fill
# positions; 2 for HDR render passes).
BASE = {"outputsize": "output_size", "format": "output_format", "pixelsize": "pixel_size",
        "pixelratio": "pixel_ratio", "tiling": "tiling", "randomseed": "seed"}
FORMATS = ("8", "16", "16f", "32f")
PIXEL_RATIOS = ("stretch", "square")
TILINGS = ("none", "h", "v", "hv")


def _node_map(d: dict, depth: str = "") -> NodeMap:
    return NodeMap(d["type"], {k: ParamMap(p["key"], {int(v): c for v, c in p["values"].items()} if "values" in p
                                           else None, p.get("order", False), p.get("scale", 1.0))
                           for k, p in d["params"].items()},
                   dict(d.get("inputs", {})), dict(d.get("outputs", {})), dict(d.get("defaults", {})), depth=depth)


_LIBRARY = json.loads((Path(__file__).parent / "library.json").read_text(encoding="utf-8"))
# The library graphs whose outputs keep a format of their own whatever they inherit (nodes inside them set to
# Absolute): 8, 16 or 32f, measured by rendering each one alone in an 8-bit graph, its inputs fed at 8 and at 32f
# bits; with several outputs, the most precise forced one.
_DEPTHS = json.loads((Path(__file__).parent / "depths.json").read_text(encoding="utf-8"))
LIBRARY: dict[str, NodeMap] = {k: _node_map(v, _DEPTHS.get(k, ""))
                               for k, v in _LIBRARY["graphs"].items()}  # "file.sbs#graph" -> map
LIBRARY_PACKAGES: set[str] = set(_LIBRARY["packages"])  # file names of Designer's library packages
# Designer's library functions ("file.sbs#function") -> our Library function nodes, whose ports are their inputs.
LIBRARY_FUNCTIONS: dict[str, str] = _LIBRARY["functions"]


def convert(value: Any, param: Param, pm: ParamMap) -> Any:
    """A Designer constant as a value of our parameter; raises ValueError (or KeyError for an enumeration value
    with no counterpart) when it can't be one."""
    k = param.kind
    if pm.values is not None:
        return param.validate(pm.values[int(value[0] if isinstance(value, list) else value)])
    if k is ParamKind.ENUM:
        raise ValueError("an enumeration without its values")
    scalar = value[0] if isinstance(value, list) and len(value) == 1 else value
    if k is ParamKind.FLOAT:
        return param.validate(float(scalar) * pm.scale)
    if k is ParamKind.INT:
        return param.validate(int(round(scalar)))
    if k is ParamKind.BOOL:
        return bool(scalar)
    n = {ParamKind.FLOAT2: 2, ParamKind.FLOAT3: 3, ParamKind.FLOAT4: 4, ParamKind.INT2: 2, ParamKind.COLOR: 4}.get(k)
    if n is not None:
        v = [float(scalar)] * n if not isinstance(scalar, list) else [float(x) for x in scalar]
        if k is ParamKind.COLOR and len(v) == 1:
            v = v * 3 + [1.0]
        if k is ParamKind.COLOR and len(v) == 3:
            v.append(1.0)
        if len(v) != n:
            raise ValueError(f"{len(v)} components for a {k.value}")
        return param.validate([int(round(x)) for x in v] if k is ParamKind.INT2 else v)
    return param.validate(value)
