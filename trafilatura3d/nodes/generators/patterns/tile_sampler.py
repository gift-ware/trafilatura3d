import math

from trafilatura3d.nodes.base import Node

from .common import (B, C, E, F, F2, F4, FILTERING, HERE, I, Input, Output, Param, PatternNode, PortType, SYMMETRY,
                     color_variant, nse, pattern_choices, prefiltered)

PATTERN_PORTS = tuple(f"pattern_input_{i}" for i in range(1, 7))
OFFSET_TYPES = ("Horizontal Quincunx", "Vertical Quincunx", "Horizontal Global", "Vertical Global")
COLOR_PARAM_MODES = ("Color Input", "Color Map", "Scale", "Row Index", "Column Index", "Pattern Index",
                     "Pattern Center Position", "Pattern Center Position (RG) B-Sphere Size (B)")


class TileSampler(PatternNode):
    """Tile Generator's grid driven by maps: Scale, Displacement, Rotation, Vector, Color, Mask and
    Pattern Distribution inputs are read at each tile centre (the mask optionally at the centre and
    the four corners of the tile) and blended in by their multipliers. Size per Size Mode: Size in cells
    (Keep Ratio: of the smaller cell side), Size (Absolute) in image units, Size (Pixel) in whole pixels. Gray by default, colour when a
    colour pattern/colour map/background is connected or with Color Variant (Tile Sampler Color).
    Blending: Add/Sub (signed: values above 0.5 of the colour add, below subtract), Max, Alpha Blend (tiles over each
    other in index order, Reverse Rendering Order flips it). Color Parametrization: the tile colour is
    mixed towards the chosen per-tile value (gray modes use its luminance); "Color Input" = the Color
    map sampled at the tile centre (as in the gray variant), "Color Map" = the same map. Then
    the random part: Luminance Random (gray) or Color Random (colour: hue shift, saturation, lightness,
    alpha towards random values).

    Ours, how it is computed: each cell's tile once (`_Cells`), then every tile drawn as a quad over the background
    with the blending done by the GPU (`_Draw`), then the result clamped (`_Layer`): the work is what the tiles cover,
    and tiles of any size or spread are whole (gathering the cells around each pixel cut them past a few cells)."""
    counted_inputs = {"pattern_input_number": "pattern_input_"}

    type = "tile_sampler"
    main_input = "background_input"  # the library graph's primary input: Relative to input follows it
    label = "Tile Sampler"
    catalog = "tile_sampler"
    unsupported = {"Size (Absolute/Pixel)": "the documentation's older parameter; the node has Size, Size "
                                            "(Absolute) and Size (Pixel)",
                   "Color/Luminance Random": "the documentation's older parameter; here Luminance Random (grayscale) "
                                             "and Color Random (color)"}
    libs = ("tile.wgsl",)
    pattern_ports = PATTERN_PORTS
    inputs = {
        **{p: Input(PortType.ANY, f"Pattern Input {i}") for i, p in enumerate(PATTERN_PORTS, 1)},
        "scale_map_input": Input(PortType.GRAY, "Scale Map Input", (1.0, 1.0, 1.0, 1.0)),
        "displacement_map_input": Input(PortType.GRAY, "Displacement Map Input", (0.5, 0.5, 0.5, 1.0)),
        "rotation_map_input": Input(PortType.GRAY, "Rotation Map Input", (0.0, 0.0, 0.0, 1.0)),
        "vector_map_input": Input(PortType.COLOR, "Vector Map Input", (0.0, 0.0, 0.0, 1.0)),
        "color_map_input": Input(PortType.ANY, "Color Map Input", (1.0, 1.0, 1.0, 1.0)),
        "mask_map_input": Input(PortType.GRAY, "Mask Map Input", (1.0, 1.0, 1.0, 1.0)),
        "pattern_distribution_map_input": Input(PortType.GRAY, "Pattern Distribution Map Input", (0.0, 0.0, 0.0, 1.0)),
        "background_input": Input(PortType.ANY, "Background Input"),
    }
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        "x_amount": Param(I, 16, "X Amount", 1, 64),
        "y_amount": Param(I, 16, "Y Amount", 1, 64),
        **nse(),
        **color_variant(),
        # Pattern
        "pattern": Param(E, "Square", "Pattern", choices=pattern_choices("Pattern Input", half_bell="Half bell"),
                         group="Pattern"),
        "pattern_input_number": Param(I, 1, "Pattern Input Number", 1, 6, group="Pattern"),
        "pattern_input_distribution": Param(E, "Random", "Pattern Input Distribution",
                                            choices=("Random", "Pattern Number", "Distribution Map"), group="Pattern"),
        "pattern_input_filtering": Param(E, FILTERING[0], "Pattern Input Filtering", choices=FILTERING, group="Pattern"),
        "pattern_specific": Param(F, 0.0, "Pattern Specific", 0.0, 1.0, group="Pattern"),
        "pattern_specific_random": Param(F, 0.0, "Pattern Specific Random", 0.0, 1.0, group="Pattern"),
        "rotation_step": Param(E, "0", "Rotation", choices=("0", "90", "180", "270"), group="Pattern"),
        "rotation_step_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Pattern"),
        "symmetry_random": Param(F, 0.0, "Symmetry Random", 0.0, 1.0, group="Pattern"),
        "symmetry_random_mode": Param(E, SYMMETRY[0], "Symmetry Random Mode", choices=SYMMETRY, group="Pattern"),
        # Size
        "size_mode": Param(E, "Normal", "Size Mode", choices=("Normal", "Keep Ratio", "Absolute", "Pixel"), group="Size"),
        "size": Param(F2, [1.0, 1.0], "Size", 0.0, 1.0, group="Size"),
        "size_absolute": Param(F2, [0.1, 0.1], "Size (Absolute)", 0.0, 1.0, group="Size"),
        "size_pixel": Param(F2, [1.0, 1.0], "Size (Pixel)", 0.0, 4.0, group="Size"),
        "size_random": Param(F2, [0.0, 0.0], "Size Random", 0.0, 1.0, group="Size"),
        "scale": Param(F, 0.75, "Scale", 0.0, 10.0, group="Size"),
        "scale_random": Param(F, 0.0, "Scale Random", 0.0, 1.0, group="Size"),
        "scale_map_multiplier": Param(F, 0.0, "Scale Map Multiplier", 0.0, 1.0, group="Size"),
        "scale_vector_map_multiplier": Param(F, 0.0, "Scale Vector Map Multiplier", 0.0, 1.0, group="Size"),
        "scale_parametrization_affect": Param(E, "X and Y", "Scale Parametrization Affect",
                                              choices=("X and Y", "X", "Y"), group="Size"),
        # Position
        "position_random": Param(F, 0.0, "Position Random", 0.0, 10.0, group="Position"),
        "offset": Param(F, 0.0, "Offset", 0.0, 1.0, group="Position"),
        "offset_type": Param(E, "Horizontal Quincunx", "Offset Type", choices=OFFSET_TYPES, group="Position"),
        "global_offset": Param(F2, [0.0, 0.0], "Global Offset", 0.0, 1.0, group="Position"),
        "displacement_map_intensity": Param(F, 0.0, "Displacement Map Intensity", 0.0, 1.0, group="Position"),
        "displacement_angle": Param(F, 0.0, "Displacement Angle", 0.0, 1.0, group="Position"),
        "vector_map_displacement": Param(F, 0.0, "Vector Map Displacement", 0.0, 1.0, group="Position"),
        # Rotation
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0, group="Rotation"),
        "rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Rotation"),
        "rotation_map_multiplier": Param(F, 0.0, "Rotation Map Multiplier", 0.0, 1.0, group="Rotation"),
        "vector_map_multiplier": Param(F, 0.0, "Vector Map Multiplier", 0.0, 1.0, group="Rotation"),
        # Color (masking included, as on the doc page)
        "mask_map_threshold": Param(F, 0.0, "Mask Map Threshold", 0.0, 1.0, group="Color"),
        "mask_map_invert": Param(B, False, "Mask Map Invert", group="Color"),
        "mask_map_sampling_technique": Param(E, "Pattern Center", "Mask Map Sampling Technique",
                                             choices=("Pattern Center", "Pattern Bounding Box (slower)"), group="Color"),
        "mask_random": Param(F, 0.0, "Mask Random", 0.0, 1.0, group="Color"),
        "invert_mask": Param(B, False, "Invert Mask", group="Color"),
        "blending_mode": Param(E, "Max", "Blending Mode", choices=("Add/Sub", "Max", "Alpha Blend"), group="Color"),
        "color": Param(C, [1.0, 1.0, 1.0, 1.0], "Color", group="Color"),
        "color_luminance_random": Param(F, 0.0, "Luminance Random", 0.0, 1.0, group="Color"),
        "color_random": Param(F4, [0.0, 0.0, 0.0, 0.0], "Color Random", 0.0, 1.0, group="Color"),
        "color_parametrization_mode": Param(E, "Color Input", "Color Parametrization Mode", choices=COLOR_PARAM_MODES,
                                            group="Color"),
        "color_parametrization_multiplier": Param(F, 0.0, "Color Parametrization Multiplier", 0.0, 1.0, group="Color"),
        "color_parametrization_affect": Param(E, "RGB+Alpha", "Color Parametrization Affect",
                                              choices=("RGB+Alpha", "RGB only", "Alpha only"), group="Color"),
        "global_opacity": Param(F, 1.0, "Global Opacity", 0.0, 1.0, group="Color"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color", group="Color"),
        "reverse_rendering_order": Param(B, False, "Reverse Rendering Order", group="Color"),
    }

    def encode(self, ctx, job):
        v = job.values
        gray = int(job.is_gray("output"))
        w, h = job.size
        n = (max(v["x_amount"], 1), max(v["y_amount"], 1))
        frame = (w / min(w, h), h / min(w, h)) if v["non_square_expansion"] else (1.0, 1.0)
        reps = (round(frame[0]), round(frame[1]))
        vert = v["offset_type"] in ("Vertical Quincunx", "Vertical Global")
        records = (n[1] * reps[1], n[0] * reps[0]) if vert else (n[0] * reps[0], n[1] * reps[1])
        cells = [ctx.scratch(job, False, "32f", records) for _ in range(3)]
        ctx.dispatch(CELLS, job, {**v, "gray_": gray}, dsts={f"r{i}": t for i, t in enumerate(cells)})
        # one channel for a grayscale output (the final pass reads its red only): a quarter of the memory
        layer = ctx.scratch(job, bool(gray), "32f" if "float32-blendable" in ctx.ev.device.features else "16f")
        ctx.dispatch(LAYER, job, {"background_color": v["background_color"], "final_": 0}, dsts={"output": layer})
        k = math.ceil(_reach(v, n, min(w, h)) / min(frame)) or 1  # periods a tile can reach past the frame
        srcs = {f"rec{i}": (t, False) for i, t in enumerate(cells)}
        if v["pattern_input_filtering"] == FILTERING[0]:  # pat_sample's footprint (tg_aa / 2) on the largest tile
            srcs |= prefiltered(ctx, job, PATTERN_PORTS, 0.5 / (min(w, h) * min(_largest_half(v, n, min(w, h)))))
        ctx.draw(DRAW, job, {**v, "gray_": gray, "copies_": [float(k)] * 2}, layer,
                 n[0] * n[1] * reps[0] * reps[1] * (2 * k + 1) ** 2, srcs=srcs)
        ctx.dispatch(LAYER, job, {"background_color": v["background_color"], "final_": 1}, srcs={"layer": (layer, bool(gray))})


def _largest_half(v: dict, n: tuple[int, int], side: int) -> tuple[float, float]:
    """A tile's half extent per axis at its largest, in frame units (tile_sampler.wgsl: the random factors at 1, the
    scale map's at 1, the vector map's length up to sqrt(2))."""
    cs = (1.0 / n[0], 1.0 / n[1])
    mode = v["size_mode"]
    if mode == "Normal":
        s = (cs[0] * v["size"][0], cs[1] * v["size"][1])
    elif mode == "Keep Ratio":
        s = (min(cs) * v["size"][0], min(cs) * v["size"][1])
    elif mode == "Absolute":
        s = tuple(v["size_absolute"])
    else:
        s = (math.ceil(v["size_pixel"][0]) / side, math.ceil(v["size_pixel"][1]) / side)
    f = 0.5 * v["scale"] * max(1.0, 1.0 + 0.415 * v["scale_vector_map_multiplier"])
    return (s[0] * f, s[1] * f)


def _reach(v: dict, n: tuple[int, int], side: int) -> float:
    """How far from its cell's centre a tile can reach, in frame units (generously: every factor at its largest,
    the square's diagonal, the vector map's length up to sqrt(2))."""
    cs = (1.0 / n[0], 1.0 / n[1])
    half = max(_largest_half(v, n, side))
    return (2.0 * 1.415 * half + 1.5 * v["position_random"] * max(cs)
            + v["displacement_map_intensity"] + 1.415 * v["vector_map_displacement"])


class _Cells(TileSampler):
    """The Tile Sampler's first pass: each cell's tile, as records (tile_sampler_cells.wgsl)."""
    type = "tile_sampler_cells"
    registered = False
    # the maps only: what it does not read has no binding (the pipeline's layout leaves it out)
    inputs = {k: i for k, i in TileSampler.inputs.items() if k not in PATTERN_PORTS and k != "background_input"}
    outputs ={f"r{i}": Output(PortType.COLOR, f"Record {i}") for i in range(3)}
    internal_uniforms = {"gray_": I}
    libs = ("tile.wgsl", "tile_sampler.wgsl")
    pattern_ports = ()
    shader = "tile_sampler_cells.wgsl"


class _Draw(TileSampler):
    """The Tile Sampler's second pass: the tiles of the records drawn over the layer (tile_sampler_draw.wgsl)."""
    type = "tile_sampler_draw"
    registered = False
    inputs = {**TileSampler.inputs, **{f"rec{i}": Input(PortType.COLOR, f"Record {i}") for i in range(3)}}
    outputs = {}
    internal_uniforms = {"gray_": I, "copies_": F2}
    libs = ("tile.wgsl", "tile_sampler.wgsl")
    shader = "tile_sampler_draw.wgsl"

    def blend(self, values):
        """The layer's blending: Add/Sub adds (signed), Max keeps the larger, Alpha Blend draws over."""
        op = {"Add/Sub": ("one", "add"), "Max": ("one", "max")}.get(values["blending_mode"], ("one-minus-src-alpha", "add"))
        state = {"src_factor": "one", "dst_factor": op[0], "operation": op[1]}
        return {"color": state, "alpha": state}


class _Layer(Node):
    """The Tile Sampler's layer: the background written into it, then it clamped into the output
    (tile_sampler_layer.wgsl)."""
    type = "tile_sampler_layer"
    label = "Tile Sampler layer"
    registered = False
    inputs = {"background_input": Input(PortType.ANY, "Background Input"), "layer": Input(PortType.COLOR, "Layer")}
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {"background_color": TileSampler.params["background_color"]}
    internal_uniforms = {"final_": I}

    def wgsl(self):
        return (HERE / "tile_sampler_layer.wgsl").read_text(encoding="utf-8")


CELLS, DRAW, LAYER = _Cells(), _Draw(), _Layer()
