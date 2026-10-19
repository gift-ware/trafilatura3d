from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect, helper, reduce_max

F, E, I, B, C = ParamKind.FLOAT, ParamKind.ENUM, ParamKind.INT, ParamKind.BOOL, ParamKind.COLOR
SIZE_MODES = ("Relative to Texture", "Relative to Shape BSphere", "Relative to Largest Shape",
              "Relative to Smallest Shape", "Fit Shape BBox")


class FloodFillMapper(Effect):
    """Places a pattern in every Flood Fill shape, like a Tile Sampler driven by the shapes. Pattern:
    one of the first Pattern Input Number inputs, at random (Jittering reshuffles), by shape size, or
    by the Distribution Map at the shape's centre. Size: the pattern's extent is the texture, the
    shape's bounding circle, the largest or smallest shape (found by a GPU reduction) or the shape's
    box, x Size (horizontally) x Scale (x Scale Map, x a random factor). Rotation (turns): global +
    map + random, Autoscale shrinks rotated patterns to fit. Position Offset (+ random) moves the
    pattern in texture or pattern units. Tiling Mode H+V repeats the pattern inside its shape.
    Grayscale: Luminance Range (contrast about 50% gray) and Offset with their randoms and the
    Luminance Offset Map; color: hue shifted by HSL Adjustment / Random, alpha by Alpha Adjustment /
    Random, normal maps' vectors turned with the pattern (Is Normal Map). Everything over Background
    Color (by the pattern's alpha in color); outside shapes and patterns the background shows."""
    counted_inputs = {"pattern_input_number": "pattern_input_"}

    type = "flood_fill_mapper"
    label = "Flood Fill Mapper"
    category = "Filters/Effects"
    catalog = "flood_fill_mapper"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill_bbox": Input(PortType.COLOR, "Flood Fill Bbox"),
              **{f"pattern_input_{k}": Input(PortType.ANY, f"Pattern Input {k}", default=(1.0, 1.0, 1.0, 1.0))
                 for k in range(1, 9)},
              "pattern_distribution_map": Input(PortType.GRAY, "Pattern Distribution Map"),
              "scale_map": Input(PortType.GRAY, "Scale Map", default=(1.0, 1.0, 1.0, 1.0)),
              "rotation_map": Input(PortType.GRAY, "Rotation Map"),
              "luminance_offset_map": Input(PortType.GRAY, "Luminance Offset Map")}
    outputs = {"output": Output()}
    params = {
        "tiling_mode": Param(E, "No Tiling", "Tiling Mode", choices=("No Tiling", "H+V")),
        "pattern_input_number": Param(I, 1, "Pattern Input Number", 1, 8, group="Pattern"),
        "pattern_distribution_mode": Param(E, "Random", "Pattern Distribution Mode",
                                           choices=("Random", "Shape Size", "Distribution Map Input"), group="Pattern"),
        "pattern_distribution_jittering": Param(F, 0.0, "Pattern Distribution Jittering", 0.0, 1.0, group="Pattern"),
        "size_mode": Param(E, "Fit Shape BBox", "Size Mode", choices=SIZE_MODES, group="Size"),
        "size": Param(F, 1.0, "Size", 0.0, 1.0, group="Size"),
        "scale": Param(F, 1.0, "Scale", 0.0, 1.0, group="Size"),
        "scale_map_multiplier": Param(F, 0.0, "Scale Map Multiplier", 0.0, 1.0, group="Size"),
        "scale_random": Param(F, 0.0, "Scale Random", -1.0, 1.0, group="Size"),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0, group="Rotation"),
        "rotation_map_multiplier": Param(F, 0.0, "Rotation Map Multiplier", 0.0, 1.0, group="Rotation"),
        "rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Rotation"),
        "rotation_autoscale": Param(B, False, "Rotation Autoscale", group="Rotation"),
        "position_offset": Param(F, 0.0, "Position Offset", 0.0, 1.0, group="Position"),
        "position_offset_alignment": Param(E, "Pattern", "Position Offset Alignment", choices=("Texture", "Pattern"),
                                           group="Position"),
        "position_offset_random": Param(F, 0.0, "Position Offset Random", 0.0, 1.0, group="Position"),
        "luminance_range": Param(F, 1.0, "Luminance Range", 0.0, 1.0, group="Color (Grayscale version)"),
        "luminance_range_random": Param(F, 0.0, "Luminance Range Random", 0.0, 1.0, group="Color (Grayscale version)"),
        "luminance_offset": Param(F, 0.0, "Luminance Offset", -1.0, 1.0, group="Color (Grayscale version)"),
        "luminance_offset_random": Param(F, 0.0, "Luminance Offset Random", 0.0, 1.0,
                                         group="Color (Grayscale version)"),
        "luminance_offset_map_multiplier": Param(F, 0.0, "Luminance Offset Map Multiplier", 0.0, 1.0,
                                                 group="Color (Grayscale version)"),
        "is_normal_map": Param(B, False, "Is Normal Map", group="Color (Color version)"),
        "normal_format": Param(E, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL"), group="Color (Color version)"),
        "hsl_adjustment": Param(F, 0.0, "HSL Adjustment", -1.0, 1.0, group="Color (Color version)"),
        "hsl_random": Param(F, 0.0, "HSL Random", -1.0, 1.0, group="Color (Color version)"),
        "alpha_adjustment": Param(F, 0.0, "Alpha Adjustment", -1.0, 1.0, group="Color (Color version)"),
        "alpha_random": Param(F, 0.0, "Alpha Random", -1.0, 1.0, group="Color (Color version)"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color"),
    }

    def encode(self, ctx, job):
        src = job.inputs.get("flood_fill_bbox")
        n = src[0].size if src else job.size
        prep = ctx.scratch(job, False, size=n)
        ctx.dispatch(PREP, job, {}, dsts={"out": prep}, size=n)
        stats = reduce_max(ctx, job, prep)
        ctx.dispatch(FINAL, job, job.values, srcs={"stats": (stats, False)})


PREP = helper("fx_ffm_prep", "_ffm_prep.wgsl", {"flood_fill_bbox": Input(PortType.COLOR)},
              {"out": Output(PortType.COLOR)})


class _Final(Effect):
    type = "fx_flood_fill_mapper_final"
    registered = False
    libs = ("_ff_read.wgsl",)
    inputs = {**FloodFillMapper.inputs, "stats": Input(PortType.COLOR)}
    outputs = FloodFillMapper.outputs
    params = FloodFillMapper.params


FINAL = _Final()
