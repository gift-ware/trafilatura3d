"""Shape Splatter and its companions (Blend, Data Extract, to Mask).

Shape Splatter places one pattern per cell of an X Amount x Y Amount grid onto a background
height map, in two passes (17 inputs exceed the 16 textures a shader may bind):

1. tiles pass (helper, one invocation per tile): samples the driving maps at the tile centre and
   writes four rgba32f X x Y textures:
   a = (centre offset x, y in uv, rotation in turns, pattern index 0-7 or -1 when masked),
   b = (half extent x, y in uv, random in (0, 1], base height at the centre),
   c = (height scale, background height at the centre, background gradient x, y per uv),
   d = (uniform scale, tile index, 0, 0);
2. render pass (helper): the tile-grid engine (tiles.wgsl) gathers the tiles around each pixel;
   each tile's height there = base (rigid, skewed along the background slope, or conformed to the
   smoothed background) + pattern value x height scale; the highest tile wins.

Splatter Data encoding (designed here, the documentation gives none), per pixel, of the topmost tile:
- Splatter Data 1: R, G = pattern uv (0-1) of the pixel in that tile; B = pattern id 1-8 (0 = no
  tile); A = the tile's random value in (0, 1].
- Splatter Data 2: R = tile rotation (turns, 0-1); G = tile uniform scale; B = tile base height;
  A = tile index / (X Amount x Y Amount).
The outputs default to 32-bit float so the data survive.

Tiles are kept within 2.5 cells of their cell (the gather reach): Out of Bounds Option "Scale
Shape" shrinks tiles that would exceed it, "Constrain Position" first pulls them back."""

from .common import B, E, F, F2, F3, I, Input, Output, Param, PatternNode, PortType

PATTERNS = tuple(f"pattern_{i}" for i in range(1, 9))
MAPS = {
    "pattern_distribution": Input(PortType.GRAY, "Pattern Distribution"),
    "shape_scale": Input(PortType.GRAY, "Shape Scale", default=(1.0, 1.0, 1.0, 1.0)),
    "shape_rotation": Input(PortType.GRAY, "Shape Rotation"),
    "height_offset": Input(PortType.GRAY, "Height Offset"),
    "height_scale": Input(PortType.GRAY, "Height Scale", default=(1.0, 1.0, 1.0, 1.0)),
    "mask_random": Input(PortType.GRAY, "Mask Random", default=(1.0, 1.0, 1.0, 1.0)),
    "vector_map": Input(PortType.COLOR, "Vector Map", default=(0.5, 0.5, 0.5, 1.0)),
}
BACKGROUND = {"background_height": Input(PortType.GRAY, "Background Height")}
TILE_TEX = ("tiles_a", "tiles_b", "tiles_c", "tiles_d")
DATA_IN = {"splatter_data_1": Input(PortType.COLOR, "Splatter Data 1"),
           "splatter_data_2": Input(PortType.COLOR, "Splatter Data 2")}


class ShapeSplatter(PatternNode):
    counted_inputs = {"pattern_input_number": "pattern_"}
    type = "shape_splatter"
    label = "Shape Splatter"
    catalog = "shape_splatter"
    unsupported = {"Size by X/Y Amount Ratio": "the documentation's name of Preserve Aspect Ratio (the node's)",
                   "Height Offset Map Mulitplier": "the documentation's typo of Height Offset Map Multiplier"}
    description = __doc__
    inputs = {**BACKGROUND, **{p: Input(PortType.GRAY, f"Pattern {i}") for i, p in enumerate(PATTERNS, 1)}, **MAPS}
    main_input = "background_height"
    outputs = {"height": Output(PortType.GRAY, "Height"),
               "splatter_data_1": Output(PortType.COLOR, "Splatter Data 1"),
               "splatter_data_2": Output(PortType.COLOR, "Splatter Data 2")}
    base_defaults = {"output_format": {"mode": "absolute", "value": "32f"}}
    params = {
        "x_amount": Param(I, 16, "X Amount", 1, 64),
        "y_amount": Param(I, 16, "Y Amount", 1, 64),
        # Pattern
        "pattern_input_number": Param(I, 1, "Pattern Input Number", 1, 8, group="Pattern"),
        "pattern_distribution_mode": Param(E, "Random", "Pattern Distribution Mode",
                                           choices=("Random", "Pattern Index", "Line Index", "Column Index"),
                                           group="Pattern"),
        "pattern_distribution_map_multiplier": Param(F, 0.0, "Pattern Distribution Map Multiplier", 0.0, 1.0,
                                                     group="Pattern"),
        "pattern_rotation": Param(E, "0", "Pattern Rotation", choices=("0", "90", "180", "270"), group="Pattern"),
        "pattern_rotation_random": Param(F, 0.0, "Pattern Rotation Random", 0.0, 1.0, group="Pattern"),
        # Size
        "scale": Param(F, 1.0, "Scale", 0.0, 5.0, group="Size"),
        "scale_random": Param(F, 0.0, "Scale Random", 0.0, 1.0, group="Size"),
        "scale_no_overlap": Param(F, 0.0, "Scale No Overlap", 0.0, 1.0, group="Size"),
        "scale_map_multiplier": Param(F, 0.0, "Scale Map Multiplier", 0.0, 1.0, group="Size"),
        "size": Param(F2, [1.0, 1.0], "Size", 0.0, 1.0, group="Size"),
        "size_ratio_from_bg_slope": Param(F, 0.0, "Size Ratio from Bg Slope", 0.0, 1.0, group="Size"),
        "preserve_aspect_ratio": Param(F, 0.0, "Preserve Aspect Ratio", 0.0, 1.0, group="Size"),
        # Position
        "position_random": Param(F, 0.0, "Position Random", 0.0, 2.0, group="Position"),
        "random_distribution": Param(E, "Gaussian", "Random Distribution", choices=("Gaussian", "Uniform"),
                                     group="Position"),
        "vector_map_multiplier": Param(F, 0.0, "Vector Map Multiplier", 0.0, 1.0, group="Position"),
        "offset_horizontal": Param(F, 0.0, "Offset Horizontal", -2.0, 2.0, group="Position"),
        "offset_vertical": Param(F, 0.0, "Offset Vertical", -2.0, 2.0, group="Position"),
        "out_of_bounds_option": Param(E, "Scale Shape", "Out of Bounds Option",
                                      choices=("Scale Shape", "Constrain Position"), group="Position"),
        # Rotation
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0, group="Rotation"),
        "rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Rotation"),
        "rotation_from_bg_slope": Param(F, 0.0, "Rotation from Bg Slope", 0.0, 1.0, group="Rotation"),
        "rotation_map_multiplier": Param(F, 0.0, "Rotation Map Multiplier", 0.0, 1.0, group="Rotation"),
        "vector_map_multiplier_rotation": Param(F, 0.0, "Vector Map Multiplier", 0.0, 1.0, group="Rotation"),
        # Height
        "height_scale_auto_adjust": Param(B, True, "Height Scale Auto Adjust", group="Height"),
        "height_offset": Param(F, 0.0, "Height Offset", -1.0, 1.0, group="Height"),
        "height_offset_random": Param(F, 0.0, "Height Offset Random", 0.0, 1.0, group="Height"),
        "height_offset_map_multiplier": Param(F, 0.0, "Height Offset Map Multiplier", 0.0, 1.0, group="Height"),
        "height_scale": Param(F, 1.0, "Height Scale", 0.0, 1.0, group="Height"),
        "height_scale_random": Param(F, 0.0, "Height Scale Random", 0.0, 1.0, group="Height"),
        "height_scale_map_multiplier": Param(F, 0.0, "Height Scale Map Multiplier", 0.0, 1.0, group="Height"),
        "conform_to_background": Param(F, 0.0, "Conform to Background", 0.0, 1.0, group="Height"),
        "smooth_conformed_background": Param(F, 0.0, "Smooth Conformed Background", 0.0, 2.0, group="Height"),
        "skew_from_bg_slope": Param(F, 0.0, "Skew from Bg Slope", 0.0, 1.0, group="Height"),
        "background_slope_smoothness": Param(F, 0.0, "Background Slope Smoothness", 0.0, 2.0, group="Height"),
        "cutout_black_pixels": Param(B, True, "Cutout Black Pixels", group="Height"),
        "flatten_pattern_base": Param(B, False, "Flatten Pattern Base", group="Height"),
        # Masking
        "mask_random": Param(F, 0.0, "Mask Random", 0.0, 1.0, group="Masking"),
        "mask_random_map_multiplier": Param(F, 0.0, "Mask Random Map Multiplier", 0.0, 1.0, group="Masking"),
        "mask_from_bg_slope": Param(F, 0.0, "Mask from Bg Slope", -1.0, 1.0, group="Masking"),
    }

    def encode(self, ctx, job):
        n = (max(1, int(job.values["x_amount"])), max(1, int(job.values["y_amount"])))
        tiles = {k: ctx.scratch(job, False, "32f", size=n) for k in TILE_TEX}
        ctx.dispatch(TILES, job, job.values, dsts=tiles, size=n)
        ctx.dispatch(RENDER, job, job.values, srcs={k: (t, False) for k, t in tiles.items()})


class _Tiles(PatternNode):
    """Per-tile data from the driving maps, one invocation per tile."""
    type = "shape_splatter_tiles"
    registered = False
    libs = ("shape_splatter_lib.wgsl",)
    shader = "shape_splatter_tiles.wgsl"
    inputs = {**BACKGROUND, **MAPS}
    outputs = {k: Output(PortType.COLOR) for k in TILE_TEX}
    params = ShapeSplatter.params


class _Render(PatternNode):
    """The tiles gathered per pixel: height and Splatter Data."""
    type = "shape_splatter_render"
    registered = False
    libs = ("tile.wgsl", "tiles.wgsl", "shape_splatter_lib.wgsl", "shape_splatter_patterns.wgsl")
    shader = "shape_splatter_render.wgsl"
    pattern_ports = PATTERNS
    inputs = {**BACKGROUND, **{p: Input(PortType.GRAY) for p in PATTERNS},
              **{k: Input(PortType.COLOR) for k in TILE_TEX}}
    outputs = ShapeSplatter.outputs
    params = ShapeSplatter.params


TILES, RENDER = _Tiles(), _Render()


class ShapeSplatterBlend(PatternNode):
    """Colour or grayscale map from Shape Splatter Data: each pixel of a tile shows its pattern
    (Pattern 1..Pattern Number, by the tile's pattern id, or a random one for a proportion of tiles)
    at the tile uv in the data, mixed with the Color Input by its opacity, then adjusted per tile:
    grayscale + Luminance Adjustment + a random [0, 1) x Luminance Random; colour in
    HSL, hue shifted by H, saturation and lightness moved towards 0 (negative) or 1 (positive) by S and L
    of HSL Adjustment + random x HSL Random. Is Normal Map: patterns are normal maps (Normal Format)
    whose XY are rotated with their tile (+ random angle). Unconnected patterns show a paraboloid, like
    Shape Splatter's default."""
    counted_inputs = {"pattern_number": "pattern_"}

    type = "shape_splatter_blend"
    label = "Shape Splatter Blend"
    catalog = "shape_splatter_blend"
    unsupported = {"HSL/Luminance Adjustment": "the documentation's parameter; the node has Luminance "
                                               "Adjustment (grayscale) and HSL Adjustment (color)",
                   "HSL/Luminance Random": "the documentation's parameter; the node has Luminance Random "
                                           "(grayscale) and HSL Random (color)"}
    shader = "shape_splatter_blend.wgsl"
    libs = ("shape_splatter_patterns.wgsl",)
    pattern_ports = PATTERNS
    inputs = {"background_color": Input(PortType.ANY, "Background Color", default=(0.0, 0.0, 0.0, 1.0)),
              **{p: Input(PortType.ANY, f"Pattern {i}") for i, p in enumerate(PATTERNS, 1)},
              "color_input": Input(PortType.ANY, "Color Input", default=(1.0, 1.0, 1.0, 1.0)),
              **DATA_IN}
    main_input = "splatter_data_1"
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        "pattern_number": Param(I, 1, "Pattern Number", 1, 8),
        "random_pattern_assignment": Param(F, 0.0, "Random Pattern Assignment", 0.0, 1.0),
        "is_normal_map": Param(B, False, "Is Normal Map"),
        "normal_format": Param(E, "DirectX", "Normal Format", choices=("DirectX", "OpenGL")),
        "luminance_adjustment": Param(F, 0.0, "Luminance Adjustment", -1.0, 1.0),
        "luminance_random": Param(F, 0.0, "Luminance Random", -1.0, 1.0),
        "hsl_adjustment": Param(F3, [0.0, 0.0, 0.0], "HSL Adjustment", -1.0, 1.0),
        "hsl_random": Param(F3, [0.0, 0.0, 0.0], "HSL Random", -1.0, 1.0),
        "normal_angle_random": Param(F, 0.0, "Normal Angle Random", 0.0, 1.0),
        "color_input_opacity": Param(F, 0.0, "Color Input Opacity", 0.0, 1.0),
    }


class ShapeSplatterDataExtract(PatternNode):
    """Splits Shape Splatter Data into readable maps (outputs designed here; a node has at most four
    outputs): Pattern ID (id / 8, 0 = no tile), Pattern UV (colour: R, G = uv in the tile, B = tile
    index / tile count), Random (the tile's random value), Transform (colour: R = rotation in turns,
    G = scale, B = tile base height)."""

    type = "shape_splatter_data_extract"
    label = "Shape Splatter Data Extract"
    catalog = "shape_splatter_data_extract"
    shader = "shape_splatter_data_extract.wgsl"
    inputs = DATA_IN
    outputs = {"pattern_id": Output(PortType.GRAY, "Pattern ID"),
               "pattern_uv": Output(PortType.COLOR, "Pattern UV"),
               "random": Output(PortType.GRAY, "Random"),
               "transform": Output(PortType.COLOR, "Transform")}
    base_defaults = {"output_format": {"mode": "relative_input", "value": "8"}}  # the Data's format: never cut at 1


class ShapeSplatterToMask(PatternNode):
    """Mask of the tiles whose pattern id is in [Start, End] (either order), minus a random
    proportion of tiles. Output: Binary Mask (0/1), Integer Mask (the pattern id, 1-8, as an HDR
    value), Grayscale Values (ids spread over (0, 1] across the range)."""

    type = "shape_splatter_to_mask"
    label = "Shape Splatter to Mask"
    catalog = "shape_splatter_to_mask"
    shader = "shape_splatter_to_mask.wgsl"
    inputs = DATA_IN
    base_defaults = {"output_format": {"mode": "relative_input", "value": "8"}}  # the Data's format: never cut at 1
    params = {
        "pattern_id_start_range": Param(I, 1, "Pattern ID Start Range", 1, 8),
        "pattern_id_end_range": Param(I, 1, "Pattern ID End Range", 1, 8),
        "random_mask": Param(F, 0.0, "Random Mask", 0.0, 1.0),
        "output_mode": Param(E, "Binary Mask", "Output", choices=("Binary Mask", "Integer Mask", "Grayscale Values")),
    }
