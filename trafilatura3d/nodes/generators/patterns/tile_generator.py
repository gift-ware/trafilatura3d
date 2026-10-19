from .common import (B, C, E, F, F2, F4, FILTERING, I, Input, Output, Param, PatternNode, PortType, SYMMETRY,
                     color_variant, nse, pattern_choices)

PATTERN_PORTS = tuple(f"pattern_input_{i}" for i in range(1, 7))
HSLA = ("None", "Hue", "Saturation", "Luminance", "Alpha")  # what By Number / By Scale multiply (colour)


class TileGenerator(PatternNode):
    """A grid of X Amount x Y Amount tiles (a built-in shape or up to six pattern inputs), gathered
    per pixel from the neighbouring cells so scaled, rotated and displaced tiles overlap their
    neighbours. Tiles are numbered row by row (Luminance By Number,
    By Pattern Number); 90-degree steps (fixed, random, Quincunx Flip = a quarter turn more per step of the
    2 x 2 block: column parity + 2 x row parity) keep the tile's footprint, then free rotation +- Rotation
    Random turns; each axis mirrored with probability Symmetry Random / 2; Middle Size resizes alternate
    columns/rows and keeps them packed; Offset Random shifts every line by up to 0.55 x the value of the
    image (spread fitted on the output); Horizontal Mask hides the
    odd rows, Vertical Mask the even columns. Five size modes; Interstice X/Y = (X, X random, Y, Y random),
    the random part shrinking the gap per tile; Size Random shrinks each axis by a random fraction.
    Gray by default (the grayscale variant): Luminance By Number/Scale multiply the luminance,
    Luminance Random pulls it towards a random value, each tile adds it minus the Background Color (Add
    Sub: signed; Add, Max: clamped to 0-1) times Global Opacity onto the background colour or the
    Background input, so these blends are order-free. Colour when a colour pattern or background is
    connected or with Color Variant: By Number/Scale multiply the hue, saturation, lightness or
    alpha chosen by their parametrisation, Color Random randomizes hue/saturation/lightness/alpha, and the tile adds
    its colour (Add) or goes over the others in tile order (Alpha Blend; Reverse Rendering Order)."""
    counted_inputs = {"pattern_input_number": "pattern_input_"}

    type = "tile_generator"
    main_input = "background"  # the library graph's primary input: Relative to input follows it
    label = "Tile Generator"
    catalog = "tile_generator"
    unsupported = {"Size (Absolute/Pixel)": "the documentation's older parameter; the node has Size, Size "
                                            "(Absolute) and Size (Pixel)",
                   "Luminance/Color Random": "the documentation's older parameter; the node has Luminance "
                                             "Random (grayscale) and Color Random (color)"}
    libs = ("tile.wgsl", "tiles.wgsl")
    pattern_ports = PATTERN_PORTS
    inputs = {**{p: Input(PortType.ANY, f"Pattern Input {i}") for i, p in enumerate(PATTERN_PORTS, 1)},
              "background": Input(PortType.ANY, "Background")}
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        "x_amount": Param(I, 10, "X Amount", 1, 64),
        "y_amount": Param(I, 10, "Y Amount", 1, 64),
        **nse(),
        **color_variant(),
        # Pattern
        "pattern": Param(E, "Square", "Pattern", choices=pattern_choices("Image Input"), group="Pattern"),
        "pattern_input_number": Param(I, 1, "Pattern Input Number", 1, 6, group="Pattern"),
        "pattern_input_distribution": Param(E, "Random", "Pattern Input Distribution",
                                            choices=("Random", "By Pattern Number"), group="Pattern"),
        # (value, random: a random fraction of the value per tile)
        "pattern_specific": Param(F2, [0.2, 0.0], "Pattern Specific", 0.0, 1.0, group="Pattern"),
        "image_input_filtering": Param(E, FILTERING[0], "Image Input Filtering", choices=FILTERING, group="Pattern"),
        "rotation_step": Param(E, "0", "Rotation", choices=("0", "90", "180", "270"), group="Pattern"),
        "rotation_step_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Pattern"),
        "quincunx_flip": Param(B, False, "Quincunx Flip", group="Pattern"),
        "symmetry_random": Param(F, 0.0, "Symmetry Random", 0.0, 1.0, group="Pattern"),
        "symmetry_random_mode": Param(E, SYMMETRY[0], "Symmetry Random Mode", choices=SYMMETRY, group="Pattern"),
        # Size
        "size_mode": Param(E, "Normal - Interstice", "Size Mode", choices=(
            "Normal - Interstice", "Normal - Size", "Keep Ratio", "Absolute", "Pixel"), group="Size"),
        "middle_size": Param(F2, [0.5, 0.5], "Middle Size", 0.0, 1.0, group="Size"),
        "interstice_x_y": Param(F4, [0.1, 0.0, 0.1, 0.0], "Interstice X/Y", 0.0, 1.0, group="Size"),
        "size": Param(F2, [1.0, 1.0], "Size", 0.0, 1.0, group="Size"),
        "size_absolute": Param(F2, [0.1, 0.1], "Size (Absolute)", 0.0, 1.0, group="Size"),
        "size_pixel": Param(F2, [1.0, 1.0], "Size (Pixel)", 0.0, 4.0, group="Size"),
        "size_random": Param(F2, [0.0, 0.0], "Size Random", 0.0, 1.0, group="Size"),
        "scale": Param(F, 1.0, "Scale", 0.0, 2.0, group="Size"),
        "scale_random": Param(F, 0.0, "Scale Random", 0.0, 1.0, group="Size"),
        "scale_random_seed": Param(I, 0, "Scale Random Seed", 0, 1000, group="Size"),
        # Position
        "offset": Param(F, 0.0, "Offset", 0.0, 1.0, group="Position"),
        "offset_random": Param(F, 0.0, "Offset Random", 0.0, 1.0, group="Position"),
        "offset_random_seed": Param(I, 0, "Offset Random Seed", 0, 1000, group="Position"),
        "vertical_offset": Param(B, False, "Vertical Offset", group="Position"),
        "position_random": Param(F2, [0.0, 0.0], "Position Random", 0.0, 1.0, group="Position"),
        "global_offset": Param(F2, [0.0, 0.0], "Global Offset", 0.0, 1.0, group="Position"),
        # Rotation
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0, group="Rotation"),
        "rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Rotation"),
        # Color
        "color": Param(C, [1.0, 1.0, 1.0, 1.0], "Color", group="Color"),
        "luminance_random": Param(F, 0.0, "Luminance Random", 0.0, 1.0, group="Color"),
        "color_random": Param(F4, [0.0, 0.0, 0.0, 0.0], "Color Random", 0.0, 1.0, group="Color"),
        "luminance_by_number": Param(B, False, "Luminance By Number", group="Color"),
        "luminance_by_scale": Param(B, False, "Luminance By Scale", group="Color"),
        "by_number_parametrisation": Param(E, "Luminance", "By Number Parametrisation", choices=HSLA, group="Color"),
        "by_scale_parametrisation": Param(E, "Luminance", "By Scale Parametrisation", choices=HSLA, group="Color"),
        "checker_mask": Param(B, False, "Checker Mask", group="Color"),
        "horizontal_mask": Param(B, False, "Horizontal Mask", group="Color"),
        "vertical_mask": Param(B, False, "Vertical Mask", group="Color"),
        "random_mask": Param(F, 0.0, "Random Mask", 0.0, 1.0, group="Color"),
        "invert_mask": Param(B, False, "Invert Mask", group="Color"),
        "blending_mode": Param(E, "Add", "Blending Mode", choices=("Add", "Max", "Add Sub", "Alpha Blend"),
                               group="Color"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color", group="Color"),
        "global_opacity": Param(F, 1.0, "Global Opacity", 0.0, 1.0, group="Color"),
        "reverse_rendering_order": Param(B, False, "Reverse Rendering Order", group="Color"),
    }
