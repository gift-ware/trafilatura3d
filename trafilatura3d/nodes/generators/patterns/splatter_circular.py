from .common import (B, C, E, F, F2, F4, FILTERING, I, Input, Output, Param, PatternNode, PortType, SYMMETRY,
                     color_variant, nse, pattern_choices)

PATTERN_PORTS = tuple(f"pattern_image_input_{i}" for i in range(1, 7))


class SplatterCircular(PatternNode):
    """Patterns on 1-10 concentric rings around the centre (+ Global Offset): ring n of R (0 = the outer one) has
    radius (R - n * Ring Radius Multiplier) / R * Radius; its Pattern Amount (randomised per ring down to the
    minimum, fractional: a gap opens at the end) sits at equal steps on an arc of Spread turns centred on the left
    (at Spread 1 the first on angle 0). Angle Random moves each pattern by its own random fraction of a step, Radius
    Random and Spiral Factor only pull patterns inwards (Spiral: the first at the centre). Center
    Orientation turns each pattern along the ring; Offset along Direction pushes it along its final rotation.
    Size is absolute (uv) or, with Connect Patterns, Size (Connected) x (the chord to the previous pattern, 1),
    the pattern spanning that chord. Gathered per pixel; the layout wraps around the image so it always tiles.
    Gray by default; colour when a colour pattern or background is connected or with Color
    Variant: the
    luminance factors scale the lightness of Color, Color Random randomizes hue, saturation, lightness (a random
    fraction of it) and alpha; Alpha Blend lays the patterns over each other in ring order."""
    counted_inputs = {"pattern_input_number": "pattern_image_input_"}

    type = "splatter_circular"
    main_input = "background"  # the library graph's primary input: Relative to input follows it
    label = "Splatter Circular"
    catalog = "splatter_circular"
    pattern_ports = PATTERN_PORTS
    inputs = {**{p: Input(PortType.ANY, f"Pattern Image Input {i}") for i, p in enumerate(PATTERN_PORTS, 1)},
              "background": Input(PortType.ANY, "Background")}
    outputs = {"output": Output(PortType.ANY, "Output")}
    params = {
        "pattern_amount": Param(I, 12, "Pattern Amount", 1, 64),
        "pattern_amount_random": Param(F, 0.0, "Pattern Amount Random", 0.0, 1.0),
        "pattern_amount_random_min": Param(I, 1, "Pattern Amount Random Min", 1, 10),
        "ring_amount": Param(I, 1, "Ring Amount", 1, 10),
        **color_variant(),
        **nse(),
        # Pattern
        "pattern": Param(E, "Square", "Pattern", choices=pattern_choices("Image Input"), group="Pattern"),
        "pattern_input_number": Param(I, 1, "Pattern Input Number", 1, 6, group="Pattern"),
        "pattern_input_distribution": Param(E, "By Pattern Number", "Pattern Input Distribution",
                                            choices=("Random", "By Pattern Number", "By Ring Number"), group="Pattern"),
        "image_input_filtering": Param(E, FILTERING[0], "Image Input Filtering", choices=FILTERING, group="Pattern"),
        "pattern_specific": Param(F, 0.0, "Pattern Specific", 0.0, 1.0, group="Pattern"),
        "symmetry_random": Param(F, 0.0, "Symmetry Random", 0.0, 1.0, group="Pattern"),
        "symmetry_random_mode": Param(E, SYMMETRY[0], "Symmetry Random Mode", choices=SYMMETRY, group="Pattern"),
        # Position
        "radius": Param(F, 0.25, "Radius", 0.0, 1.0, group="Position"),
        "radius_random": Param(F, 0.0, "Radius Random", 0.0, 1.0, group="Position"),
        "ring_radius_multiplier": Param(F, 1.0, "Ring Radius Multiplier", 0.0, 1.0, group="Position"),
        "angle_random": Param(F, 0.0, "Angle Random", 0.0, 1.0, group="Position"),
        "spiral_factor": Param(F, 0.0, "Spiral Factor", 0.0, 1.0, group="Position"),
        "spread": Param(F, 1.0, "Spread", 0.0, 2.0, group="Position"),
        "offset_along_direction": Param(F, 0.0, "Offset along Direction", 0.0, 1.0, group="Position"),
        "global_offset": Param(F2, [0.0, 0.0], "Global Offset", 0.0, 1.0, group="Position"),
        # Size
        "connect_patterns": Param(B, False, "Connect Patterns", group="Size"),
        "size_connected": Param(F2, [1.0, 0.1], "Size (Connected)", 0.0, 1.0, group="Size"),
        "size": Param(F2, [0.1, 0.1], "Size", 0.0, 1.0, group="Size"),
        "size_random": Param(F2, [0.0, 0.0], "Size Random", 0.0, 1.0, group="Size"),
        "scale": Param(F, 1.0, "Scale", 0.0, 2.0, group="Size"),
        "scale_random": Param(F, 0.0, "Scale Random", 0.0, 1.0, group="Size"),
        "scale_by_pattern_number": Param(F, 0.0, "Scale by Pattern Number", 0.0, 1.0, group="Size"),
        "invert_pattern_number": Param(B, False, "Invert Pattern Number", group="Size"),
        "scale_by_ring_number": Param(F, 0.0, "Scale by Ring Number", 0.0, 1.0, group="Size"),
        "invert_ring_number": Param(B, False, "Invert Ring Number", group="Size"),
        # Rotation
        "pattern_rotation": Param(F, 0.0, "Pattern Rotation", 0.0, 1.0, group="Rotation"),
        "pattern_rotation_random": Param(F, 0.0, "Pattern Rotation Random", 0.0, 1.0, group="Rotation"),
        "pattern_rotation_pivot": Param(E, "Center", "Pattern Rotation Pivot",
                                        choices=("Center", "Min X", "Max X", "Min Y", "Max Y"), group="Rotation"),
        "center_orientation": Param(B, False, "Center Orientation", group="Rotation"),
        "ring_rotation": Param(F, 0.0, "Ring Rotation", 0.0, 1.0, group="Rotation"),
        "ring_rotation_random": Param(F, 0.0, "Ring Rotation Random", 0.0, 1.0, group="Rotation"),
        "ring_rotation_offset": Param(F, 0.0, "Ring Rotation Offset", 0.0, 1.0, group="Rotation"),
        # Color
        "color": Param(C, [1.0, 1.0, 1.0, 1.0], "Color", group="Color"),
        "luminance_random": Param(F, 0.0, "Luminance Random", 0.0, 1.0, group="Color"),
        "color_random": Param(F4, [0.0, 0.0, 0.0, 0.0], "Color Random", 0.0, 1.0, group="Color"),
        "luminance_by_scale": Param(F, 0.0, "Luminance By Scale", 0.0, 1.0, group="Color"),
        "luminance_by_pattern_number": Param(F, 0.0, "Luminance by Pattern Number", 0.0, 1.0, group="Color"),
        "luminance_invert_pattern_number": Param(B, False, "Invert Pattern Number", group="Color"),
        "luminance_by_ring_number": Param(F, 0.0, "Luminance by Ring Number", 0.0, 1.0, group="Color"),
        "luminance_invert_ring_number": Param(B, False, "Invert Ring Number", group="Color"),
        "random_mask": Param(F, 0.0, "Random Mask", 0.0, 1.0, group="Color"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color", group="Color"),
        "blending_mode": Param(E, "Max", "Blending Mode", choices=("Add", "Max", "Add Sub", "Alpha Blend"),
                               group="Color"),
        "global_opacity": Param(F, 1.0, "Global Opacity", 0.0, 1.0, group="Color"),
    }
