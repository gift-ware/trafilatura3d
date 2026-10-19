from .common import B, E, F, I, Input, Output, Param, PatternNode, PortType

DISTRIBUTION = ("Uniform", "Blue Noise")
MAP = (1.0, 1.0, 1.0, 1.0)  # unconnected maps read white: the multipliers then act as plain factors


class TileRandom2(PatternNode):
    """Adjacent cells of random sizes and ratios: the vertices of an Amount X x Amount Y grid are
    moved by the random sizes (up to half a cell; Uniform values, or Blue Noise = an R2
    low-discrepancy sequence, more even) and each carries a horizontal and a vertical line, tilted by
    the random slant. The cells pinwheel around the vertices: each side lies on the line of one of
    its four vertices, alternating with the cell's parity, so neighbours share their sides and meet in
    T junctions (rectangles without slant, quadrilaterals with it). Each shape is its cell, rotated by sliding
    its corners along the cell border (Rotation, random; Corners Position Random slides each corner
    on its own), scaled (+ random) about its centre, moved within the free space (Random Position),
    eroded by half the interstice and rounded (radius in cells), then bevelled: distance mode
    relative to the grid cell, the shape (its inradius) or the image (a quarter of it), rounded
    curve, or per-axis distances (no rounded corners then). Shapes are masked by a random range,
    by cell area (area / 8 cells, clamped to 1) and by the Mask Map at the cell centre: each range (start/end
    in either order, default empty) hides the shapes in it, Invert the others. Input maps scale their parameter (size and slant at each
    vertex, corner radius at the cell centre, bevel per pixel).

    Outputs: `output` the bevelled shapes (gray); `uv` (color, designed for Flood Fill style
    workflows): R, G = position in the shape's bounding box (0-1), B = a random value per cell,
    A = the shape mask."""

    type = "tile_random_sizes"
    label = "Tile Random Sizes"
    catalog = "tile_random_2"
    inputs = {
        "random_size_map": Input(PortType.GRAY, "Random Size Map", MAP),
        "random_slant_map": Input(PortType.GRAY, "Random Slant Map", MAP),
        "round_corners_radius_map": Input(PortType.GRAY, "Round Corners Radius Map", MAP),
        "bevel_distance_map": Input(PortType.GRAY, "Bevel Distance Map", MAP),
        "mask_map": Input(PortType.GRAY, "Mask Map", MAP),
    }
    main_input = None
    outputs = {"output": Output(PortType.GRAY, "Output"), "uv": Output(PortType.COLOR, "UV")}
    params = {
        "amount_x": Param(I, 10, "Amount X", 2, 16),
        "amount_y": Param(I, 10, "Amount Y", 2, 16),
        # Size
        "random_size_multiplier": Param(F, 1.0, "Random Size Multiplier", 0.0, 1.0, group="Size"),
        "random_size_input_map_multiplier": Param(F, 0.0, "Random Size Input Map Multiplier", 0.0, 1.0, group="Size"),
        "random_size_x": Param(F, 1.0, "Random Size X", 0.0, 1.0, group="Size"),
        "random_size_y": Param(F, 1.0, "Random Size Y", 0.0, 1.0, group="Size"),
        "random_size_distribution": Param(E, "Uniform", "Random Size Distribution", choices=DISTRIBUTION, group="Size"),
        # Shape Aspect - Transform
        "interstice_thickness": Param(F, 0.0, "Interstice Thickness", 0.0, 1.0, group="Shape Aspect - Transform"),
        "random_position_multiplier": Param(F, 0.0, "Random Position Multiplier", 0.0, 1.0,
                                            group="Shape Aspect - Transform"),
        "round_corners_radius": Param(F, 0.0, "Round Corners Radius", 0.0, 1.0, group="Shape Aspect - Transform"),
        "round_corners_radius_input_map_mult": Param(F, 0.0, "Round Corners Radius Input Map Mult.", 0.0, 1.0,
                                                     group="Shape Aspect - Transform"),
        "scale_multiplier": Param(F, 1.0, "Scale Multiplier", 0.0, 1.0, group="Shape Aspect - Transform"),
        "scale_random": Param(F, 0.0, "Scale Random", 0.0, 1.0, group="Shape Aspect - Transform"),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 0.25, group="Shape Aspect - Transform"),
        "rotation_random": Param(F, 0.0, "Rotation Random", 0.0, 1.0, group="Shape Aspect - Transform"),
        "corners_position_random": Param(F, 0.0, "Corners Position Random", 0.0, 1.0, group="Shape Aspect - Transform"),
        # Slant
        "random_slant_multiplier": Param(F, 0.0, "Random Slant Multiplier", 0.0, 1.0, group="Slant"),
        "random_slant_input_map_multiplier": Param(F, 0.0, "Random Slant Input Map Multiplier", 0.0, 1.0, group="Slant"),
        "random_slant_x": Param(F, 1.0, "Random Slant X", 0.0, 1.0, group="Slant"),
        "random_slant_y": Param(F, 1.0, "Random Slant Y", 0.0, 1.0, group="Slant"),
        "random_slant_distribution": Param(E, "Uniform", "Random Slant Distribution", choices=DISTRIBUTION, group="Slant"),
        # Bevel
        "bevel_distance_mode": Param(E, "Relative to Grid Size", "Bevel Distance Mode", choices=(
            "Relative to Grid Size", "Relative to Shape Size", "Relative to Image Size"), group="Bevel"),
        "bevel_distance_multiplier": Param(F, 0.2, "Bevel Distance Multiplier", 0.0, 1.0, group="Bevel"),
        "bevel_distance_input_map_mult": Param(F, 0.0, "Bevel Distance Input Map Mult.", 0.0, 1.0, group="Bevel"),
        "bevel_rounded_curve": Param(F, 0.0, "Bevel Rounded Curve", 0.0, 1.0, group="Bevel"),
        "enable_per_axis_bevel_control": Param(B, False, "Enable per Axis Bevel Control", group="Bevel"),
        "bevel_distance_x": Param(F, 1.0, "Bevel Distance X", 0.0, 1.0, group="Bevel"),
        "bevel_distance_y": Param(F, 1.0, "Bevel Distance Y", 0.0, 1.0, group="Bevel"),
        # Mask
        "mask_random_invert": Param(B, False, "Mask Random Invert", group="Mask"),
        "mask_random_start": Param(F, 0.0, "Mask Random Start", 0.0, 1.0, group="Mask"),
        "mask_random_end": Param(F, 0.0, "Mask Random End", 0.0, 1.0, group="Mask"),
        "mask_by_cell_area_invert": Param(B, False, "Mask by Cell Area Invert", group="Mask"),
        "mask_by_cell_area_start": Param(F, 0.0, "Mask by Cell Area Start", 0.0, 1.0, group="Mask"),
        "mask_by_cell_area_end": Param(F, 0.0, "Mask by Cell Area End", 0.0, 1.0, group="Mask"),
        "mask_map_input_invert": Param(B, False, "Mask Map Input Invert", group="Mask"),
        "mask_map_input_start": Param(F, 0.0, "Mask Map Input Start", 0.0, 1.0, group="Mask"),
        "mask_map_input_end": Param(F, 0.0, "Mask Map Input End", 0.0, 1.0, group="Mask"),
    }
