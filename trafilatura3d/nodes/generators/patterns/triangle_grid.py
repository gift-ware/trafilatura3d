from .common import C, E, F, F2, I, Input, Output, Param, PatternNode, PortType, nse


class TriangleGrid(PatternNode):
    """A periodic grid of X Amount x Y Amount vertices, triangulated and seen from above (Z-down
    orthographic). Vertex offset, in cells, = (Random Position x Random Position Multiplier x random +
    Vector Map Displacement x (0.5 - vector map rg)) rotated by Rotation plus a random turn of up to +- half
    a turn x Rotation Disorder, plus the quincunx offsets on alternate rows/columns, at most a cell's
    diagonal; Z = lerp(0.5, Height input, multiplier) + Height Random x random. Each pixel tests the
    two triangles of the quads around it. Overlaps (from large offsets) blend per output: Per Vertex /
    Per Triangle by Max or Depth Test (highest Z wins), Thin Line also by Alpha Blend (over, in grid
    order; the only antialiased mode), Distance to Edge by Min (nearest edge of any triangle) or Depth
    Test; Center always by depth. Thin Line is 1 px wide at 256 px (scales with the image); Distance to
    Edge = the distance in uv / Thickness; Center = 3 x the smallest barycentric weight (1 at the
    barycenter, 0 on the edges). Where no triangle covers a pixel the Background Color shows."""

    type = "triangle_grid"
    label = "Triangle Grid"
    catalog = "triangle_grid"
    inputs = {"height": Input(PortType.GRAY, "Height"),
              "vector_map": Input(PortType.COLOR, "Vector map", default=(0.5, 0.5, 0.5, 1.0)),
              "color_input": Input(PortType.COLOR, "Color input", default=(1.0, 1.0, 1.0, 1.0))}
    main_input = "height"
    outputs = {"output": Output(PortType.COLOR, "Output")}
    unsupported = {"Quincux Offset X": "documentation typo of Quincunx Offset X",
                   "Quincux Offset Y": "documentation typo of Quincunx Offset Y"}
    params = {
        "color_output": Param(E, "Per Vertex", "Color Output",
                              choices=("Per Vertex", "Per Triangle", "Thin Line", "Distance to Edge", "Center")),
        "triangulation": Param(E, "Auto", "Triangulation",
                               choices=("Auto", "45°", "-45°", "Quincunx horizontal", "Quincunx vertical")),
        "x_amount": Param(I, 16, "X Amount", 1, 64),
        "y_amount": Param(I, 16, "Y Amount", 1, 64),
        "random_position_multiplier": Param(F, 1.0, "Random Position Multiplier", 0.0, 2.0),
        "random_position": Param(F2, [1.0, 1.0], "Random Position", 0.0, 1.0),
        "vector_map_displacement": Param(F, 0.0, "Vector Map Displacement", 0.0, 2.0),
        "quincunx_offset_x": Param(F, 0.0, "Quincunx Offset X", 0.0, 1.0),
        "quincunx_offset_y": Param(F, 0.0, "Quincunx Offset Y", 0.0, 1.0),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0),
        "rotation_disorder": Param(F, 0.5, "Rotation Disorder", 0.0, 1.0),
        "height_input_multiplier": Param(F, 0.0, "Height Input Multiplier", 0.0, 1.0),
        "height_random": Param(F, 1.0, "Height Random", 0.0, 1.0),
        "blend_mode": Param(E, "Max", "Blend Mode", choices=("Max", "Depth Test")),
        "thin_line_blend_mode": Param(E, "Max", "Thin Line Blend Mode", choices=("Max", "Depth Test", "Alpha Blend")),
        "distance_blend_mode": Param(E, "Min", "Distance Blend Mode", choices=("Min", "Depth Test")),
        "color_source": Param(E, "Random Grayscale", "Color Source",
                              choices=("Height", "Color Input", "Random Grayscale", "Random Color")),
        "color_source_opacity": Param(F, 0.0, "Color Source Opacity", 0.0, 1.0),
        "distance_to_edge_thickness": Param(F, 0.025, "Distance to Edge Thickness", 0.0, 0.1),
        "line_color": Param(C, [1.0, 1.0, 1.0, 1.0], "Line Color"),
        "background_color": Param(C, [0.0, 0.0, 0.0, 1.0], "Background Color"),
        "random_color_seed_mode": Param(E, "Global Random Seed", "Random Color Seed Mode",
                                        choices=("Global Random Seed", "Manual Seed")),
        "random_color_seed": Param(I, 0, "Random Color Seed", 0, 100),
        **nse(),
    }
