from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class CrossSection(Effect):
    """Plots one row (Section axis X) or column (Y) of the input, at Cross section coordinate, as a
    profile: height = value x Height scale + Height offset, bottom to top. Drawing style: Solid fill,
    Gradient (dark at the bottom, Color at the profile), Gradient mirrored (the profile mirrored about
    the middle line, bright in the middle), or Line (Segment amount segments, Line thickness in UV;
    Line style Smooth / Solid / Segment ID / Gradient along line). Smooth / Polygonal: off = the exact
    profile, on = straight segments. Show helper overlays the input with the section marked, scaled by
    Helper scale, placed by Helper position across the free space (0 = top left,
    1 = bottom right corner)."""

    type = "cross_section"
    label = "Cross Section"
    category = "Filters/Effects"
    catalog = "cross_section"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "cross_section_coordinate": Param(ParamKind.FLOAT, 0.5, "Cross section coordinate", 0.0, 1.0),
        "section_axis": Param(ParamKind.ENUM, "X", "Section axis", choices=("X", "Y")),
        "show_helper": Param(ParamKind.BOOL, False, "Show helper"),
        "helper_scale": Param(ParamKind.FLOAT, 0.25, "Helper scale", 0.0, 1.0, group="Helper Settings"),
        "helper_position": Param(ParamKind.FLOAT2, [1.0, 1.0], "Helper position", 0.0, 1.0, group="Helper Settings"),
        "height_scale": Param(ParamKind.FLOAT, 1.0, "Height scale", 0.0, 1.0),
        "height_offset": Param(ParamKind.FLOAT, 0.0, "Height offset", -1.0, 1.0),
        "drawing_style": Param(ParamKind.ENUM, "Solid", "Drawing style",
                               choices=("Solid", "Gradient", "Gradient mirrored", "Line")),
        "invert_gradient": Param(ParamKind.BOOL, False, "Invert Gradient"),
        "smooth_polygonal": Param(ParamKind.BOOL, False, "Smooth / Polygonal"),
        "segment_amount": Param(ParamKind.INT, 32, "Segment amount", 2, 256),
        "line_thickness": Param(ParamKind.FLOAT, 0.01, "Line thickness", 0.0, 0.1),
        "line_style": Param(ParamKind.ENUM, "Smooth", "Line style",
                            choices=("Smooth", "Solid", "Segment ID", "Gradient along line")),
        "line_smoothness": Param(ParamKind.FLOAT, 1.0, "Line smoothness", 0.0, 1.0),
        "color": Param(ParamKind.FLOAT, 1.0, "Color", 0.0, 1.0),
        "background_color": Param(ParamKind.FLOAT, 0.0, "Background color", 0.0, 1.0),
    }
