from .common import B, E, F, I, Input, Output, Param, PatternNode, PortType


class ShapeMapper(PatternNode):
    """Projects the input along a circle or a regular polygon, Pattern amount times without gaps.
    The input's U runs along the outline (clockwise), its V across the band: V = 0 on the outer
    border, 1 on the inner one. Radius and Width are fractions of half the smaller image side; Rotation is
    in turns, clockwise, starting from the +x axis. On a polygon the outline is parametrised by arc
    length; the band is centred on the edges (perpendicular distance from the centre)."""

    type = "shape_mapper"
    label = "Shape mapper"
    catalog = "shape_mapper"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY, "Output")}
    params = {
        "shape": Param(E, "Circle", "Shape", choices=("Circle", "Polygon")),
        "pattern_amount": Param(I, 8, "Pattern amount", 1, 10),
        "link_segments_with_pattern_amount": Param(B, False, "Link segments with pattern amount"),
        "segments": Param(I, 8, "Segments", 3, 16),
        "radius": Param(F, 0.5, "Radius", 0.0, 1.0),
        "width": Param(F, 0.5, "Width", 0.0, 1.0),
        "rotation": Param(F, 0.0, "Rotation", 0.0, 1.0),
        "flip_one_on_two": Param(B, False, "Flip one on two"),
        "filtering_mode": Param(E, "Bilinear", "Filtering mode", choices=("Nearest", "Bilinear")),
        "non_square_expansion": Param(B, True, "Non-square expansion"),
    }
