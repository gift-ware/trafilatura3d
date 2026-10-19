from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class FloodFillToGradient(Effect):
    """A linear gradient across each shape's bounding box (0 to 1 corner to corner along its
    direction), for tiles tilted at random. Direction (turns) = Angle + a random angle x Angle
    Variation + the Angle Input at the shape's centre x its multiplier. Slope strength: the Slope
    Input at the centre x its multiplier (1 without it) blends from Flat Slope Color (no slope) to
    the full gradient. Multiply by Bounding Box Size darkens smaller shapes (x their size), Multiply
    by Slope Intensity darkens gentler slopes (x their strength); both from 0 (off) to 1 (full)."""

    type = "flood_fill_to_gradient"
    label = "Flood Fill to Gradient"
    category = "Filters/Effects"
    catalog = "flood_fill_to_gradient"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill": Input(PortType.COLOR, "Flood Fill"),
              "angle_input": Input(PortType.GRAY, "Angle Input"),
              "slope_input": Input(PortType.GRAY, "Slope Input", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "angle": Param(ParamKind.FLOAT, 0.0, "Angle", 0.0, 1.0),
        "angle_variation": Param(ParamKind.FLOAT, 0.0, "Angle Variation", 0.0, 1.0),
        "multiply_by_bounding_box_size": Param(ParamKind.FLOAT, 0.0, "Multiply by Bounding Box Size", 0.0, 1.0),
        "angle_image_input_multiplier": Param(ParamKind.FLOAT, 0.0, "Angle Image Input Multiplier", 0.0, 1.0),
        "slope_image_input_multiplier": Param(ParamKind.FLOAT, 0.0, "Slope Image Input Multiplier", 0.0, 1.0),
        "multiply_by_slope_intensity": Param(ParamKind.FLOAT, 1.0, "Multiply by Slope Intensity", 0.0, 1.0),
        "flat_slope_color": Param(ParamKind.FLOAT, 0.5, "Flat Slope Color", 0.0, 1.0),
    }
