from trafilatura3d.nodes.base import Input, Output, PortType

from ._fx import Effect


class FloodFillToPosition(Effect):
    """Each shape's position: the UV of its bounding box centre in red (X) and green (Y), from the top left; blue 0, black outside the shapes."""

    type = "flood_fill_to_position"
    label = "Flood Fill to Position"
    category = "Filters/Effects"
    catalog = "flood_fill_to_position"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill": Input(PortType.COLOR, "Flood Fill")}
    outputs = {"output": Output(PortType.COLOR)}
