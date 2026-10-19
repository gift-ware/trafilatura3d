from trafilatura3d.nodes.base import Input, Output, PortType

from ._fx import Effect


class FloodFillToRandomColor(Effect):
    """A random RGB color per shape (each channel uniform in 0-1, from the Random Seed and the shape's bounding box), black outside the shapes."""

    type = "flood_fill_to_random_color"
    label = "Flood Fill to Random Color"
    category = "Filters/Effects"
    catalog = "flood_fill_to_random_color"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill": Input(PortType.COLOR, "Flood Fill")}
    outputs = {"output": Output(PortType.COLOR)}
