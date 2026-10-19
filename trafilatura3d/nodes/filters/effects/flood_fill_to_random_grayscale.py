from trafilatura3d.nodes.base import Input, Output, PortType

from ._fx import Effect


class FloodFillToRandomGrayscale(Effect):
    """A random gray value per shape (uniform in 0-1, from the Random Seed and the shape's bounding box), black outside the shapes."""

    type = "flood_fill_to_random_grayscale"
    label = "Flood Fill to Random Grayscale"
    category = "Filters/Effects"
    catalog = "flood_fill_to_random_grayscale"
    libs = ("_ff_read.wgsl",)
    inputs = {"flood_fill": Input(PortType.COLOR, "Flood Fill")}
    outputs = {"output": Output(PortType.GRAY)}
