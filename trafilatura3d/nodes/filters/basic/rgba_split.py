from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class RGBASplit(BasicNode):
    """Unpack a colour image into four grayscale images R, G, B, A."""

    type = "rgba_split"
    label = "RGBA Split"
    category = "Filters/Channels"
    catalog = "rgba_split"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"r": Output(PortType.GRAY, "R"), "g": Output(PortType.GRAY, "G"), "b": Output(PortType.GRAY, "B"),
               "a": Output(PortType.GRAY, "A")}
