from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class AlphaSplit(BasicNode):
    """The image without its alpha (alpha set to 1) and the alpha as a grayscale image."""

    type = "alpha_split"
    label = "Alpha Split"
    category = "Filters/Channels"
    catalog = "alpha_split"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"rgb": Output(PortType.COLOR, "RGB"), "a": Output(PortType.GRAY, "A")}
