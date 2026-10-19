from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class RGBAMerge(BasicNode):
    """Pack four grayscale images into R, G, B, A (unconnected: 0 for colour channels, 1 for alpha)."""

    type = "rgba_merge"
    label = "RGBA Merge"
    category = "Filters/Channels"
    catalog = "rgba_merge"
    inputs = {"r": Input(PortType.GRAY, "R"), "g": Input(PortType.GRAY, "G"), "b": Input(PortType.GRAY, "B"),
              "a": Input(PortType.GRAY, "A", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
