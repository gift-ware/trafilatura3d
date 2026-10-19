from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class AlphaMerge(BasicNode):
    """RGB from the colour input, alpha from the grayscale A input (replacing any existing alpha)."""

    type = "alpha_merge"
    label = "Alpha Merge"
    category = "Filters/Channels"
    catalog = "alpha_merge"
    inputs = {"rgb": Input(PortType.COLOR, "RGB"), "a": Input(PortType.GRAY, "A", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
