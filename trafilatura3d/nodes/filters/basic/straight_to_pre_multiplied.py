from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class StraightToPreMultiplied(BasicNode):
    """Premultiply: rgb x alpha, alpha kept."""

    type = "straight_to_pre_multiplied"
    label = "Straight to Pre-multiplied"
    category = "Filters/Channels"
    catalog = "straight_to_pre_multiplied"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
