from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class PreMultipliedToStraight(BasicNode):
    """Unpremultiply: rgb / alpha (unchanged where alpha is 0), and the alpha is dropped (set to 1)."""

    type = "pre_multiplied_to_straight"
    label = "Pre-Multiplied to Straight"
    category = "Filters/Channels"
    catalog = "pre_multiplied_to_straight"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
