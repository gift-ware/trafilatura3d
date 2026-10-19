from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class ChrominanceExtract(BasicNode):
    """The colour with its luminance removed: rgb - luminance + 0.5 (chrominance around mid gray).
    Alpha passes through."""

    type = "chrominance_extract"
    label = "Chrominance Extract"
    category = "Filters/Adjustments"
    catalog = "chrominance_extract"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
