from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.hdri._common import HdriNode


class Exposure(HdriNode):
    """Brightens or darkens a (linear, HDR) image by stops: x 2^EV."""

    type = "exposure"
    label = "Exposure"
    catalog = "exposure"
    inputs = {"input": Input(label="Input")}
    outputs = {"output": Output()}
    params = {"exposure": F(0.0, "Exposure (EV)", -8.0, 8.0)}
