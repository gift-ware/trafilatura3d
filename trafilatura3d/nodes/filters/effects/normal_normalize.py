from trafilatura3d.nodes.base import Input, Output, PortType

from ._fx import Effect


class NormalNormalize(Effect):
    """Rescales the vector of every pixel of a normal map to unit length (alpha kept)."""

    type = "normal_normalize"
    label = "Normal Normalize"
    category = "Filters/Normal Map"
    catalog = "normal_normalize"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
