from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class ReplaceColorRange(BasicNode):
    """Replace Color's result (the HSL offset from source to target added: the source itself becomes exactly the
    target, shading of nearby colours is kept) over the input where the colour is near Source Color: the Euclidean
    RGB distance with a smooth falloff to 0 at a radius growing with Threshold and Source Range (0: no effect,
    0.5: as is, 1: eight times wider). Ours, fitted on the output."""

    type = "replace_color_range"
    label = "Replace Color Range"
    category = "Filters/Adjustments"
    catalog = "replace_color_range"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "source_color": Param(ParamKind.COLOR, [0.502, 0.502, 0.502, 1.0], "Source Color"),
        "target_color": Param(ParamKind.COLOR, [0.502, 0.502, 0.502, 1.0], "Target Color"),
        "source_range": Param(ParamKind.FLOAT, 0.5, "Source Range", 0.0, 1.0),
        "threshold": Param(ParamKind.FLOAT, 1.0, "Threshold", 0.0, 1.0),
    }
