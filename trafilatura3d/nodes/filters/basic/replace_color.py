from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class ReplaceColor(BasicNode):
    """The HSL difference from Source Color to Target Color added to every pixel (hue
    rotated, saturation and lightness moved), so the source colour becomes the target. Alpha passes
    through."""

    type = "replace_color"
    label = "Replace Color"
    category = "Filters/Adjustments"
    catalog = "replace_color"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "source_color": Param(ParamKind.COLOR, [0.5, 0.5, 0.5, 1.0], "Source Color"),
        "target_color": Param(ParamKind.COLOR, [0.5, 0.5, 0.5, 1.0], "Target Color"),
    }
