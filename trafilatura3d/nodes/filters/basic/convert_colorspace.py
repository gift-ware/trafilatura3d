from trafilatura3d.nodes.base import Input

from ._lib import BasicNode


class ConvertToLinear(BasicNode):
    """sRGB -> linear with the exact piecewise sRGB transfer function. Alpha passes through."""

    type = "convert_to_linear"
    label = "Convert To Linear"
    category = "Filters/Adjustments"
    catalog = "convert_to_linear"
    inputs = {"input": Input(label="Input")}
    defines = {"TO_LINEAR": "true"}


class ConvertToSRGB(BasicNode):
    """Linear -> sRGB with the exact piecewise sRGB transfer function. Alpha passes through."""

    type = "convert_to_srgb"
    label = "Convert to sRGB"
    category = "Filters/Adjustments"
    catalog = "convert_to_srgb"
    inputs = {"input": Input(label="Input")}
    defines = {"TO_LINEAR": "false"}
