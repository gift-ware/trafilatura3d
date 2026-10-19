from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HaldCLUT(BasicNode):
    """Colour lookup through a Hald CLUT image (level L: an L^3 x L^3 image holding an L^2 cube;
    4096 px = level 16, 256 steps per channel), trilinear between entries. With LUT Intensity by
    Alpha the result is blended with the input by the input's alpha. No LUT connected: unchanged."""

    type = "hald_clut"
    label = "Hald CLUT"
    category = "Filters/Adjustments"
    catalog = "hald_clut"
    inputs = {"input": Input(PortType.COLOR, "input"), "lut": Input(PortType.COLOR, "lut")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"lut_intensity_by_alpha": Param(ParamKind.BOOL, True, "LUT Intensity by Alpha")}
