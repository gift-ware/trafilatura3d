from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class ContrastLuminosity(BasicNode):
    """Contrast around mid gray (positive: steeper, 1 = hard step; negative: flatter, -1 = flat
    gray) then Luminosity added; 0 and 0 leave the image unchanged. Alpha passes through."""

    type = "contrast_luminosity"
    label = "ContrastLuminosity"
    category = "Filters/Adjustments"
    catalog = "contrast_luminosity"
    inputs = {"input": Input(label="Input")}
    params = {
        "contrast": Param(ParamKind.FLOAT, 0.0, "Contrast", -1.0, 1.0),
        "luminosity": Param(ParamKind.FLOAT, 0.0, "Luminosity", -1.0, 1.0),
    }
