"""Color to mask (Legacy): the first Color to mask, found in older materials; the current library replaces it with
Color to mask (Mask range and softness become Threshold and Selection fade). The documentation doesn't list it. Ours,
fitted on the legacy node's outputs."""

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType
from trafilatura3d.nodes.filters.basic._lib import BasicNode

KEYING = ("RGB", "Chrominance", "Luminance")


class ColorToMaskLegacy(BasicNode):
    """Mask of the colours near Color: white up to (1 - Mask softness) x T, a linear fall to black at T, T = Mask range
    / 4 (softness 0: the colours closer than T, Color itself even at range 0). Distance by Keying type: RGB = the
    sum of the channels' absolute differences; Chrominance = the same on the colours minus their Rec. 601 luma;
    Luminance = the difference of the Rec. 601 lumas. The input's alpha is ignored; Flatten alpha changes nothing in
    the grayscale mask (kept for the files that set it)."""

    type = "color_to_mask_legacy"
    label = "Color to mask (Legacy)"
    category = "Filters/Legacy"
    inputs = {"input": Input(PortType.COLOR, "Color input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "color": Param(ParamKind.FLOAT3, [0.0, 1.0, 0.0], "Color", 0.0, 1.0),
        "mask_range": Param(ParamKind.FLOAT, 0.0, "Mask range", 0.0, 1.0),
        "mask_softness": Param(ParamKind.FLOAT, 0.0, "Mask softness", 0.0, 1.0),
        "keying_type": Param(ParamKind.ENUM, "RGB", "Keying type", choices=KEYING),
        "flatten_alpha": Param(ParamKind.BOOL, False, "Flatten alpha"),
    }
