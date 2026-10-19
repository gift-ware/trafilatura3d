"""Noise Upscale 1 / 2 / 3: show a tiling noise at twice its frequency (each output pixel maps to
one input pixel when the node's output is twice the input size: detail kept, features half as big
in the image) while hiding the repetition. 1 and 2: two copies of the input tiled twice, the second one
also turned half a turn, each moved by its Offset (in output UV); a soft X mask shows the first copy in the
top and bottom quarters, as the documentation's Offset1 says, the second in the left and right ones."""

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode

_PARAMS = {
    "offset1x": Param(ParamKind.FLOAT, 0.0, "Offset1X", 0.0, 1.0),
    "offset1y": Param(ParamKind.FLOAT, 0.0, "Offset1Y", 0.0, 1.0),
    "offset2x": Param(ParamKind.FLOAT, 0.0, "Offset2X", 0.0, 1.0),
    "offset2y": Param(ParamKind.FLOAT, 0.0, "Offset2Y", 0.0, 1.0),
}


class NoiseUpscale1(BasicNode):
    """The two copies joined through the X mask (contrast like the input)."""
    type = "noise_upscale_masked"
    label = "Noise Upscale Masked"
    category = "Filters/Transforms"
    catalog = "noise_upscale_1"
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = _PARAMS
    defines = {"MODE": 1}


class NoiseUpscale2(NoiseUpscale1):
    """Noise Upscale 1's result mixed with the input itself stretched to the output (its features twice as
    big), 0.53 and 0.47 (fitted), so the two scales meet with softer
    contrast."""
    type = "noise_upscale_mixed"
    label = "Noise Upscale Mixed"
    catalog = "noise_upscale_2"
    defines = {"MODE": 2}
