from trafilatura3d.nodes.base import Input, Output, PortType

from ._lib import BasicNode


class NoiseUpscale3(BasicNode):
    """The noise tiled 2x in four copies: as is and turned by a quarter, a half and three quarters of a turn,
    each moved by its own offset. The Mask picks between the copies in two pairs and the Mask turned by a
    quarter of a turn between the pairs, so a noisy mask hides the repetition (unconnected: white, the plain
    copy). The result tiles."""

    type = "noise_upscale_quad"
    label = "Noise Upscale Quad"
    category = "Filters/Transforms"
    catalog = "noise_upscale_3"
    inputs = {"grayscale": Input(PortType.GRAY, "Grayscale"),
              "mask": Input(PortType.GRAY, "Mask", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
