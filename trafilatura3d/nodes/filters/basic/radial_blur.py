from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class RadialBlur(BasicNode):
    """Spin blur (an FX-Map stamping the image Samples times, each turned by Angle x k /
    Samples turns, k = 0 .. Samples - 1, at opacity 1 / Samples): each pixel averages the taps along the arc
    of Angle turns that starts on it, around Center Position (an offset from the image's centre, -0.5..0.5;
    circles stay round on non-square images)."""

    type = "radial_blur"
    label = "Radial Blur"
    category = "Filters/Blurs"
    catalog = "radial_blur"
    inputs = {"input": Input(label="Input")}
    params = {
        "samples": Param(ParamKind.INT, 64, "Samples", 1, 128),
        "angle": Param(ParamKind.FLOAT, 0.13, "Angle", 0.0, 0.5),
        "center_position": Param(ParamKind.FLOAT2, [0.0, 0.0], "Center Position", -0.5, 0.5),
    }
