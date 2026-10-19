from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode

SPACES = ("RGB (Data)", "Linear sRGB (Color)", "Luminance (Color)", "Lab (Color)", "Angle (Normal)")


class ColorToMask(BasicNode):
    """Mask of the colours near a reference (Color, or Color Input per pixel): white up to (1 - Selection fade) x
    Threshold, a linear fall to black at Threshold (fade 0: exactly the colours closer than Threshold). Distances,
    each normalised by the largest the space allows: RGB and linear RGB euclidean over the cube diagonal; luminance
    as the Michelson contrast of the Rec. 601 lumas; Lab CIEDE2000 / 100 with Distance weights (kL, kC, kH); angle
    between tangent-space normals over a half turn."""

    type = "color_to_mask"
    label = "Color to mask"
    category = "Filters/Adjustments"
    catalog = "color_to_mask"
    inputs = {"input": Input(PortType.COLOR, "Input"), "color_input": Input(PortType.COLOR, "Color Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "use_color_input": Param(ParamKind.BOOL, False, "Use color input"),
        "color": Param(ParamKind.FLOAT3, [0.5, 0.5, 0.5], "Color", 0.0, 1.0),
        "threshold": Param(ParamKind.FLOAT, 0.01, "Threshold", 0.001, 1.0),
        "selection_fade": Param(ParamKind.FLOAT, 0.0, "Selection fade", 0.0, 1.0),
        "distance_color_space": Param(ParamKind.ENUM, "RGB (Data)", "Distance color space", choices=SPACES),
        "distance_weights": Param(ParamKind.FLOAT3, [1.0, 1.0, 1.0], "Distance weights", 0.0, 1.0),
    }
