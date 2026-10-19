from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._lib import BasicNode


class TrapezoidTransform(BasicNode):
    """Trapezoid warp: the image's width is 1 - Top Stretch at the top and 1 - Bottom Stretch at the
    bottom (0 = unchanged, positive squashes towards the centre, negative widens; beyond 1 it
    flips), varying linearly in between. Off-image areas show Background Color unless tiling H."""

    type = "trapezoid_transform"
    label = "Trapezoid Transform"
    category = "Filters/Transforms"
    catalog = "trapezoid_transform"
    inputs = {"input": Input(label="Input")}
    params = {
        "top_stretch": Param(ParamKind.FLOAT, 0.0, "Top Stretch", 0.0, 1.0),
        "bottom_stretch": Param(ParamKind.FLOAT, 0.0, "Bottom Stretch", 0.0, 1.0),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Background Color"),
        "sampling": Param(ParamKind.ENUM, "Bilinear", "Sampling", choices=("Bilinear", "Nearest")),
    }
