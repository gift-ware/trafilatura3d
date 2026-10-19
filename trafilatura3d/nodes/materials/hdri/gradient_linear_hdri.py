from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, C, F, F2
from trafilatura3d.nodes.materials.hdri._common import HdrGenerator


class GradientLinearHdri(HdrGenerator):
    """Linear gradient through the centre of the sphere towards a point: Top color at the point,
    Bottom color opposite, following the cosine of the angle (correct in spherical projection)."""

    type = "gradient_linear_hdri"
    label = "Gradient Linear (HDRI)"
    catalog = "gradient_linear_hdri"
    outputs = {"output": Output(COLOR)}
    params = {
        "point_position": F2([0.5, 0.0], "Point Position"),
        "top_color": C([0.55, 0.7, 1.0], "Top Color"),
        "bottom_color": C([0.2, 0.17, 0.14], "Bottom Color"),
        "contrast": F(0.0, "Contrast", 0.0, 1.0),
    }
