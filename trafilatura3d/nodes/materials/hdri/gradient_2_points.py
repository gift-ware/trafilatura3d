from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, C, F, F2
from trafilatura3d.nodes.materials.hdri._common import HdrGenerator


class Gradient2Points(HdrGenerator):
    """Panorama gradient between two points on the sphere: the blend follows the angular distance
    to each point (great circles), so it stays correct once projected. Each point's Contrast
    tightens the area its color holds."""

    type = "gradient_2_points"
    label = "Gradient 2 Points"
    catalog = "gradient_2_points"
    outputs = {"output": Output(COLOR)}
    params = {
        "point_1_position": F2([0.5757, 0.2996], "Point 1 Position"),
        "point_1_color": C([1.0, 0.4392, 0.3843], "Point 1 Color"),
        "point_1_contrast": F(0.0, "Point 1 Contrast", 0.0, 1.0),
        "point_2_position": F2([0.3991, 0.7595], "Point 2 Position"),
        "point_2_color": C([0.0, 0.7797, 1.0], "Point 2 Color"),
        "point_2_contrast": F(0.0, "Point 2 Contrast", 0.0, 1.0),
    }
