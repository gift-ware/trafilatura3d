from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, F
from trafilatura3d.nodes.materials.hdri._common import HdriNode


class PanoramaRotation(HdriNode):
    """Rotates a panorama on the sphere, keeping the projection correct: a spin around the pole
    (Pre), a tilt by Rotation around a horizontal axis at Direction Angle, a spin around the pole
    (Post). Angles in turns."""

    type = "panorama_rotation"
    label = "Panorama Rotation"
    catalog = "panorama_rotation"
    inputs = {"input": Input(COLOR, "Input", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "rotation": F(0.0, "Rotation", 0.0, 1.0),
        "direction_angle": F(0.0, "Direction Angle", 0.0, 1.0),
        "pre_rotation_around_pole": F(0.0, "Pre Rotation Around Pole", -1.0, 1.0),
        "post_rotation_around_pole": F(0.0, "Post Rotation Around Pole", 0.0, 1.0),
    }
