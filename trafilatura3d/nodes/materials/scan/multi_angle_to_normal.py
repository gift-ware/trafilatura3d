from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, NORMAL_FORMATS, E, F, I, MatNode
from trafilatura3d.nodes.materials.scan._common import CATEGORY

N = 8


class MultiAngleToNormal(MatNode):
    """Photometric stereo: a normal map from photos lit from evenly spaced azimuths (the first at
    First Sample Light Angle, the next ones turning counterclockwise or clockwise) at 45 degrees of
    elevation. Per pixel, the least-squares solution of intensity = albedo x dot(normal, light)
    over the samples; Intensity scales the slopes."""

    type = "multi_angle_to_normal"
    label = "Multi-Angle to Normal"
    category = CATEGORY
    catalog = "multi_angle_to_normal"
    channels = tuple(f"input_{i}" for i in range(1, N + 1))
    inputs = {f"input_{i}": Input(COLOR, f"Input {i}", (0.0, 0.0, 0.0, 1.0)) for i in range(1, N + 1)}
    outputs = {"output": Output(COLOR)}
    params = {
        "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS),
        "samples_amount": I(4, "Samples Amount", 2, 8),
        "intensity": F(0.5, "Intensity", 0.0, 1.0),
        "first_sample_light_angle": F(0.0, "First Sample Light Angle", 0.0, 360.0),
        "next_sample_light_angle": E("Counterclockwise", "Next Sample Light Angle", ("Counterclockwise", "Clockwise")),
    }
