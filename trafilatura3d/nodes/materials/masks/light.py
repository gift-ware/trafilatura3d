from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.masks._common import WSN, MaskNode


class Light(MaskNode):
    """Fake lighting from the World Space Normal map: a directional light (Horizontal Angle in
    turns around the up axis, from -X at 0 through +Z at 0.25; Vertical Angle in turns around the
    horizontal, horizon at 0, zenith at 0.25, opposite horizon at 0.5), a half-Lambert term
    windowed by Highlight Glossiness (start) and Highlight Level (how close the end comes)."""

    type = "light"
    label = "Light"
    catalog = "light"
    inputs = {"world_space_normal": WSN}
    params = {
        "horizontal_angle": F(0.25, "Horizontal Angle", 0.0, 1.0),
        "vertical_angle": F(0.125, "Vertical Angle", 0.0, 1.0),
        "highlight_glossiness": F(0.5, "Highlight Glossiness", 0.0, 0.999),
        "highlight_level": F(0.0, "Highlight Level", 0.0, 1.0),
    }
