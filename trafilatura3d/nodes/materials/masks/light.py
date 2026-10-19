from trafilatura3d.nodes.materials._base import F
from trafilatura3d.nodes.materials.masks._common import WSN, MaskNode


class Light(MaskNode):
    """Fake lighting from the World Space Normal map: a directional light (Horizontal Angle in
    turns around the up axis, Vertical Angle from the horizon at 0 to the zenith at 1), Lambert
    shading plus a highlight whose spread is Highlight Glossiness."""

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
