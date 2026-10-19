from trafilatura3d.nodes.base import Input, Output, PortType

from ._fx import Effect


class FacingNormal(Effect):
    """Facing ratio of the normal map: the cosine between each (normalized)
    normal and the surface's own normal +Z; flat is white, normals at grazing angles go to black."""

    type = "facing_normal"
    label = "Facing Normal"
    category = "Filters/Normal Map"
    catalog = "facing_normal"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
