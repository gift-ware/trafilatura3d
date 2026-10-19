from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, E
from trafilatura3d.nodes.materials.hdri._common import UP_VECTORS, HdriNode


class Panorama3dPosition(HdriNode):
    """World-space direction of every pixel of a latitude/longitude panorama, encoded like a
    position map (0.5 + 0.5 x unit vector), with Z or Y as the up axis."""

    type = "panorama_3d_position"
    label = "Panorama 3D Position"
    catalog = "panorama_3d_position"
    outputs = {"output": Output(COLOR)}
    params = {"up_vector": E("Z Up", "Up Vector", UP_VECTORS)}
