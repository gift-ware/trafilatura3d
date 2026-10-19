from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class NormalVectorRotation(Effect):
    """Rotates every normal around the surface normal (Z) by Rotation Angle (turns, counter-clockwise),
    times the Rotation Map when connected; pixels do not move."""

    type = "normal_vector_rotation"
    label = "Normal Vector Rotation"
    category = "Filters/Normal Map"
    catalog = "normal_vector_rotation"
    inputs = {"normal": Input(PortType.COLOR, "Normal", default=(0.5, 0.5, 1.0, 1.0)),
              "rotation_map": Input(PortType.GRAY, "Rotation Map (optional)", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "rotation_angle": Param(ParamKind.FLOAT, 0.0, "Rotation Angle", 0.0, 1.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
    }
