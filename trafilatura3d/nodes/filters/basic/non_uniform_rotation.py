from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._lib import BasicNode


class NonUniformRotation(BasicNode):
    """Rotate each pixel by its own angle: turns = Rotation Map x Rotation Angle Multiplier +
    Rotation Angle Offset, positive clockwise, around PIvot Position (UV) or the R/G of the pivot
    map. Off-image areas show Background Color on the axes that do not tile."""

    type = "non_uniform_rotation"
    label = "Non-Uniform Rotation"
    category = "Filters/Transforms"
    catalog = "non_uniform_rotation"
    inputs = {"input": Input(label="Input"), "rotation_map": Input(PortType.GRAY, "Rotation Map"),
              "rotation_pivot_position_map": Input(PortType.COLOR, "Rotation Pivot Position Map",
                                                   (0.5, 0.5, 0.0, 1.0))}
    params = {
        "rotation_angle_multiplier": Param(ParamKind.FLOAT, 1.0, "Rotation Angle Multiplier", -2.0, 2.0),
        "rotation_angle_offset": Param(ParamKind.FLOAT, 0.0, "Rotation Angle Offset", -1.0, 1.0),
        "use_pivot_position_map": Param(ParamKind.BOOL, False, "Use Pivot Position Map"),
        "pivot_position": Param(ParamKind.FLOAT2, [0.5, 0.5], "PIvot Position", 0.0, 1.0),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0], "Background Color"),
        "filtering_mode": Param(ParamKind.ENUM, "Bilinear", "Filtering Mode", choices=("Nearest", "Bilinear")),
    }
