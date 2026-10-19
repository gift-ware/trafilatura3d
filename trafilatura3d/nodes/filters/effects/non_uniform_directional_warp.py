from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._fx import Effect


class NonUniformDirectionalWarp(Effect):
    """Directional warp whose distance (Intensity x Intensity Input, pixels at 256) and direction
    (Warp Angle + Warp Angle Input x its multiplier, turns) vary per pixel. The displacement follows a
    streamline: 24 steps, each re-reading the angle where it is, so paths curve. Trail Length keeps
    the samples of the last part of the path, combined with Min / Max / Average; Trail Fade fades them
    out along the trail, with Trail Curve bending that fade."""

    type = "non_uniform_directional_warp"
    label = "Non Uniform Directional Warp"
    category = "Filters/Effects"
    catalog = "non_uniform_directional_warp"
    inputs = {"input": Input(PortType.GRAY, "Input"),
              "intensity_input": Input(PortType.GRAY, "Intensity Input", default=(1.0, 1.0, 1.0, 1.0)),
              "warp_angle_input": Input(PortType.GRAY, "Warp Angle Input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 20.0),
        "warp_angle": Param(ParamKind.FLOAT, 0.0, "Warp Angle", 0.0, 1.0),
        "warp_angle_input_multiplier": Param(ParamKind.FLOAT, 1.0, "Warp Angle Input Multiplier", 0.0, 1.0),
        "trail_mode": Param(ParamKind.ENUM, "Max", "Trail Mode", choices=("Min", "Max", "Average")),
        "trail_length": Param(ParamKind.FLOAT, 0.0, "Trail Length", 0.0, 1.0),
        "trail_fade": Param(ParamKind.FLOAT, 1.0, "Trail Fade", 0.0, 1.0),
        "trail_curve": Param(ParamKind.FLOAT, 0.0, "Trail Curve", -1.0, 1.0),
    }
