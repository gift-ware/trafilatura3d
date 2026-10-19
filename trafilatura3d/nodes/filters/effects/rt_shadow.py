from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect

MODES = ("Shaded", "Shadow Only", "Gradient", "Gradient Inverted")


class RTShadows(Effect):
    """Ray traced shadows of a height map. Height: the input x Height Scale (white
    = that fraction of the width). Light Position is a point over the image: the light comes from its
    side, and its distance from the centre sets the elevation (centre = zenith, 0.5 away = the horizon;
    Keep Light Above Horizon keeps the height at least Light Size Y x Light Scale). Light Scale (Soft
    Shadows) > 0 spreads Samples rays over a rectangle of Light Size x Light Scale across the light
    direction, else one ray. A ray is shadowed when it hits the surface within Max Shadows Length (UV).
    Shaded: Lambert x Light Intensity, darkened by the shadow x Shadows Opacity (faded with the distance
    to the caster by Shadows Attenuation), sRGB; Shadow Only: 1 - shadow; Gradient: 1 - shadow x (1 -
    distance / length); Gradient Inverted: 1 - shadow x distance / length; Invert flips the last three."""

    type = "rt_shadow"
    label = "RT Shadows"
    category = "Filters/Effects"
    catalog = "rt_shadow"
    libs = ("_rays.wgsl",)
    inputs = {"input": Input(PortType.GRAY, "Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "samples": Param(ParamKind.INT, 64, "Samples", 1, 64),
        "mode": Param(ParamKind.ENUM, "Shaded", "Mode", choices=MODES),
        "height_scale": Param(ParamKind.FLOAT, 0.1, "Height Scale", 0.0, 1.0),
        "light_position": Param(ParamKind.FLOAT2, [0.8, 0.2], "Light Position", 0.0, 1.0),
        "light_intensity": Param(ParamKind.FLOAT, 1.0, "Light Intensity", 0.0, 1.0),
        "light_size": Param(ParamKind.FLOAT2, [1.0, 1.0], "Light Size", 0.0, 1.0),
        "light_scale_soft_shadows": Param(ParamKind.FLOAT, 0.0, "Light Scale (Soft Shadows)", 0.0, 0.25),
        "keep_light_above_horizon": Param(ParamKind.BOOL, True, "Keep Light Above Horizon"),
        "shadows_opacity": Param(ParamKind.FLOAT, 0.9, "Shadow Opacity", 0.0, 1.0),
        "shadows_attenuation": Param(ParamKind.FLOAT, 0.5, "Shadow Attenuation", 0.0, 1.0),
        "invert": Param(ParamKind.BOOL, False, "Invert"),
        "max_shadows_length": Param(ParamKind.FLOAT, 1.0, "Max Shadows Length", 0.0, 1.0),
    }
