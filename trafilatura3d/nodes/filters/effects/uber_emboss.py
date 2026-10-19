from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class UberEmboss(Effect):
    """Fake 2D lighting of the Color input from the Height input: Ambient Color everywhere, Diffuse
    Color from an Oren-Nayar diffuse term (Diffuse Roughness) scaled by Light Intensity, Specular
    Color from a Blinn-Phong highlight (Specular Glossiness = size, Specular Intensity), and shadows
    cast by the height field (marched towards the light, Shadows Opacity). The light comes from
    Light Angle (turns) at 45 degrees elevation."""

    type = "uber_emboss"
    label = "Uber Emboss"
    category = "Filters/Effects"
    catalog = "uber_emboss"
    libs = ("_shade.wgsl",)
    inputs = {"color": Input(PortType.COLOR, "Color", default=(0.5, 0.5, 0.5, 1.0)),
              "height": Input(PortType.GRAY, "Height")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "ambient_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Ambient Color"),
        "diffuse_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Diffuse Color"),
        "specular_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Specular Color"),
        "light_intensity": Param(ParamKind.FLOAT, 1.0, "Light Intensity", 0.0, 1.0),
        "light_angle": Param(ParamKind.FLOAT, 0.0, "Light Angle", 0.0, 1.0),
        "specular_intensity": Param(ParamKind.FLOAT, 1.0, "Specular Intensity", 0.0, 1.0),
        "specular_glossiness": Param(ParamKind.FLOAT, 0.0, "Specular Glossiness", 0.0, 1.0),
        "diffuse_roughness": Param(ParamKind.FLOAT, 0.0, "Diffuse Roughness", 0.0, 1.0),
        "shadows_opacity": Param(ParamKind.FLOAT, 1.0, "Shadows Opacity", 0.0, 1.0),
    }
