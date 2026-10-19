from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, MATRIX, B, C, E, F, F2, F4
from trafilatura3d.nodes.materials.hdri._common import COLOR_MODES, GAMMAS, HDR, HdriNode


class ShapeLight(HdriNode):
    """A rectangle or disc light on the sphere: Offset places its centre on the panorama (from the
    image centre), Matrix rotates and scales it in the plane tangent to the sphere there (gnomonic
    projection, so it stays undistorted). Color from RGB, Kelvin or the Shape Image; Shape Exposure
    in stops; a hotspot adds Hotspot Exposure at its own position inside the shape."""

    type = "shape_light"
    label = "Shape Light"
    catalog = "shape_light"
    unsupported = {"Enable Backgound Input": "the documentation's typo of Enable Background Input"}
    base_defaults = HDR
    includes = ("_light.wgsl",)
    inputs = {"background_image_input": Input(COLOR, "Background Image Input", (0.0, 0.0, 0.0, 1.0)),
              "shape_image_input": Input(COLOR, "Shape Image Input", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "matrix": F4(MATRIX, "Matrix", group="Shape Matrix"),
        "offset": F2([0.0, -0.15], "Offset", -2.0, 2.0, group="Shape Matrix"),
        "shape": E("Rectangle", "Shape", ("Rectangle", "Disc"), group="Shape Matrix"),
        "shape_color_mode": E("Temperature (Kelvin)", "Shape Color Mode", COLOR_MODES, group="Shape Matrix"),
        "color": C([1.0, 1.0, 1.0], "Color", group="Shape Matrix"),
        "shape_temperature": F(6500.0, "Shape Temperature", 1000.0, 27000.0, group="Shape Matrix"),
        "shape_image_input_gamma": E("sRGB", "Shape Image Input Gamma", GAMMAS, group="Shape Matrix"),
        "shape_exposure": F(1.0, "Shape Exposure (EV)", 0.0, 10.0, group="Shape Matrix"),
        "shape_hardness": F(0.9, "Shape Hardness", 0.0, 1.0, group="Shape Matrix"),
        "hotspot_exposure": F(1.0, "Hotspot Exposure (EV)", 0.0, 10.0, group="Shape Matrix"),
        "hotspot_size": F(0.5, "Hotspot Size", 0.0, 1.0, group="Shape Matrix"),
        "hotspot_falloff": F(0.9, "Hotspot Falloff", 0.0, 1.0, group="Shape Matrix"),
        "hotspot_position": F2([0.5, 0.5], "Hotspot Position", 0.0, 1.0, group="Shape Matrix"),
        "enable_background_input": B(False, "Enable Background Input", group="Shape Matrix"),
        "background_color": C([0.0, 0.0, 0.0], "Background Color", group="Shape Matrix"),
        "background_gamma": E("Linear", "Background Gamma", GAMMAS, group="Shape Matrix"),
    }
