from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, B, C, E, F, F2, F3
from trafilatura3d.nodes.materials.hdri._common import COLOR_MODES, GAMMAS, HDR, UP_VECTORS, HdriNode
from trafilatura3d.nodes.materials.hdri.plane_light import PATTERNS, POSITION_MODES, UV_MODES


class LineLight(HdriNode):
    """A line light between two points in 3D: a flat card of Line Thickness along the segment,
    facing the origin and turned by Line Rotation around its length, ray traced from the panorama
    centre. Points come from the panorama on the ceiling/ground plane (Line Absolute Height), at a
    distance, or from world coordinates. The Pattern masks the card across and along its length
    (stretched, ends kept square, or repeated with spacing), the Shape Image colors it."""

    type = "line_light"
    label = "Line Light"
    catalog = "line_light"
    unsupported = {"Enable Backgound Input": "the documentation's typo of Enable Background Input"}
    base_defaults = HDR
    includes = ("_light.wgsl",)
    inputs = {"background_image_input": Input(COLOR, "Background Image Input", (0.0, 0.0, 0.0, 1.0)),
              "shape_image_input": Input(COLOR, "Shape Image Input", (1.0, 1.0, 1.0, 1.0)),
              "pattern_image_input": Input(GRAY, "Pattern Image Input", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    G = "Position Coordinates"
    params = {
        "position_mode": E("Ground / Ceiling", "Position Mode", POSITION_MODES),
        "show_ground_grid": B(False, "Show Ground Grid"),
        "up_vector": E("Z Up", "Up Vector", UP_VECTORS, group=G),
        "point_1_uv_position": F2([0.25, 0.25], "Point 1 UV Position", group=G),
        "point_2_uv_position": F2([0.75, 0.25], "Point 2 UV Position", group=G),
        "point_1_world_position": F3([-1.0, -1.0, 1.0], "Point 1 World Position", -2.0, 2.0, group=G),
        "point_2_world_position": F3([1.0, 1.0, 1.0], "Point 2 World Position", -2.0, 2.0, group=G),
        "line_absolute_height": F(1.0, "Line Absolute Height", 0.0, 1.0, group=G),
        "distance_from_origin": F(1.0, "Distance from Origin", 0.0, 1.0, group=G),
        "camera_position": F3([0.0, 0.0, 0.0], "Camera Position", 0.0, 1.0, group=G),
        "shape_color_mode": E("Temperature (Kelvin)", "Shape Color Mode", COLOR_MODES, group=G),
        "color": C([1.0, 1.0, 1.0], "Color", group=G),
        "temperature": F(6500.0, "Temperature", 1000.0, 27000.0, group=G),
        "shape_image_uv_mode": E("Stretch Middle only", "Shape Image UV Mode", UV_MODES, group=G),
        "shape_image_repeat_spacing": F(0.0, "Shape Image Repeat Spacing", 0.0, 1.0, group=G),
        "shape_image_gamma": E("sRGB", "Shape Image Gamma", GAMMAS, group=G),
        "exposure": F(1.0, "Exposure (EV)", 0.0, 10.0, group=G),
        "line_rotation": F(0.0, "Line Rotation", 0.0, 1.0, group=G),
        "line_thickness": F(0.1, "Line Thickness", 0.0, 1.0, group=G),
        "pattern": E("Smooth Square", "Pattern", PATTERNS, group=G),
        "pattern_hardness": F(0.9, "Pattern Hardness", 0.0, 1.0, group=G),
        "pattern_uv_mode": E("Stretch Middle only", "Pattern UV Mode", UV_MODES, group=G),
        "pattern_repeat_spacing": F(0.0, "Pattern Repeat Spacing", 0.0, 1.0, group=G),
        "enable_ground_clipping": B(False, "Enable Ground Clipping", group=G),
        "ground_height": F(-1.0, "Ground Height", -2.0, 0.0, group=G),
        "enable_background_input": B(False, "Enable Background Input", group=G),
        "background_color": C([0.0, 0.0, 0.0], "Background Color", group=G),
        "background_gamma": E("Linear", "Background Gamma", GAMMAS, group=G),
    }
