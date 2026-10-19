from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, GRAY, B, C, E, F, F2, F3
from trafilatura3d.nodes.materials.hdri._common import COLOR_MODES, GAMMAS, HDR, UP_VECTORS, HdriNode

UV_MODES = ("Stretch", "Stretch Middle only", "Repeat + Spacing")
PATTERNS = ("Smooth Square", "Sharp Square", "Cone", "Hemisphere", "Image Input")
POSITION_MODES = ("Ground / Ceiling", "Distance from Origin", "World Positions")


class PlaneLight(HdriNode):
    """A rectangular light card in 3D, ray traced from the panorama centre. Ground / Ceiling: the
    card lies flat on the ceiling (point above the horizon) or the ground (below) at Plane Absolute
    Height; Distance from Origin and World Positions: the card faces the origin. Plane Scale and
    Size set its extent, Plane Rotation spins it; a Pattern masks it, the Shape Image or a
    temperature colors it. Ground clipping hides what lies under Ground Height."""

    type = "plane_light"
    label = "Plane Light"
    catalog = "plane_light"
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
        "plane_uv_position": F2([0.5, 0.25], "Plane UV Position", group=G),
        "plane_world_position": F3([-1.0, -1.0, 1.0], "Plane World Position", -2.0, 2.0, group=G),
        "plane_absolute_height": F(1.0, "Plane Absolute Height", 0.0, 1.0, group=G),
        "distance_from_origin": F(1.0, "Distance from Origin", 0.0, 1.0, group=G),
        "camera_position": F3([0.0, 0.0, 0.0], "Camera Position", 0.0, 1.0, group=G),
        "shape_color_mode": E("Temperature (Kelvin)", "Shape Color Mode", COLOR_MODES, group=G),
        "color": C([1.0, 1.0, 1.0], "Color", group=G),
        "temperature": F(6500.0, "Temperature", 1000.0, 27000.0, group=G),
        "shape_image_uv_mode": E("Stretch Middle only", "Shape Image UV Mode", UV_MODES, group=G),
        "shape_image_repeat_spacing": F(0.2, "Shape Image Repeat Spacing", 0.0, 1.0, group=G),
        "shape_image_gamma": E("sRGB", "Shape Image Gamma", GAMMAS, group=G),
        "exposure": F(1.0, "Exposure (EV)", 0.0, 10.0, group=G),
        "plane_scale": F(0.5, "Plane Scale", 0.0, 1.0, group=G),
        "plane_size": F2([1.0, 1.0], "Plane Size", 0.0, 1.0, group=G),
        "plane_rotation": F3([0.0, 0.0, 0.0], "Plane Rotation", 0.0, 1.0, group=G),
        "pattern": E("Smooth Square", "Pattern", PATTERNS, group=G),
        "pattern_hardness": F(0.9, "Pattern Hardness", 0.0, 1.0, group=G),
        "pattern_uv_mode": E("Stretch Middle only", "Pattern UV Mode", UV_MODES[:2], group=G),
        "enable_ground_clipping": B(False, "Enable Ground Clipping", group=G),
        "ground_height": F(-1.0, "Ground Height", -2.0, 0.0, group=G),
        "enable_background_input": B(False, "Enable Background Input", group=G),
        "background_color": C([0.0, 0.0, 0.0], "Background Color", group=G),
        "background_gamma": E("Linear", "Background Gamma", GAMMAS, group=G),
    }
