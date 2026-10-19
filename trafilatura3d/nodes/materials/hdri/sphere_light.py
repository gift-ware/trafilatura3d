from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, B, C, E, F, F2, F3
from trafilatura3d.nodes.materials.hdri._common import SPHERE_COLOR_MODES, GAMMAS, HDR, UP_VECTORS, HdriNode


class SphereLight(HdriNode):
    """A sphere seen from the panorama centre (ray traced): placed by direction (Position on the
    panorama) and Distance from Origin, or at a world position. Shading: none (flat emitter), limb
    darkening (a sun, a star), or lit by a light at Shading Light World Position (a planet, with a
    soft terminator). The Shape Image maps onto the sphere like a globe, spun and tilted by Sphere Rotation;
    Camera Position moves the panorama's viewpoint."""

    type = "sphere_light"
    label = "Sphere Light"
    catalog = "sphere_light"
    unsupported = {"Enable Backgound Input": "the documentation's typo of Enable Background Input"}
    base_defaults = HDR
    includes = ("_light.wgsl",)
    inputs = {"background_image_input": Input(COLOR, "Background Image Input", (0.0, 0.0, 0.0, 1.0)),
              "shape_image_input": Input(COLOR, "Shape Image Input", (1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    G = "Position Coordinates"
    params = {
        "position_mode": E("Distance from Origin", "Position Mode", ("Distance from Origin", "World Position")),
        "up_vector": E("Z Up", "Up Vector", UP_VECTORS, group=G),
        "sphere_world_position": F3([-1.0, -1.0, 1.0], "Sphere World Position", -2.0, 2.0, group=G),
        "position": F2([0.6086, 0.3928], "Position", group=G),
        "distance_from_origin": F(5.0, "Distance from Origin", 0.0, 20.0, group=G),
        "camera_position": F3([0.0, 0.0, 0.0], "Camera Position", 0.0, 1.0, group=G),
        "shape_color_mode": E("Temperature (Kelvin)", "Shape Color Mode", SPHERE_COLOR_MODES, group=G),
        "color": C([1.0, 1.0, 1.0], "Color", group=G),
        "shape_temperature": F(6500.0, "Shape Temperature", 1000.0, 27000.0, group=G),
        "sphere_image_input_gamma": E("sRGB", "Sphere Image Input Gamma", GAMMAS, group=G),
        "sphere_rotation": F2([0.0, 0.0], "Sphere Rotation", 0.0, 1.0, group=G),
        "exposure": F(1.0, "Exposure (EV)", 0.0, 10.0, group=G),
        "sphere_radius": F(0.5, "Sphere Radius", 0.0, 1.0, group=G),
        "sphere_hardness": F(0.9, "Sphere Hardness", 0.0, 1.0, group=G),
        "shading": E("None", "Shading", ("None", "Limb Darkening", "Shading Light"), group=G),
        "shading_light_world_position": F3([0.0, 0.0, 1.0], "Shading Light World Position", -1.0, 1.0, group=G),
        "penombra_transparency": F(0.0, "Penombra Transparency", 0.0, 1.0, group=G),
        "enable_background_input": B(False, "Enable Background Input", group=G),
        "background_color": C([0.0, 0.0, 0.0], "Background Color", group=G),
        "background_gamma": E("Linear", "Background Gamma", GAMMAS, group=G),
    }
