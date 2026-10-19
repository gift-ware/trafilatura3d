from trafilatura3d.nodes.paint._lib import DOC, B, F, Generator, mesh_inputs


class Light(Generator):
    """Lambert lighting plus a highlight from a direction (angles), on the world space normals; attenuated along
    the light with the position map."""
    type = "paint_light"
    label = "Light"
    catalog = "painter.generator.light"
    doc = DOC + "generators/light"
    inputs = mesh_inputs("world_space_normals", "position")
    params = {"invert": B("Invert", False), "horizontal_angle": F("Horizontal Angle", 0.0, 0.0, 360.0),
              "vertical_angle": F("Vertical Angle", 45.0, -90.0, 90.0),
              "highlight_glossiness": F("Highlight Glossiness", 0.3, 0.0, 1.0),
              "highlight_level": F("Highlight Level", 0.5, 0.0, 1.0),
              "light_attenuation": F("Light Attenuation", 0.0, 0.0, 1.0)}
