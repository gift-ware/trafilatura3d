from trafilatura3d.nodes.paint._lib import DOC, B, F, Generator, mesh_inputs


class Distance3D(Generator):
    """Distance from a point (position map): 0 within Offset, 1 at Offset + Radius."""
    type = "paint_distance_3d"
    label = "3D Distance"
    catalog = "painter.generator.3d_distance"
    doc = DOC + "generators/3d-distance"
    inputs = mesh_inputs("position")
    params = {"invert": B("Invert", False), "position_x": F("Position X", 0.5, 0.0, 1.0),
              "position_y": F("Position Y", 0.5, 0.0, 1.0), "position_z": F("Position Z", 0.5, 0.0, 1.0),
              "radius": F("Radius", 0.5, 0.0, 2.0), "offset": F("Offset", 0.0, 0.0, 1.0),
              "contrast": F("Contrast", 0.0, 0.0, 1.0)}
