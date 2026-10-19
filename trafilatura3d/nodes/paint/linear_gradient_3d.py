from trafilatura3d.nodes.paint._lib import DOC, B, C, F, Generator, mesh_inputs


class LinearGradient3D(Generator):
    """0 at the start position, 1 at the end position, along the line between them (position map)."""
    type = "paint_linear_gradient_3d"
    label = "3D Linear Gradient"
    catalog = "painter.generator.3d_linear_gradient"
    doc = DOC + "generators/3d-linear-gradient"
    inputs = mesh_inputs("position")
    params = {"invert": B("Invert", False), "balance": F("Balance", 0.5, 0.0, 1.0),
              "contrast": F("Contrast", 0.0, 0.0, 1.0),
              "position_start": C("3D Position Start", (0.5, 0.0, 0.5)),
              "position_end": C("3D Position End", (0.5, 1.0, 0.5))}
