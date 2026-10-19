from trafilatura3d.nodes.base import Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, SEED, C, E, F, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import BorderGenerator


class UVChecker(BorderGenerator):
    """A checker of square tiles over the UVs (Checker Grid Tile tiles across), alternating the checker color and a
    darker shade of it; a grid of lines on the tile edges and their quarters (Checker Grid Opacity); the UV island
    borders drawn in UV Borders Color, UV Borders Width px at 256 wide. Random: each island's checker color is a
    random hue from the Seed."""
    type = "paint_uv_checker"
    label = "UV Checker"
    catalog = "painter.generator.uv_checker"
    doc = DOC + "generators/uv-checker"
    inputs = mesh_inputs("uv_islands")
    outputs = {"output": Output(PortType.COLOR)}
    params = {"checker_color_mode": E("Checker Color Mode", ("Uniform", "Random")),
              "checker_color": C("Checker Color", (0.85, 0.45, 0.1)),
              "checker_grid_tile": F("Checker Grid Tile", 8.0, 1.0, 64.0),
              "checker_grid_opacity": F("Checker Grid Opacity", 0.5, 0.0, 1.0),
              "uv_borders_color": C("UV Borders Color", (1.0, 1.0, 1.0)),
              "uv_borders_width": F("UV Borders Width", 1.0, 0.0, 8.0)}
    unsupported = {**SEED}
