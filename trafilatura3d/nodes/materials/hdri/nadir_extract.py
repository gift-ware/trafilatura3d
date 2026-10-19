from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, E, F
from trafilatura3d.nodes.materials.hdri._common import GAMMAS, HdriNode


class NadirExtract(HdriNode):
    """View straight down at the nadir of a panorama (gnomonic projection): Scale zooms (1 = 90
    degree field of view), Rotation spins around the vertical axis. Output Gamma sRGB encodes the
    (linear) panorama for display, Linear keeps its values."""

    type = "nadir_extract"
    label = "Nadir Extract"
    catalog = "nadir_extract"
    inputs = {"input": Input(COLOR, "Input", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "scale": F(5.0, "Scale", 0.0, 10.0),
        "rotation": F(0.0, "Rotation", 0.0, 1.0),
        "output_gamma": E("sRGB", "Output Gamma", GAMMAS),
    }
