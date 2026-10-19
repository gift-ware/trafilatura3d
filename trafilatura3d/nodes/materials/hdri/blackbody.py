from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, F
from trafilatura3d.nodes.materials.hdri._common import HdrGenerator


class Blackbody(HdrGenerator):
    """Uniform color of a black body at a temperature in Kelvin (linear RGB, brightest channel 1):
    red-orange at 1000 K, white near 6600 K, blue above."""

    type = "blackbody"
    label = "Blackbody"
    catalog = "blackbody"
    seamless = True
    outputs = {"output": Output(COLOR)}
    params = {"temperature": F(3000.0, "Temperature", 1000.0, 27000.0)}
