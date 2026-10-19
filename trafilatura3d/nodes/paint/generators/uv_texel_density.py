from trafilatura3d.nodes.base import Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, C, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import GenNode


class UVTexelDensity(GenNode):
    """The computed `texel_density` map (log2 of the density relative to the Texture Set's mean) as colors: half
    the mean or less = Color Low, the mean = Color Medium, twice the mean or more = Color High, blended between."""
    type = "paint_uv_texel_density"
    label = "UV Texel Density"
    catalog = "painter.generator.uv_texel_density"
    doc = DOC + "generators/uv-texel-density"
    inputs = mesh_inputs("texel_density")
    outputs = {"output": Output(PortType.COLOR)}
    params = {"color_low": C("Color Low", (0.1, 0.25, 0.9)), "color_medium": C("Color Medium", (0.15, 0.8, 0.2)),
              "color_high": C("Color High", (0.9, 0.15, 0.1))}
