from trafilatura3d.nodes.base import Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, SEED, E, mesh_inputs
from trafilatura3d.nodes.paint.generators._common import GenNode, custom


class UVRandomColor(GenNode):
    """One flat color per UV island (computed `uv_islands` map: two random numbers per island) drawn from the
    Seed: a random hue, saturation and value (Random) or a color picked along the Custom Gradient image (its middle
    row, left to right). Black outside the UVs."""
    type = "paint_uv_random_color"
    label = "UV Random Color"
    catalog = "painter.generator.uv_random_color"
    doc = DOC + "generators/uv-random-color"
    inputs = {**mesh_inputs("uv_islands"), "custom_gradient": custom("Custom Gradient")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"color_source_mode": E("Color Source Mode", ("Random", "Custom Gradient"))}
    unsupported = {**SEED}
