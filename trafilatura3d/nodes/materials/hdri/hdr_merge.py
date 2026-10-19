from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, F, I
from trafilatura3d.nodes.materials.hdri._common import HDR, HdriNode

N = 16


class HdrMerge(HdriNode):
    """Merges bracketed LDR exposures (sRGB) into one linear HDR image. Input 1 is the most
    under-exposed, each next one Exposure Delta stops brighter; radiance is expressed in the units
    of the brightest exposure (its whites = 1). Every pixel is a weighted average favouring well
    exposed values (hat weight, neither crushed nor clipped). Radiance above 2^White Point is
    clipped."""

    type = "hdr_merge"
    label = "HDR Merge"
    catalog = "hdr_merge"
    base_defaults = HDR
    channels = tuple(f"input_{i}" for i in range(1, N + 1))
    inputs = {f"input_{i}": Input(COLOR, f"Input {i}", (0.0, 0.0, 0.0, 1.0)) for i in range(1, N + 1)}
    outputs = {"output": Output(COLOR)}
    params = {
        "inputs": I(12, "Inputs", 2, 16),
        "exposure_delta": F(2.0, "Exposure Delta (EV)", 0.0, 4.0),
        "white_point": F(9.0, "White Point", 0.0, 13.0),
    }
