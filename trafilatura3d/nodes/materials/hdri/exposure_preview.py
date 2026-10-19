from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import COLOR, F
from trafilatura3d.nodes.materials.hdri._common import HdriNode


class ExposurePreview(HdriNode):
    """Bands of the input at evenly spaced exposures, brightest (Max) at the top, darkest (Min) at
    the bottom; the number of bands grows with the output resolution (2 at 64 px, 8 from 4K)."""

    type = "exposure_preview"
    label = "Exposure Preview"
    catalog = "exposure_preview"
    inputs = {"input": Input(COLOR, "Input", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(COLOR)}
    params = {
        "max_exposure": F(0.0, "Max Exposure (EV)", -8.0, 8.0),
        "min_exposure": F(-6.0, "Min Exposure (EV)", -8.0, 8.0),
    }
