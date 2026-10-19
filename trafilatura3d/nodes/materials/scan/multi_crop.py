from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import B, I, MatNode
from trafilatura3d.nodes.materials.scan._common import CATEGORY, crop_params

N = 8


class MultiCrop(MatNode):
    """The same Crop on up to eight inputs (Input Count), e.g. a set of multi-angle photos. Area: Input 1
    with the crop lightened."""
    counted_inputs = {"input_count": "input_"}

    type = "multi_crop"
    label = "Multi Crop"
    category = CATEGORY
    catalog = "multi_crop"
    includes = ("_crop.wgsl",)
    channels = tuple(f"input_{i}" for i in range(1, N + 1))
    inputs = {f"input_{i}": Input(label=f"Input {i}") for i in range(1, N + 1)}
    outputs = {**{f"output_{i}": Output(label=f"Output {i}") for i in range(1, N + 1)}, "area": Output(label="Area")}
    params = {"input_count": I(1, "Input Count", 1, 8), **crop_params(),
              "is_normal": B(False, "Is Normal (only for Color version)")}
