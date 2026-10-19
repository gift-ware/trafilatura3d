from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import NORMAL_FORMATS, B, E, MatNode
from trafilatura3d.nodes.materials.scan._common import CATEGORY, crop_params


class Crop(MatNode):
    """Returns an area of the input, the rest discarded: the crop is the
    rectangle that T(x) = 0.5 + Transform (x - 0.5) + Offset maps onto the output (x in units of the
    input's short side, Input Size aspect), stretched to the output; outside the input: Background.
    Filtering Nearest samples whole texels. Normal maps (Is Normal, in Normal Format) keep their
    vectors aligned with the crop. Area: the input with the crop lightened (Show Crop Area)."""

    type = "crop"
    label = "Crop"
    category = CATEGORY
    catalog = "crop"
    includes = ("_crop.wgsl",)
    inputs = {"input": Input(label="Input")}
    outputs = {"output": Output(), "area": Output(label="Area")}
    params = {**crop_params(), "is_normal": B(False, "Is Normal (only for Color version)"),
              "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS),
              "filtering": E("Bilinear", "Filtering", ("Bilinear", "Nearest"))}
