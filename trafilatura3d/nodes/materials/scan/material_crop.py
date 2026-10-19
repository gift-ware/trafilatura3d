from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import (COLOR, NORMAL_FORMATS, E, MatNode, channel_toggles, material_inputs,
                                           material_outputs)
from trafilatura3d.nodes.materials.scan._common import CATEGORY, crop_params

# The defaults for the channel switches of its scan material nodes (crop, clone patch).
SCAN_ON = ("basecolor", "normal", "roughness", "metallic", "height")


class MaterialCrop(MatNode):
    """Crop on every channel of a material at once. Outside the input, color channels take the
    Background color, grayscale ones its red value, the normal stays flat. Area: the base color with
    the crop lightened."""

    type = "material_crop"
    main_input = "basecolor"  # the library graph's primary input: Relative to input follows it
    label = "Material Crop"
    category = CATEGORY
    catalog = "material_crop"
    includes = ("_crop.wgsl",)
    inputs = material_inputs()
    outputs = {**material_outputs(), "area": Output(COLOR, "Area")}
    params = {**channel_toggles(SCAN_ON), **crop_params(),
              "normal_format": E("OpenGL", "Normal Format", NORMAL_FORMATS)}
