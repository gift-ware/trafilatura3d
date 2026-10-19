from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import COLOR, F, MatNode, color_in, gray_in
from trafilatura3d.nodes.materials.scan._common import CATEGORY


class AoCancellation(MatNode):
    """Removes ambient occlusion baked in an albedo: divides the color by the AO (weighted by AO
    Cancellation), the lifted dark areas re-saturated by AO Saturation to recover lost color."""

    type = "ao_cancellation"
    label = "AO Cancellation"
    category = CATEGORY
    catalog = "ao_cancellation"
    inputs = {"basecolor": color_in("Base Color", (0.5, 0.5, 0.5, 1.0)),
              "ambient_occlusion": gray_in("Ambient Occlusion", 1.0)}
    outputs = {"basecolor": Output(COLOR, "Base Color")}
    params = {
        "ao_cancellation": F(0.25, "AO Cancellation", 0.0, 1.0),
        "ao_saturation": F(0.24, "AO Saturation", 0.0, 1.0),
    }
