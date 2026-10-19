from trafilatura3d.nodes.base import Output
from trafilatura3d.nodes.materials._base import GRAY, MatNode, E, F, gray_in

MODES = ("Balanced height", "Bottom height priority")


class HeightBlend(MatNode):
    """Combines two heightmaps by their heights (ours, `_hblend.wgsl`): Height Offset favours one of them first
    (below 0.5 the top is scaled down; above it Balanced height scales the bottom down, Bottom height priority
    raises the top towards 1). The top shows where it is at or above the bottom, falling off linearly over
    1 - Contrast below it; the mask is that times Opacity, and the blended height mixes the moved heights by it.
    The Mask input scales the top and the bottom's lowering (unconnected: white, the blend everywhere)."""

    type = "height_blend"
    main_input = "height_bottom"  # the library graph's primary input: Relative to input follows it
    label = "Height Blend"
    category = "Material filters/Effects (Material)"
    catalog = "height_blend"
    includes = ("_hblend.wgsl",)
    inputs = {"height_top": gray_in("Height Top", 0.5), "height_bottom": gray_in("Height Bottom", 0.5),
              "mask": gray_in("Mask (optional)", 1.0)}
    outputs = {"height": Output(GRAY, "Height"), "mask": Output(GRAY, "Mask")}
    params = {
        "height_offset": F(0.5, "Height Offset", 0.0, 1.0),
        "contrast": F(0.9, "Contrast", 0.0, 1.0),
        "mode": E("Balanced height", "Mode", MODES),
        "opacity": F(1.0, "Opacity", 0.0, 1.0),
    }
