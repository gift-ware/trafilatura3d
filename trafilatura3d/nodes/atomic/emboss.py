from trafilatura3d.nodes.base import Input, Node, Param, ParamKind, PortType


class Emboss(Node):
    """Light the input with the slopes of the Intensity input height map: on slopes facing the light (Light angle,
    turns) the Highlight color is added, on the others the input is multiplied by the Shadow color, both in
    proportion to the slope. The documentation gives no formula; both colors can be
    driven from one light intensity I as highlight I and shadow 1 - I (Creases filter), or the shadow left white to
    light one side only (Ambient Occlusion)."""

    type = "emboss"
    label = "Emboss"
    category = "Atomic nodes"
    catalog = "emboss"
    inputs = {"input": Input(label="Input"), "intensity_input": Input(PortType.GRAY, "Intensity input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 5.0, "Intensity", 0.0, 32.0),
        "light_angle": Param(ParamKind.FLOAT, 0.0, "Light angle", 0.0, 1.0),
        "highlight_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Highlight color"),
        "shadow_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Shadow color"),
    }
