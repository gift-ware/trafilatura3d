from trafilatura3d.nodes.base import Input, Node, Param, ParamKind


class Sharpen(Node):
    """Unsharp mask: adds the difference between the input and its 3x3 blur, scaled by Intensity.
    Punchthrough Alpha leaves the alpha untouched."""

    type = "sharpen"
    label = "Sharpen"
    category = "Atomic nodes"
    catalog = "sharpen"
    inputs = {"input": Input(label="Input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 1.0, "Intensity", 0.0, 5.0),
        "punchthrough_alpha": Param(ParamKind.BOOL, True, "Punchthrough Alpha"),
    }
