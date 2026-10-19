from trafilatura3d.nodes.base import Input, Node, Param, ParamKind

IDENTITY = [[0.0, 0.0], [1.0, 1.0]]


class Curve(Node):
    """output = curve(input). Color images go through the per-channel curves, then the RGB curve.
    With Apply/Expose curve on, the output is the curve itself (x across the image)."""

    type = "curve"
    label = "Curve"
    category = "Atomic nodes"
    catalog = "curve"
    inputs = {"input": Input(label="Input")}
    params = {
        "expose_curve": Param(ParamKind.BOOL, False, "Apply/Expose curve"),
        "curve_addressing": Param(ParamKind.ENUM, "Clamp", "Curve addressing", choices=("Clamp", "Fold/repeat")),
        "curve": Param(ParamKind.CURVE, IDENTITY, "Curve"),
        "curve_red": Param(ParamKind.CURVE, IDENTITY, "Curve Red"),
        "curve_green": Param(ParamKind.CURVE, IDENTITY, "Curve Green"),
        "curve_blue": Param(ParamKind.CURVE, IDENTITY, "Curve Blue"),
        "curve_alpha": Param(ParamKind.CURVE, IDENTITY, "Curve Alpha"),
    }
