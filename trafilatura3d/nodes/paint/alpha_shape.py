from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.paint._lib import B, E, F, PaintNode

ORDERS = ("Levels, then falloff", "Falloff, then levels")


class AlphaShape(PaintNode):
    """The three parameters every brush alpha shares, applied to its base texture: Invert (1 - base), Hardness
    (levels: input black at 1 - 2h below one half, input white at 2 (1 - h) above it) and Shape (a square falloff
    M = S((1 - |x|) / s) S((1 - |y|) / s), S smoothstep, x and y in -1..1, subtracted as clamp(base + M - 1);
    0 = none). Order: the levels before or after the falloff, as each alpha does. Threshold: white wherever the result is
    above zero once stored in 8 bits (0.5 / 255), black elsewhere (the hard-edged alphas)."""
    type = "paint_alpha_shape"
    label = "Alpha Shape"
    category = "Paint/Alphas"
    inputs = {"input": Input(PortType.GRAY, label="Input")}
    outputs = {"output": Output(PortType.GRAY)}
    params = {"hardness": F("Hardness", 0.5, 0.0, 1.0), "shape": F("Shape", 0.75, 0.0, 1.0),
              "invert": B("Invert"), "order": E("Order", ORDERS), "threshold": B("Threshold")}
