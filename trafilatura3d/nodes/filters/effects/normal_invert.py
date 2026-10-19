from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect


class NormalInvert(Effect):
    """Inverts the chosen channels of a normal map (c -> 1 - c)."""

    type = "normal_invert"
    label = "Normal Invert"
    category = "Filters/Normal Map"
    catalog = "normal_invert"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "invert_red": Param(ParamKind.BOOL, False, "Invert Red"),
        "invert_green": Param(ParamKind.BOOL, True, "Invert Green"),
        "invert_blue": Param(ParamKind.BOOL, False, "Invert Blue"),
        "invert_alpha": Param(ParamKind.BOOL, False, "Invert Alpha"),
    }
