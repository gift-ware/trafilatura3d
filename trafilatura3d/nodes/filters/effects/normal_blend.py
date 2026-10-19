from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect

FLAT = (0.5, 0.5, 1.0, 1.0)


class NormalBlend(Effect):
    """NormalFG over NormalBG by Opacity (times the Mask when Use Mask is on), interpolating the
    vectors and renormalizing them, so the result stays a valid normal map."""

    type = "normal_blend"
    main_input = "normal_bg"  # the library graph's primary input: Relative to input follows it
    label = "Normal Blend"
    category = "Filters/Normal Map"
    catalog = "normal_blend"
    inputs = {"normal_fg": Input(PortType.COLOR, "NormalFG", default=FLAT),
              "normal_bg": Input(PortType.COLOR, "NormalBG", default=FLAT),
              "mask": Input(PortType.GRAY, "Mask", default=(1.0, 1.0, 1.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "opacity": Param(ParamKind.FLOAT, 1.0, "Opacity", 0.0, 1.0),
        "use_mask": Param(ParamKind.BOOL, True, "Use Mask"),
    }
