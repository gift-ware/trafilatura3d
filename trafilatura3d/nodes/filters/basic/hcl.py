from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class HCL(BasicNode):
    """Hue, chroma and luma: hexagonal hue, chroma max - min, luma Rec. 709 kept while hue and chroma change. The
    hue turns by Hue (degrees or turns), wrapped into the circle once (past it: gray below, red above); Chrominance
    and Luminance add to chroma and luma, or with Proportional the two percentages move them towards their limits
    (towards 1 when positive, scaled down when negative). Clamp Chroma keeps chroma from going negative (else the
    hue flips). RGB is clipped to [0, 1]; alpha passes through."""

    type = "hcl"
    label = "HCL"
    category = "Filters/Adjustments"
    pointwise = True
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "hue_mode": Param(ParamKind.ENUM, "Degrees", "Hue Mode", choices=("Turns", "Degrees")),
        "hue_degrees": Param(ParamKind.FLOAT, 0.0, "Hue (Degrees)", -180.0, 180.0),
        "hue_turns": Param(ParamKind.FLOAT, 0.0, "Hue (Turns)", -1.0, 1.0),
        "proportional": Param(ParamKind.BOOL, False, "Proportional"),
        "chrominance": Param(ParamKind.FLOAT, 0.0, "Chrominance", -1.0, 1.0),
        "luminance": Param(ParamKind.FLOAT, 0.0, "Luminance", -1.0, 1.0),
        "proportional_chrominance": Param(ParamKind.FLOAT, 0.0, "Proportional Chrominance", -100.0, 100.0),
        "proportional_luminance": Param(ParamKind.FLOAT, 0.0, "Proportional Luminance", -100.0, 100.0),
        "clamp_chroma": Param(ParamKind.BOOL, True, "Clamp Chroma"),
    }
