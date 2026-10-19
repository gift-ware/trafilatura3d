from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode


class ChannelMixer(BasicNode):
    """Each output channel is a weighted sum of the input R, G, B plus a Constant (the
    four components of each parameter are Red, Green, Blue, Constant), all in percent (100 = 1.0),
    clipped to 0..1. Monochrome: every channel takes the Red Channel mix. Alpha passes through."""

    type = "channel_mixer"
    label = "Channel Mixer"
    category = "Filters/Adjustments"
    catalog = "channel_mixer"
    inputs = {"input": Input(PortType.COLOR, "Input")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "red_channel": Param(ParamKind.FLOAT4, [100.0, 0.0, 0.0, 0.0], "Red Channel", -200.0, 200.0),
        "green_channel": Param(ParamKind.FLOAT4, [0.0, 100.0, 0.0, 0.0], "Green Channel", -200.0, 200.0),
        "blue_channel": Param(ParamKind.FLOAT4, [0.0, 0.0, 100.0, 0.0], "Blue Channel", -200.0, 200.0),
        "monochrome": Param(ParamKind.BOOL, False, "Monochrome"),
    }
