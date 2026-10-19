from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

SOURCES = tuple(f"Input {i} {ch}" for i in (1, 2) for ch in ("Red", "Green", "Blue", "Alpha"))


class ChannelShuffle(Node):
    """Every channel of the output is taken from a chosen R, G, B or A of either input (a grayscale input reads as
    R = G = B = A = its value)."""

    type = "channel_shuffle"
    label = "Channels shuffle"
    category = "Atomic nodes"
    catalog = "channel_shuffle"
    inputs = {"input_1": Input(label="Input 1"), "input_2": Input(label="Input 2")}
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "red_channel": Param(ParamKind.ENUM, SOURCES[0], "Red channel", choices=SOURCES),
        "green_channel": Param(ParamKind.ENUM, SOURCES[1], "Green channel", choices=SOURCES),
        "blue_channel": Param(ParamKind.ENUM, SOURCES[2], "Blue channel", choices=SOURCES),
        "alpha_channel": Param(ParamKind.ENUM, SOURCES[3], "Alpha channel", choices=SOURCES),
    }
