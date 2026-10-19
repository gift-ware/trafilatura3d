from trafilatura3d.nodes.base import Input, Node, Param, ParamKind

# The Levels parameters (FLOAT4, per channel): a node that has all of them is edited with the levels editor.
LEVELS_PARAMS = ("level_in_low", "level_in_mid", "level_in_high", "level_out_low", "level_out_high")


class Levels(Node):
    """Remap the input range [in low, in high] to [out low, out high] through a gamma set by In mid: 0.5 linear, above darker (t^(1 + 16 (mid - 0.5))), below lighter. Values are per channel (RGBA); a
    grayscale image uses the first."""

    type = "levels"
    label = "Levels"
    category = "Atomic nodes"
    catalog = "levels"
    inputs = {"input": Input(label="Input")}
    params = {
        "level_in_low": Param(ParamKind.FLOAT4, [0.0, 0.0, 0.0, 0.0], "Level in low", 0.0, 1.0),
        "level_in_high": Param(ParamKind.FLOAT4, [1.0, 1.0, 1.0, 1.0], "Level in high", 0.0, 1.0),
        "level_in_mid": Param(ParamKind.FLOAT4, [0.5, 0.5, 0.5, 0.5], "Level in mid", 0.0, 1.0),
        "level_out_low": Param(ParamKind.FLOAT4, [0.0, 0.0, 0.0, 0.0], "Level out low", 0.0, 1.0),
        "level_out_high": Param(ParamKind.FLOAT4, [1.0, 1.0, 1.0, 1.0], "Level out high", 0.0, 1.0),
        "intermediary_clamp": Param(ParamKind.BOOL, True, "Intermediary clamp"),
    }
