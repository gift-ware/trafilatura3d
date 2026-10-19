from trafilatura3d.nodes.base import Input, Param, ParamKind

from ._fx import Effect


class Swirl(Effect):
    """Twists the input around a centre: positions are taken into the swirl's frame by the Matrix (local =
    Matrix x (position - centre), the centre at 0.5 - Matrix^-1 x Offset, its nearest copy when tiling), where
    the disc of radius 0.5 turns by Amount x pi / 2 x (1 - smoothstep(0, 0.5, r))^2 radians (Amount / 4 turns
    at the centre, easing out to nothing at the rim), then taken back."""

    type = "swirl"
    label = "Swirl"
    category = "Filters/Effects"
    catalog = "swirl"
    inputs = {"input": Input(label="Input")}
    params = {
        "matrix": Param(ParamKind.FLOAT4, [2.0, 0.0, 0.0, 2.0], "Matrix", -2.0, 2.0, group="Matrix"),
        "offset": Param(ParamKind.FLOAT2, [0.0, 0.0], "Offset", 0.0, 1.0, group="Matrix"),
        "amount": Param(ParamKind.FLOAT, 8.0, "Amount", -16.0, 16.0),
    }
