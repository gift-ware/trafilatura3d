from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect
from ._n2h import relief


class NormalToHeight(Effect):
    """Height map from a tangent-space normal map by integrating its slopes on the GPU (see _n2h.py),
    the quick version: few Jacobi iterations per level from an 8 px coarsest level. Relief Balance
    weighs the low, mid and high frequencies (0.5 = as integrated); Global Opacity scales the relief around
    50% gray (default 0.36 = as integrated)."""

    type = "normal_to_height"
    label = "Normal to Height"
    category = "Filters/Normal Map"
    catalog = "normal_to_height"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "relief_balance": Param(ParamKind.FLOAT3, [0.5, 0.5, 0.5], "Relief Balance", 0.0, 1.0),
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
        "global_opacity": Param(ParamKind.FLOAT, 0.36, "Global Opacity", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        v = job.values
        h, _ = relief(ctx, job, v["normal_format"] == "DirectX", 6, 8, v["relief_balance"], False)
        ctx.dispatch(FINAL, job, v, srcs={"relief": (h, True)})


class _Final(Effect):
    type = "fx_normal_to_height_final"
    registered = False
    inputs = {"relief": Input(PortType.GRAY)}
    outputs = NormalToHeight.outputs
    params = NormalToHeight.params


FINAL = _Final()
