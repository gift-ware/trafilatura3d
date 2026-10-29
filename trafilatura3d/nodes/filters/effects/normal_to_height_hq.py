from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect
from ._n2h import relief


class NormalToHeightHQ(Effect):
    """Height map from a tangent-space normal map by integrating its slopes on the GPU (see _n2h.py),
    the thorough version: a 4 px coarsest level and 16 (Normal) or 48 (High) Jacobi iterations per
    level. Relief Balance weighs details against broad forms (0: broad forms only, 1: details only), Height Normalize stretches the result to
    the full 0-1 range (auto levels), Height Intensity fades it towards 50% gray."""

    type = "normal_to_height_hq"
    label = "Normal To Height HQ"
    category = "Filters/Normal Map"
    catalog = "normal_to_height_hq"
    inputs = {"input": Input(PortType.COLOR, "Input", default=(0.5, 0.5, 1.0, 1.0))}
    outputs = {"output": Output(PortType.GRAY)}
    params = {
        "normal_format": Param(ParamKind.ENUM, "OpenGL", "Normal Format", choices=("DirectX", "OpenGL")),
        "relief_balance": Param(ParamKind.FLOAT, 0.5, "Relief Balance", 0.0, 1.0),
        "height_intensity": Param(ParamKind.FLOAT, 1.0, "Height Intensity", 0.0, 1.0),
        "height_normalize": Param(ParamKind.BOOL, False, "Height Normalize"),
        "quality": Param(ParamKind.ENUM, "Normal", "Quality", choices=("Normal", "High")),
    }

    def encode(self, ctx, job):
        v = job.values
        iterations = 48 if v["quality"] == "High" else 16
        b = v["relief_balance"] * 0.5  # bands at half weight: the relief's strength in the original's range
        h, mm = relief(ctx, job, v["normal_format"] == "DirectX", iterations, 4, (0.5 - b, b, b), True)
        ctx.dispatch(FINAL, job, v, srcs={"relief": (h, True), "stats": (mm, False)})


class _Final(Effect):
    type = "fx_normal_to_height_hq_final"
    registered = False
    inputs = {"relief": Input(PortType.GRAY), "stats": Input(PortType.COLOR)}
    outputs = NormalToHeightHQ.outputs
    params = NormalToHeightHQ.params


FINAL = _Final()
