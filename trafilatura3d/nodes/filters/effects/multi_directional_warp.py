from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._fx import Effect

DIRECTIONS = ("1", "2", "4")


class MultiDirectionalWarp(Effect):
    """Directional warps in opposite directions while the content stays in place: 1 = Warp Angle
    and its opposite, 2 = also the perpendicular axis, 4 = also the diagonals. The passes are averaged
    or combined with Max / Min, or Chained (each pass warps the previous result); Mode applies with
    2 or 4 directions, 1 is always averaged. Distance = Intensity x Intensity Input, pixels at 256."""

    type = "multi_directional_warp"
    label = "Multi Directional Warp"
    category = "Filters/Effects"
    catalog = "multi_directional_warp"
    inputs = {"input": Input(label="Input"),
              "intensity_input": Input(PortType.GRAY, "Intensity Input", default=(1.0, 1.0, 1.0, 1.0))}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 20.0),
        "warp_angle": Param(ParamKind.FLOAT, 0.0, "Warp Angle", 0.0, 1.0),
        "mode": Param(ParamKind.ENUM, "Average", "Mode", choices=("Average", "Max", "Min", "Chain")),
        "directions": Param(ParamKind.ENUM, "4", "Directions", choices=DIRECTIONS),
    }
    internal_uniforms = {"pass_angle": ParamKind.FLOAT}  # >= 0: one chained pass at this angle

    def encode(self, ctx, job):
        v = job.values
        n = 2 * int(v["directions"])
        if v["mode"] != "Chain" or n == 2:
            ctx.dispatch(self, job, {**v, "pass_angle": -1.0})
            return
        gray = job.is_gray("output")
        angles = [v["warp_angle"] + k / n for k in range(n)]
        bufs = [ctx.scratch(job, gray), ctx.scratch(job, gray)]
        cur = None
        for i, a in enumerate(angles):
            last = i == len(angles) - 1
            dst = {} if last else {"output": bufs[i % 2]}
            srcs = {"input": (cur, gray)} if cur is not None else None
            ctx.dispatch(self, job, {**v, "pass_angle": a % 1.0}, srcs=srcs, dsts=dst)
            cur = bufs[i % 2]
