import math

from trafilatura3d.nodes.base import Input, Node, Param, ParamKind


class DirectionalBlur(Node):
    """Blur along one direction: Angle in turns, clockwise from +X; Intensity in pixels of a 256 px
    image (the blur spans Intensity on each side). The box along the line is computed by doubling
    (`directional_blur`): its cost grows with the log of the length, not with the length."""

    type = "directional_blur"
    label = "Directional blur"
    category = "Atomic nodes"
    catalog = "directional_blur"
    inputs = {"input": Input(label="Input")}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 64.0),
        "angle": Param(ParamKind.FLOAT, 0.0, "Angle", 0.0, 1.0),
    }
    internal_uniforms = {"a_": ParamKind.FLOAT2, "b_": ParamKind.FLOAT2}

    def encode(self, ctx, job):
        s = job.inputs.get("input")
        src = (ctx.ev.texture(s[0], s[1]), s[0].is_gray(s[1])) if s else (ctx.ev.constant(self.inputs["input"].default),
                                                                          False)
        directional_blur(ctx, job, src, job.is_gray("output"), job.values["intensity"], job.values["angle"])


PASS = DirectionalBlur()


def directional_blur(ctx, job, src, gray: bool, intensity: float, angle: float, dst=None):
    """The average of `src` ((texture, gray)) over the segment of half-length `intensity` (pixels at 256) along
    `angle` (turns), into `dst` (default: the job's output); returns the texture written. 2^k evenly spaced samples
    (about one per pixel) by k passes of two: the first averages the segment's start and the next sample, each
    next one the result and itself shifted by twice the previous span."""
    out = dst if dst is not None else ctx.ev.texture(job, "output")
    a = angle * 2.0 * math.pi
    d = (math.cos(a), math.sin(a))  # clockwise on screen: +y is down
    length = max(intensity, 0.0) / 256.0  # in uv
    k = max(1, math.ceil(math.log2(2 * math.ceil(length * job.size[0]) + 1)))
    step = 2.0 * length / (2 ** k - 1)
    cur = src
    for j in range(k):
        start = [-length * d[0], -length * d[1]] if j == 0 else [0.0, 0.0]
        shift = step * 2 ** j
        values = {**job.values, "intensity": intensity, "angle": angle, "a_": start,
                  "b_": [start[0] + shift * d[0], start[1] + shift * d[1]]}
        target = out if j == k - 1 else ctx.scratch(job, gray)
        ctx.dispatch(PASS, job, values, srcs={"input": cur}, dsts={"output": target})
        cur = (target, gray)
    return out
