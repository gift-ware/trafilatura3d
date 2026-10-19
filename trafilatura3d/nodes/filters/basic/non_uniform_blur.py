from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._lib import BasicNode

SHRINK = 2.57  # each pass's radius is ((Samples - 1) / (Samples + 1))^SHRINK x the previous one's (fitted)
TURN = 0.381966  # each pass's ring is turned by the golden angle (in turns) from the previous one


class NonUniformBlur(BasicNode):
    """Blur whose radius is Intensity x Blur Map (pixels at 256 px; black = sharp): iterated sparse rings (as
    in real-time multi-pass blurs), each pass the mean of the pixel and Blades samples on an ellipse around it,
    on the previous pass's result, its radius ((Samples - 1) / (Samples + 1))^SHRINK times the previous one and
    its ring turned by the golden angle. Samples passes, those under a quarter of a pixel skipped. Anisotropy squeezes the ellipse across
    Angle, Asymmetry folds its back half forward (a one-sided smear)."""

    type = "non_uniform_blur"
    label = "Non Uniform Blur"
    category = "Filters/Blurs"
    catalog = "non_uniform_blur"
    inputs = {"input": Input(label="Input"),
              "blur_map": Input(PortType.GRAY, "Blur Map", (1.0, 1.0, 1.0, 1.0))}
    params = {
        "intensity": Param(ParamKind.FLOAT, 10.0, "Intensity", 0.0, 50.0),
        "anisotropy": Param(ParamKind.FLOAT, 0.0, "Anisotropy", 0.0, 1.0),
        "asymmetry": Param(ParamKind.FLOAT, 0.0, "Asymmetry", 0.0, 1.0),
        "angle": Param(ParamKind.FLOAT, 0.0, "Angle", 0.0, 1.0),
        "samples": Param(ParamKind.INT, 4, "Samples", 1, 16),
        "blades": Param(ParamKind.INT, 5, "Blades", 1, 9),
    }
    internal_uniforms = {"radius": ParamKind.FLOAT, "inner": ParamKind.FLOAT}

    def encode(self, ctx, job):
        non_uniform_blur(ctx, job, None, None, job.values, None)


NODE = NonUniformBlur()


def non_uniform_blur(ctx, job, src, blur_map, values: dict, dst) -> None:
    """The passes of Non Uniform Blur with `values` (its parameters) inside another node's `job`: `src` and
    `blur_map` are (texture, is_gray), None = the job's own input of that name; the result goes to `dst`
    (texture, None = the job's output)."""
    intensity = max(values["intensity"], 0.0)
    samples = min(max(values["samples"], 1), 16)
    px = job.size[0] / 256.0  # output pixels per pixel at 256
    ratio = ((samples - 1) / (samples + 1)) ** SHRINK  # more passes, each closer to the previous
    radii = [intensity * ratio ** k for k in range(samples)]
    radii = [r for r in radii if r * px >= 0.25] or [intensity]
    gray = job.is_gray("output")
    bufs = [ctx.scratch(job, gray), ctx.scratch(job, gray)]
    cur = src
    for k, radius in enumerate(radii):
        srcs = {port: s for port, s in (("input", cur), ("blur_map", blur_map)) if s is not None}
        out = dst if k == len(radii) - 1 else bufs[k % 2]
        ctx.dispatch(NODE, job, {**values, "radius": radius, "inner": (k * TURN) % 1.0}, srcs=srcs,
                     dsts={"output": out} if out is not None else None)
        cur = (bufs[k % 2], gray)
