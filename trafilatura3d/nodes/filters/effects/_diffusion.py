"""Diffusion of known pixels (Mask white, cut at 0.5) into the unknown ones, on the GPU: normalized
convolution iterated coarse to fine. Each of the Iterations passes averages the 8 neighbours at a
distance that shrinks geometrically from Distance (fraction of the image) to one pixel, weighted by
how much known value has reached them; known pixels are reset to the source every pass. Dithering
turns the neighbourhood randomly per pixel and pass (fewer passes, some noise). Shared by Diffusion
Color / Grayscale / UV; the Intensity input scales the reach locally."""

from trafilatura3d.nodes.base import Input, Output, ParamKind, PortType

from ._fx import helper

MODE_COLOR, MODE_GRAY, MODE_UV = 0, 1, 2
_UNIFORMS = {"mode": ParamKind.INT, "normal": ParamKind.INT, "alpha_mask": ParamKind.INT,
             "reach": ParamKind.FLOAT, "dither": ParamKind.INT, "pass_": ParamKind.INT}
_IN = {"source": Input(PortType.COLOR), "mask": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0)),
       "intensity": Input(PortType.GRAY, default=(1.0, 1.0, 1.0, 1.0))}
_STEP = helper("fx_diffusion_step", "_diff_step.wgsl",
               {**_IN, "v": Input(PortType.COLOR), "w": Input(PortType.GRAY)},
               {"v": Output(PortType.COLOR), "w": Output(PortType.GRAY)}, _UNIFORMS, libs=("_diff.wgsl",))


def diffuse(ctx, job, mode, iterations, distance, dither, normal=False, alpha_mask=False):
    """Returns the diffused value texture (encoded, see _diff.wgsl)."""
    w, h = job.size
    v = [ctx.scratch(job, False), ctx.scratch(job, False)]
    wt = [ctx.scratch(job, True), ctx.scratch(job, True)]
    base = {"mode": mode, "normal": int(normal), "alpha_mask": int(alpha_mask), "dither": int(dither)}
    zero_v, zero_w = ctx.ev.constant((0.0, 0.0, 0.0, 0.0)), ctx.ev.constant((0.0, 0.0, 0.0, 0.0))
    n = max(iterations, 0)
    dmax = max(distance * max(w, h) * 0.5, 1.0)
    cur_v, cur_w = (zero_v, False), (zero_w, True)
    for i in range(max(n, 1)):
        reach = dmax ** (1.0 - i / (n - 1)) if n > 1 else 1.0
        if n == 0:
            reach = 0.0  # no iteration: known pixels only
        ctx.dispatch(_STEP, job, {**base, "reach": float(reach), "pass_": i},
                     srcs={"v": cur_v, "w": cur_w}, dsts={"v": v[i % 2], "w": wt[i % 2]})
        cur_v, cur_w = (v[i % 2], False), (wt[i % 2], True)
    return cur_v[0]
