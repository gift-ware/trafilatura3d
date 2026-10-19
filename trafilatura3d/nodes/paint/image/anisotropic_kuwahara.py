from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.filters.effects._fx import gauss
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, E, F, Filter

from ._passes import helper, px


class AnisotropicKuwahara(Filter):
    """Anisotropic Kuwahara filter with polynomial sector weights (Kyprianidis et al.): each pixel looks at 8
    overlapping sectors of an ellipse aligned with the local direction and keeps the mean of the most uniform ones,
    which flattens areas into painterly strokes while keeping edges. The direction comes from the structure
    tensor of the input or of the Custom Input (Extract Direction; Sobel gradients, blurred by Tensor Smoothness
    in pixels at 256), or from the Anisotropy Angle Map (turns); Anisotropy Angle (turns) rotates it. Radius
    (pixels at 256, max 32) x the Radius Map gives the ellipse; Anisotropy stretches it with the local anisotropy
    (0: a circle, still split into sectors along the direction). Smoothness widens the sectors' overlap (more
    blending), Sharpness favours the most uniform sectors (flatter, crisper areas). 21 x 21 samples at most per
    pixel, whatever the radius."""
    type = "paint_anisotropic_kuwahara"
    label = "Anisotropic Kuwahara"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.anisotropic_kuwahara"
    doc = DOC + "filters/advanced-filters/anisotropic-kuwahara"
    inputs = {"input": Input(label="Input"),
              "radius_map": Input(PortType.GRAY, "Radius Map", (1.0, 1.0, 1.0, 1.0)),
              "custom_input": Input(PortType.COLOR, "Custom Input", (0.0, 0.0, 0.0, 1.0)),
              "anisotropy_angle_map": Input(PortType.GRAY, "Anisotropy Angle Map", (0.0, 0.0, 0.0, 1.0))}
    params = {"extract_direction": E("Extract Direction", ("Input", "Custom Input", "Anisotropy Angle Map")),
              "radius": F("Radius", 6.0, 0.0, 32.0), "smoothness": F("Smoothness", 0.5, 0.0, 1.0),
              "sharpness": F("Sharpness", 0.5, 0.0, 1.0), "tensor_smoothness": F("Tensor Smoothness", 2.0, 0.0, 16.0),
              "anisotropy": F("Anisotropy", 1.0, 0.0, 4.0), "anisotropy_angle": F("Anisotropy Angle", 0.0, 0.0, 1.0)}

    def encode(self, ctx, job):
        v = job.values
        tensor = ctx.scratch(job, False)
        ctx.dispatch(TENSOR, job, v, dsts={"out": tensor})
        if v["tensor_smoothness"] > 0.0:
            tensor = gauss(ctx, job, (tensor, False), v["tensor_smoothness"] * px(job), gray=False)
        ctx.dispatch(FINAL, job, v, srcs={"tensor": (tensor, False)})


TENSOR = helper("paint_kuwahara_tensor", "anisotropic_kuwahara_tensor.wgsl",
                {k: AnisotropicKuwahara.inputs[k] for k in ("input", "custom_input", "anisotropy_angle_map")},
                {"out": Output(PortType.COLOR)}, AnisotropicKuwahara.params)
FINAL = helper("paint_kuwahara_final", "anisotropic_kuwahara_final.wgsl",
               {"input": Input(), "radius_map": AnisotropicKuwahara.inputs["radius_map"],
                "tensor": Input(PortType.COLOR)},
               {"output": Output()}, AnisotropicKuwahara.params, main=True)
