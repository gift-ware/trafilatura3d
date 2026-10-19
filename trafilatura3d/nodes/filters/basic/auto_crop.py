from trafilatura3d.nodes.base import Input, Param, ParamKind, PortType

from ._lib import BOX_ALPHA, BOX_RGB, BasicNode, helper, reduce, source, source_size

MODES = ("Crop square", "Crop auto", "Fit (Keep ratio)", "Fill (Stretch)")


class AutoCrop(BasicNode):
    """Find the content's box (pixels above 0; colour: any of R, G, B, or alpha with Use alpha;
    computed on the GPU) and re-frame it. The output size is static, so the crop modes centre the
    content without resizing it and clear everything outside the smallest square (Crop square) or
    the box itself (Crop auto); Fit scales it to the full span keeping its ratio, Fill stretches it.
    Filtering Auto: Nearest for the crop modes (moves by whole pixels), Bilinear for Fit and Fill."""

    type = "auto_crop"
    label = "Auto Crop"
    category = "Filters/Transforms"
    catalog = "auto_crop"
    inputs = {"input": Input(label="Input")}
    params = {
        "mode": Param(ParamKind.ENUM, "Crop square", "Mode", choices=MODES),
        "use_alpha": Param(ParamKind.BOOL, False, "Use alpha"),
        "filtering_mode": Param(ParamKind.ENUM, "Auto", "Filtering Mode", choices=("Nearest", "Bilinear", "Auto")),
    }

    def encode(self, ctx, job):
        src = source(ctx, job, "input")
        kind = BOX_ALPHA if job.values["use_alpha"] and not src[1] else BOX_RGB
        lo, hi, _ = reduce(ctx, job, src, source_size(job, "input"), kind)
        ctx.dispatch(FINAL, job, job.values, srcs={"lo": (lo, False), "hi": (hi, False)})


FINAL = helper("auto_crop_final", "auto_crop.wgsl",
               {"input": Input(), "lo": Input(PortType.COLOR), "hi": Input(PortType.COLOR)},
               {"output": AutoCrop.outputs["output"]}, params=AutoCrop.params)
