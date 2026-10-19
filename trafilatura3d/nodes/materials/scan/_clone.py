from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import GRAY, B, E, F, F2, F4, I, Pass, passes_of, px256

SHAPES = ("Square", "Disc")


def clone_params(threshold_label: str = "Threshold", shape_group: str = "") -> dict:
    """The clone patch parameters and defaults: a target stamp a third of the image wide (Target
    Matrix 3) on the right, its source half that size (Source Matrix 6) on the left."""
    return {
        "show_patch_areas": B(True, "Show Patch Areas"),
        "shape": E("Disc", "Shape", SHAPES, group=shape_group),
        "threshold": F(0.5, threshold_label, 0.0, 1.0, group="Edge"),
        "blur": F(0.0, "Blur", 0.0, 2.0, group="Edge"),
        "smoothness": F(0.0, "Smoothness", 0.0, 2.0, group="Edge"),
        "grid_resolution": I(9, "Grid Resolution", 1, 11, group="Edge"),
        "source_matrix": F4([6.0, 0.0, 0.0, 6.0], "Source Matrix", group="Transformations"),
        "source_offset": F2([1.0, 0.0], "Source Offset", -0.5, 0.5, group="Transformations"),
        "target_matrix": F4([3.0, 0.0, 0.0, 3.0], "Target Matrix", group="Transformations"),
        "target_offset": F2([-0.5, 0.0], "Target Offset", -0.5, 0.5, group="Transformations"),
    }


def clone_encode(node, ctx, job, analysis: dict, composite_file: str, composite_inputs: dict, outputs: dict,
                 channels=None):
    """Mask pass (`<analysis inputs>` -> grid-evaluated patch mask, `node.band_expr`), blur, composite pass."""
    mk, comp = passes_of(node, lambda n: (
        Pass(n, "mask", "clone_mask.wgsl", analysis, {"m": Output(GRAY)}, includes=("_clone.wgsl",),
             subst={"$BAND": n.band_expr}),
        Pass(n, "comp", composite_file, {**composite_inputs, "m": Input(GRAY)}, outputs, includes=("_clone.wgsl",),
             channels=channels)))
    m = ctx.scratch(job, True)
    ctx.dispatch(mk, job, job.values, dsts={"m": m})
    mb = gaussian(ctx, job, m, True, (job.values["blur"] * 4.0 + 0.7) * px256(job))
    ctx.dispatch(comp, job, job.values, srcs={"m": (mb, True)})
