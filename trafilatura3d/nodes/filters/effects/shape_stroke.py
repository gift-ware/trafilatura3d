from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._fx import Effect
from ._shapefx import shape_seeds

SHAPE_PARAMS = {
    "mask_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Mask Color"),
    "input_is_pre_multiplied": Param(ParamKind.BOOL, True, "Input Is Pre-Multiplied"),
    "pre_multiply_output": Param(ParamKind.BOOL, False, "Pre-Multiply Output"),
}
SHAPE_OUTPUTS = {"output": Output(PortType.COLOR, "Output"), "mask": Output(PortType.GRAY, "Mask")}


class ShapeStroke(Effect):
    """Outline around a shape: a black and white mask (shown in Mask Color) or a color image's alpha.
    A solid band from the shape's edge (exact distance, jump flooding) out to 128 x Width^2 pixels at
    256, its outer edge antialiased over a pixel, minus the shape itself. Width > 0 draws it outside the shape (under
    it), < 0 inside (over it). Opacity and (Outline) Color tint it. Output: the composite with its
    transparency, optionally premultiplied; Mask: the stroke alone."""

    type = "shape_stroke"
    label = "Shape Stroke"
    category = "Filters/Effects"
    catalog = "shape_stroke"
    libs = ("_shape.wgsl", "_shape_sd.wgsl")
    inputs = {"input": Input(label="Input")}
    outputs = SHAPE_OUTPUTS
    params = {
        "width": Param(ParamKind.FLOAT, 0.25, "Width", -1.0, 1.0),
        "opacity": Param(ParamKind.FLOAT, 1.0, "Opacity", 0.0, 1.0),
        "outline_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "(Outline) Color"),
        **SHAPE_PARAMS,
        "mask_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Mask Color"),
    }

    def encode(self, ctx, job):
        seeds = shape_seeds(ctx, job)
        ctx.dispatch(FINAL, job, job.values, srcs={"seeds": (seeds, False)})


class _Final(Effect):
    type = "fx_shape_stroke_final"
    registered = False
    libs = ("_shape.wgsl", "_shape_sd.wgsl")
    inputs = {"input": Input(), "seeds": Input(PortType.COLOR)}
    outputs = ShapeStroke.outputs
    params = ShapeStroke.params


FINAL = _Final()
