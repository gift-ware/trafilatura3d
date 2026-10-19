from dataclasses import replace

from trafilatura3d.nodes.base import Output, PortType
from trafilatura3d.nodes.paint._lib import DOC, B, C, E, F, I, Filter, blur

MAX_COLORS = 8
# Default keys: black to white with the first two; the others ready to be enabled with Color Quantity.
KEYS = [((0.0, 0.0, 0.0), 0.0), ((1.0, 1.0, 1.0), 1.0), ((0.8, 0.25, 0.1), 0.5), ((0.15, 0.3, 0.75), 0.25),
        ((0.95, 0.8, 0.3), 0.75), ((0.2, 0.6, 0.3), 0.375), ((0.6, 0.2, 0.6), 0.625), ((0.5, 0.8, 0.9), 0.875)]
INPUT = "Input Transformation"


class Gradient(Filter):
    """The input made grayscale (Grayscale Mode), blurred, balanced, contrasted, inverted, then mapped through the
    gradient of the first Color Quantity colors: each color at its Position (any order), linear between the
    neighbouring positions (in linear color space with Blend Color in Linear), constant beyond the ends. Display
    Gradient (Debug) shows the gradient itself across the image (left 0, right 1)."""
    type = "paint_gradient"
    label = "Gradient"
    catalog = "painter.filter.gradient"
    doc = DOC + "filters/standard-filters/gradient"
    includes = ("color/color",)
    outputs = {"output": Output(PortType.COLOR)}
    params = {
        "display_gradient_debug": B("Display Gradient (Debug)", False),
        "color_quantity": I("Color Quantity", 2, 2, MAX_COLORS),
        "blend_color_in_linear": B("Blend Color in Linear", False),
        **{k: v for n, (col, pos) in enumerate(KEYS, 1)
           for k, v in ((f"color_{n}", C(f"Color {n}", col)), (f"color_{n}_position", F(f"Color {n} Position", pos, 0.0, 1.0)))},
        "grayscale_mode": replace(E("Grayscale Mode", ("Desaturation", "Luma", "Average", "Max", "Min"), "Luma"), group=INPUT),
        "blur_intensity": replace(F("Blur Intensity", 0.0, 0.0, 16.0), group=INPUT),
        "balance": replace(F("Balance", 0.5, 0.0, 1.0), group=INPUT),
        "contrast": replace(F("Contrast", 0.0, 0.0, 1.0), group=INPUT),
        "invert": replace(B("Invert", False), group=INPUT),
    }
    unsupported = {"Color N": f"the documentation's placeholder for the colors after Color 2: offered as Color 3 to "
                              f"Color {MAX_COLORS}",
                   "Color N Position": f"the documentation's placeholder: offered as Color 3 Position to Color "
                                       f"{MAX_COLORS} Position"}

    def encode(self, ctx, job):
        if job.values["blur_intensity"] <= 0.0:
            return ctx.dispatch(self, job, job.values)
        blurred = blur(ctx, job, job.values["blur_intensity"], True, ctx.scratch(job, False))
        ctx.dispatch(self, job, job.values, srcs={"input": (blurred, False)})
