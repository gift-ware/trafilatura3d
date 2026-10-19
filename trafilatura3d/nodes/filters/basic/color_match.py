"""Color Match (in the HCY model: hexagonal hue, chroma = max - min, Rec. 601 luma): every pixel takes
the Target colour plus its own deviation from the Source colour, scaled by the variation (Custom Color
Variation off: hue 0, chroma 1, luma 1, so hues collapse to the target's while chroma and luma keep their
spread; on: Hue in degrees / 360, Chroma, Luma). The source is the image average (GPU reduction), a parameter
or an input; the target a parameter or an input. With Use Mask the result is copied over the input through
the mask: in Parameter mode the pixels whose hue, chroma and luma lie within half of Hue range (degrees),
Chroma range and Luma range of the source; Smoothness rounds that selection (a pixel stays selected where most
of its gaussian neighbourhood of that Blur HQ size is) and Blur softens it (Blur HQ); in Input mode the Mask
input. The Mask output is that mask."""

from trafilatura3d.nodes.base import Input, Output, Param, ParamKind, PortType

from ._lib import BasicNode, helper, reduce, source, source_size
from .blur_hq import blur_hq

BLACK = (0.0, 0.0, 0.0, 0.0)


class ColorMatch(BasicNode):
    type = "color_match"
    label = "Color Match"
    category = "Filters/Adjustments"
    catalog = "color_match"
    inputs = {"input": Input(PortType.COLOR, "Input"), "source_color": Input(PortType.COLOR, "Source Color", BLACK),
              "target_color": Input(PortType.COLOR, "Target Color", BLACK),
              "mask": Input(PortType.GRAY, "Mask", (0.0, 0.0, 0.0, 1.0))}
    outputs = {"output": Output(PortType.COLOR, "Output"), "mask": Output(PortType.GRAY, "Mask")}
    params = {
        "source_color_mode": Param(ParamKind.ENUM, "Average", "Source Color Mode",
                                   choices=("Average", "Parameter", "Input")),
        "source_color": Param(ParamKind.COLOR, [0.5, 0.5, 0.5, 1.0], "Source Color"),
        "target_color_mode": Param(ParamKind.ENUM, "Parameter", "Target Color Mode",
                                   choices=("Parameter", "Image Input")),
        "target_color": Param(ParamKind.COLOR, [0.5, 0.5, 0.5, 1.0], "Target Color"),
        "custom_color_variation": Param(ParamKind.BOOL, False, "Custom Color Variation"),
        "variation_hue": Param(ParamKind.FLOAT, 0.0, "Hue", 0.0, 360.0, group="Color Variation"),
        "variation_chroma": Param(ParamKind.FLOAT, 1.0, "Chroma", 0.0, 1.0, group="Color Variation"),
        "variation_luma": Param(ParamKind.FLOAT, 1.0, "Luma", 0.0, 1.0, group="Color Variation"),
        "use_mask": Param(ParamKind.BOOL, True, "Use Mask"),
        "mask_mode": Param(ParamKind.ENUM, "Parameter", "Mask Mode", choices=("Parameter", "Input")),
        "mask_hue_range": Param(ParamKind.FLOAT, 30.0, "Hue range", 0.0, 360.0, group="Mask"),
        "mask_chroma_range": Param(ParamKind.FLOAT, 0.5, "Chroma range", 0.0, 1.0, group="Mask"),
        "mask_luma_range": Param(ParamKind.FLOAT, 0.5, "Luma range", 0.0, 1.0, group="Mask"),
        "mask_blur": Param(ParamKind.FLOAT, 0.0, "Blur", 0.0, 2.0, group="Mask"),
        "mask_smoothness": Param(ParamKind.FLOAT, 0.0, "Smoothness", 0.0, 2.0, group="Mask"),
    }
    unsupported = {"Color Variation": "a group: Hue, Chroma, Luma",
                   "Mask": "a group: Hue range, Chroma range, Luma range, Blur, Smoothness"}

    def encode(self, ctx, job):
        v = job.values
        src = source(ctx, job, "input")
        _, _, mean = reduce(ctx, job, src, source_size(job, "input"))
        color, sel = ctx.scratch(job, False), ctx.scratch(job, True)
        ctx.dispatch(PIXEL, job, v, srcs={"mean": (mean, False)}, dsts={"color": color, "selection": sel})
        if v["mask_smoothness"] > 0.0:
            sel = blur_hq(ctx, job, (sel, True), True, v["mask_smoothness"], 1, ctx.scratch(job, True))
            if v["mask_blur"] > 0.0:  # without Blur the final pass makes the cut
                cut = ctx.scratch(job, True)
                ctx.dispatch(CUT, job, {}, srcs={"selection": (sel, True)}, dsts={"output": cut})
                sel = cut
        if v["mask_blur"] > 0.0:
            sel = blur_hq(ctx, job, (sel, True), True, v["mask_blur"], 1, ctx.scratch(job, True))
        ctx.dispatch(FINAL, job, v, srcs={"color": (color, False), "selection": (sel, True)})


PIXEL = helper("color_match_pixel", ("color_match_hcl.wgsl", "color_match_pixel.wgsl"),
               {"input": Input(PortType.COLOR), "source_color": Input(PortType.COLOR),
                "target_color": Input(PortType.COLOR), "mean": Input(PortType.COLOR)},
               {"color": Output(PortType.COLOR), "selection": Output(PortType.GRAY)}, params=ColorMatch.params)
CUT = helper("color_match_cut", ("color_match_cut.wgsl", "color_match_cut_pass.wgsl"),
             {"selection": Input(PortType.GRAY)}, {"output": Output(PortType.GRAY)})
FINAL = helper("color_match_final", ("color_match_cut.wgsl", "color_match.wgsl"),
               {"input": Input(PortType.COLOR), "color": Input(PortType.COLOR), "selection": Input(PortType.GRAY),
                "mask": Input(PortType.GRAY)},
               {"output": Output(PortType.COLOR), "mask": Output(PortType.GRAY)}, params=ColorMatch.params)
