from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.filters.basic._lib import reduce, source, source_size
from trafilatura3d.nodes.paint._lib import DOC, FILTERS, B, C, E, F, Filter
from trafilatura3d.nodes.paint.color._passes import helper

GRAY50 = (0.5, 0.5, 0.5, 1.0)


class ColorMatch(Filter):
    """Moves the colors from the source color to the target color keeping their details, in OKLab: each pixel's
    deviation from the source (lightness; chroma along the source's hue direction; hue across it) is laid on the
    target, scaled by Luma, Chroma and Hue when Custom Color Variation is on (1 = kept, 0 = flattened to the
    target, 2 = doubled). The source is the input's average (GPU reduction), a parameter or the Source Color input;
    the target a parameter or the Target Color input."""
    type = "paint_color_match"
    label = "Color Match"
    category = FILTERS + "/Advanced"
    catalog = "painter.filter.color_match"
    doc = DOC + "filters/advanced-filters/color-match"
    inputs = {"input": Input(PortType.COLOR, "Input"), "source_color": Input(PortType.COLOR, "Source Color", GRAY50),
              "target_color": Input(PortType.COLOR, "Target Color", GRAY50)}
    outputs = {"output": Output(PortType.COLOR)}
    params = {"source_color_mode": E("Source Color Mode", ("Average", "Parameter", "Input")),
              "source_color": C("Source Color", (0.5, 0.5, 0.5)),
              "target_color_mode": E("Target Color Mode", ("Parameter", "Input")),
              "target_color": C("Target Color", (0.7, 0.35, 0.2)),
              "custom_color_variation": B("Custom Color Variation", False),
              "hue": F("Hue", 1.0, 0.0, 2.0), "chroma": F("Chroma", 1.0, 0.0, 2.0), "luma": F("Luma", 1.0, 0.0, 2.0)}

    def encode(self, ctx, job):
        if job.values["source_color_mode"] == "Average":
            _, _, mean = reduce(ctx, job, source(ctx, job, "input"), source_size(job, "input"))
        else:
            mean = ctx.ev.constant(GRAY50)
        ctx.dispatch(FINAL, job, job.values, srcs={"mean": (mean, False)})


FINAL = helper("paint_color_match_final", "color_match.wgsl",
               {**ColorMatch.inputs, "mean": Input(PortType.COLOR)}, ColorMatch.outputs, ColorMatch.params, main=True)
