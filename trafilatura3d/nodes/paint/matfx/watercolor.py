from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, E, F, MatFX, stage


class Watercolor(MatFX):
    """Watercolor look of the base color: the paint bleeds (a noise warp growing with Paper Absorption), then is
    laid on paper as a transparent glaze (Paint Opacity, Beer-Lambert: color^density). The pigment accumulates
    (Accumulation Type) on the borders found by a blur over Borders Spread, darker with Borders Density: the
    coverage's borders, and the borders between colors with Show cross paint edges. The paper (generated fibers,
    or the `paper` input with Custom Paper) tints the result and granulates the pigment where it absorbs; Hide
    Paper leaves only the paint."""
    type = "paint_material_fx_watercolor"
    label = "Material FX Watercolor"
    catalog = "painter.filter.matfx_watercolor"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-watercolor"
    inputs = {**material_inputs(("basecolor",)), "paper": Input(PortType.GRAY, "Paper", (1.0, 1.0, 1.0, 1.0))}
    outputs = material_outputs(("basecolor",))
    params = {"paint_opacity": F("Paint Opacity", 0.85), "paper_absorption": F("Paper Absorption", 0.5),
              "accumulation_type": E("Accumulation Type", ("Borders", "Center", "Uniform")),
              "borders_spread": F("Borders Spread", 3.0, 0.0, 16.0), "borders_density": F("Borders Density", 0.5),
              "show_cross_paint_edges": B("Show cross paint edges", True), "hide_paper": B("Hide Paper", False),
              "custom_paper": B("Custom Paper", False)}
    unsupported = {}

    def encode(self, ctx, job):
        t = ctx.scratch(job, False)
        ctx.dispatch(BLEED, job, job.values, dsts={"bleed": t})
        b = soft_blur(ctx, job, t, False, max(job.values["borders_spread"], 0.1))
        ctx.dispatch(FINAL, job, job.values, srcs={"bleed": (t, False), "spread": (b, False)})


P = Watercolor.params
BLEED = stage("paint_material_fx_watercolor_bleed", "watercolor_bleed.wgsl", material_inputs(("basecolor",)),
              {"bleed": Output(PortType.COLOR)}, {"paper_absorption": P["paper_absorption"]})
FINAL = stage("paint_material_fx_watercolor_final", "watercolor.wgsl",
              {"paper": Watercolor.inputs["paper"], "bleed": Input(PortType.COLOR),
               "spread": Input(PortType.COLOR)}, Watercolor.outputs, P)
