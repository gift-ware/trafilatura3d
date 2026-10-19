from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import C, F, MatFX, stage

CHANNELS = ("basecolor", "normal", "height")


class ShutLine(MatFX):
    """Panel lines along the steps of the height channel below (the shapes of the layers under the filter): a
    groove Cavity Width wide (pixels at 256 px) with walls as steep as Slope Intensity, raised lips on both sides
    (Emboss Strength), both scaled by Height Intensity into the height and the normal; the groove takes Line Color
    in the base color. Source Material Opacity is the coverage kept outside the lines (0: only the lines remain)."""
    type = "paint_material_fx_shut_line"
    label = "Material FX Shut Line"
    catalog = "painter.filter.matfx_shut_line"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-shut-line"
    inputs = material_inputs(CHANNELS)
    outputs = material_outputs(CHANNELS)
    params = {"cavity_width": F("Cavity Width", 1.0, 0.0, 8.0), "slope_intensity": F("Slope Intensity", 0.5),
              "emboss_strength": F("Emboss Strength", 0.3), "height_intensity": F("Height Intensity", 0.5),
              "line_color": C("Line Color", (0.04, 0.04, 0.04), "Material"),
              "source_material_opacity": F("Source Material Opacity", 1.0, group="Material")}
    unsupported = {}

    def encode(self, ctx, job):
        e = ctx.scratch(job, True)
        ctx.dispatch(STEPS, job, job.values, dsts={"steps": e})
        w = max(job.values["cavity_width"], 0.05)
        groove = soft_blur(ctx, job, e, True, w)
        lips = soft_blur(ctx, job, e, True, w * 2.5)
        ctx.dispatch(FINAL, job, job.values, srcs={"groove": (groove, True), "lips": (lips, True)})


STEPS = stage("paint_material_fx_shut_line_steps", "shut_line_steps.wgsl", material_inputs(("height",)),
              {"steps": Output(PortType.GRAY)})
FINAL = stage("paint_material_fx_shut_line_final", "shut_line.wgsl",
              {**ShutLine.inputs, "groove": Input(PortType.GRAY), "lips": Input(PortType.GRAY)}, ShutLine.outputs,
              ShutLine.params)
