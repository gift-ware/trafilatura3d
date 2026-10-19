from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs, material_outputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, F, MatFX, stage

CHANNELS = ("roughness", "normal", "height")


class EdgeDamages(MatFX):
    """Chips on the convex edges found in the stack below (curvature of the normal channel plus the height
    channel's), smoothed by Blur Intensity, broken by a chip pattern (Level, Contrast) and scratches. Damaged areas
    get Damages Roughness, are carved Damages Depth into the height and their walls show in the normal. Base color
    is not an output: it is left as it is. Where the height stack below has no coverage the height output has
    none either, so the height channel only changes where it has data (documentation)."""
    type = "paint_material_fx_edge_damages"
    label = "Material FX Edge Damages"
    catalog = "painter.filter.matfx_edge_damages"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-edge-damages"
    inputs = material_inputs(CHANNELS)
    outputs = material_outputs(CHANNELS)
    params = {"blur_intensity": F("Blur Intensity", 3.0, 0.0, 16.0), "blur_wrap": B("Blur Wrap", True),
              "level": F("Level", 0.5), "contrast": F("Contrast", 0.6),
              "scratches_intensity": F("Scratches Intensity", 0.3), "damages_roughness": F("Damages Roughness", 0.75),
              "damages_depth": F("Damages Depth", 0.5)}
    unsupported = {}

    def encode(self, ctx, job):
        e = ctx.scratch(job, True)
        ctx.dispatch(EDGES, job, job.values, dsts={"edges": e})
        eb = soft_blur(ctx, job, e, True, job.values["blur_intensity"], job.values["blur_wrap"])
        ctx.dispatch(FINAL, job, job.values, srcs={"edges": (eb, True)})


EDGES = stage("paint_material_fx_edge_damages_edges", "edges_convexity.wgsl", material_inputs(("normal", "height")),
              {"edges": Output(PortType.GRAY)})
FINAL = stage("paint_material_fx_edge_damages_final", "edge_damages.wgsl",
              {**EdgeDamages.inputs, "edges": Input(PortType.GRAY)}, EdgeDamages.outputs, EdgeDamages.params)
