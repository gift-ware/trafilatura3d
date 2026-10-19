from trafilatura3d.nodes.base import Input, Output, PortType
from trafilatura3d.nodes.materials._base import material_inputs
from trafilatura3d.nodes.paint._lib import soft_blur
from trafilatura3d.nodes.paint.matfx._common import B, E, F, MatFX, stage


class HBAO(MatFX):
    """Horizon based ambient occlusion of the height channel (Channel Source Height) or of the channel the filter
    is on (Current Channel: its luminance as a height, for masks): 8 directions searched up to Radius (fraction
    of the texture), heights scaled by Height Depth, each rise of the horizon occludes (Intensity); Relief Balance
    weighs the near small reliefs (0) against the far large forms (1). The occlusion is blurred (Blur Intensity)
    and multiplies the channel (a white mask or the AO channel become the occlusion itself)."""
    type = "paint_material_fx_hbao"
    label = "Material FX HBAO"
    catalog = "painter.filter.matfx_hbao"
    doc = "https://experienceleague.adobe.com/en/docs/substance-3d-painter/using/effects/filters/matfx-filters/matfx-hbao"
    inputs = {"input": Input(label="Input"), **material_inputs(("height",))}
    outputs = {"output": Output()}
    params = {"blur_intensity": F("Blur Intensity", 1.0, 0.0, 16.0), "blur_wrap": B("Blur Wrap", True),
              "channel_source": E("Channel Source", ("Height", "Current Channel")),
              "height_depth": F("Height Depth", 0.1), "radius": F("Radius", 0.05, 0.0, 0.5),
              "intensity": F("Intensity", 1.0, 0.0, 4.0), "relief_balance": F("Relief Balance", 0.5)}
    unsupported = {"Use World Units": "needs the Texture Set's size in world units (scale of the UV space "
                   "on the mesh), which effects do not receive"}

    def encode(self, ctx, job):
        ao = ctx.scratch(job, True)
        ctx.dispatch(AO, job, job.values, dsts={"ao": ao})
        b = soft_blur(ctx, job, ao, True, job.values["blur_intensity"], job.values["blur_wrap"])
        ctx.dispatch(FINAL, job, job.values, srcs={"ao": (b, True)})


P = HBAO.params
AO = stage("paint_material_fx_hbao_ao", "hbao_ao.wgsl", HBAO.inputs, {"ao": Output(PortType.GRAY)},
           {k: P[k] for k in ("channel_source", "height_depth", "radius", "intensity", "relief_balance")})
FINAL = stage("paint_material_fx_hbao_final", "hbao.wgsl", {"input": Input(), "ao": Input(PortType.GRAY)}, HBAO.outputs)
