from trafilatura3d.nodes.base import Input
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import (GRAY, F, Pass, channel_toggles, gray_in, input_texture, material_inputs,
                                           passes_of, px256)
from trafilatura3d.nodes.materials.filters._effect import SURFACE_CHANNELS, SURFACE_ON, EffectNode


class SnowCover(EffectNode):
    """Snow buildup on a full material, driven by its height: fresh snow on the raised areas (above
    the local average of the height), melted slush in the lowered corners; Buildup piles snow up in
    the height, Smoothness fills the relief under it, Flakes Intensity adds flake relief to the
    normal. PBR values: snow albedo ~0.85, roughness 0.6; slush darker and glossier."""

    type = "snow_cover"
    main_input = "basecolor"  # the library graph's primary input: Relative to input follows it
    label = "Snow Cover"
    category = "Material filters/Effects (Material)"
    catalog = "snow_cover"
    inputs = {**material_inputs(), "mask": gray_in("Mask (optional)", 1.0)}
    params = {
        **channel_toggles(SURFACE_ON, SURFACE_CHANNELS),
        "fresh_snow": F(0.5, "Fresh Snow", 0.0, 1.0),
        "melted_snow": F(0.0, "Melted Snow", 0.0, 1.0),
        "buildup": F(0.5, "Buildup", 0.0, 1.0),
        "smoothness": F(0.5, "Smoothness", 0.0, 1.0),
        "flakes_intensity": F(0.5, "Flakes Intensity", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        main = passes_of(self, lambda n: Pass(n, "main", "snow_cover.wgsl", {**n.inputs, "hb": Input(GRAY)}, n.outputs))
        h = input_texture(ctx, job, "height")
        hb = gaussian(ctx, job, h[0], h[1], 12.0 * px256(job)) if h else ctx.ev.constant((0.5, 0.5, 0.5, 1.0))
        ctx.dispatch(main, job, job.values, srcs={"hb": (hb, True)})
