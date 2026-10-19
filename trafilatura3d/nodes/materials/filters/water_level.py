from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.filters.basic._lib import gaussian
from trafilatura3d.nodes.materials._base import (COLOR, GRAY, NORMAL_FORMATS, C, E, F, Pass, channel_toggles, gray_in,
                                           input_texture, material_inputs, material_outputs, passes_of, px256)
from trafilatura3d.nodes.materials.filters._effect import SURFACE_CHANNELS, SURFACE_ON, EffectNode


class WaterLevel(EffectNode):
    """Floods a full material up to Water Level of its height: below it a flat, glossy water surface
    over the material darkened with depth (Water Darkness) and blurred with depth (Depth Blur), with
    sludge settling deeper than Sludge Depth; wet, darker edges above the shore (Edges Wetness over
    Edges Wetness Distance); frost growing from the shore inwards with cracks, its relief in the
    Frost Normal Format. The Mask scales the water level; the Mask output is the wetness (water and
    shore)."""

    type = "water_level"
    main_input = "basecolor"  # the library graph's primary input: Relative to input follows it
    label = "Water Level"
    category = "Material filters/Effects (Material)"
    catalog = "water_level"
    inputs = {**material_inputs(), "mask": gray_in("Mask", 1.0)}
    outputs = {**material_outputs(), "mask": Output(GRAY, "Mask")}
    params = {
        **channel_toggles(SURFACE_ON, SURFACE_CHANNELS),
        "water_level": F(0.5, "Water Level", 0.0, 1.0),
        "water_darkness": F(0.5, "Water Darkness", 0.0, 1.0),
        "edges_wetness": F(0.75, "Edges Wetness", 0.0, 1.0),
        "edges_wetness_distance": F(0.1, "Edges Wetness Distance", 0.0, 1.0),
        "depth_blur_amount": F(0.5, "Depth Blur Amount", 0.0, 1.0),
        "depth_blur_opacity": F(0.75, "Depth Blur Opacity", 0.0, 1.0),
        "sludge_color": C([0.25, 0.22, 0.12], "Sludge Color"),
        "sludge_depth": F(0.5, "Sludge Depth", 0.0, 1.0),
        "sludge_opacity": F(0.5, "Sludge Opacity", 0.0, 1.0),
        "frost": F(0.0, "Frost", 0.0, 1.0),
        "frost_intensity": F(0.5, "Frost Intensity", 0.0, 1.0),
        "frost_cracks": F(0.25, "Frost Cracks", 0.0, 1.0),
        "frost_normal_format": E("OpenGL", "Frost Normal Format", NORMAL_FORMATS),
    }

    def encode(self, ctx, job):
        main = passes_of(self, lambda n: Pass(n, "main", "water_level.wgsl",
                                              {**n.inputs, "bb": Input(COLOR), "db": Input(COLOR)}, n.outputs))
        sigma = job.values["depth_blur_amount"] * 10.0 * px256(job)
        srcs = {}
        for port, name in (("basecolor", "bb"), ("diffuse", "db")):
            t = input_texture(ctx, job, port)
            srcs[name] = (gaussian(ctx, job, t[0], t[1], sigma), t[1]) if t else (ctx.ev.constant((0.5, 0.5, 0.5, 1.0)), False)
        ctx.dispatch(main, job, job.values, srcs=srcs)
