from trafilatura3d.nodes.base import Input, Output
from trafilatura3d.nodes.materials._base import (GRAY, MatNode, Pass, channel_passes, channel_toggles, material_inputs,
                                           material_outputs, mean, passes_of, E, F)
from trafilatura3d.nodes.materials.filters._common import BLEND_ON
from trafilatura3d.nodes.materials.filters.height_blend import MODES


class MaterialHeightBlend(MatNode):
    """Blends two full materials by their heightmaps (no user mask): the Top material shows where its
    height (shifted by Height Offset) rises above the Bottom's, every channel through the same mask;
    normals blend as vectors, heights combine. Albedo Matching pulls the top's colors towards the
    bottom's average color. Also outputs the mask."""

    type = "material_height_blend"
    label = "Material Height Blend"
    category = "Material filters/Effects (Material)"
    catalog = "material_height_blend"
    includes = ("_hblend.wgsl",)
    inputs = {**material_inputs(suffix="_fg", label="Top"), **material_inputs(suffix="_bg", label="Bottom")}
    main_input = "diffuse_bg"  # the library graph's primary input: Relative to input follows it
    outputs = {**material_outputs(), "mask": Output(GRAY, "Mask")}
    params = {
        **channel_toggles(BLEND_ON),
        "height_offset": F(0.5, "Height Offset", 0.0, 1.0),
        "contrast": F(0.9, "Contrast", 0.0, 1.0),
        "mode": E("Balanced height", "Mode", MODES),
        "opacity": F(1.0, "Opacity", 0.0, 1.0),
        "albedo_matching": F(0.5, "Albedo Matching", 0.0, 1.0),
    }

    def encode(self, ctx, job):
        mk = passes_of(self, lambda n: Pass(n, "mask", "material_height_blend_mask.wgsl",
                                            {"height_fg": n.inputs["height_fg"], "height_bg": n.inputs["height_bg"]},
                                            {"mask": n.outputs["mask"]}))
        ps = channel_passes(self, lambda n, ch: Pass(
            n, ch, "material_height_blend.wgsl",
            {f"{ch}_fg": n.inputs[f"{ch}_fg"], f"{ch}_bg": n.inputs[f"{ch}_bg"], "m": Input(GRAY),
             "mean_fg": Input(), "mean_bg": Input()}, {ch: n.outputs[ch]}, channel=ch))
        ctx.dispatch(mk, job, job.values)
        m = (ctx.ev.texture(job, "mask"), True)
        means = {}
        for ch in ("diffuse", "basecolor"):
            src = [job.inputs.get(f"{ch}_{s}") for s in ("fg", "bg")]
            if all(src):
                means[ch] = [mean(ctx, job, ctx.ev.texture(j, o), j.is_gray(o)) for j, o in src]
        one = ctx.ev.constant((1.0, 1.0, 1.0, 1.0))
        for ch, p in ps.items():
            fg, bg = means.get(ch, (one, one))
            ctx.dispatch(p, job, job.values, srcs={"m": m, "mean_fg": (fg, False), "mean_bg": (bg, False)})
