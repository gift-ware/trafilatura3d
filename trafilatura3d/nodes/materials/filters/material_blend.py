from trafilatura3d.nodes.materials._base import (FULL, MatNode, Pass, channel_passes, channel_toggles, color_in, gray_in,
                                           material_inputs, material_outputs, E, F)
from trafilatura3d.nodes.materials.filters._common import BLEND_MODES, BLEND_ON, group, id_mask_params


def _params():
    ps = channel_toggles(BLEND_ON)
    for ch in FULL:
        ps[f"{ch}_opacity"] = F(1.0, "Opacity", 0.0, 1.0, group=group(ch))
        if ch != "normal":
            ps[f"{ch}_blending_mode"] = E("Normal", "Blending Mode", BLEND_MODES, group=group(ch))
    return {**ps, **id_mask_params()}


class MaterialBlend(MatNode):
    """Blends a Foreground material over a Background material, channel by channel, through the
    Greyscale Mask (1 = foreground) or one color of the ColorID map. Each channel has its own
    opacity and blending mode (Switch: the mask thresholded at 0.5); normals are interpolated as
    vectors. A channel switched off gives its default value."""

    type = "material_blend"
    label = "Material Blend"
    category = "Material filters/Blending (Material)"
    catalog = "material_blend"
    inputs = {**material_inputs(suffix="_fg", label="Foreground"), **material_inputs(suffix="_bg", label="Background"),
              "colorid": color_in("ColorID", (0.0, 0.0, 0.0, 1.0)), "greyscale_mask": gray_in("Greyscale Mask", 1.0)}
    main_input = "diffuse_bg"  # the library graph's primary input: Relative to input follows it
    outputs = material_outputs()
    params = _params()

    def encode(self, ctx, job):
        ps = channel_passes(self, lambda n, ch: Pass(
            n, ch, "material_blend.wgsl",
            {f"{ch}_fg": n.inputs[f"{ch}_fg"], f"{ch}_bg": n.inputs[f"{ch}_bg"], "colorid": n.inputs["colorid"],
             "greyscale_mask": n.inputs["greyscale_mask"]},
            {ch: n.outputs[ch]}, channel=ch, subst={"$mode": "0" if ch == "normal" else f"p.{ch}_blending_mode"}))
        for p in ps.values():
            ctx.dispatch(p, job, job.values)
