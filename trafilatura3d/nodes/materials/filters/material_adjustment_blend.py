from trafilatura3d.nodes.materials._base import (CHANNELS, FULL, MatNode, Pass, channel_passes, channel_toggles, color_in,
                                           gray_in, material_inputs, material_outputs, E, F)
from trafilatura3d.nodes.materials.filters._common import BLEND_MODES, BLEND_ON, group, id_mask_params


def _params():
    ps = channel_toggles(BLEND_ON)
    for ch in FULL:
        g = group(ch)
        if ch == "normal":
            ps["intensity"] = F(1.0, "Intensity", 0.0, 1.0, group=g)
            continue
        ps[f"{ch}_contrast"] = F(0.0, "Contrast", -1.0, 1.0, group=g)
        ps[f"{ch}_luminosity"] = F(0.0, "Luminosity", -1.0, 1.0, group=g)
        if CHANNELS[ch][3] == 0:
            ps[f"{ch}_hue"] = F(0.5, "Hue", 0.0, 1.0, group=g)
            ps[f"{ch}_saturation"] = F(0.5, "Saturation", 0.0, 1.0, group=g)
        ps[f"{ch}_opacity"] = F(1.0, "Opacity", 0.0, 1.0, group=g)
        ps[f"{ch}_blending_mode"] = E("Normal", "Blending Mode", BLEND_MODES, group=g)
    ps.update(id_mask_params(fuzziness=0.11))
    return ps


class MaterialAdjustmentBlend(MatNode):
    """Blends a material with an adjusted copy of itself through one mask (the Greyscale Mask, or one
    color picked from the Color ID Mask): per channel Contrast / Luminosity (the
    ContrastLuminosity filter), then Hue / Saturation (the HSL node, 0.5 = unchanged) on color
    channels, laid over the original with the channel's Opacity times the mask in its Blending Mode;
    Intensity tones the normal down towards flat through the mask."""

    type = "material_adjustment_blend"
    label = "Material Adjustment Blend"
    category = "Material filters/Blending (Material)"
    catalog = "material_adjustment_blend"
    inputs = {**material_inputs(), "color_id_mask": color_in("Color ID Mask", (0.0, 0.0, 0.0, 1.0)),
              "greyscale_mask": gray_in("Greyscale Mask", 1.0)}
    outputs = material_outputs()
    params = _params()

    def encode(self, ctx, job):
        def make(n, ch):
            color = CHANNELS[ch][3] == 0
            sub = {"$hue": f"p.{ch}_hue" if color else "0.5", "$sat": f"p.{ch}_saturation" if color else "0.5"}
            if ch == "normal":
                sub.update({"$con": "0.0", "$lum": "0.0", "$opacity": "1.0", "$mode": "0"})
            else:
                sub.update({"$con": f"p.{ch}_contrast", "$lum": f"p.{ch}_luminosity", "$opacity": f"p.{ch}_opacity",
                            "$mode": f"p.{ch}_blending_mode"})
            return Pass(n, ch, "material_adjustment_blend.wgsl",
                        {ch: n.inputs[ch], "color_id_mask": n.inputs["color_id_mask"],
                         "greyscale_mask": n.inputs["greyscale_mask"]}, {ch: n.outputs[ch]}, channel=ch, subst=sub)
        for p in channel_passes(self, make).values():
            ctx.dispatch(p, job, job.values)
