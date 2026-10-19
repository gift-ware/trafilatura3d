from trafilatura3d.nodes.materials._base import (CHANNELS, FULL, NORMAL_FORMATS, MatNode, Pass, channel_passes,
                                           channel_toggles, color_in, gray_in, material_inputs, material_outputs, C, E,
                                           F)
from trafilatura3d.nodes.materials.filters._common import BLEND_MODES, BLEND_ON, group, id_mask_params

GRAY_DEFAULTS = {"metallic": 0.0, "specular_level": 0.5}  # the other grayscale colors are white


def _params():
    ps = channel_toggles(BLEND_ON)
    for ch in FULL:
        g = group(ch)
        if ch == "normal":
            ps.update({
                "source": E("Height", "Source", ("Height", "Mask"), group=g),
                "normal_blending_mode": E("Combine", "Blending Mode", ("Combine", "Blend"), group=g),
                "normal_height_intensity": F(1.0, "Height Intensity", 0.0, 1.0, group=g),
                "normal_height_opacity": F(1.0, "Height Opacity", 0.0, 1.0, group=g),
                "format": E("OpenGL", "Format", NORMAL_FORMATS, group=g),
            })
            continue
        if CHANNELS[ch][3] == 0:
            ps[f"{ch}_color"] = C([1.0, 1.0, 1.0], "Color", group=g)
        else:
            ps[f"{ch}_color"] = F(GRAY_DEFAULTS.get(ch, 1.0), "Color", 0.0, 1.0, group=g)
        ps[f"{ch}_opacity"] = F(1.0, "Opacity", 0.0, 1.0, group=g)
        ps[f"{ch}_blending_mode"] = E("Normal", "Blending Mode", BLEND_MODES, group=g)
    ps.update(id_mask_params(fuzziness=0.11))
    return ps


class MaterialColorBlend(MatNode):
    """Blends solid colors (or values) over the channels of a material through one mask (Greyscale
    Mask or one color of the ColorID map), each channel with its own color, opacity and blending
    mode. The normal gets the relief of the material Height (or of the mask) through the mask,
    combined with the existing normal or blended over it. Defaults: white colors at full opacity
    (a grayscale channel's Color is a value)."""

    type = "material_color_blend"
    label = "Material Color Blend"
    category = "Material filters/Blending (Material)"
    catalog = "material_color_blend"
    inputs = {**material_inputs(), "colorid": color_in("ColorID", (0.0, 0.0, 0.0, 1.0)),
              "greyscale_mask": gray_in("Greyscale Mask", 1.0)}
    outputs = material_outputs()
    params = _params()

    def encode(self, ctx, job):
        def make(n, ch):
            masks = {"colorid": n.inputs["colorid"], "greyscale_mask": n.inputs["greyscale_mask"]}
            if ch == "normal":
                return Pass(n, ch, "material_color_blend_normal.wgsl",
                            {"normal": n.inputs["normal"], "height": n.inputs["height"], **masks},
                            {ch: n.outputs[ch]}, channel=ch)
            val = f"p.{ch}_color.rgb" if CHANNELS[ch][3] == 0 else f"vec3f(p.{ch}_color)"
            return Pass(n, ch, "material_color_blend.wgsl", {ch: n.inputs[ch], **masks}, {ch: n.outputs[ch]},
                        channel=ch, subst={"$value": val})
        for p in channel_passes(self, make).values():
            ctx.dispatch(p, job, job.values)
