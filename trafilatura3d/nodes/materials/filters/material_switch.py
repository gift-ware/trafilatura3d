from trafilatura3d.nodes.materials._base import (B, MatNode, Pass, channel_passes, channel_toggles, material_inputs,
                                           material_outputs)
from trafilatura3d.nodes.materials.filters._common import BLEND_ON


class MaterialSwitch(MatNode):
    """Returns Material 1 or Material 2 (all channels) depending on Switch."""

    type = "material_switch"
    main_input = "diffuse_2"  # the library graph's primary input: Relative to input follows it
    label = "Material Switch"
    category = "Material filters/Blending (Material)"
    catalog = "material_switch"
    inputs = {**material_inputs(suffix="_1", label="Material 1"), **material_inputs(suffix="_2", label="Material 2")}
    outputs = material_outputs()
    params = {**channel_toggles(BLEND_ON), "switch_": B(False, "Switch")}

    def encode(self, ctx, job):
        ps = channel_passes(self, lambda n, ch: Pass(
            n, ch, "material_switch.wgsl", {f"{ch}_1": n.inputs[f"{ch}_1"], f"{ch}_2": n.inputs[f"{ch}_2"]},
            {ch: n.outputs[ch]}, channel=ch))
        for p in ps.values():
            ctx.dispatch(p, job, job.values)
